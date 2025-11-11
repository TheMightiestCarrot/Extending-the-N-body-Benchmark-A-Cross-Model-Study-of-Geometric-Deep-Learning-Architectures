import math
from typing import Sequence

import torch
import torch.nn as nn
from torch_scatter import scatter


def gaussian_rbf(
    inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor
) -> torch.Tensor:
    """Compute standard Gaussian radial basis values."""
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    return torch.exp(coeff * torch.pow(diff, 2))


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions mirroring the original PaiNN formulation."""

    def __init__(
        self,
        n_rbf: int,
        cutoff: float,
        start: float = 0.0,
        trainable: bool = False,
    ):
        super().__init__()
        self.n_rbf = n_rbf

        offsets = torch.linspace(start, cutoff, n_rbf)
        step = (
            torch.abs(offsets[1] - offsets[0])
            if n_rbf > 1
            else offsets.new_tensor(cutoff - start)
        )
        widths = torch.full_like(offsets, step)

        if trainable:
            self.offsets = nn.Parameter(offsets)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("offsets", offsets)
            self.register_buffer("widths", widths)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return gaussian_rbf(inputs, self.offsets, self.widths)


def cosine_cutoff(input_tensor: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    """Behler-style cosine cutoff."""
    cutoff_values = 0.5 * (torch.cos(input_tensor * math.pi / cutoff) + 1.0)
    return cutoff_values * (input_tensor < cutoff).float()


class CosineCutoff(nn.Module):
    """Behler-style cosine cutoff module."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=torch.float32))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return cosine_cutoff(input_tensor, self.cutoff)


class EquivariantLinear(nn.Module):
    """Linear layer acting on the feature dimension of vector features."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, F_in)
        return torch.einsum("ncf,fo->nco", x, self.weight)


class PaiNNInteraction(nn.Module):
    """Implements the message passing block of PaiNN."""

    def __init__(self, hidden_channels: int, num_rbf: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.inter_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 3),
            nn.SiLU(),
            nn.Linear(hidden_channels * 3, hidden_channels * 3),
        )
        self.filter_network = nn.Sequential(
            nn.Linear(num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        unit_vectors: torch.Tensor,
        cutoff_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = q.size(0)
        source = edge_index[1]
        target = edge_index[0]

        filters = self.filter_network(rbf) * cutoff_values.unsqueeze(-1)
        filter_q, filter_r, filter_mu = filters.chunk(3, dim=-1)

        x = self.inter_mlp(q)
        x_q, x_r, x_mu = x.chunk(3, dim=-1)

        x_q_src = x_q[source] * filter_q
        scalar_msg = scatter(
            x_q_src, target, dim=0, dim_size=num_nodes, reduce="sum"
        )

        x_r_src = x_r[source] * filter_r
        x_mu_src = x_mu[source] * filter_mu
        mu_src = mu[source]

        vec_new = unit_vectors.unsqueeze(-1) * x_r_src.unsqueeze(-2)
        vec_propagated = mu_src * x_mu_src.unsqueeze(-2)
        vector_msg = scatter(
            vec_new + vec_propagated,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )

        q = q + scalar_msg
        mu = mu + vector_msg
        return q, mu


class PaiNNMixing(nn.Module):
    """Implements the equivariant mixing block."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.vec_linear = EquivariantLinear(hidden_channels, hidden_channels * 2)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 3),
            nn.SiLU(),
            nn.Linear(hidden_channels * 3, hidden_channels * 3),
        )

    def forward(self, q: torch.Tensor, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_cat = self.vec_linear(mu)
        mu_v, mu_w = mu_cat.chunk(2, dim=-1)

        mu_v_norm = torch.sqrt((mu_v**2).sum(dim=1) + 1e-8)
        scalar_input = torch.cat([q, mu_v_norm], dim=-1)
        delta = self.scalar_mlp(scalar_input)
        dq, dmu_scale, dqmu = delta.chunk(3, dim=-1)

        inner = (mu_v * mu_w).sum(dim=1)
        q = q + dq + dqmu * inner
        mu = mu + mu_w * dmu_scale.unsqueeze(1)
        return q, mu


class PaiNNBlock(nn.Module):
    """Full interaction + mixing block."""

    def __init__(self, hidden_channels: int, num_rbf: int):
        super().__init__()
        self.interaction = PaiNNInteraction(hidden_channels, num_rbf)
        self.mixing = PaiNNMixing(hidden_channels)

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        unit_vectors: torch.Tensor,
        cutoff_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, mu = self.interaction(q, mu, edge_index, rbf, unit_vectors, cutoff_values)
        q, mu = self.mixing(q, mu)
        return q, mu


class PaiNNReadout(nn.Module):
    """Vector readout head gated by scalar context."""

    def __init__(self, hidden_channels: int, vector_outputs: int = 1):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.vector_linear = EquivariantLinear(hidden_channels, hidden_channels)
        self.out_linear = EquivariantLinear(hidden_channels, vector_outputs)

    def forward(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        gate = self.gate_mlp(q).unsqueeze(1)
        mu_gated = mu * gate
        mu_proj = self.vector_linear(mu_gated)
        out = self.out_linear(mu_proj)
        return out  # (N, 3, vector_outputs)


class PaiNN(nn.Module):
    """PaiNN model adapted for N-body rollout targets (pos_dt + vel)."""

    def __init__(
        self,
        hidden_features: int = 128,
        num_layers: int = 6,
        num_rbf: int = 64,
        cutoff: float = 10.0,
        targets: Sequence[str] | None = None,
        use_velocity_input: bool = True,
        include_velocity_norm: bool = True,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.targets = tuple(targets or ("pos_dt", "vel"))
        self.use_velocity_input = use_velocity_input
        self.include_velocity_norm = include_velocity_norm

        scalar_inputs = 1  # mass
        if include_velocity_norm:
            scalar_inputs += 1

        self.scalar_embedding = nn.Sequential(
            nn.Linear(scalar_inputs, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )

        if use_velocity_input:
            self.vector_gate = nn.Sequential(
                nn.Linear(scalar_inputs, hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, hidden_features),
            )
        else:
            self.vector_gate = None

        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.blocks = nn.ModuleList(
            PaiNNBlock(hidden_features, num_rbf) for _ in range(num_layers)
        )

        # Heads (each output is a single 3D vector)
        self.pos_head = PaiNNReadout(hidden_features, vector_outputs=1)
        self.vel_head = PaiNNReadout(hidden_features, vector_outputs=1)

    def _prepare_scalar_inputs(self, graph) -> torch.Tensor:
        features = [graph.mass]
        if self.include_velocity_norm:
            vel_norm = torch.linalg.norm(graph.vel, dim=-1, keepdim=True)
            features.append(vel_norm)
        return torch.cat(features, dim=-1)

    @staticmethod
    def _unit_vectors(edge_vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        distances = torch.linalg.norm(edge_vectors, dim=-1, keepdim=True)
        safe_denominator = torch.where(
            distances > eps, distances, torch.ones_like(distances)
        )
        unit_vec = edge_vectors / safe_denominator
        unit_vec = torch.where(distances > eps, unit_vec, torch.zeros_like(unit_vec))
        return unit_vec

    def forward(self, graph) -> torch.Tensor:
        pos = graph.pos
        vel = graph.vel
        mass = graph.mass
        edge_index = graph.edge_index
        if edge_index is None:
            raise ValueError("edge_index must be provided by the dataloader for PaiNN.")

        scalar_inputs = self._prepare_scalar_inputs(graph)
        q = self.scalar_embedding(scalar_inputs)

        if self.use_velocity_input:
            velocity_scale = self.vector_gate(scalar_inputs)
            mu = vel.unsqueeze(-1) * velocity_scale.unsqueeze(1)
        else:
            mu = torch.zeros(
                pos.size(0),
                3,
                self.hidden_features,
                device=pos.device,
                dtype=pos.dtype,
            )

        row, col = edge_index
        edge_vectors = pos[col] - pos[row]
        distances = torch.linalg.norm(edge_vectors, dim=-1)
        unit_vectors = self._unit_vectors(edge_vectors)
        rbf = self.rbf(distances)
        cutoff_values = self.cutoff_fn(distances)

        for block in self.blocks:
            q, mu = block(q, mu, edge_index, rbf, unit_vectors, cutoff_values)

        pos_delta = self.pos_head(q, mu).squeeze(-1)
        vel_delta = self.vel_head(q, mu).squeeze(-1)
        vel_pred = vel + vel_delta

        outputs = []
        for target in self.targets:
            if target == "pos_dt":
                outputs.append(pos_delta)
            elif target == "vel":
                outputs.append(vel_pred)
            else:
                raise NotImplementedError(f"Unsupported target '{target}' for PaiNN.")

        return torch.cat(outputs, dim=-1)

    def get_model_size(self) -> int:
        return self.hidden_features

    def get_serializable_attributes(self) -> dict:
        return {
            "hidden_features": self.hidden_features,
            "num_layers": self.num_layers,
            "num_rbf": self.num_rbf,
            "cutoff": self.cutoff,
            "targets": list(self.targets),
            "use_velocity_input": self.use_velocity_input,
            "include_velocity_norm": self.include_velocity_norm,
            "num_params": sum(p.numel() for p in self.parameters()),
        }
