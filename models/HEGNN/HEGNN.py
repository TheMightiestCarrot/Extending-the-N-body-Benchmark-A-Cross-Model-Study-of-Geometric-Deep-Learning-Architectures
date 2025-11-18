# HEGNN with dual equivariant vector heads (pos & vel)
# ----------------------------------------------------
# Requirements:
#   torch, torch_scatter, e3nn
#
# Notes:
# - Uses only invariant scalars to gate vector differences, so updates are O(3)-equivariant
#   and translation-invariant.
# - Replaces the old velocity "nudge" with a proper velocity head.
# - Forward returns {"pos": node_pos, "vel": node_vel} for rollouts.

from functools import partial
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
from torch_scatter import scatter

import e3nn

from .modules import SH_INIT, SH_Msg, BaseMLP, SmoothBesselBasis


class HEGNN_Layer(nn.Module):
    """
    One message-passing layer with:
      - Scalar message trunk (invariant)
      - SH-based steerable update (all degrees)
      - Two vector heads (position & velocity) that produce equivariant updates by
        gating vector differences with invariant scalars.
    """

    def __init__(
        self,
        radial_dim: int,
        hidden_dim: int,
        sh_irreps: e3nn.o3.Irreps,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.sh_irreps = sh_irreps
        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)

        # Invariant scalar message (includes per-degree inner products)
        self.mlp_msg = MLP(
            input_dim=2 * hidden_dim + radial_dim + sh_irreps.lmax + 1,
            output_dim=hidden_dim,
            last_act=True,
        )

        # Two equivariant vector heads: each outputs 2 scalar gates per edge
        #   pos_head: weights for (x_i - x_j) and (v_i - v_j)
        #   vel_head: weights for (v_i - v_j) and (x_i - x_j)
        self.mlp_pos_basis = MLP(input_dim=hidden_dim, output_dim=2)
        self.mlp_vel_basis = MLP(input_dim=hidden_dim, output_dim=2)

        # Node feature update (invariant)
        self.mlp_node_feat = MLP(
            input_dim=hidden_dim + hidden_dim, output_dim=hidden_dim
        )

        # SH machinery for steerable residuals
        self.sh_msg = SH_Msg(sh_irreps)
        self.sh_coff = e3nn.o3.FullyConnectedTensorProduct(
            self.sh_irreps, "1x0e", self.sh_irreps, shared_weights=False
        )
        self.mlp_sh = MLP(input_dim=hidden_dim, output_dim=self.sh_coff.weight_numel)

    def forward(
        self,
        node_feat: torch.Tensor,  # (N, H)
        node_sh: torch.Tensor,  # (N, irreps.dim)
        edge_index: torch.Tensor,  # (2, E)
        diff_pos: torch.Tensor,  # (E, 3)
        diff_vel: torch.Tensor,  # (E, 3)
        radial: torch.Tensor,  # (E, R)
    ):
        msg, edge_vec_pos, edge_vec_vel, diff_sh = self.Msg(
            edge_index,
            node_feat,
            node_sh,
            diff_pos,
            diff_vel,
            radial,
        )
        msg_agg, pos_agg, vel_agg, sh_agg = self.Agg(
            edge_index, node_feat.size(0), msg, edge_vec_pos, edge_vec_vel, diff_sh
        )
        node_feat, node_sh = self.Upd(node_feat, node_sh, msg_agg, sh_agg)
        return node_feat, node_sh, pos_agg, vel_agg

    def Msg(
        self,
        edge_index: torch.Tensor,
        node_feat: torch.Tensor,
        node_sh: torch.Tensor,
        diff_pos: torch.Tensor,
        diff_vel: torch.Tensor,
        radial: torch.Tensor,
    ):
        row, col = edge_index
        sh_ip = self.sh_msg(edge_index, node_sh)  # (E, lmax+1)

        # Invariant edge message
        msg = torch.cat(
            [node_feat[row], node_feat[col], radial, sh_ip], dim=-1
        )  # (E, *)
        msg = self.mlp_msg(msg)  # (E, H)

        # Vector heads: produce invariant gates -> equivariant vectors via bases
        pos_gates = self.mlp_pos_basis(msg)  # (E, 2)
        vel_gates = self.mlp_vel_basis(msg)  # (E, 2)

        # Equivariant edge vectors
        edge_vec_pos = (
            pos_gates[:, 0:1] * diff_pos + pos_gates[:, 1:2] * diff_vel
        )  # (E, 3)
        edge_vec_vel = (
            vel_gates[:, 0:1] * diff_vel + vel_gates[:, 1:2] * diff_pos
        )  # (E, 3)

        # Steerable (all-degree) residual via CG with scalar weights
        diff_sh_edge = node_sh[row] - node_sh[col]  # (E, irreps.dim)
        one = torch.ones(
            (diff_sh_edge.size(0), 1),
            device=diff_sh_edge.device,
            dtype=diff_sh_edge.dtype,
        )
        diff_sh = self.sh_coff(diff_sh_edge, one, self.mlp_sh(msg))  # (E, irreps.dim)

        return msg, edge_vec_pos, edge_vec_vel, diff_sh

    def Agg(
        self,
        edge_index: torch.Tensor,
        dim_size: int,
        msg: torch.Tensor,
        edge_vec_pos: torch.Tensor,
        edge_vec_vel: torch.Tensor,
        diff_sh: torch.Tensor,
    ):
        row, _ = edge_index
        msg_agg = scatter(
            src=msg, index=row, dim=0, dim_size=dim_size, reduce="mean"
        )  # (N, H)
        pos_agg = scatter(
            src=edge_vec_pos, index=row, dim=0, dim_size=dim_size, reduce="mean"
        )  # (N, 3)
        vel_agg = scatter(
            src=edge_vec_vel, index=row, dim=0, dim_size=dim_size, reduce="mean"
        )  # (N, 3)
        sh_agg = scatter(
            src=diff_sh, index=row, dim=0, dim_size=dim_size, reduce="mean"
        )  # (N, irreps.dim)
        return msg_agg, pos_agg, vel_agg, sh_agg

    def Upd(
        self,
        node_feat: torch.Tensor,
        node_sh: torch.Tensor,
        msg_agg: torch.Tensor,
        sh_agg: torch.Tensor,
    ):
        node_sh = node_sh + sh_agg
        node_feat = self.mlp_node_feat(torch.cat([node_feat, msg_agg], dim=-1))
        return node_feat, node_sh


class HEGNN(nn.Module):
    """
    Full HEGNN with L layers and two equivariant vector heads.
    forward(data) expects attributes:
        - data.node_feat: (N, node_input_dim)
        - data.pos:       (N, 3)
        - data.vel:       (N, 3)
        - data.edge_index:(2, E)
        - data.rel_pos:   (E, 3)
        - data.edge_length: (E, 1) optional precomputed norms
    Returns:
        Tensor of shape (N, 3 * len(targets)) matching target order.
    """

    def __init__(
        self,
        num_layers: int,
        node_input_dim: int,
        edge_attr_dim: int,
        hidden_dim: int,
        max_ell: int,
        radial_cutoff: float = 0.4,
        envelope_power: int = 5,
        activation: nn.Module = nn.SiLU(),
        device: str = "cpu",
        targets: Sequence[str] | Tuple[str, ...] = ("pos_dt", "vel"),
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Linear(node_input_dim, hidden_dim)
        self.radial_basis = SmoothBesselBasis(
            num_basis=edge_attr_dim, cutoff=radial_cutoff, envelope_power=envelope_power
        )
        self.sh_init = SH_INIT(edge_attr_dim, hidden_dim, max_ell, activation)

        self.layers = nn.ModuleList(
            [
                HEGNN_Layer(
                    edge_attr_dim,
                    hidden_dim,
                    self.sh_init.sh_irreps,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 3),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 3),
        )
        self.to(device)
        targets = tuple(targets)
        if len(targets) == 0:
            targets = ("pos_dt", "vel")
        self.targets = targets

    @torch.no_grad()
    def get_model_size(self) -> int:
        return self.embedding.out_features

    def get_serializable_attributes(self) -> Dict[str, int]:
        radial_dim = (
            self.sh_init.mlp_sh.mlp[0].in_features - 2 * self.embedding.out_features
        )
        return {
            "num_layers": self.num_layers,
            "node_input_dim": self.embedding.in_features,
            "edge_attr_dim": radial_dim,
            "hidden_dim": self.embedding.out_features,
            "max_ell": self.sh_init.sh_irreps.lmax,
            "radial_cutoff": float(self.radial_basis.cutoff),
            "envelope_power": self.radial_basis.envelope_power,
            "targets": list(self.targets),
        }

    def forward(self, data, batch=None) -> torch.Tensor:
        node_feat = getattr(data, "node_feat", None)
        if node_feat is None and hasattr(data, "x"):
            node_feat = data.x
        inferred_feat = False
        if node_feat is None:
            node_mass = getattr(data, "mass", None)
            node_vel = getattr(data, "vel", None)
            if node_mass is not None and node_vel is not None:
                inferred_feat = True
                speed = torch.linalg.norm(node_vel, dim=-1, keepdim=True)
                kinetic = 0.5 * node_mass * speed.square()
                ones = torch.ones_like(speed)
                node_feat = torch.cat([node_mass, speed, kinetic, ones], dim=-1)
            else:
                raise ValueError(
                    "HEGNN requires 'node_feat' (or inferable features from mass & vel)."
                )
        node_pos = data.pos
        node_vel = data.vel
        edge_index = data.edge_index
        rel_pos = getattr(data, "rel_pos", None)
        edge_length = getattr(data, "edge_length", None)
        radial_cached = getattr(data, "hegnn_radial", None)
        sh_cached = getattr(data, "hegnn_diff_sh", None)

        dtype = self.embedding.weight.dtype
        if node_feat.dtype != dtype:
            node_feat = node_feat.to(dtype)
        node_feat = self.embedding(node_feat)  # (N, H)
        if node_pos.dtype != dtype:
            node_pos = node_pos.to(dtype)
        if node_vel.dtype != dtype:
            node_vel = node_vel.to(dtype)
        if inferred_feat and node_feat.dtype != dtype:
            node_feat = node_feat.to(dtype)

        if rel_pos is None:
            row, col = edge_index
            rel_pos = node_pos[row] - node_pos[col]
        elif rel_pos.dtype != dtype:
            rel_pos = rel_pos.to(dtype)

        if edge_length is None:
            edge_length = torch.norm(rel_pos, dim=-1)
        else:
            edge_length = edge_length
            if edge_length.dim() > 1:
                edge_length = edge_length.squeeze(-1)
            if edge_length.dtype != dtype:
                edge_length = edge_length.to(dtype)

        diff_vel = node_vel[edge_index[0]] - node_vel[edge_index[1]]
        if radial_cached is not None:
            radial = radial_cached
            if radial.dtype != dtype:
                radial = radial.to(dtype)
        else:
            radial = self.radial_basis(edge_length)

        # Initialize steerable features from positions
        node_sh = self.sh_init(
            node_feat, rel_pos, edge_index, radial, precomputed_sh=sh_cached
        )  # (N, irreps.dim)

        delta_pos = torch.zeros_like(node_pos)
        delta_vel = torch.zeros_like(node_vel)

        # Message passing
        for layer in self.layers:
            node_feat, node_sh, pos_inc, vel_inc = layer(
                node_feat,
                node_sh,
                edge_index,
                rel_pos,
                diff_vel,
                radial,
            )
            delta_pos = delta_pos + pos_inc
            delta_vel = delta_vel + vel_inc

        pos_input = torch.cat([node_feat, delta_pos], dim=-1)
        vel_input = torch.cat([node_feat, delta_vel, node_vel], dim=-1)

        pos_dt = self.pos_head(pos_input)
        vel_pred = self.vel_head(vel_input)

        target_map = {
            "pos_dt": pos_dt,
            "pos": node_pos + pos_dt,
            "vel": vel_pred,
            "vel_dt": vel_pred - node_vel,
        }

        outputs = []
        for target in self.targets:
            if target not in target_map:
                raise NotImplementedError(
                    f"Unsupported target '{target}' for HEGNN."
                )
            outputs.append(target_map[target])

        return torch.cat(outputs, dim=-1)
