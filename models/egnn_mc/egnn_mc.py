from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

__all__ = ["EGNNMultiChannel"]


def _make_activation(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == "silu":
        return nn.SiLU
    if name == "relu":
        return nn.ReLU
    if name in {"leaky_relu", "lrelu"}:
        return lambda: nn.LeakyReLU(negative_slope=0.2)
    raise ValueError(f"Unsupported activation '{name}'.")


def _unsorted_segment_sum(
    data: Tensor, segment_ids: Tensor, num_segments: int
) -> Tensor:
    result_shape = (num_segments,) + data.shape[1:]
    result = data.new_zeros(result_shape)
    expanded_index = segment_ids.view(-1, *([1] * (data.dim() - 1))).expand_as(data)
    result.scatter_add_(0, expanded_index, data)
    return result


def _unsorted_segment_mean(
    data: Tensor, segment_ids: Tensor, num_segments: int
) -> Tensor:
    result_shape = (num_segments,) + data.shape[1:]
    result = data.new_zeros(result_shape)
    count = data.new_zeros(result_shape)
    expanded_index = segment_ids.view(-1, *([1] * (data.dim() - 1))).expand_as(data)
    result.scatter_add_(0, expanded_index, data)
    count.scatter_add_(0, expanded_index, torch.ones_like(data))
    return result / count.clamp(min=1)


class _EGNNMessageBlock(nn.Module):
    """Differentiable equivariant block adapted from the channels_egnn project."""

    def __init__(
        self,
        node_input_dim: int,
        node_output_dim: int,
        hidden_edge_dim: int,
        hidden_node_dim: int,
        hidden_coord_dim: int,
        *,
        edge_attr_dim: int = 0,
        act_factory: Callable[[], nn.Module] = nn.SiLU,
        coords_weight: float = 1.0,
        recurrent: bool = True,
        attention: bool = False,
        norm_diff: bool = False,
        tanh: bool = False,
        num_vectors_in: int = 1,
        num_vectors_out: int = 1,
    ) -> None:
        super().__init__()
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.num_vectors_in = num_vectors_in
        self.num_vectors_out = num_vectors_out

        edge_input_dim = node_input_dim * 2 + num_vectors_in + edge_attr_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_edge_dim),
            act_factory(),
            nn.Linear(hidden_edge_dim, hidden_edge_dim),
            act_factory(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_edge_dim + node_input_dim, hidden_node_dim),
            act_factory(),
            nn.Linear(hidden_node_dim, node_output_dim),
        )

        coord_layers: list[nn.Module] = [
            nn.Linear(hidden_edge_dim, hidden_coord_dim),
            act_factory(),
            nn.Linear(hidden_coord_dim, num_vectors_in * num_vectors_out, bias=False),
        ]
        nn.init.xavier_uniform_(coord_layers[-1].weight, gain=0.001)
        if tanh:
            coord_layers.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_layers)

        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(node_input_dim, hidden_coord_dim),
            act_factory(),
            nn.Linear(hidden_coord_dim, num_vectors_in * num_vectors_out),
        )

        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_edge_dim, 1), nn.Sigmoid())

    def edge_model(
        self,
        source: Tensor,
        target: Tensor,
        radial: Tensor,
        edge_attr: Tensor | None,
    ) -> Tensor:
        pieces = [source, target, radial]
        if edge_attr is not None:
            pieces.append(edge_attr)
        edge_input = torch.cat(pieces, dim=-1)
        messages = self.edge_mlp(edge_input)
        if self.attention:
            messages = messages * self.att_mlp(messages)
        return messages

    def node_model(
        self, node_state: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        row, _ = edge_index
        aggregated = _unsorted_segment_mean(edge_attr, row, num_segments=node_state.size(0))
        node_input = torch.cat([node_state, aggregated], dim=-1)
        node_output = self.node_mlp(node_input)
        if self.recurrent:
            node_output = node_state + node_output
        return node_output

    def coord_model(
        self,
        coord: Tensor,
        edge_index: Tensor,
        coord_diff: Tensor,
        edge_feat: Tensor,
    ) -> Tensor:
        row, _ = edge_index
        coord_matrix = self.coord_mlp(edge_feat).view(
            -1, self.num_vectors_in, self.num_vectors_out
        )
        if coord_diff.dim() == 2:
            coord_diff = coord_diff.unsqueeze(-1)
            coord = coord.unsqueeze(-1).repeat(1, 1, self.num_vectors_out)
        trans = torch.einsum("bij,bci->bcj", coord_matrix, coord_diff)
        trans = torch.clamp(trans, min=-100.0, max=100.0)
        aggregated = _unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + aggregated * self.coords_weight
        return coord

    def coord2radial(self, edge_index: Tensor, coord: Tensor) -> tuple[Tensor, Tensor]:
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, dim=1).unsqueeze(1)
        if self.norm_diff:
            norm = torch.sqrt(radial).clamp_min(1.0)
            coord_diff = coord_diff / norm
        if radial.dim() == 3:
            radial = radial.squeeze(1)
        return radial, coord_diff

    def forward(
        self,
        node_state: Tensor,
        edge_index: Tensor,
        coord: Tensor,
        velocity: Tensor,
        edge_attr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(node_state[edge_index[0]], node_state[edge_index[1]], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        coord_vel_matrix = self.coord_mlp_vel(node_state).view(
            -1, self.num_vectors_in, self.num_vectors_out
        )
        if velocity.dim() == 2:
            velocity = velocity.unsqueeze(-1)
        coord = coord + torch.einsum("bij,bci->bcj", coord_vel_matrix, velocity)

        node_state = self.node_model(node_state, edge_index, edge_feat)
        return node_state, coord, velocity


class _VectorHead(nn.Module):
    """Simple two-layer head producing a 3D vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        act_factory: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_factory(),
            nn.Linear(hidden_dim, hidden_dim),
            act_factory(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EGNNMultiChannel(nn.Module):
    """EGNN variant with learnable vector heads for multiple targets."""

    def __init__(
        self,
        *,
        node_input_dim: int = 2,
        edge_attr_dim: int = 3,
        hidden_node_dim: int = 128,
        hidden_edge_dim: int = 128,
        hidden_coord_dim: int = 128,
        num_layers: int = 4,
        target_names: Sequence[str] | None = None,
        activation: str = "silu",
        coords_weight: float = 1.0,
        recurrent: bool = True,
        norm_diff: bool = False,
        tanh: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if target_names is None or len(target_names) == 0:
            raise ValueError("EGNNMultiChannel requires at least one target.")

        self.device = torch.device(device)
        self.target_names = tuple(target_names)
        self.hidden_node_dim = hidden_node_dim
        self.num_layers = num_layers

        act_factory = _make_activation(activation)

        self.embedding = nn.Linear(node_input_dim, hidden_node_dim)
        layers: list[_EGNNMessageBlock] = []
        for _ in range(num_layers):
            layers.append(
                _EGNNMessageBlock(
                    hidden_node_dim,
                    hidden_node_dim,
                    hidden_edge_dim,
                    hidden_node_dim,
                    hidden_coord_dim,
                    edge_attr_dim=edge_attr_dim,
                    act_factory=act_factory,
                    coords_weight=coords_weight,
                    recurrent=recurrent,
                    attention=False,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    num_vectors_in=1,
                    num_vectors_out=1,
                )
            )
        self.layers = nn.ModuleList(layers)

        head_input_dim = hidden_node_dim + 6  # pos_dt (3) + vel (3)
        self.heads = nn.ModuleList(
            [_VectorHead(head_input_dim, hidden_node_dim, act_factory) for _ in self.target_names]
        )

        self.to(self.device)

    def forward(self, graph: Data) -> Tensor:
        dtype = self.embedding.weight.dtype
        node_feat = graph.x.to(dtype=dtype)
        node_pos = graph.pos.to(dtype=dtype)
        node_vel = graph.vel.to(dtype=dtype)
        edge_index = graph.edge_index
        edge_attr = getattr(graph, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(dtype=dtype)

        h = self.embedding(node_feat)
        coord = node_pos
        vel_state = node_vel

        for layer in self.layers:
            h, coord, vel_state = layer(h, edge_index, coord, vel_state, edge_attr=edge_attr)

        coord = coord.squeeze(-1) if coord.dim() == 3 else coord
        vel_state = vel_state.squeeze(-1) if vel_state.dim() == 3 else vel_state
        pos_dt = coord - node_pos

        head_input = torch.cat([h, pos_dt, vel_state], dim=-1)
        outputs = [head(head_input) for head in self.heads]
        return torch.cat(outputs, dim=-1)

    def get_serializable_attributes(self) -> dict[str, object]:
        return {
            "hidden_node_dim": self.hidden_node_dim,
            "num_layers": self.num_layers,
            "target_names": list(self.target_names),
            "num_params": sum(p.numel() for p in self.parameters()),
        }

    def get_model_size(self) -> int:
        return self.hidden_node_dim
