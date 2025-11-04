# gmn.py
import torch
from torch import nn
import torch.nn.functional as F


# ---------- utilities ----------
def unsorted_segment_sum(data, segment_ids, num_segments):
    """Replicates TensorFlow's unsorted_segment_sum for 2D tensors."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    idx = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, idx, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    """Mean with implicit counts via scatter_add."""
    result_shape = (num_segments, data.size(1))
    idx = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, idx, data)
    count.scatter_add_(0, idx, torch.ones_like(data))
    return result / count.clamp(min=1)


# ---------- GMN layer ----------
class GMNLayer(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
        norm_diff=False,
        tanh=False,
        learnable=False,
    ):
        """
        Graph Mechanics Networks layer.

        Args:
            input_nf: node feature dim in
            output_nf: node feature dim out
            hidden_nf: hidden width
            edges_in_d: edge feature dim in
            nodes_att_dim: extra per-node attrs (optional)
            act_fn: activation function
            recurrent: residual on x in node_model
            coords_weight: scaling for coordinate updates
            attention: optional edge attention
            norm_diff: normalize coord differences
            tanh: optional Tanh on coord_mlp output
            learnable: use learnable forward kinematics
        """
        super().__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.learnable = learnable
        edge_coords_nf = 1  # squared distance

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        n_basis_stick = 1
        n_basis_hinge = 3

        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )
        self.coord_mlp_w_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )
        self.center_mlp = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, input_nf),
        )

        self.f_stick_mlp = nn.Sequential(
            nn.Linear(n_basis_stick * n_basis_stick, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, n_basis_stick),
        )
        self.f_hinge_mlp = nn.Sequential(
            nn.Linear(n_basis_hinge * n_basis_hinge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, n_basis_hinge),
        )

        if self.learnable:
            self.stick_v_fk_mlp = nn.Sequential(
                nn.Linear(3 * 3, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 3),
            )
            self.hinge_v_fk_mlp = nn.Sequential(
                nn.Linear(3 * 3, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 3),
            )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1) * 3)
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    # --- message passing pieces ---
    def edge_model(self, source, target, radial, edge_attr):
        out = torch.cat(
            (
                [source, target, radial]
                if edge_attr is None
                else [source, target, radial, edge_attr]
            ),
            dim=1,
        )
        out = self.edge_mlp(out)
        if self.attention:
            out = out * self.att_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr, others=None):
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        if others is not None:
            agg = torch.cat([others, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, _ = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        f = agg * self.coords_weight
        return coord + f, f

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
        return radial, coord_diff

    # --- rigid body updates ---
    def update(self, x, v, f, h, node_index, type="Isolated"):
        """
        Update X and V given the current X, V, and force F
        Args:
            x: [N, 3] positions
            v: [N, 3] velocities
            f: [N, 3] forces (from message passing)
            h: [N, hidden] node features
            node_index: indices describing the rigid object
            type: 'Isolated' | 'Stick' | 'Hinge'
        """
        if type == "Isolated":
            _x, _v, _f, _h = x[node_index], v[node_index], f[node_index], h[node_index]
            _a = _f / 1.0
            _v = self.coord_mlp_vel(_h) * _v + _a
            _x = _x + _v
            x[node_index], v[node_index] = _x, _v
            return x, v

        elif type == "Stick":
            id1, id2 = node_index[..., 0], node_index[..., 1]
            _x1, _v1, _f1, _h1 = x[id1], v[id1], f[id1], h[id1]
            _x2, _v2, _f2, _h2 = x[id2], v[id2], f[id2], h[id2]
            _x0, _v0 = (_x1 + _x2) / 2, (_v1 + _v2) / 2

            def apply_f(cur_x, cur_v, cur_f):
                _X = torch.stack((cur_f,), dim=-1)
                _inv = torch.einsum("bij,bjk->bik", _X.permute(0, 2, 1), _X).reshape(
                    _X.size(0), -1
                )
                _inv = F.normalize(_inv, dim=-1, p=2)
                msg = self.f_stick_mlp(_inv)
                return torch.einsum("bij,bjk->bik", _X, msg.unsqueeze(-1)).squeeze(-1)

            messages = [apply_f(_x1, _v1, _f1), apply_f(_x2, _v2, _f2)]
            _a0 = sum(messages) / len(messages)

            if self.learnable:
                _X = torch.stack((_a0, _x1 - _x0, _f1), dim=-1)
                _inv = torch.einsum("bij,bjk->bik", _X.permute(0, 2, 1), _X).reshape(
                    _X.size(0), -1
                )
                _inv = F.normalize(_inv, dim=-1, p=2)
                delta_v = torch.einsum(
                    "bij,bjk->bik", _X, self.stick_v_fk_mlp(_inv).unsqueeze(-1)
                ).squeeze(-1)
                _v1 = self.coord_mlp_vel(_h1) * _v1 + delta_v
                _x1 = _x1 + _v1

                _X = torch.stack((_a0, _x2 - _x0, _f2), dim=-1)
                _inv = torch.einsum("bij,bjk->bik", _X.permute(0, 2, 1), _X).reshape(
                    _X.size(0), -1
                )
                _inv = F.normalize(_inv, dim=-1, p=2)
                delta_v = torch.einsum(
                    "bij,bjk->bik", _X, self.stick_v_fk_mlp(_inv).unsqueeze(-1)
                ).squeeze(-1)
                _v2 = self.coord_mlp_vel(_h2) * _v2 + delta_v
                _x2 = _x2 + _v2

            else:
                J = torch.sum((_x1 - _x0) ** 2, dim=-1, keepdim=True) + torch.sum(
                    (_x2 - _x0) ** 2, dim=-1, keepdim=True
                )
                _beta1 = torch.cross((_x1 - _x0), _f1) / J
                _beta2 = torch.cross((_x2 - _x0), _f2) / J
                _beta = _beta1 + _beta2

                _r, _v = (_x1 - _x2) / 2, (_v1 - _v2) / 2
                _w = torch.cross(F.normalize(_r, dim=-1, p=2), _v) / torch.norm(
                    _r, dim=-1, p=2, keepdim=True
                ).clamp_min(1e-5)

                _h_c = self.center_mlp(_h1) + self.center_mlp(_h2)

                _w = self.coord_mlp_w_vel(_h_c) * _w + _beta
                _v0 = self.coord_mlp_vel(_h_c) * _v0 + _a0
                _x0 = _x0 + _v0

                _theta = torch.norm(_w, p=2, dim=-1)
                rot = self.compute_rotation_matrix(_theta, F.normalize(_w, p=2, dim=-1))
                _r = torch.einsum("bij,bjk->bik", rot, _r.unsqueeze(-1)).squeeze(-1)
                _x1, _x2 = _x0 + _r, _x0 - _r
                _v1, _v2 = _v0 + torch.cross(_w, _r), _v0 + torch.cross(_w, -_r)

            x[id1], x[id2] = _x1, _x2
            v[id1], v[id2] = _v1, _v2
            return x, v

        elif type == "Hinge":
            id0, id1, id2 = node_index[..., 0], node_index[..., 1], node_index[..., 2]
            _x0, _v0, _f0, _h0 = x[id0], v[id0], f[id0], h[id0]
            _x1, _v1, _f1, _h1 = x[id1], v[id1], f[id1], h[id1]
            _x2, _v2, _f2, _h2 = x[id2], v[id2], f[id2], h[id2]

            def apply_f(cur_x, cur_v, cur_f):
                _X = torch.stack((cur_f, cur_x - _x0, cur_v - _v0), dim=-1)
                _inv = torch.einsum("bij,bjk->bik", _X.permute(0, 2, 1), _X).reshape(
                    _X.size(0), -1
                )
                _inv = F.normalize(_inv, dim=-1, p=2)
                msg = self.f_hinge_mlp(_inv)
                return torch.einsum("bij,bjk->bik", _X, msg.unsqueeze(-1)).squeeze(-1)

            messages = [
                apply_f(_x0, _v0, _f0),
                apply_f(_x1, _v1, _f1),
                apply_f(_x2, _v2, _f2),
            ]
            _a0 = sum(messages) / len(messages)

            def apply_g(cur_x, cur_f):
                return torch.cross(cur_x - _x0, cur_f - _a0) / torch.sum(
                    (cur_x - _x0) ** 2, dim=-1, keepdim=True
                )

            _beta1, _beta2 = apply_g(_x1, _f1), apply_g(_x2, _f2)

            def compute_c_metrics(cur_x, cur_v):
                cur_r, relative_v = cur_x - _x0, cur_v - _v0
                cur_w = torch.cross(
                    F.normalize(cur_r, dim=-1, p=2), relative_v
                ) / torch.norm(cur_r, dim=-1, p=2, keepdim=True).clamp_min(1e-5)
                return cur_r, cur_w

            _r1, _w1 = compute_c_metrics(_x1, _v1)
            _r2, _w2 = compute_c_metrics(_x2, _v2)

            _h_c = self.center_mlp(_h1) + self.center_mlp(_h2)
            _v0 = self.coord_mlp_vel(_h_c) * _v0 + _a0
            _x0 = _x0 + _v0

            if self.learnable:
                _X = torch.stack((_a0, _x1 - _x0, _f1), dim=-1)
                _inv = torch.einsum("bij,bjk->bik", _X.permute(0, 2, 1), _X).reshape(
                    _X.size(0), -1
                )
                _inv = F.normalize(_inv, dim=-1, p=2)
                delta_v = torch.einsum(
                    "bij,bjk->bik", _X, self.hinge_v_fk_mlp(_inv).unsqueeze(-1)
                ).squeeze(-1)
                _v1 = self.coord_mlp_vel(_h1) * _v1 + delta_v
                _x1 = _x1 + _v1

                _X = torch.stack((_a0, _x2 - _x0, _f2), dim=-1)
                _inv = torch.einsum("bij,bjk->bik", _X.permute(0, 2, 1), _X).reshape(
                    _X.size(0), -1
                )
                _inv = F.normalize(_inv, dim=-1, p=2)
                delta_v = torch.einsum(
                    "bij,bjk->bik", _X, self.hinge_v_fk_mlp(_inv).unsqueeze(-1)
                ).squeeze(-1)
                _v2 = self.coord_mlp_vel(_h2) * _v2 + delta_v
                _x2 = _x2 + _v2
            else:

                def update_c_metrics(rot_func, cur_w, cur_beta, cur_r, cur_h):
                    cur_w = self.coord_mlp_w_vel(cur_h) * cur_w + cur_beta
                    cur_theta = torch.norm(cur_w, p=2, dim=-1)
                    cur_rot = rot_func(cur_theta, F.normalize(cur_w, p=2, dim=-1))
                    cur_r = torch.einsum(
                        "bij,bjk->bik", cur_rot, cur_r.unsqueeze(-1)
                    ).squeeze(-1)
                    return cur_r, cur_w

                _r1, _w1 = update_c_metrics(
                    self.compute_rotation_matrix, _w1, _beta1, _r1, _h1
                )
                _r2, _w2 = update_c_metrics(
                    self.compute_rotation_matrix, _w2, _beta2, _r2, _h2
                )

                _x1, _x2 = _x0 + _r1, _x0 + _r2
                _v1, _v2 = _v0 + torch.cross(_w1, _r1), _v0 + torch.cross(_w2, _r2)

            x[id0], x[id1], x[id2] = _x0, _x1, _x2
            v[id0], v[id1], v[id2] = _v0, _v1, _v2
            return x, v

        else:
            raise NotImplementedError(f"Unknown object type: {type}")

    def forward(self, h, edge_index, x, v, cfg, edge_attr=None, node_attr=None):
        """
        Args:
            h: [N, hidden] node features
            edge_index: LongTensor [2, M]
            x: [N, 3] coordinates
            v: [N, 3] velocities
            cfg: dict mapping rigid type -> indices
                 keys: 'Isolated', 'Stick', 'Hinge'
                 values: indices shaped [K], [K,2], [K,3] respectively
            edge_attr: [M, E] optional edge features
            node_attr: [N, F] optional original node features
        Returns:
            h, x, v, edge_attr (updated)
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, x)
        edge_feat = self.edge_model(
            h[row], h[col], radial, edge_attr
        )  # invariant messages
        _, f = self.coord_model(
            x, edge_index, coord_diff, edge_feat
        )  # equivariant "force"

        # Update each rigid object group
        for type in cfg:
            x, v = self.update(x, v, f, h, node_index=cfg[type], type=type)

        h, _ = self.node_model(h, edge_index, edge_feat, node_attr, others=h)
        return h, x, v, edge_attr

    @staticmethod
    def compute_rotation_matrix(theta, d):
        """Rodrigues' rotation formula batched for unit axis d and angle theta."""
        x, y, z = torch.unbind(d, dim=-1)
        cos, sin = torch.cos(theta), torch.sin(theta)
        ret = torch.stack(
            (
                cos + (1 - cos) * x * x,
                (1 - cos) * x * y - sin * z,
                (1 - cos) * x * z + sin * y,
                (1 - cos) * x * y + sin * z,
                cos + (1 - cos) * y * y,
                (1 - cos) * y * z - sin * x,
                (1 - cos) * x * z - sin * y,
                (1 - cos) * y * z + sin * x,
                cos + (1 - cos) * z * z,
            ),
            dim=-1,
        )
        return ret.reshape(-1, 3, 3)


# ---------- GMN model ----------
class GMN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        recurrent=False,
        norm_diff=False,
        tanh=False,
        learnable=False,
    ):
        """
        Graph Mechanics Networks (stack of GMNLayer).
        """
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.layers = nn.ModuleList(
            [
                GMNLayer(
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    coords_weight=coords_weight,
                    recurrent=recurrent,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    learnable=learnable,
                )
                for _ in range(n_layers)
            ]
        )
        self.to(self.device)

    def forward(self, h, x, edge_index, v, cfg, edge_attr=None, node_attr=None):
        """
        Args:
            h: [N, in_node_nf] input node features
            x: [N, 3] coordinates
            edge_index: [2, M] graph connectivity
            v: [N, 3] velocities
            cfg: dict with keys 'Isolated', 'Stick', 'Hinge' and index tensors
            edge_attr: [M, in_edge_nf] edge attributes (optional)
            node_attr: [N, in_node_nf] original node attrs for node_model (optional)
        Returns:
            x, v: updated coordinates and velocities
        """
        h = self.embedding(h)
        for layer in self.layers:
            h, x, v, _ = layer(
                h, edge_index, x, v, cfg, edge_attr=edge_attr, node_attr=node_attr
            )
        return x, v
