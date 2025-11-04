import time

import torch
import torch.nn.functional as F
from torch import nn

from models.CGENN.algebra.cliffordalgebra import CliffordAlgebra
from models.CGENN.gp import SteerableGeometricProductLayer
from models.CGENN.linear import MVLinear
from models.CGENN.mvlayernorm import MVLayerNorm
from models.CGENN.mvsilu import MVSiLU


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class CEMLP(nn.Module):
    def __init__(
        self,
        algebra,
        metric,
        in_features,
        hidden_features,
        out_features,
        n_layers=2,
        normalization_init=0,
    ):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.metric = metric

        print("in_features", in_features)
        print("out_features", out_features)

        layers = []

        # Add geometric product layers.
        for i in range(n_layers - 1):
            layers.append(
                nn.Sequential(
                    MVLinear(self.algebra, in_features, hidden_features),
                    MVSiLU(self.algebra, self.metric, hidden_features),
                    SteerableGeometricProductLayer(
                        self.algebra,
                        self.metric,
                        hidden_features,
                        normalization_init=normalization_init,
                    ),
                    MVLayerNorm(self.algebra, self.metric, hidden_features),
                )
            )
            in_features = hidden_features

        # Add final layer.
        layers.append(
            nn.Sequential(
                MVLinear(self.algebra, in_features, out_features),
                MVSiLU(self.algebra, self.metric, out_features),
                SteerableGeometricProductLayer(
                    self.algebra,
                    self.metric,
                    out_features,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, self.metric, out_features),
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EGCL(nn.Module):
    def __init__(
        self,
        algebra,
        metric,
        in_features,
        hidden_features,
        out_features,
        edge_attr_features=0,
        node_attr_features=0,
        residual=True,
        normalization_init=0,
    ):
        super().__init__()
        self.residual = residual
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.edge_attr_features = edge_attr_features
        self.node_attr_features = node_attr_features

        self.edge_model = CEMLP(
            algebra,
            metric,
            self.in_features + self.edge_attr_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )

        self.node_model = CEMLP(
            algebra,
            metric,
            self.in_features + self.out_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )

    def message(self, h_i, h_j, edge_attr=None):
        if edge_attr is None:  # Unused.
            input = h_i - h_j
        else:
            input = torch.cat([h_i - h_j, edge_attr], dim=1)

        h_msg = self.edge_model(input)
        return h_msg

    def aggregate(self, h_msg, segment_ids, num_segments):
        h_agg = unsorted_segment_mean(h_msg, segment_ids, num_segments=num_segments)
        return h_agg

    def update(self, h_agg, h, node_attr=None):
        if node_attr is not None:
            input_h = torch.cat([h, h_agg, node_attr], dim=1)
        else:
            input_h = torch.cat([h, h_agg], dim=1)

        out_h = self.node_model(input_h)

        if self.residual:
            out_h = h + out_h

        return out_h

    def forward(self, h, edge_index, edge_attr=None, node_attr=None):
        # Message
        rows, cols = edge_index

        h_i, h_j = h[rows], h[cols]

        h_msg = self.message(h_i, h_j, edge_attr)

        # Aggregate
        agg_h = self.aggregate(h_msg.flatten(1), rows, num_segments=len(h)).view(
            len(h), *h_msg.shape[1:]
        )

        # Update
        h = self.update(agg_h, h, node_attr)

        return h


class NBodyCGENN(nn.Module):
    def __init__(
        self,
        in_features=3,
        hidden_features=28,
        out_features=2,
        edge_features_in=0,
        n_layers=3,
        normalization_init=0,
        residual=True,
    ):
        super().__init__()

        # # Fixed
        # # print("*"*60)
        # # print("Using Fixed Metric")
        # # print("*"*60)
        # # self.metric = self.algebra.metric

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.edge_features_in = edge_features_in
        self.n_layers = n_layers
        self.normalization_init = normalization_init
        self.residual = residual

        # time.sleep(5)
        # self.algebra = CliffordAlgebra((1., 1., 1.)).to('cuda')
        # self.metric = torch.diag(self.algebra.metric)
        # print("*"*60)
        # print("Using Fixed Metric")
        # print("*"*60)
        # time.sleep(5)

        time.sleep(5)
        # Initialize without device, will be moved when model.to(device) is called
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        expanded_metric = 0.5 * torch.diag(self.algebra.metric)
        print("*" * 60)
        print("Using Learnable Metric!!!")
        print("*" * 60)
        # Create random tensor on the same device as expanded_metric
        random_tensor = torch.rand(3, 3).to(expanded_metric.device)
        self.metric = nn.Parameter(
            expanded_metric + 1e-4 * random_tensor, requires_grad=False
        )
        time.sleep(5)

        self.embedding = MVLinear(
            self.algebra, in_features, hidden_features, subspaces=False
        )

        layers = []

        for i in range(0, n_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    self.metric + self.metric.T,
                    hidden_features,
                    hidden_features,
                    hidden_features,
                    edge_features_in,
                    residual=residual,
                    normalization_init=normalization_init,
                )
            )

        self.projection = nn.Sequential(
            MVLinear(self.algebra, hidden_features, out_features),
        )

        self.layers = nn.Sequential(*layers)

    def _forward(self, h, edges, edge_attr):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges, edge_attr=edge_attr)
        h = self.projection(h)
        return h

    def forward(self, data):
        data = data.tuple
        loc, vel, edge_attr, charges, _, edges = data

        batch_size, n_nodes, _ = loc.size()

        loc_mean = loc - loc.mean(dim=1, keepdim=True)

        loc_mean = loc_mean.reshape(-1, *loc_mean.shape[2:])
        loc = loc.reshape(-1, *loc.shape[2:])
        vel = vel.reshape(-1, *vel.shape[2:])
        charges = charges.reshape(-1, *charges.shape[2:])

        # Cast input into orthogonal space with respect to the eigenvalues of the metric:
        metric = self.metric + self.metric.T
        _, P = torch.linalg.eig(metric)
        P = P.real
        loc = torch.matmul(loc, P)
        loc_mean = torch.matmul(loc_mean, P)
        vel = torch.matmul(vel, P)

        # Add batch to graph.
        batch_index = torch.arange(batch_size, device=loc_mean.device)
        edges = edges + n_nodes * batch_index[:, None, None]
        edges = tuple(edges.transpose(0, 1).flatten(1))

        # edge_attr = edge_attr
        # edge_attr = self.algebra.embed(edge_attr[..., None], (0,))

        invariants = charges
        invariants = self.algebra.embed(invariants, (0,))

        xv = torch.stack([loc_mean, vel], dim=1)
        covariants = self.algebra.embed(xv, (1, 2, 3))

        input = torch.cat([invariants[:, None], covariants], dim=1)

        pred = self._forward(input, edges, edge_attr)

        loc_pred = pred[..., 0, 1:4]
        vel_pred = pred[..., 1, 1:4]
        # Compute absolute predictions first
        loc_pred_abs = loc + loc_pred
        vel_pred_abs = vel + vel_pred
        # Cast output back into original space:
        loc_pred_abs = torch.matmul(loc_pred_abs, torch.linalg.inv(P))
        vel_pred_abs = torch.matmul(vel_pred_abs, torch.linalg.inv(P))
        # Convert absolute position to displacement (pos_dt)
        pos_dt_pred = loc_pred_abs - torch.matmul(loc, torch.linalg.inv(P))
        # Flatten
        pos_dt_pred = pos_dt_pred.view(-1, 3)
        vel_pred_abs = vel_pred_abs.view(-1, 3)
        pred = torch.cat((pos_dt_pred, vel_pred_abs), dim=1)

        return pred

    def get_serializable_attributes(self):
        return {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "out_features": self.out_features,
            "edge_features_in": self.edge_features_in,
            "n_layers": self.n_layers,
            "metric": self.metric.tolist(),  # Convert tensor to list for serialization
            "normalization_init": self.normalization_init,
            "residual": self.residual,
        }
    
    def get_model_size(self):
        return self.hidden_features
