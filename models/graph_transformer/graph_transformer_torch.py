import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from models.set_transformer.models import EncoderWithMLP


class GraphTransformerTorch(nn.Module):
    """Lightweight Torch implementation of the Graph Transformer full-attention model.

    - Consumes nodewise features built from graph fields (pos/vel/force)
    - Applies a Transformer encoder over nodes per graph (full attention)
    - Predicts targets with dimensionality matching args.target (3 per field)
    """

    def __init__(
        self,
        hidden_features: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        args=None,
    ) -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.args = args

        # Determine which target components are predicted and total output size
        self.target_components = [comp.strip() for comp in (args.target.split("+") if args and hasattr(args, "target") else ["pos_dt"]) ]
        self.output_dim = 3 * len(self.target_components)

        # We will feed the encoder with exactly output_dim features to keep
        # the interface simple (pos only, vel only, force only, or their concat)
        # This mirrors the original Graph Transformer idea of conditioning on current state.
        self.encoder = EncoderWithMLP(
            particle_dim=self.output_dim,
            model_dim=self.hidden_features,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            activation="relu",
            hparams={"precision": "double" if getattr(args, "precision_mode", "double") == "double" else "single"},
            hidden_dims=[self.hidden_features, self.hidden_features],
            mlp_act=nn.ReLU,
            mlp_type="output",
            mlp_output_act=None,
        )

    @staticmethod
    def _gather_features(graph, components):
        parts = []
        for comp in components:
            if comp in ("pos_dt", "pos"):
                parts.append(graph.pos)
            elif comp in ("vel", "vel_dt"):
                # Some datasets may miss velocities; fall back to zeros
                parts.append(getattr(graph, "vel", torch.zeros_like(graph.pos)))
            elif comp in ("force", "force_dt", "current_force"):
                parts.append(getattr(graph, "force", torch.zeros_like(graph.pos)))
            else:
                # Default to zeros for unknown component to keep shapes consistent
                parts.append(torch.zeros_like(graph.pos))
        return torch.cat(parts, dim=-1)

    def forward(self, graph, *args, **kwargs):
        # Build per-node input features matching output dimensionality
        x = self._gather_features(graph, self.target_components)

        # Densify per-graph with padding; mask indicates valid nodes
        x_dense, mask = to_dense_batch(x, graph.batch)

        # Run transformer encoder (batch_first=True inside EncoderWithMLP)
        y_dense = self.encoder(x_dense)

        # Return only valid nodes, flatten back to (total_nodes, output_dim)
        y = y_dense[mask]
        return y

    def get_serializable_attributes(self):
        return {
            "hidden_features": self.hidden_features,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "targets": "+".join(self.target_components),
        }

    def get_model_size(self):
        return self.hidden_features

