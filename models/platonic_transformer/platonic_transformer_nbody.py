import torch
import torch.nn as nn

from models.platonic_transformer.platoformer.platoformer import PlatonicTransformer


class PlatonicTransformerNBody(nn.Module):
    """Thin N-body wrapper around the PlatonicTransformer core architecture."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        solid_name: str = "icosahedron",
        input_dim: int = 1,
        input_dim_vec: int = 1,
        scalar_task_level: str = "node",
        vector_task_level: str = "node",
        dense_mode: bool = False,
        ffn_readout: bool = True,
        mean_aggregation: bool = False,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        ffn_dim_factor: int = 4,
        rope_sigma: float = 1.0,
        ape_sigma=None,
        learned_freqs: bool = True,
        freq_init: str = "random",
        use_key: bool = False,
        attention: bool = False,
        spatial_dim: int = 3,
        args=None,
    ) -> None:
        super().__init__()
        self.args = args

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.solid_name = solid_name
        self.input_dim = input_dim
        self.input_dim_vec = input_dim_vec
        self.vector_task_level = vector_task_level
        self.scalar_task_level = scalar_task_level

        # Determine which target vectors to predict (pos_dt, vel, force, ...)
        self.target_components = (
            [comp.strip() for comp in args.target.split("+")]
            if args is not None and hasattr(args, "target")
            else ["pos_dt", "vel"]
        )
        self.output_dim_vec = len(self.target_components)

        self.model = PlatonicTransformer(
            input_dim=self.input_dim,
            input_dim_vec=self.input_dim_vec,
            hidden_dim=self.hidden_dim,
            output_dim=0,
            output_dim_vec=self.output_dim_vec,
            nhead=self.num_heads,
            num_layers=self.num_layers,
            solid_name=self.solid_name,
            spatial_dim=spatial_dim,
            dense_mode=dense_mode,
            scalar_task_level=self.scalar_task_level,
            vector_task_level=self.vector_task_level,
            ffn_readout=ffn_readout,
            mean_aggregation=mean_aggregation,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=None,
            attention=attention,
            ffn_dim_factor=ffn_dim_factor,
            rope_sigma=rope_sigma,
            ape_sigma=ape_sigma,
            learned_freqs=learned_freqs,
            freq_init=freq_init,
            use_key=use_key,
        )

    def forward(self, graph):
        batch_indices = getattr(graph, "batch", None)

        # Scalar inputs: default to masses if present.
        x = getattr(graph, "x", None)
        if x is None:
            mass = getattr(graph, "mass", None)
            if mass is not None:
                x = mass
            else:
                x = torch.ones(
                    graph.pos.size(0),
                    self.input_dim,
                    device=graph.pos.device,
                    dtype=graph.pos.dtype,
                )

        # Vector inputs: velocities are natural vector features.
        vec = getattr(graph, "vec", None)
        if vec is None:
            vel = getattr(graph, "vel", None)
            if vel is not None:
                vec = vel.unsqueeze(1)
            else:
                vec = torch.zeros(
                    graph.pos.size(0),
                    self.input_dim_vec,
                    3,
                    device=graph.pos.device,
                    dtype=graph.pos.dtype,
                )

        # Average node count is stable for OTF n-body (num_atoms).
        avg_num_nodes = float(getattr(self.args, "num_atoms", vec.shape[0]))

        _, vectors = self.model(
            x,
            graph.pos,
            batch=batch_indices,
            mask=None,
            vec=vec,
            avg_num_nodes=avg_num_nodes,
        )

        # Flatten vector outputs to match trainer expectations: (N, 3 * len(targets))
        vectors = vectors.reshape(vectors.shape[0], -1)
        return vectors

    def get_serializable_attributes(self):
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "solid_name": self.solid_name,
            "input_dim": self.input_dim,
            "input_dim_vec": self.input_dim_vec,
            "targets": "+".join(self.target_components),
        }

    def get_model_size(self):
        return self.hidden_dim
