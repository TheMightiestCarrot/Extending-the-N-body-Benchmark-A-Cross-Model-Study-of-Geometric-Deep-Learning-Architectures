import importlib

from e3nn.o3 import Irreps

from models import PONITA_NBODY, SEGNN, GraphTransformerTorch, PaiNN
from models.CGENN.nbody_cgenn import NBodyCGENN
from models.equiformer_v2.architecture.equiformer_v2_nbody import (
    EquiformerV2_nbody,
)
from utils.get_device import get_device


def load_class_from_args(args, section: str):
    class_path = getattr(args, section).class_path
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    loaded_class = getattr(module, class_name)
    return loaded_class


def create_model(args, train_dataloader=None):
    model = None

    if args.model_type == "equiformer" or args.model_type == "equiformer_v2":
        model = EquiformerV2_nbody(
            device=get_device(args.gpu_id),
            use_pbc=args.use_pbc,
            max_neighbors=args.max_neighbors,
            max_radius=args.max_radius,
            num_layers=args.num_layers,
            attn_hidden_channels=args.attn_hidden_channels,
            sphere_channels=args.sphere_channels,
            num_heads=args.num_heads,
            attn_alpha_channels=args.attn_alpha_channels,
            attn_value_channels=args.attn_value_channels,
            ffn_hidden_channels=args.ffn_hidden_channels,
            lmax_list=args.lmax_list,
            mmax_list=args.mmax_list,
            grid_resolution=args.grid_resolution,
            edge_channels=args.edge_channels,
            use_atom_edge_embedding=args.use_atom_edge_embedding,
            share_atom_edge_embedding=args.share_atom_edge_embedding,
            distance_function=args.distance_function,
            num_distance_basis=args.num_distance_basis,
            attn_activation=args.attn_activation,
            use_s2_act_attn=args.use_s2_act_attn,
            ffn_activation=args.ffn_activation,
        )

    elif args.model_type == "segnn":
        if args.dataloader_type in ("segnn_nbody", "segnn_nbody_offline"):
            model = SEGNN(
                num_layers=args.num_layers,
                hidden_features=args.hidden_features,
                lmax_h=args.lmax_h,
            )
        else:
            raise ValueError(
                f"""
                Unknown combination of model {args.model_type}
                and dataloader {args.dataloader_type}
                """
            )

    elif args.model_type == "ponita":
        model = PONITA_NBODY(
            layers=args.num_layers,
            hidden_dim=args.hidden_features,
            lr=args.learning_rate,
        )

    elif args.model_type == "cgenn":
        model = NBodyCGENN(
            n_layers=args.num_layers, hidden_features=args.hidden_features
        ).to(get_device(args.gpu_id))
    elif args.model_type == "graph_transformer":
        # Lightweight Torch-only Graph Transformer full-attention model
        model = GraphTransformerTorch(
            hidden_features=getattr(args, "hidden_features", 128),
            num_layers=getattr(args, "num_layers", 4),
            num_heads=getattr(args, "graph_transformer_num_heads", 4),
            args=args,
        )
    elif args.model_type == "painn":
        targets = tuple(
            args.target.split("+")
        ) if hasattr(args, "target") and isinstance(args.target, str) else ("pos_dt", "vel")
        model = PaiNN(
            hidden_features=getattr(args, "hidden_features", 128),
            num_layers=getattr(args, "num_layers", 6),
            num_rbf=getattr(args, "num_rbf", 64),
            cutoff=getattr(args, "cutoff", 10.0),
            targets=targets,
            use_velocity_input=getattr(args, "use_velocity_input", True),
            include_velocity_norm=getattr(args, "include_velocity_norm", True),
        )
    else:
        raise ValueError(f"Unknown model {args.model_type}")

    return model
