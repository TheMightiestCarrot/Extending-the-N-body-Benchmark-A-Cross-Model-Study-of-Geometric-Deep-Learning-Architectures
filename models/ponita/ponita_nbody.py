import torch
import torch.nn as nn
import torchmetrics

from models.ponita.models.ponita_pg import PonitaFiberBundle
from models.ponita.transforms.random_rotate import RandomRotate


class PONITA_NBODY(nn.Module):
    """Graph Neural Network module"""

    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-5,
        warmup=10,
        layer_scale=1e-6,
        train_augm=False,
        hidden_dim=64,
        layers=4,
        radius=None,
        num_ori=20,
        basis_dim=128,
        degree=3,
        widening_factor=4,
        multiple_readouts=True,
        # Input/output specifications:
        in_channels_scalar=1,  # Mass
        in_channels_vec=1,  # Velocity
        out_channels_scalar=0,  # None
        out_channels_vec=2,  # Change of positions, velocities
    ):
        super().__init__()

        # Store all arguments as attributes
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.layer_scale = layer_scale
        self.train_augm = train_augm
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.radius = radius
        self.num_ori = num_ori
        self.basis_dim = basis_dim
        self.degree = degree
        self.widening_factor = widening_factor
        self.multiple_readouts = multiple_readouts
        self.in_channels_scalar = in_channels_scalar
        self.in_channels_vec = in_channels_vec
        self.out_channels_scalar = out_channels_scalar
        self.out_channels_vec = out_channels_vec

        if layer_scale == 0.0:
            layer_scale = None

        # For rotation augmentations during training and testing
        self.rotation_transform = RandomRotate(["pos", "vec", "y"], n=3)

        # The metrics to log
        self.train_metric = torchmetrics.MeanSquaredError()
        self.valid_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()

        # Make the model
        self.model = PonitaFiberBundle(
            in_channels_scalar + in_channels_vec,
            hidden_dim,
            out_channels_scalar,
            layers,
            output_dim_vec=out_channels_vec,
            radius=radius,
            num_ori=num_ori,
            basis_dim=basis_dim,
            degree=degree,
            widening_factor=widening_factor,
            layer_scale=layer_scale,
            task_level="node",
            multiple_readouts=multiple_readouts,
        )

    def forward(self, graph):
        scalars, vectors = self.model(graph)

        tensors_to_concat = []

        if self.out_channels_scalar > 0:
            tensors_to_concat.append(scalars)

        if self.out_channels_vec > 0:
            tensors_to_concat.append(vectors.view(vectors.shape[0], -1))

        res = torch.cat(tensors_to_concat, dim=-1)

        return res

    def get_serializable_attributes(self):
        return {
            # "num_params": sum(p.numel() for p in self.parameters()),
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "warmup": self.warmup,
            "layer_scale": self.layer_scale,
            "train_augm": self.train_augm,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "radius": self.radius,
            "num_ori": self.num_ori,
            "basis_dim": self.basis_dim,
            "degree": self.degree,
            "widening_factor": self.widening_factor,
            "multiple_readouts": self.multiple_readouts,
        }

    def get_model_size(self):
        return self.hidden_dim
