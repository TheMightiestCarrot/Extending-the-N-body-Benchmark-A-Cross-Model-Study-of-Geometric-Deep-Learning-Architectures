import importlib
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class BaseConfig(BaseModel):
    class_path: str = Field(
        ..., description="Full path to the class, e.g. models.SEGNN"
    )

    @staticmethod
    def import_class(class_path: str):
        """Helper function to dynamically import a class from a string reference."""
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    # pylint: disable=no-self-argument
    @field_validator("class_path", mode="before")
    def validate_class_path(cls, v: str):
        return cls.import_class(v)


class PonitaModelConfig(BaseConfig):
    name: Literal["ponita"] = "ponita"
    num_layers: int = Field(4, description="Number of layers in the model")
    hidden_features: int = Field(64, description="Number of hidden features")
    learning_rate: float = Field(0.01, description="Starting learning rate of model")


class SegnnModelConfig(BaseConfig):
    name: Literal["segnn"] = "segnn"
    lmax_attr: int = Field(1, description="Maximum l value for attribute harmonics")
    lmax_h: int = Field(1, description="Maximum l value for harmonics")
    num_layers: int = Field(4, description="Number of layers in the model")
    hidden_features: int = Field(64, description="Number of hidden features")
    model_type: str = Field("segnn", description="Name of the model type")
    normalization_type: Optional[str] = Field(
        None, description='Normalization type: "batch", "instance", or None'
    )
    # input_irreps: not actually a parameter, dependent entirely on other inputs
    # output_irreps: same
    # additional_message_irreps: same

    model_config = {"protected_namespaces": ()}  # Disable namespace protection


class CGENNModelConfig(BaseConfig):
    name: Literal["cgenn"] = "cgenn"
    num_layers: int = Field(4, description="Number of layers in the model")
    hidden_features: int = Field(96, description="Number of hidden features")


class EquiformerV2Config(BaseConfig):
    name: str = "equiformer_v2"
    class_path: str = (
        "models.equiformer_v2.architecture.equiformer_v2_nbody.EquiformerV2_nbody"
    )
    use_pbc: bool = False
    max_neighbors: int = 5
    max_radius: float = 4096.0
    num_layers: int = 3
    attn_hidden_channels: int = 32
    sphere_channels: int = 32
    num_heads: int = 2
    attn_alpha_channels: int = 8
    attn_value_channels: int = 4
    ffn_hidden_channels: int = 64
    lmax_list: list = Field(default_factory=lambda: [2])
    mmax_list: list = Field(default_factory=lambda: [1])
    grid_resolution: Optional[int] = None
    edge_channels: int = 32
    use_atom_edge_embedding: bool = True
    share_atom_edge_embedding: bool = False
    distance_function: str = "projection"
    num_distance_basis: int = 64
    attn_activation: str = "scaled_silu"
    use_s2_act_attn: bool = False
    ffn_activation: str = "scaled_silu"


class GraphTransformerModelConfig(BaseConfig):
    name: Literal["graph_transformer"] = "graph_transformer"
    class_path: str = Field(
        "models.graph_transformer.graph_transformer_torch.GraphTransformerTorch"
    )
    hidden_features: int = Field(128)
    num_layers: int = Field(4)
    graph_transformer_num_heads: int = Field(4)

class GravityDatasetOtfConfig(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset")
    num_atoms: int = Field(5, description="Number of atoms")
    target: str = Field("pos_dt+vel", description="Target variable")
    sample_freq: int = Field(10, description="Sampling frequency")
    center_of_mass: bool = Field(False, description="Use center of mass")
    interaction_strength: float = Field(2)
    softening: float = Field(0.2)


class PonitaNBodyDataLoaderConfig(BaseConfig):
    name: Literal["ponita_nbody"] = "ponita_nbody"
    num_neighbors: int = Field(description="Number of neighbors to use for the model")
    batch_size: int = Field(128, description="Batch size for training")
    double_precision: bool = Field(True, description="Use double precision")
    gravity_dataset: GravityDatasetOtfConfig = Field(
        ..., description="Gravity dataset config"
    )
    model_path: Optional[str] = Field(None, description="Path to model checkpoint")

    model_config = {"protected_namespaces": ()}  # Disable namespace protection


class SegnnNBodyDataLoaderConfig(BaseConfig):
    name: Literal["segnn_nbody"] = "segnn_nbody"
    num_neighbors: int = Field(description="Number of neighbors to use for the model")
    gravity_dataset: GravityDatasetOtfConfig = Field(
        ..., description="Gravity dataset config"
    )
    batch_size: int = Field(128, description="Batch size for training")
    dataset_name: str = Field("nbody_small", description="Dataset name")

    model_config = {"protected_namespaces": ()}  # Disable namespace protection


MODEL_CONFIG_NAMES = {
    "ponita": PonitaModelConfig,
    "segnn": SegnnModelConfig,
    "cgenn": CGENNModelConfig,
    "equiformer_v2": EquiformerV2Config,
    "graph_transformer": GraphTransformerModelConfig,
}

class CgennNBodyDataLoaderConfig(BaseConfig):
    name: Literal["cgenn_nbody"] = "cgenn_nbody"
    batch_size: int = Field(128, description="Batch size for training")
    gravity_dataset: GravityDatasetOtfConfig = Field(
        ..., description="Gravity dataset config"
    )


class EquiformerV2NBodyDataLoaderConfig(BaseConfig):
    name: Literal["equiformer_v2_nbody"] = "equiformer_v2_nbody"
    batch_size: int = Field(128, description="Batch size for training")
    gravity_dataset: GravityDatasetOtfConfig = Field(
        ..., description="Gravity dataset config"
    )
    max_neighbors: int = Field(5, description="Maximum number of neighbors")
    max_radius: float = Field(4096.0, description="Maximum radius for neighbor search")

class GraphTransformerNBodyDataLoaderConfig(BaseConfig):
    name: Literal["graph_transformer_nbody"] = "graph_transformer_nbody"
    batch_size: int = Field(128)
    gravity_dataset: GravityDatasetOtfConfig = Field(...)


class SegnnNbodyOfflineDataLoaderConfig(BaseConfig):
    """Offline N-body dataset config for SEGNN."""

    name: Literal["segnn_nbody_offline"] = "segnn_nbody_offline"
    batch_size: int = Field(128, description="Batch size for training")
    dataset_name: str = Field(..., description="Dataset name (e.g., '5_0_0')")
    data_directory: str = Field(..., description="Directory with data")
    virtual_channels: int = Field(1, description="Virtual channels for dataset")
    max_samples: int = Field(1000, description="Max samples to load")
    frame_0: int = Field(30)
    frame_T: int = Field(40)
    cutoff_rate: float = Field(0.0)
    target: str = Field("pos_dt+vel", description="Training target for SEGNN")


DATALOADER_CONFIG_NAMES = {
    "ponita_nbody": PonitaNBodyDataLoaderConfig,
    "segnn_nbody": SegnnNBodyDataLoaderConfig,
    "segnn_nbody_offline": SegnnNbodyOfflineDataLoaderConfig,
    "cgenn_nbody": CgennNBodyDataLoaderConfig,
    "equiformer_v2_nbody": EquiformerV2NBodyDataLoaderConfig,
    "graph_transformer_nbody": GraphTransformerNBodyDataLoaderConfig,
}


class ValidationConfig(BaseModel):
    do_validation: bool = Field(
        False, description="If True, validation dataset is used"
    )
    split_ratio: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Ratio of training data to use for training (rest is validation)",
    )
    validation_frequency: int = Field(1, description="Validation frequency in epochs")


class PrecisionMode(str, Enum):
    DOUBLE = "double"
    SINGLE = "single"
    AUTOCAST = "autocast"


class BaseTrainerConfig(BaseConfig):
    com_loss: bool = Field(False, description="Use center of mass loss")
    precision_mode: PrecisionMode = Field(
        PrecisionMode.DOUBLE, description="Use double precision"
    )
    energy_loss: bool = Field(False, description="Use energy loss")
    learning_rate: float = Field(1e-2, description="Learning rate")
    learning_rate_factor: float = Field(1.0, description="LambdaLR factor")
    learning_rate_warmup_steps: int = Field(
        1000,
        description="Number of LR scheduler warmup steps (LR going up to the starting value)",
    )
    model_path: Optional[str] = Field(
        None, description="Path to a pre-trained model (optional)"
    )
    run_name: Optional[str] = Field(
        None,
        description="Name of the run for wandb, if None, save_dir_path is used",
    )
    save_model_every: int = Field(10, description="Save model every N steps")
    test_macros_every: int = Field(1024, description="Test macros every N steps")
    train_steps: Optional[int] = Field(None, description="Number of training steps")
    model_config = {"protected_namespaces": ()}  # Disable namespace protection
    sync_cuda_cores: bool = Field(
        True,
        description="Disabling synchronization improves speed but reduces the precision of time measurements.",
    )
    steps_per_epoch: int = Field(
        1, description="Number of learning steps in epoch. -1 stands for whole dataset"
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="If configured, validation set will be used",
    )
    seed: Optional[int] = Field(None, description="Seed for whole operation.")
    clip_gradients_norm: Optional[float] = Field(
        None,
        description="During training, this is the max allowed value of norm of all gradients",
    )
    clip_gradients_value: Optional[float] = Field(
        None,
        description="During training, this is the max allowed value of a single gradient",
    )
    discard_nan_gradients: bool = Field(
        False,
        description="If true, if any gradient is nan during training, we won't apply it and continue to the next batch",
    )
    per_atom_loss: bool = Field(
        False,
        description="Calculate per-atom losses during training",
    )


class TrainerNBodyConfig(BaseTrainerConfig):
    name: Literal["trainer_nbody"] = "trainer_nbody"
    momentum_loss: bool = Field(False, description="Use momentum loss")
    momentum_loss_weight: float = Field(0.0001, description="Weight for momentum loss")
    position_loss_weight: float = Field(1.0, description="Weight for position loss")
    velocity_loss_weight: float = Field(1.0, description="Weight for velocity loss")
    force_loss_weight: float = Field(1.0, description="Weight for force loss")


TRAINER_CONFIG_NAMES = {
    "trainer_nbody": TrainerNBodyConfig,
}


class MainConfig(BaseModel):
    model_type: str = Field(..., description="Name of the selected model")
    dataloader_type: str = Field(..., description="Dataloader configuration")
    trainer_type: str = Field(..., description="Trainer configuration")
    gpu_id: Union[int, str] = Field(0, description="GPU ID (number or 'auto')")

    model_config = {"protected_namespaces": ()}  # Disable namespace protection
