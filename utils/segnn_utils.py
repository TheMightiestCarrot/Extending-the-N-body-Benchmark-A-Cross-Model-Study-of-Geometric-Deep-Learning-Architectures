import argparse


def create_argparser():
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument("--experiment_name", type=str, default="segnn_runs")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size. Does not scale with number of GPUs.",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="weight decay")
    parser.add_argument("--print", type=int, default=100, help="print interval")
    parser.add_argument("--log", action="store_true", help="Enable logging.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers in dataloader."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models",
        help="Directory in which to save models.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the training.")
    parser.add_argument(
        "--log_dataset", action="store_true", help="Save the dataset after training."
    )
    parser.add_argument("--criterion", type=str, default="MSE", help="Loss function.")

    # Data parameters
    parser.add_argument("--dataset", type=str, default="qm9", help="Dataset name.")
    parser.add_argument(
        "--root", type=str, default="datasets", help="Dataset location."
    )
    parser.add_argument("--download", action="store_true", help="Download the dataset.")

    parser.add_argument("--dataset_name", type=str, help="Name of the dataset.")

    # QM9 parameters
    parser.add_argument(
        "--target",
        type=str,
        default="alpha",
        help="Target value, also used for gravity dataset [pos, force].",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2,
        help="Radius (Angstrom) within which atoms are connected.",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="one_hot",
        help="Type of input feature: one-hot or Cormorant's charge representation.",
    )

    # N-body parameters:
    parser.add_argument(
        "--nbody_name",
        type=str,
        default="nbody_small",
        help="Name of N-body dataset [nbody, nbody_small].",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=3000,
        help="Maximum number of samples in N-body dataset.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of steps in N-body dataset.",
    )
    parser.add_argument(
        "--time_exp", action="store_true", help="Flag for timing experiment."
    )
    parser.add_argument(
        "--test_interval", type=int, default=5, help="Test every N epochs."
    )
    parser.add_argument(
        "--n_nodes", type=int, default=5, help="Number of nodes in the graph."
    )
    parser.add_argument(
        "--limit_output",
        type=int,
        default=None,
        help="Limits how many bodies the dataset and model predict.",
    )

    # Gravity parameters:
    parser.add_argument(
        "--neighbours",
        type=int,
        default=6,
        help="Number of connected nearest neighbours.",
    )
    parser.add_argument(
        "--steps_to_predict", type=int, default=2, help="Number of steps to predict."
    )

    parser.add_argument(
        "--data_loading_limit",
        type=int,
        default=None,
        help="Restrict how many steps are loaded from h5 file.",
    )

    parser.add_argument(
        "--random_trajectory_sampling",
        action="store_true",
        help="Enable random trajectory sampling.",
    )
    parser.add_argument(
        "--use_force", action="store_true", help="Include force in the model."
    )
    parser.add_argument(
        "--use_charge", action="store_true", help="Include charge in the model."
    )

    # Model parameters
    parser.add_argument("--model", type=str, default="segnn", help="Model name.")
    parser.add_argument(
        "--hidden_features", type=int, default=128, help="Number of hidden features."
    )
    parser.add_argument(
        "--lmax_h", type=int, default=2, help="Max degree of hidden representation."
    )
    parser.add_argument(
        "--lmax_attr",
        type=int,
        default=3,
        help="Max degree of geometric attribute embedding.",
    )
    parser.add_argument(
        "--subspace_type",
        type=str,
        default="weightbalanced",
        help="How to divide spherical harmonic subspaces.",
    )
    parser.add_argument(
        "--layers", type=int, default=7, help="Number of message passing layers."
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="instance",
        help="Normalization type [instance, batch].",
    )
    parser.add_argument(
        "--pool", type=str, default="avg", help="Pooling type [avg, sum]."
    )
    parser.add_argument(
        "--conv_type",
        type=str,
        default="linear",
        help="Aggregation method in SEConv [linear, non-linear].",
    )
    parser.add_argument(
        "--double_precision", action="store_true", help="Enable double precision."
    )
    parser.add_argument(
        "--cutoff_radius",
        type=float,
        default=0.32,
        help="Cutoff radius for interactions.",
    )

    # Parallel computing parameters
    parser.add_argument(
        "-g",
        "--gpus",
        default=0,
        type=int,
        help="Number of GPUs to use (assumes all are on one node).",
    )

    return parser
