import argparse
import os

import jsonargparse
import yaml

from utils.config_models import (
    DATALOADER_CONFIG_NAMES,
    MODEL_CONFIG_NAMES,
    TRAINER_CONFIG_NAMES,
    MainConfig,
    PrecisionMode,
)

DEFAULT_CONFIG_PATH = "config.yaml"


def reconstruct_config(args):
    trainer_args = vars(args.trainer) if hasattr(args, "trainer") else {}
    if 'precision_mode' in trainer_args and isinstance(trainer_args['precision_mode'], PrecisionMode):
        trainer_args['precision_mode'] = trainer_args['precision_mode'].value
    return {
        "main": {
            "model_type": args.model_type,
            "dataloader_type": args.dataloader_type,
            "trainer_type": args.trainer_type,
            "gpu_id": args.gpu_id,
        },
        "models": {args.model_type: vars(args.model) if hasattr(args, "model") else {}},
        "dataloaders": {
            args.dataloader_type: (
                vars(args.dataloader) if hasattr(args, "dataloader") else {}
            )
        },
        "trainers": {
            args.trainer_type: trainer_args
        },
    }


def save_config(save_dir_path, config):
    # Save the reconstructed config as YAML
    final_config_path = os.path.join(save_dir_path, "config.yaml")
    with open(final_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"config saved to `{final_config_path}`")


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_config_path():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the configuration file",
    )
    args, _ = parser.parse_known_args()
    return args.config


def parse_main_config_only(defaults):
    """
    Parse only the high-level config args needed to determine which specific config classes to load.

    We need these args (model_type, dataloader_type, trainer_type) before we can load their
    respective config classes for full argument parsing. While pydantic/jsonargparse are used
    for the full config, they don't support partial arg parsing well, hence this separate step.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        help="Name of the selected model",
    )
    parser.add_argument(
        "--dataloader_type",
        type=str,
        help="Name of the dataloader configuration",
    )
    parser.add_argument(
        "--trainer_type",
        type=str,
        help="Name of the trainer configuration",
    )
    parser.add_argument("--gpu_id", default='auto', help="GPU ID to use (number or 'auto')")
    parser.set_defaults(**defaults)
    args, _ = parser.parse_known_args()

    return args


def flatten_dict(d, parent_key="", sep=".", skip_keys=None):
    """Recursively flattens a nested dictionary using dot notation.
    ``skip_keys`` can be used to specify dictionary keys that should not be
    flattened any further. This is useful for configuration entries that expect
    a dictionary object (e.g. ``atom_type_mapper``) rather than individual
    command line options.
    """
    items = []
    skip_keys = set(skip_keys or [])
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and k not in skip_keys:
            items.extend(flatten_dict(v, new_key, sep=sep, skip_keys=skip_keys).items())
        else:
            items.append((new_key, v))
    return dict(items)


def namespace_to_dict(namespace):
    result = {}
    for key, value in vars(namespace).items():
        if isinstance(value, argparse.Namespace):
            result[key] = namespace_to_dict(value)
        else:
            result[key] = value
    return result


def strip_nested_args(args: jsonargparse.Namespace):
    """
    Strip formerly nested args paths to just the argument name.
    Exception: class_path remains with full path
    Example:
    model.learning_rate -> learning_rate
    model.class_path -> model.class_path
    """
    stripped_args = {}

    def _strip_key(key):
        # If the key is for 'class_path', keep the full key
        if key.endswith(".class_path"):
            return key
        # Otherwise, just keep the argument name (the last part)
        return key.split(".")[-1]

    for key, value in vars(args).items():
        if key.endswith("class_path"):
            stripped_args[key] = value
        else:
            stripped_key = _strip_key(key)
            stripped_args[stripped_key] = value

    return jsonargparse.dict_to_namespace(stripped_args)


def add_section_args(
    parser,
    config,
    object_name,
    section_name,
    section_config,
    nesting_key,
    *,
    skip_keys=None,
):
    parser.add_class_arguments(section_config, nesting_key)
    model_config_values = flatten_dict(
        config[section_name][object_name], nesting_key, skip_keys=skip_keys
    )

    parser.set_defaults(model_config_values)
    return parser


def parse_args():
    config_path = parse_config_path()
    config = load_config(config_path)

    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to the configuration file",
    )
    parser.add_class_arguments(MainConfig)
    parser.set_defaults(config["main"])
    args = parse_main_config_only(config["main"])

    model_name = args.model_type
    model_config = MODEL_CONFIG_NAMES[model_name]
    parser = add_section_args(
        parser, config, model_name, "models", model_config, "model"
    )

    dataloader_type = args.dataloader_type
    dataloader_config = DATALOADER_CONFIG_NAMES[dataloader_type]
    parser = add_section_args(
        parser,
        config,
        dataloader_type,
        "dataloaders",
        dataloader_config,
        "dataloader",
        skip_keys={"atom_type_mapper"},
    )

    trainer_type = args.trainer_type
    trainer_config = TRAINER_CONFIG_NAMES[trainer_type]
    parser = add_section_args(
        parser,
        config,
        trainer_type,
        "trainers",
        trainer_config,
        "trainer",
    )
    args = parser.parse_args()

    reconstructed_config = reconstruct_config(args)

    # Ideally we should refactor our classes parameters
    # so that this doesn't need to be done
    args = strip_nested_args(args.as_flat())
    print(f"Args used: {args}")

    return args, reconstructed_config


if __name__ == "__main__":
    print(parse_args())