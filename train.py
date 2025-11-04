import os
import random
import warnings

import numpy as np
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # change to 1 for debugging
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from utils.config import parse_args, save_config
from utils.utils_train import create_model, load_class_from_args

warnings.filterwarnings(
    "ignore",
    message=".*The TorchScript type system doesn't support instance-level annotations.*",
    category=UserWarning,
    module="torch.jit._check",
)


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # For CPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # For current GPU
            torch.cuda.manual_seed_all(seed)  # For all GPUs
        # this may slow down the training
        # torch.backends.cudnn.deterministic = True  # For deterministic behavior
        # torch.backends.cudnn.benchmark = False  # Disable benchmark for deterministic


def main():
    args, yaml_config_to_save = parse_args()
    set_seed(args.seed)
    dataloader = load_class_from_args(args, "dataloader")
    if args.do_validation:
        train_dataloader = dataloader(args, partition="train")
        validation_dataloader = dataloader(args, partition="valid")
    else:
        train_dataloader = dataloader(args)
        validation_dataloader = None

    model = create_model(args, train_dataloader)

    # loud parameter count print (initialize lazy params if needed)
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    except ValueError:
        try:
            # initialize lazy parameters with a real preprocessed batch
            batch_list, _ = train_dataloader.get_batch()
            dummy_batch = train_dataloader.preprocess_batch(
                batch_list[0], "cpu"
            )

            # match device and dtype to dataloader/precision
            model = model.to("cpu")
            if getattr(args, "precision_mode", "single") == "double":
                model = model.double()
            else:
                model = model.float()

            with torch.no_grad():
                _ = model(dummy_batch)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        except Exception as e:
            raise e
    banner = "=" * 80
    print(banner)
    print(
        f"MODEL PARAMETER COUNT -> TOTAL: {total_params:,} | TRAINABLE: {trainable_params:,}"
    )
    print(banner)

    trainer = load_class_from_args(args, "trainer")
    trainer = trainer(
        model, train_dataloader, validation_dataloader=validation_dataloader, args=args
    )

    # Store parameter counts on trainer for wandb logging
    trainer.total_params = total_params
    trainer.trainable_params = trainable_params

    save_config(trainer.save_dir_path, yaml_config_to_save)
    trainer.train()


if __name__ == "__main__":
    main()
