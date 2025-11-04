import warnings

import torch

from utils.config import parse_args
from utils.utils_train import create_model, load_class_from_args

warnings.filterwarnings(
    "ignore",
    message=".*The TorchScript type system doesn't support instance-level annotations.*",
    category=UserWarning,
    module="torch.jit._check",
)


def main():
    args, _ = parse_args()
    model = create_model(args)

    dataloader = load_class_from_args(args, "dataloader")
    train_dataloader = dataloader(args, partition="train")
    validation_dataloader = dataloader(args, partition="valid")
    trainer = load_class_from_args(args, "trainer")
    trainer = trainer(
        model, train_dataloader, validation_dataloader=validation_dataloader, args=args
    )
    trainer.validate_one_epoch(save_data=True)


if __name__ == "__main__":
    main()
