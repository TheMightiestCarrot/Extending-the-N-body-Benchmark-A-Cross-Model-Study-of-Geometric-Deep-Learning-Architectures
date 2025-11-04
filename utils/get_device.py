import torch
from .gpu_selector import get_device_auto


def get_device(gpu_id=0):
    if gpu_id == "auto":
        return get_device_auto()
    elif gpu_id == -1:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device
