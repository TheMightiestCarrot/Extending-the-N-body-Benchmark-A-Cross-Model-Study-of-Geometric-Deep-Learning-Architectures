"""
GPU selection utility for automatic allocation of training to available GPUs.
"""

import torch
import subprocess
import numpy as np
from typing import Optional, Tuple


def get_gpu_memory_usage() -> list[tuple[int, float, float]]:
    """
    Get GPU memory usage for all available GPUs.
    
    Returns:
        List of tuples (gpu_id, used_memory_mb, total_memory_mb)
    """
    if not torch.cuda.is_available():
        return []
    
    gpu_info = []
    
    try:
        # Use nvidia-smi to get accurate memory usage
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) == 3:
                gpu_id = int(parts[0])
                used_memory = float(parts[1])
                total_memory = float(parts[2])
                gpu_info.append((gpu_id, used_memory, total_memory))
                
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to PyTorch if nvidia-smi is not available
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            used_memory = torch.cuda.memory_allocated(i) / 1024**2  # Convert to MB
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**2
            gpu_info.append((i, used_memory, total_memory))
    
    return gpu_info


def select_best_gpu(memory_threshold_mb: float = 1000.0) -> Tuple[Optional[int], str]:
    """
    Select the best available GPU based on memory usage.
    
    Args:
        memory_threshold_mb: Minimum free memory required (in MB) to consider a GPU as "idle"
    
    Returns:
        Tuple of (gpu_id, status_message)
        gpu_id is None if no suitable GPU is found
    """
    if not torch.cuda.is_available():
        return None, "No GPUs available. Using CPU."
    
    gpu_info = get_gpu_memory_usage()
    
    if not gpu_info:
        return None, "Could not retrieve GPU information. Using CPU."
    
    # Calculate free memory and usage percentage for each GPU
    gpu_stats = []
    for gpu_id, used_memory, total_memory in gpu_info:
        free_memory = total_memory - used_memory
        usage_percent = (used_memory / total_memory) * 100
        gpu_stats.append({
            'id': gpu_id,
            'free_memory': free_memory,
            'usage_percent': usage_percent,
            'used_memory': used_memory,
            'total_memory': total_memory
        })
    
    # Sort by free memory (descending)
    gpu_stats.sort(key=lambda x: x['free_memory'], reverse=True)
    
    # Find idle GPUs (those with enough free memory)
    idle_gpus = [gpu for gpu in gpu_stats if gpu['free_memory'] >= memory_threshold_mb]
    
    if idle_gpus:
        # Select the GPU with the most free memory
        best_gpu = idle_gpus[0]
        return best_gpu['id'], f"Selected GPU {best_gpu['id']} (idle) - {best_gpu['free_memory']:.0f}MB free ({best_gpu['usage_percent']:.1f}% used)"
    else:
        # No idle GPUs, select the one with the least load
        best_gpu = gpu_stats[0]  # Already sorted by free memory
        warning_msg = (f"WARNING: No idle GPUs found (threshold: {memory_threshold_mb}MB free). "
                      f"Using GPU {best_gpu['id']} with least load - "
                      f"{best_gpu['free_memory']:.0f}MB free ({best_gpu['usage_percent']:.1f}% used)")
        return best_gpu['id'], warning_msg


def get_device_auto(fallback_gpu_id: int = 0, memory_threshold_mb: float = 1000.0) -> torch.device:
    """
    Automatically select the best available device (GPU or CPU).
    
    Args:
        fallback_gpu_id: GPU ID to use if automatic selection fails
        memory_threshold_mb: Minimum free memory required to consider a GPU as "idle"
    
    Returns:
        torch.device object
    """
    gpu_id, message = select_best_gpu(memory_threshold_mb)
    
    print(message)
    
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using device: {device}")
        return device
    elif torch.cuda.is_available():
        # Fallback to specified GPU if selection failed but CUDA is available
        device = torch.device(f"cuda:{fallback_gpu_id}")
        print(f"Using fallback device: {device}")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
        return device
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
        return device


def print_gpu_status():
    """Print detailed status of all available GPUs."""
    if not torch.cuda.is_available():
        print("No GPUs available.")
        return
    
    gpu_info = get_gpu_memory_usage()
    
    print("\nGPU Status:")
    print("-" * 60)
    print(f"{'GPU':<5} {'Used (MB)':<12} {'Total (MB)':<12} {'Free (MB)':<12} {'Usage %':<10}")
    print("-" * 60)
    
    for gpu_id, used_memory, total_memory in gpu_info:
        free_memory = total_memory - used_memory
        usage_percent = (used_memory / total_memory) * 100
        print(f"{gpu_id:<5} {used_memory:<12.0f} {total_memory:<12.0f} {free_memory:<12.0f} {usage_percent:<10.1f}")
    
    print("-" * 60)


if __name__ == "__main__":
    # Test the GPU selector
    print_gpu_status()
    print("\nSelecting best GPU...")
    device = get_device_auto()
    print(f"\nSelected device: {device}")