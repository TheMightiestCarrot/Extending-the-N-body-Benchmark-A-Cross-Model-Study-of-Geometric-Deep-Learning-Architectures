#!/usr/bin/env python3
"""
Test script for GPU selection functionality.
"""

import subprocess
import sys

def test_gpu_selection():
    """Test different GPU selection scenarios."""
    
    print("Testing GPU Selection Functionality")
    print("=" * 60)
    
    # Test 1: Auto selection (default)
    print("\nTest 1: Auto selection (default config)")
    cmd = ["python", "-m", "train", "--trainer.train_steps=1", "--trainer.save_model_every=10000"]
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if "Selected GPU" in result.stdout or "Using device" in result.stdout:
        print("✓ Auto selection working")
        print("Output excerpt:", [line for line in result.stdout.split('\n') if 'GPU' in line or 'device' in line][:3])
    else:
        print("✗ Auto selection may not be working")
    
    # Test 2: Explicit GPU selection
    print("\n\nTest 2: Explicit GPU selection (GPU 1)")
    cmd = ["python", "-m", "train", "--gpu_id=1", "--trainer.train_steps=1", "--trainer.save_model_every=10000"]
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if "Using device: cuda:1" in result.stdout:
        print("✓ Explicit GPU selection working")
    else:
        print("✗ Explicit GPU selection may not be working")
    
    # Test 3: String 'auto' parameter
    print("\n\nTest 3: String 'auto' parameter")
    cmd = ["python", "-m", "train", "--gpu_id=auto", "--trainer.train_steps=1", "--trainer.save_model_every=10000"]
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if "Selected GPU" in result.stdout:
        print("✓ String 'auto' parameter working")
        print("Output excerpt:", [line for line in result.stdout.split('\n') if 'Selected GPU' in line][:1])
    else:
        print("✗ String 'auto' parameter may not be working")
    
    print("\n" + "=" * 60)
    print("GPU selection tests completed!")

if __name__ == "__main__":
    test_gpu_selection()