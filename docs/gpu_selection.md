# Automatic GPU Selection

The training scripts now support automatic GPU selection based on availability and current GPU memory usage.

## Features

- **Automatic Selection**: Finds idle GPUs and selects the one with the most free memory
- **Load Balancing**: If no idle GPUs are found, selects the GPU with the least load
- **Configurable Threshold**: Define what constitutes an "idle" GPU (default: 1000MB free memory)
- **Fallback Support**: Falls back to CPU if no GPUs are available

## Usage

### Default Behavior (Automatic Selection)

By default, the system will automatically select the best available GPU:

```bash
python -m train
# or
python train_equiformer.py
```

### Explicit GPU Selection

You can still specify a particular GPU if needed:

```bash
# Use GPU 0
python -m train --gpu_id=0

# Use GPU 2
python -m train --gpu_id=2
```

### Force Automatic Selection

You can explicitly request automatic selection:

```bash
python -m train --gpu_id=auto
```

### Configuration File

In `config.yaml`, set the gpu_id to enable automatic selection by default:

```yaml
main:
  gpu_id: auto  # Can be: auto, 0, 1, 2, etc.
```

## GPU Status

To check the current status of all GPUs:

```bash
python -m utils.gpu_selector
```

This will display:
- Memory usage for each GPU
- Which GPU would be selected
- Current device selection

## How It Works

1. **Query GPU Status**: Uses `nvidia-smi` to get accurate memory usage
2. **Identify Idle GPUs**: GPUs with more than 1000MB free memory (configurable)
3. **Select Best GPU**: 
   - If idle GPUs exist: Select the one with most free memory
   - If no idle GPUs: Select the one with least load (most free memory)
4. **Fallback**: If GPU selection fails, falls back to the default behavior

## Customization

### Memory Threshold

You can customize what constitutes an "idle" GPU by modifying the threshold in `utils/gpu_selector.py`:

```python
# Default: 1000MB free memory required
memory_threshold_mb = 1000.0
```

### Integration in Custom Scripts

```python
from utils.nbody_utils import get_device

# Automatic selection
device = get_device()  # or get_device('auto')

# Explicit selection
device = get_device(0)  # Use GPU 0
```

## Troubleshooting

- **"No idle GPUs found" warning**: All GPUs are in use. The system will use the least loaded GPU.
- **"No GPUs available"**: No CUDA devices detected. System will use CPU.
- **nvidia-smi not found**: The system will fall back to PyTorch's memory queries (less accurate).

## Notes

- The selection happens at the start of training and doesn't change during execution
- Memory usage is checked at selection time; subsequent changes won't affect the selected GPU
- Compatible with all existing training scripts and models