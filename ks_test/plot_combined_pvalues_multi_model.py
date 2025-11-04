import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import torch
import csv

# Add parent directory to path to import from ks_test_model_checkpoints
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ks_test_model_checkpoints import perform_ks_tests

def get_parameter_count(run_path):
    """Get parameter count from model_params.json or by loading the model"""
    # First try model_params.json (for SEGNN)
    model_params_path = os.path.join(run_path, "model_params.json")
    if os.path.exists(model_params_path):
        with open(model_params_path, "r") as f:
            data = json.load(f)
        if "num_params" in data:
            return data["num_params"]
    
    # Try to count parameters from the model file
    model_path = os.path.join(run_path, "model.pth")
    if not os.path.exists(model_path):
        return None
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Extract state dict (could be directly the state dict or nested in checkpoint)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            return None
        
        # Count parameters
        num_params = 0
        for param in state_dict.values():
            if isinstance(param, torch.Tensor):
                num_params += param.numel()
        
        return num_params if num_params > 0 else None
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def load_8h_cutoff_config():
    """Load the 8-hour cutoff configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), "run_8h_cutoffs.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def get_max_checkpoint_for_run(run_path, max_hours=8):
    """Get the maximum checkpoint number for a run within the time limit"""
    # Load 8-hour cutoff config
    cutoff_config = load_8h_cutoff_config()
    
    # Normalize run path for matching
    normalized_path = os.path.normpath(run_path)
    
    # Check if we have a configured cutoff for this run
    for config_key, max_ckpt in cutoff_config.items():
        if config_key == "comment":
            continue
        if os.path.normpath(config_key) == normalized_path or config_key in normalized_path:
            print(f"  Using configured 8h cutoff: checkpoint {max_ckpt}")
            return max_ckpt
    
    # Fallback: try to estimate from filesystem timestamps (unreliable)
    checkpoint_path = os.path.join(normalized_path, "checkpoints")
    if not os.path.isdir(checkpoint_path):
        return None
    
    checkpoint_dirs = [d for d in os.listdir(checkpoint_path) if d.isdigit()]
    if not checkpoint_dirs:
        return None
    
    timestamps = {}
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_dir_path = os.path.join(checkpoint_path, checkpoint_dir)
        if os.path.isdir(checkpoint_dir_path):
            stat_info = os.stat(checkpoint_dir_path)
            timestamps[int(checkpoint_dir)] = stat_info.st_mtime
    
    if not timestamps:
        return None
    
    checkpoint_numbers = sorted(timestamps.keys())
    first_timestamp = timestamps[checkpoint_numbers[0]]
    max_seconds = max_hours * 3600
    
    max_checkpoint = None
    for checkpoint_num in checkpoint_numbers:
        if timestamps[checkpoint_num] - first_timestamp <= max_seconds:
            max_checkpoint = checkpoint_num
        else:
            break
    
    print(f"  Using filesystem-based estimate: checkpoint {max_checkpoint} (may be inaccurate)")
    return max_checkpoint


def get_combined_p_values_for_run(run_path, max_hours=8):
    """Get combined p-values for a single run, filtered by time"""
    from scipy.stats import chi2
    from mpmath import log, mp
    
    # Set precision for mpmath
    mp.dps = 200
    
    # Normalize run path
    normalized_run_path = os.path.normpath(run_path)
    checkpoint_path = os.path.join(normalized_run_path, "checkpoints")
    
    if not os.path.isdir(checkpoint_path):
        print(f"No checkpoint directory found at: {run_path}")
        return None, None, None
    
    # Get maximum checkpoint for this run within time limit
    max_checkpoint = get_max_checkpoint_for_run(normalized_run_path, max_hours)
    
    if max_checkpoint is None:
        print(f"No valid checkpoints found within {max_hours} hours for {run_path}")
        return None, None, None
    
    print(f"Processing {run_path} up to checkpoint {max_checkpoint} (within {max_hours}h)")
    
    # Perform KS tests
    checkpoint_numbers, p_values_by_key, ground_truth_data, predicted_data, output_dir = (
        perform_ks_tests(run_path, max_steps=max_checkpoint)
    )
    
    if checkpoint_numbers is None or len(checkpoint_numbers) == 0:
        print(f"No checkpoints were processed for {run_path}")
        return None, None, None
    
    # Convert p_values_by_key to a numpy array
    p_values_matrix = np.array(list(p_values_by_key.values()))
    
    # Initialize combined p-values array
    combined_p_values = []
    min_positive = 1e-300
    
    for i in range(p_values_matrix.shape[1]):  # Iterate over checkpoints
        p_values = p_values_matrix[:, i]
        # Remove NaN values
        p_values = p_values[~np.isnan(p_values)]
        if len(p_values) == 0:
            combined_p_values.append(np.nan)
            continue
        # Use mpmath for high-precision logarithms
        chi_stat = -2 * mp.fsum([log(mp.mpf(p)) for p in p_values])
        chi_stat = float(chi_stat)
        # Degrees of freedom: 2 times the number of p-values combined
        dof = 2 * len(p_values)
        # Combined p-value using the survival function
        combined_p_value = chi2.sf(chi_stat, dof)
        # Handle zeros
        if combined_p_value == 0:
            combined_p_value = min_positive
        combined_p_values.append(combined_p_value)
    
    return checkpoint_numbers, np.array(combined_p_values), max_checkpoint


def categorize_runs(run_paths):
    """Categorize runs into 2M and 10M parameter groups"""
    runs_2m = {}
    runs_10m = {}
    
    for run_path in run_paths:
        num_params = get_parameter_count(run_path)
        if num_params is None:
            print(f"Warning: Could not get parameter count for {run_path}")
            continue
        
        # Extract model name from path
        model_name = run_path.split("/")[-2]
        
        # Categorize by parameter count
        if num_params < 5_000_000:  # < 5M means it's a 2M model
            print(f"  {model_name}: {num_params:,} params -> 2M category")
            runs_2m[model_name] = run_path
        else:  # >= 5M means it's a 10M model
            print(f"  {model_name}: {num_params:,} params -> 10M category")
            runs_10m[model_name] = run_path
    
    return runs_2m, runs_10m


def plot_combined_p_values(runs_dict, size_label, output_path, max_hours=8, color_scheme="option1_vibrant"):
    """Plot combined p-values for multiple models"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Model name mapping for display
    model_display_names = {
        "segnn": "SEGNN",
        "ponita": "Ponita",
        "equiformer_v2": "EquiformerV2",
        "cgenn": "CGENN",
        "graph_transformer": "Graph Transformer"
    }
    
    # Color scheme options
    COLOR_SCHEMES = {
        "option1_vibrant": {
            "colors": {"segnn": "#0173B2", "ponita": "#DE8F05", "equiformer_v2": "#029E73", 
                      "cgenn": "#CC78BC", "graph_transformer": "#D62728"},  # Changed to red
            "linewidth": 2.5, "markersize": 5, "star_size": 200
        },
        "option2_colorblind": {
            "colors": {"segnn": "#0173B2", "ponita": "#ECE133", "equiformer_v2": "#029E73",
                      "cgenn": "#DE8F05", "graph_transformer": "#CC78BC"},
            "linewidth": 2.5, "markersize": 5, "star_size": 200
        },
        "option3_bold": {
            "colors": {"segnn": "#1f77b4", "ponita": "#ff7f0e", "equiformer_v2": "#2ca02c",
                      "cgenn": "#d62728", "graph_transformer": "#9467bd"},
            "linewidth": 3.5, "markersize": 7, "star_size": 250
        },
        "option4_distinct": {
            "colors": {"segnn": "#1E88E5", "ponita": "#FFC107", "equiformer_v2": "#00C853",
                      "cgenn": "#E91E63", "graph_transformer": "#9C27B0"},
            "linewidth": 2.5, "markersize": 6, "star_size": 200,
            "line_styles": {"segnn": "-", "ponita": "-", "equiformer_v2": "--", 
                          "cgenn": "-.", "graph_transformer": ":"}
        },
        "option5_seaborn": {
            "colors": {"segnn": "#4C72B0", "ponita": "#DD8452", "equiformer_v2": "#55A868",
                      "cgenn": "#C44E52", "graph_transformer": "#8172B3"},
            "linewidth": 2.5, "markersize": 5, "star_size": 180
        }
    }
    
    # Select color scheme
    scheme = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["option1_vibrant"])
    colors = scheme["colors"]
    linewidth = scheme["linewidth"]
    markersize = scheme["markersize"]
    star_size = scheme["star_size"]
    line_styles = scheme.get("line_styles", {})
    
    min_positive = 1e-300
    has_any_data = False
    all_p_values = []  # Track all p-values to determine y-axis range
    summary = {}
    
    for model_name, run_path in sorted(runs_dict.items()):
        checkpoint_numbers, combined_p_values, max_checkpoint = get_combined_p_values_for_run(
            run_path, max_hours
        )
        
        if checkpoint_numbers is None or len(checkpoint_numbers) == 0:
            print(f"Skipping {model_name} - no valid data")
            continue
        
        has_any_data = True
        
        raw_values = np.array(combined_p_values, copy=True)
        combined_p_values = np.clip(raw_values, min_positive, None)
        all_p_values.extend(combined_p_values)
        
        display_name = model_display_names.get(model_name, model_name)
        color = colors.get(model_name, None)
        line_style = line_styles.get(model_name, "-")
        
        ax.plot(
            checkpoint_numbers,
            combined_p_values,
            label=display_name,
            marker="o",
            markersize=markersize,
            linewidth=linewidth,
            color=color,
            linestyle=line_style
        )

        # Determine best checkpoint (max p-value)
        if np.all(np.isnan(raw_values)):
            best_idx = None
            best_checkpoint = None
            best_p = np.nan
        else:
            best_idx = int(np.nanargmax(raw_values))
            best_checkpoint = int(checkpoint_numbers[best_idx])
            best_p = float(raw_values[best_idx])

            # Highlight best point with a star
            star = ax.scatter(
                checkpoint_numbers[best_idx],
                combined_p_values[best_idx],
                marker="*",
                s=star_size,
                color=color,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            star.set_clip_on(False)

        # Find earliest checkpoint with p >= 0.05
        threshold_idx = None
        if not np.all(np.isnan(raw_values)):
            for idx, val in enumerate(raw_values):
                if not np.isnan(val) and val >= 0.05:
                    threshold_idx = idx
                    break

        threshold_checkpoint = (
            int(checkpoint_numbers[threshold_idx]) if threshold_idx is not None else None
        )

        summary[model_name] = {
            "display_name": display_name,
            "num_checkpoints": int(len(checkpoint_numbers)),
            "max_checkpoint": int(max_checkpoint),
            "best_checkpoint": best_checkpoint,
            "best_p_value": best_p,
            "threshold_checkpoint": threshold_checkpoint,
        }
        
        print(f"{display_name}: {len(checkpoint_numbers)} checkpoints, "
              f"final p-value: {combined_p_values[-1]:.4e}")
    
    if not has_any_data:
        print(f"No data to plot for {size_label}")
        return
    
    # Add reference line
    ax.axhline(y=0.05, color="r", linestyle=":", linewidth=1.5, label="p-value = 0.05")
    
    ax.set_xlabel("Checkpoint Number", fontsize=12)
    ax.set_ylabel("Combined p-value (Fisher's method)", fontsize=12)
    ax.set_title(f"KS Test Combined p-value Evolution - {size_label} Models ({max_hours}h training)", 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which='both')
    
    # Set y-axis limits based on actual data range
    if all_p_values:
        min_p = min(all_p_values)
        max_p = max(all_p_values)
        # Add some padding in log space
        y_min = max(min_p / 10, 1e-300)  # One order of magnitude below min
        y_max = min(max_p * 10, 1.5)     # One order of magnitude above max, capped at 1.5
        # Add extra headroom at top to prevent star markers from being clipped
        # In log space, multiply by factor for headroom (3x gives enough space for stars)
        y_max_with_headroom = min(y_max * 5.0, 10.0)
        ax.set_ylim(y_min, y_max_with_headroom)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{size_label} plot saved to: {output_path}")
    return summary


def main():
    import argparse
    
    # Default runs for the paper (8-hour training budget)
    DEFAULT_RUNS = [
        "runs/cgenn/2025-10-01_03-31-26",      # CGENN 2M
        "runs/cgenn/2025-10-01_03-34-34",      # CGENN 10M
        "runs/segnn/2025-09-30_03-23-40",      # SEGNN 2M
        "runs/segnn/2025-09-30_05-21-25",      # SEGNN 10M
        "runs/ponita/2025-10-03_00-05-28",     # Ponita 2M
        "runs/ponita/2025-09-29_12-31-52",     # Ponita 10M
        "runs/equiformer_v2/2025-10-02_23-59-51",  # EquiformerV2 2M
        "runs/equiformer_v2/2025-10-01_14-19-00",  # EquiformerV2 10M
        "runs/graph_transformer/2025-10-01_03-10-47",        # Graph Transformer 2M
        "runs/graph_transformer/2025-10-01_03-16-16",        # Graph Transformer 10M
    ]
    
    parser = argparse.ArgumentParser(
        description="Plot combined KS test p-values for multiple model runs"
    )
    parser.add_argument(
        "--run-paths",
        type=str,
        nargs="+",
        default=DEFAULT_RUNS,
        help=f"Paths to run directories (default: {len(DEFAULT_RUNS)} paper runs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory for plots (default: figures)"
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=8.0,
        help="Maximum training hours to consider (default: 8.0)"
    )
    parser.add_argument(
        "--color-scheme",
        type=str,
        default="option1_vibrant",
        choices=["option1_vibrant", "option2_colorblind", "option3_bold", 
                 "option4_distinct", "option5_seaborn"],
        help="Color scheme to use (default: option1_vibrant)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Categorize runs
    runs_2m, runs_10m = categorize_runs(args.run_paths)
    
    print(f"\n2M Parameter Models: {list(runs_2m.keys())}")
    print(f"10M Parameter Models: {list(runs_10m.keys())}\n")
    
    # Plot 2M models
    summary_rows = []

    if runs_2m:
        output_path_2m = os.path.join(args.output_dir, "combined_pvalues_2M_multi.png")
        summary_2m = plot_combined_p_values(runs_2m, "≈2M", output_path_2m, args.max_hours, args.color_scheme)
        for model_name, info in summary_2m.items():
            summary_rows.append({
                "size": "≈2M",
                "model": info["display_name"],
                "best_checkpoint": info["best_checkpoint"],
                "best_p_value": info["best_p_value"],
                "time_to_threshold": info["threshold_checkpoint"],
            })
    else:
        print("No 2M parameter models found")
    
    # Plot 10M models
    if runs_10m:
        output_path_10m = os.path.join(args.output_dir, "combined_pvalues_10M_multi.png")
        summary_10m = plot_combined_p_values(runs_10m, "≈10M", output_path_10m, args.max_hours, args.color_scheme)
        for model_name, info in summary_10m.items():
            summary_rows.append({
                "size": "≈10M",
                "model": info["display_name"],
                "best_checkpoint": info["best_checkpoint"],
                "best_p_value": info["best_p_value"],
                "time_to_threshold": info["threshold_checkpoint"],
            })
    else:
        print("No 10M parameter models found")

    # Write summary table if available
    if summary_rows:
        summary_path = os.path.join(args.output_dir, "combined_pvalues_summary.csv")
        with open(summary_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["size", "model", "best_checkpoint", "best_p_value", "time_to_p_ge_0.05"])
            for row in summary_rows:
                writer.writerow([
                    row["size"],
                    row["model"],
                    row["best_checkpoint"] if row["best_checkpoint"] is not None else "",
                    f"{row['best_p_value']:.6g}" if row["best_p_value"] is not None and not np.isnan(row["best_p_value"]) else "",
                    row["time_to_threshold"] if row["time_to_threshold"] is not None else ""
                ])
        print(f"Summary table saved to: {summary_path}")


if __name__ == "__main__":
    main()

