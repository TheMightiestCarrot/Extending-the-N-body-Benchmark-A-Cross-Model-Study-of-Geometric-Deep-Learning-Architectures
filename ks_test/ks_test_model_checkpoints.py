import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpmath import log, mp
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2


def perform_ks_tests(run_path, max_steps=None):
    normalized_run_path = os.path.normpath(run_path)
    candidate_checkpoint_path = os.path.join(normalized_run_path, "checkpoints")

    if os.path.isdir(candidate_checkpoint_path):
        checkpoint_path = candidate_checkpoint_path
        output_dir = normalized_run_path
    elif os.path.isdir(normalized_run_path) and os.path.basename(normalized_run_path) == "checkpoints":
        checkpoint_path = normalized_run_path
        output_dir = os.path.dirname(normalized_run_path)
    else:
        print(
            f"No checkpoint directory found at: {run_path}."
            " Expected a run directory containing a 'checkpoints' subfolder or"
            " pass the 'checkpoints' directory directly."
        )
        return None, None, None, None, None

    checkpoint_numbers = []
    p_values_by_key = {}
    ground_truth_data = {}
    predicted_data = {}

    # Define filename and dict_key pairs
    file_info_list = [
        {"filename": "collision_distributions", "dict_key": "collision_histogram"},
        {
            "filename": "group_collision_distribution",
            "dict_key": "group_collision_histogram",
        },
        {"filename": "leaving_distribution", "dict_key": "leaving_count"},
        {"filename": "sharp_turn_30_distribution", "dict_key": "sharp_turn_count_30"},
        {"filename": "sharp_turn_45_distribution", "dict_key": "sharp_turn_count_45"},
        {"filename": "sticking_distributions", "dict_key": "sticking_histogram"},
    ]

    # Initialize dictionaries for each dict_key
    for file_info in file_info_list:
        dict_key = file_info["dict_key"]
        p_values_by_key[dict_key] = []
        ground_truth_data[dict_key] = []
        predicted_data[dict_key] = []

    checkpoint_dirs = [d for d in os.listdir(checkpoint_path) if d.isdigit()]
    if not checkpoint_dirs:
        print(f"No numeric checkpoint directories found in: {checkpoint_path}")
        return np.array([]), {k: np.array([]) for k in p_values_by_key}, ground_truth_data, predicted_data, output_dir

    for checkpoint_dir in sorted(checkpoint_dirs, key=int):
        checkpoint_number = int(checkpoint_dir)
        if max_steps is not None and checkpoint_number > max_steps:
            break

        checkpoint_dir_path = os.path.join(checkpoint_path, checkpoint_dir)
        if not os.path.isdir(checkpoint_dir_path):
            continue

        generated_dirs = [
            d
            for d in os.listdir(checkpoint_dir_path)
            if d.startswith("generated_trajectories")
        ]

        plots_candidates = []

        if generated_dirs:
            latest_generated_dir = max(generated_dirs)
            generated_root = os.path.join(checkpoint_dir_path, latest_generated_dir)

            timestamp_dirs = [
                d
                for d in os.listdir(generated_root)
                if d.startswith("20") and os.path.isdir(os.path.join(generated_root, d))
            ]
            if timestamp_dirs:
                latest_timestamp_dir = max(timestamp_dirs)
                plots_candidates.append(
                    os.path.join(generated_root, latest_timestamp_dir, "plots")
                )
                plots_candidates.append(os.path.join(generated_root, latest_timestamp_dir))

        # Fallback: generated files stored directly under the checkpoint directory
        plots_candidates.append(checkpoint_dir_path)

        plots_path = None
        for candidate in plots_candidates:
            if not os.path.isdir(candidate):
                continue
            has_expected_files = any(
                os.path.exists(os.path.join(candidate, f"{info['filename']}.json"))
                for info in file_info_list
            )
            if has_expected_files:
                plots_path = candidate
                break

        if plots_path is None:
            print(
                f"No macro-distribution JSON files found for checkpoint {checkpoint_dir} in"
                f" any known location. Skipping."
            )
            continue

        checkpoint_numbers.append(checkpoint_number)

        for file_info in file_info_list:
            filename = file_info["filename"]
            dict_key = file_info["dict_key"]
            file_path = os.path.join(plots_path, f"{filename}.json")
            print("Processing file:", file_path)
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist. Skipping.")
                p_values_by_key[dict_key].append(np.nan)
                ground_truth_data[dict_key].append(None)
                predicted_data[dict_key].append(None)
                continue

            with open(file_path, "r") as f:
                data = json.load(f)

            try:
                ground_truth = np.array(data["ground truth"][dict_key])
                predicted = np.array(data["predicted"][dict_key])
            except KeyError as e:
                print(
                    f"KeyError: {e} not found in {file_path}. Available keys: {data.keys()}. Skipping."
                )
                p_values_by_key[dict_key].append(np.nan)
                ground_truth_data[dict_key].append(None)
                predicted_data[dict_key].append(None)
                continue

            if len(ground_truth) == 0 or len(predicted) == 0:
                print(f"Empty data for '{dict_key}' in {file_path}. Skipping.")
                p_values_by_key[dict_key].append(np.nan)
                ground_truth_data[dict_key].append(None)
                predicted_data[dict_key].append(None)
                continue

            ground_truth_data[dict_key].append(ground_truth)
            predicted_data[dict_key].append(predicted)

            _, p_value = stats.ks_2samp(ground_truth, predicted)
            p_values_by_key[dict_key].append(p_value)

    # Sort checkpoint numbers and corresponding data
    if checkpoint_numbers:
        sorted_indices = np.argsort(checkpoint_numbers)
        checkpoint_numbers = np.array(checkpoint_numbers)[sorted_indices]
        for key in p_values_by_key.keys():
            p_values_by_key[key] = np.array(p_values_by_key[key])[sorted_indices]
            ground_truth_data[key] = [ground_truth_data[key][i] for i in sorted_indices]
            predicted_data[key] = [predicted_data[key][i] for i in sorted_indices]

    return checkpoint_numbers, p_values_by_key, ground_truth_data, predicted_data, output_dir


def calculate_baseline_p_values(checkpoint_numbers, ground_truth_data):
    baseline_p_values = {key: [] for key in ground_truth_data.keys()}
    num_checkpoints = len(checkpoint_numbers)

    for key in ground_truth_data.keys():
        for _ in range(num_checkpoints):
            valid_gt_data = [gt for gt in ground_truth_data[key] if gt is not None]
            if len(valid_gt_data) >= 2:
                gt1, gt2 = random.sample(valid_gt_data, 2)
                _, p_value = stats.ks_2samp(gt1, gt2)
                baseline_p_values[key].append(p_value)
            else:
                baseline_p_values[key].append(np.nan)

        # Ensure all arrays in baseline_p_values have the same length as checkpoint_numbers
        baseline_p_values[key] = baseline_p_values[key][:num_checkpoints]

    return baseline_p_values


def plot_results(
    checkpoint_numbers,
    combined_p_values,
    p_values_by_key,
    combined_baseline_p_values,
    run_path,
):
    # Check if we have write permissions for the output directory
    if not os.access(run_path, os.W_OK):
        print(f"\n{'='*80}")
        print(f"ERROR: No write permission for directory: {run_path}")
        print(f"{'='*80}")
        print(f"\nThis directory was likely created by Docker with restricted permissions.")
        print(f"To fix this, run the following command:\n")
        print(f"  sudo chmod -R a+w {run_path}\n")
        print(f"{'='*80}\n")
        raise PermissionError(f"No write permission for directory: {run_path}")
    
    min_positive = 1e-300  # Smallest positive value to display

    # Prepare combined p-values for plotting
    combined_p_values = np.array(combined_p_values)
    combined_p_values = np.clip(combined_p_values, min_positive, None)

    # Plot combined p-values as a function of checkpoint number
    plt.figure(figsize=(12, 6))
    plt.plot(
        checkpoint_numbers,
        combined_p_values,
        label="Combined GT-P p-value (Fisher's method)",
        marker="o",
    )
    if combined_baseline_p_values is not None:
        combined_baseline_p_values = np.array(combined_baseline_p_values)
        combined_baseline_p_values = np.clip(
            combined_baseline_p_values, min_positive, None
        )
        plt.plot(
            checkpoint_numbers,
            combined_baseline_p_values,
            label="Combined GT-GT p-value (Fisher's method)",
            linestyle="--",
            alpha=0.5,
            marker="x",
        )
    plt.axhline(y=0.05, color="r", linestyle=":", label="p-value = 0.05")
    plt.xlabel("Checkpoint Number")
    plt.ylabel("Combined p-value")
    plt.title("Combined p-value vs. Checkpoint Number using Fisher's Method")
    plt.legend()
    plt.yscale("log")  # Use log scale for y-axis to better visualize small p-values
    # plt.ylim(min_positive, 1)  # Adjust y-axis limits
    plt.grid(True)
    plt.tight_layout()
    combined_p_values_plot_path = os.path.join(
        run_path, "combined_p_values_vs_checkpoints.png"
    )
    plt.savefig(combined_p_values_plot_path)
    plt.close()

    print(f"Combined p-values plot saved in: {combined_p_values_plot_path}")

    # Plot individual p-values for each macroproperty distribution
    plt.figure(figsize=(12, 6))
    for key, p_values in p_values_by_key.items():
        plt.plot(checkpoint_numbers, p_values, label=key)
    plt.axhline(y=0.05, color="r", linestyle=":", label="p-value = 0.05")
    plt.xlabel("Checkpoint Number")
    plt.ylabel("p-value")
    plt.title(
        "Individual p-values for Each Macroproperty Distribution vs. Checkpoint Number"
    )
    plt.legend()
    plt.yscale("log")  # Use log scale for y-axis to better visualize small p-values
    plt.grid(True)
    plt.tight_layout()
    individual_p_values_plot_path = os.path.join(
        run_path, "individual_p_values_vs_checkpoints.png"
    )
    plt.savefig(individual_p_values_plot_path)
    plt.close()

    print(f"Individual p-values plot saved in: {individual_p_values_plot_path}")

    # Create an interactive version of the combined p-values plot
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    fig.add_trace(
        go.Scatter(
            x=checkpoint_numbers,
            y=combined_p_values,
            mode="lines+markers",
            name="Combined GT-P p-value (Fisher's method)",
        )
    )
    if combined_baseline_p_values is not None:
        fig.add_trace(
            go.Scatter(
                x=checkpoint_numbers,
                y=combined_baseline_p_values,
                mode="lines+markers",
                line=dict(dash="dash"),
                name="Combined GT-GT p-value (Fisher's method)",
                opacity=0.5,
            )
        )
    fig.add_hline(
        y=0.05,
        line_dash="dot",
        annotation_text="p-value = 0.05",
        annotation_position="bottom right",
        line_color="green",
    )

    fig.update_layout(
        title="Combined p-value vs. Checkpoint Number using Fisher's Method",
        xaxis_title="Checkpoint Number",
        yaxis_title="Combined p-value",
        yaxis_type="log",
    )

    interactive_plot_path = os.path.join(
        run_path, "interactive_combined_p_values_vs_checkpoints.html"
    )
    fig.write_html(interactive_plot_path)

    print(f"Interactive combined p-values plot saved in: {interactive_plot_path}")


def ks_test_model_checkpoints(run_path, training=False, max_steps=None):
    # Set the desired precision for mpmath
    mp.dps = 200  # Adjust as needed for higher precision

    checkpoint_numbers, p_values_by_key, ground_truth_data, predicted_data, output_dir = (
        perform_ks_tests(run_path, max_steps)
    )

    if checkpoint_numbers is None or len(checkpoint_numbers) == 0:
        print(
            "No checkpoints were processed. Ensure that the run path is correct and contains valid data."
        )
        return

    # Calculate baseline p-values
    if not training:
        baseline_p_values = calculate_baseline_p_values(
            checkpoint_numbers, ground_truth_data
        )
    else:
        baseline_p_values = None

    # Convert p_values_by_key to a numpy array for easier manipulation
    p_values_matrix = np.array(
        list(p_values_by_key.values())
    )  # Shape: (num_macros, num_checkpoints)

    # Initialize combined p-values array
    combined_p_values = []

    min_positive = 1e-300  # Minimum positive value to avoid zeros in plotting

    for i in range(p_values_matrix.shape[1]):  # Iterate over checkpoints
        p_values = p_values_matrix[:, i]
        # Remove NaN values
        p_values = p_values[~np.isnan(p_values)]
        if len(p_values) == 0:
            combined_p_values.append(np.nan)
            continue
        # Use mpmath for high-precision logarithms
        chi_stat = -2 * mp.fsum([log(mp.mpf(p)) for p in p_values])
        chi_stat = float(chi_stat)  # Convert back to float
        # Degrees of freedom: 2 times the number of p-values combined
        dof = 2 * len(p_values)
        # Combined p-value using the survival function for better precision
        combined_p_value = chi2.sf(chi_stat, dof)
        # Handle zeros
        if combined_p_value == 0:
            combined_p_value = min_positive
        combined_p_values.append(combined_p_value)

    # For baseline p-values (if applicable), perform the same combination
    if not training and baseline_p_values is not None:
        baseline_p_values_matrix = np.array(list(baseline_p_values.values()))
        combined_baseline_p_values = []

        for i in range(baseline_p_values_matrix.shape[1]):
            p_values = baseline_p_values_matrix[:, i]
            p_values = p_values[~np.isnan(p_values)]
            if len(p_values) == 0:
                combined_baseline_p_values.append(np.nan)
                continue
            chi_stat = -2 * mp.fsum([log(mp.mpf(p)) for p in p_values])
            chi_stat = float(chi_stat)
            dof = 2 * len(p_values)
            combined_p_value = chi2.sf(chi_stat, dof)
            if combined_p_value == 0:
                combined_p_value = min_positive
            combined_baseline_p_values.append(combined_p_value)
    else:
        combined_baseline_p_values = None

    # Plot results with combined p-values
    plot_results(
        checkpoint_numbers,
        combined_p_values,
        p_values_by_key,
        combined_baseline_p_values,
        output_dir,
    )

    # Print best checkpoints for individual macros and overall combined p-value
    print("\nBest checkpoints:")
    for key, p_values in p_values_by_key.items():
        if np.all(np.isnan(p_values)):
            print(f"{key}: No valid p-values found.")
        else:
            best_checkpoint_index = np.nanargmax(p_values)
            best_checkpoint = checkpoint_numbers[best_checkpoint_index]
            best_p_value = p_values[best_checkpoint_index]
            print(f"{key}: Checkpoint {best_checkpoint} (p-value: {best_p_value:.4f})")

    if np.all(np.isnan(combined_p_values)):
        print("Overall (combined): No valid p-values found.")
    else:
        best_overall_checkpoint_index = np.nanargmax(combined_p_values)
        best_overall_checkpoint = checkpoint_numbers[best_overall_checkpoint_index]
        best_overall_p_value = combined_p_values[best_overall_checkpoint_index]
        print(
            f"Overall (combined): Checkpoint {best_overall_checkpoint} (combined p-value: {best_overall_p_value:.4e})"
        )

    # Optionally, print data for the last checkpoint if available
    if len(checkpoint_numbers) > 0:
        last_checkpoint = checkpoint_numbers[-1]
        print(f"\nData for the last checkpoint ({last_checkpoint}):")
        for key in p_values_by_key.keys():
            gt_data = ground_truth_data[key][-1]
            pred_data = predicted_data[key][-1]
            p_value = p_values_by_key[key][-1]
            print(f"{key}:")
            print(f"  Ground Truth: {gt_data}")
            print(f"  Predicted: {pred_data}")
            print(f"  p-value: {p_value:.4f}")
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run KS test on model checkpoints")
    parser.add_argument("--run-path", type=str, help="Path to the run directory")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to consider (default: use all)",
    )
    args = parser.parse_args()

    ks_test_model_checkpoints(args.run_path, training=False, max_steps=args.max_steps)
