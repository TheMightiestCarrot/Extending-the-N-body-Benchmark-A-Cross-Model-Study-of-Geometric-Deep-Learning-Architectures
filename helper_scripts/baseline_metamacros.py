import json
import os

import numpy as np

from datasets.nbody.dataset_gravity_otf import GravityDatasetOtf
from helper_scripts.plot_macros import plot_macros
from helper_scripts.plot_metamacros import (
    plot_energy_conservation_over_time,
    plot_kl_divergence_over_time,
)

dataset = GravityDatasetOtf()

output_dir = "baseline_metamacros"
num_batches = 10
num_sims_per_batch = 64


def generate_trajectories():
    for i in range(num_batches):
        batch_dir = os.path.join(output_dir, f"batch_{i}")
        os.makedirs(batch_dir, exist_ok=True)
        # Assuming a function `simulate_batch` that takes a directory and number of simulations
        batch_data, metadata = dataset.get_ground_truth_trajectories(num_sims_per_batch)
        for sim_index, trajectory in enumerate(batch_data):
            loc, vel, _, _ = trajectory
            np.save(os.path.join(batch_dir, f"loc_pred_sim_{sim_index}.npy"), loc)
            np.save(  # Saving predicted velocities for each simulation
                os.path.join(batch_dir, f"vel_pred_sim_{sim_index}.npy"), vel
            )
            metadata_save_path = (
                f"{batch_dir}/{metadata['dataset_name']}_dataset/metadata.json"
            )
            os.makedirs(os.path.dirname(metadata_save_path), exist_ok=True)
            with open(metadata_save_path, "w") as f:
                json.dump(metadata, f)


def calculate_macros():
    counter = 1
    for i in range(num_batches):
        for j in range(i + 1, num_batches):
            if i == j:
                continue
            batch_dir_1 = os.path.join(output_dir, f"batch_{i}")
            batch_dir_2 = os.path.join(output_dir, f"batch_{j}")

            # Initialize arrays to store positions and velocities from both batches
            all_positions = np.zeros((2, num_sims_per_batch, 500, 5, 3))
            all_velocities = np.zeros((2, num_sims_per_batch, 500, 5, 3))

            # Load data for each simulation from both batches
            for sim_index in range(num_sims_per_batch):
                positions_1 = np.load(
                    os.path.join(batch_dir_1, f"loc_pred_sim_{sim_index}.npy")
                )
                velocities_1 = np.load(
                    os.path.join(batch_dir_1, f"vel_pred_sim_{sim_index}.npy")
                )
                positions_2 = np.load(
                    os.path.join(batch_dir_2, f"loc_pred_sim_{sim_index}.npy")
                )
                velocities_2 = np.load(
                    os.path.join(batch_dir_2, f"vel_pred_sim_{sim_index}.npy")
                )

                all_positions[0, sim_index] = positions_1
                all_velocities[0, sim_index] = velocities_1
                all_positions[1, sim_index] = positions_2
                all_velocities[1, sim_index] = velocities_2

            plot_macros(
                all_positions,
                all_velocities,
                dataset=dataset,
                plot_dir=f"{output_dir}/pair_{counter}/plots",
            )
            counter += 1
    return counter


def load_data(file_name, dir):
    # Load sticking distributions
    file_path = os.path.join(dir, "plots", file_name)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data


def compute_metamacros(num_pairs):
    sticking_distributions = {}
    collision_distributions = {}
    difference_distributions = {}
    feature_distributions = {}
    energy_statistics = {}

    for i in range(num_pairs):
        pair_dir = os.path.join(output_dir, f"pair_{i}")
        # Load macro data, assuming it's saved in a JSON file
        sticking_distributions[i] = load_data("sticking_distributions.json", pair_dir)
        collision_distributions[i] = load_data("collision_distributions.json", pair_dir)
        difference_distributions[i] = load_data(
            "difference_distributions.json", pair_dir
        )
        feature_distributions[i] = load_data("feature_distributions.json", pair_dir)
        energy_statistics[i] = load_data("energy_statistics.json", pair_dir)

    save_dir = f"{output_dir}/metaplots"
    os.makedirs(save_dir, exist_ok=True)
    plot_kl_divergence_over_time(sticking_distributions, "sticking_histogram", save_dir)
    plot_kl_divergence_over_time(
        collision_distributions, "collision_histogram", save_dir
    )

    plot_kl_divergence_over_time(
        difference_distributions, "position_difference", save_dir
    )
    plot_kl_divergence_over_time(
        difference_distributions, "velocity_difference", save_dir
    )

    plot_kl_divergence_over_time(feature_distributions, "position", save_dir)
    plot_kl_divergence_over_time(feature_distributions, "velocity", save_dir)

    plot_energy_conservation_over_time(energy_statistics, save_dir)
    print(f"Saved metamacros to {save_dir}")


def main():
    os.makedirs(output_dir, exist_ok=True)
    generate_trajectories()
    num_pairs = calculate_macros()
    compute_metamacros(num_pairs)


if __name__ == "__main__":
    main()
