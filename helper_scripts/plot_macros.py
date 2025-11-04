import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from datasets.nbody.visualization_utils import (
    load_dataset_from_metadata_file,
    plot_differences_distribution_multiplot,
    plot_energies_of_all_sims_multiplot,
    plot_energy_distributions_across_all_sims_multiplot,
    plot_energy_statistics_multiplot,
    plot_feature_distribution_multiplot,
    plot_group_collision_distribution_multiplot,
    plot_max_com_distance_multiplot,
    plot_momentum_statistics,
    plot_nbodies_leaving_area_multiplot,
    plot_sharp_turns_distribution_multiplot,
    plot_sticking_and_collision_distribution_multiplot,
    plot_trajectories_static_3D_to_2D_multiplot,
    plot_trajectories_static_multiplot,
)
from utils.nbody_utils import get_dataset_metadata_path


def plot_macros(
    combined_positions,
    combined_velocities,
    dataset,
    title_suffixes=["ground truth", "predicted"],
    plot_dir=None,
):
    print("### PLOTTING MACROS ###")
    os.makedirs(plot_dir, exist_ok=True)

    # List of plotting functions to iterate over
    plot_functions = [
        plot_sticking_and_collision_distribution_multiplot,
        # plot_trajectories_static_multiplot,
        plot_trajectories_static_3D_to_2D_multiplot,
        plot_feature_distribution_multiplot,
        plot_differences_distribution_multiplot,
        plot_group_collision_distribution_multiplot,
        plot_energies_of_all_sims_multiplot,
        plot_nbodies_leaving_area_multiplot,
        plot_max_com_distance_multiplot,
        plot_momentum_statistics,
        plot_sharp_turns_distribution_multiplot,
    ]

    # First batch of plots
    with ProcessPoolExecutor() as executor:
        futures = []
        for plot_func in plot_functions:
            print(f"Plotting: {plot_func.__name__}")
            args = {
                "save_dir": plot_dir,
                "title_suffixes": title_suffixes,
            }

            # Assign appropriate arguments based on the plotting function
            if plot_func.__name__ in [
                "plot_sticking_and_collision_distribution_multiplot",
                "plot_trajectories_static_multiplot",
                "plot_trajectories_static_3D_to_2D_multiplot",
                "plot_group_collision_distribution_multiplot",
                "plot_nbodies_leaving_area_multiplot",
                "plot_max_com_distance_multiplot",
            ]:
                args["loc"] = combined_positions
            elif plot_func.__name__ == "plot_feature_distribution_multiplot":
                args["loc"] = combined_positions
                args["vel"] = combined_velocities
            elif plot_func.__name__ == "plot_differences_distribution_multiplot":
                args["loc"] = combined_positions
                args["vel"] = combined_velocities
            elif plot_func.__name__ == "plot_energies_of_all_sims_multiplot":
                args["dataset"] = dataset
                args["loc"] = combined_positions
                args["vel"] = combined_velocities
            elif plot_func.__name__ == "plot_momentum_statistics":
                args["vel"] = combined_velocities
            elif plot_func.__name__ == "plot_sharp_turns_distribution_multiplot":
                args["vel"] = combined_velocities

            futures.append(executor.submit(plot_func, **args))

        # Retrieve energies_array from the energies plotting function
        energies_array = None
        for future in futures:
            result = future.result()
            if (
                future.done()
                and plot_func.__name__ == "plot_energies_of_all_sims_multiplot"
            ):
                energies_array = result

    # Second batch of plots that depend on energies_array
    with ProcessPoolExecutor() as executor:
        futures = []

        print("Plotting energy distributions across all simulations multiplot")
        futures.append(
            executor.submit(
                plot_energy_distributions_across_all_sims_multiplot,
                dataset=dataset,
                loc=combined_positions,
                vel=combined_velocities,
                save_dir=plot_dir,
                energies_array=energies_array,
                title_suffixes=title_suffixes,
            )
        )

        print("Plotting energy statistics multiplot")
        futures.append(
            executor.submit(
                plot_energy_statistics_multiplot,
                dataset=dataset,
                loc=combined_positions,
                vel=combined_velocities,
                save_dir=plot_dir,
                energies_array=energies_array,
                title_suffixes=title_suffixes,
            )
        )

        for future in futures:
            future.result()

    print(f"Saved plots to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize simulation trajectories")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder containing the generated trajectories",
    )
    args = parser.parse_args()
    folder = args.folder

    # Load data
    sim_files = glob.glob(os.path.join(folder, "loc_pred_sim_*.npy"))
    num_sims = len(sim_files)
    print(f"Number of simulations: {num_sims}")

    positions_pred = []
    velocities_pred = []
    positions_actual = []
    velocities_actual = []

    # Load predicted and actual positions and velocities
    for i in range(num_sims):
        loc_pred_path = f"{folder}/loc_pred_sim_{i}.npy"
        vel_pred_path = f"{folder}/vel_pred_sim_{i}.npy"
        loc_actual_path = f"{folder}/loc_actual_sim_{i}.npy"
        vel_actual_path = f"{folder}/vel_actual_sim_{i}.npy"

        if (
            os.path.exists(loc_pred_path)
            and os.path.exists(vel_pred_path)
            and os.path.exists(loc_actual_path)
            and os.path.exists(vel_actual_path)
        ):
            loc_pred = np.load(loc_pred_path)
            vel_pred = np.load(vel_pred_path)
            loc_actual = np.load(loc_actual_path)
            vel_actual = np.load(vel_actual_path)

            positions_pred.append(loc_pred)
            velocities_pred.append(vel_pred)
            positions_actual.append(loc_actual)
            velocities_actual.append(vel_actual)
        else:
            print(f"Missing data for simulation {i}, skipping.")

    positions_pred = np.array(positions_pred)
    velocities_pred = np.array(velocities_pred)
    positions_actual = np.array(positions_actual)
    velocities_actual = np.array(velocities_actual)

    # Load the dataset metadata
    dataset_metadata_path = get_dataset_metadata_path(folder)
    dataset = load_dataset_from_metadata_file(dataset_metadata_path)

    combined_positions = np.stack([positions_actual, positions_pred], axis=0)
    combined_velocities = np.stack([velocities_actual, velocities_pred], axis=0)

    print(f"Plotting simulations from {folder}")
    plot_macros(
        combined_positions=combined_positions,
        combined_velocities=combined_velocities,
        dataset=dataset,
        title_suffixes=["ground truth", "predicted"],
        plot_dir=os.path.join(os.path.dirname(folder), "plots"),
    )


if __name__ == "__main__":
    main()
