import argparse
import glob
import os
import concurrent.futures

import numpy as np

from datasets.nbody.visualization_utils import (
    interactive_plotly_offline_plot_multi_trajectory,
    load_dataset_from_metadata_file,
    plot_energies_of_all_sims_multiplot,
)
from utils.nbody_utils import get_dataset_metadata_path


def visualize_simulation_by_sim_index(args):
    folder, sim_index = args
    try:
        print(f"visualizing sim index {sim_index} from {folder}")
        # load data
        loc_actual = np.load(f"{folder}/loc_actual_sim_{sim_index}.npy")
        loc_pred = np.load(f"{folder}/loc_pred_sim_{sim_index}.npy")
        vel_actual = np.load(f"{folder}/vel_actual_sim_{sim_index}.npy")
        vel_pred = np.load(f"{folder}/vel_pred_sim_{sim_index}.npy")

        # if lengths aren't the same, throw exception
        if len(loc_actual) != len(loc_pred):
            raise ValueError("Lengths of actual and predicted locations are not the same")
        if len(vel_actual) != len(vel_pred):
            raise ValueError("Lengths of actual and predicted velocities are not the same")

        save_dir = os.path.join(folder, "plots", f"sim_{sim_index}")
        os.makedirs(save_dir, exist_ok=True)

        combined_locations = np.stack([loc_actual, loc_pred], axis=0)
        combined_velocities = np.stack([vel_actual, vel_pred], axis=0)
        interactive_plotly_offline_plot_multi_trajectory(
            combined_locations, labels=["actual", "predicted"], save_dir=save_dir
        )

        dataset_metadata_path = get_dataset_metadata_path(folder)
        dataset = load_dataset_from_metadata_file(dataset_metadata_path)

        # add a dimension of 1 (one simulation)
        loc_pred = np.expand_dims(loc_pred, axis=0)
        vel_pred = np.expand_dims(vel_pred, axis=0)
        loc_actual = np.expand_dims(loc_actual, axis=0)
        vel_actual = np.expand_dims(vel_actual, axis=0)

        combined_locations = np.expand_dims(combined_locations, axis=1)
        combined_velocities = np.expand_dims(combined_velocities, axis=1)

        plot_energies_of_all_sims_multiplot(
            dataset,
            combined_locations,
            combined_velocities,
            save_dir=save_dir,
            title_suffixes=["ground truth", "predicted"],
        )
    except Exception as e:
        print(f"Error processing simulation {sim_index}: {e}")
        return None
    return save_dir


def expand_index_list(index_list):
    expanded = []
    for item in index_list:
        if '-' in item:
            start, end = map(int, item.split('-'))
            expanded.extend(range(start, end + 1))
        else:
            expanded.append(int(item))
    return sorted(set(expanded))


def main():
    parser = argparse.ArgumentParser(description="Visualize simulation trajectories")
    parser.add_argument(
        "--folder",
        type=str,
        help="Folder containing the generated trajectories",
    )
    parser.add_argument(
        "--sim-index",
        nargs='+',
        required=True,
        help="Index(es) of simulation(s) to visualize. Can be single numbers or ranges (e.g., 0-3)",
    )
    args = parser.parse_args()

    folder = args.folder
    if not folder:
        folder = max(
            filter(
                os.path.isdir,
                glob.glob("models/equiformer_v2/runs/*/generated_trajectories"),
            ),
            key=os.path.getmtime,
        )
    
    # Expand index list
    sim_indexes = expand_index_list(args.sim_index)
    
    # Create argument tuples
    args_list = [(folder, idx) for idx in sim_indexes]
    
    # Process in parallel using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(visualize_simulation_by_sim_index, args_list))
    
    print(f"Processed {len([r for r in results if r])}/{len(sim_indexes)} simulations")


if __name__ == "__main__":
    main()
