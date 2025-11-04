import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from datasets.nbody.dataset.synthetic_sim import GravitySim
from datasets.nbody.dataset_gravity_otf import GravityDatasetOtf
from datasets.nbody.visualization_utils import (
    interactive_plotly_offline_plot_multi_trajectory,
    plot_differences_distribution_multiplot,
    plot_energies_of_all_sims_multiplot,
    plot_energy_distributions_across_all_sims_multiplot,
    plot_energy_statistics_multiplot,
    plot_feature_distribution_multiplot,
    plot_nbodies_leaving_area_multiplot,
    plot_sharp_turns_distribution_multiplot,
    plot_sticking_and_collision_distribution_multiplot,
    plot_trajectories_static_3D_to_2D_multiplot,
    plot_trajectories_static_multiplot,
)

# Constants used in simulation
BASE_SAMPLE_FREQUENCY = 100
BASE_DT = 0.001
BASE_SIM_LEN = 500
DTS = np.linspace(0.01, 0.1, 10)
G = 2
N_BALLS = 5
SOFTENING = 0.2
NUM_SIMS_FOR_MACROS = 256
NUM_TRAJECTORIES_FOR_INTERACTIVE_PLOTS = 5


def simulate_for_dt(sim_params):
    (
        dt,
        n_balls,
        G,
        softening,
        base_sim_len,
        base_sample_frequency,
        base_dt,
        log_progress,
        seed,
    ) = sim_params
    sim = GravitySim(
        n_balls=n_balls, interaction_strength=G, dt=dt, softening=softening
    )

    T = int(base_sim_len * (base_sample_frequency / (dt / base_dt)))
    sample_freq = int(base_sample_frequency / (dt / base_dt))
    if T % sample_freq != 0:
        T = T - (T % sample_freq)

    result_pos, result_vel, _, result_mass = sim.sample_trajectory(
        T=T,
        sample_freq=sample_freq,
        random_seed=seed,
        log_progress=log_progress,
    )
    return np.array(result_pos), np.array(result_vel), np.array(result_mass)


def main():
    # Generate all trajectories
    all_trajectories = {}
    for dt in DTS:
        sim_params = [
            (
                dt,
                N_BALLS,
                G,
                SOFTENING,
                BASE_SIM_LEN,
                BASE_SAMPLE_FREQUENCY,
                BASE_DT,
                True,
                seed,
            )
            for seed in range(NUM_SIMS_FOR_MACROS)
        ]
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(simulate_for_dt, param) for param in sim_params]
            results = list(
                tqdm(
                    as_completed(futures),
                    total=len(sim_params),
                    desc=f"Simulating for dt={dt}",
                )
            )
            all_trajectories[dt] = [future.result() for future in results]

    # Find the shortest trajectory length across all dt values
    min_length = min(
        min(len(traj[0]) for traj in trajectories)
        for trajectories in all_trajectories.values()
    )

    # Truncate all trajectories to the shortest length
    for dt in all_trajectories:
        all_trajectories[dt] = [
            (traj[0][:min_length], traj[1][:min_length], traj[2])
            for traj in all_trajectories[dt]
        ]

    print(f"All trajectories have been truncated to length {min_length}")

    # Select trajectories for detailed plotting
    plot_trajectories = {
        dt: all_trajectories[dt][:NUM_TRAJECTORIES_FOR_INTERACTIVE_PLOTS] for dt in DTS
    }

    # Plotting detailed trajectories
    print("plotting detailed trajectories")
    plot_dir = (
        f"datasets/nbody/dataset/gravity/compare_dt/{'_'.join([str(dt) for dt in DTS])}"
    )
    os.makedirs(plot_dir, exist_ok=True)
    for trajectory_index in range(NUM_TRAJECTORIES_FOR_INTERACTIVE_PLOTS):
        trajectories_to_plot = []
        labels = []
        for dt, trajectories in plot_trajectories.items():
            trajectories_to_plot.append(
                trajectories[trajectory_index][0]
            )  # Assuming result_pos is what we plot
            labels.append(f"dt={dt}")
        save_dir_for_trajectory = os.path.join(
            plot_dir, f"trajectory_{trajectory_index + 1}"
        )
        os.makedirs(save_dir_for_trajectory, exist_ok=True)

        interactive_plotly_offline_plot_multi_trajectory(
            np.stack(trajectories_to_plot),
            labels=labels,
            save_dir=save_dir_for_trajectory,
        )

    # Use all trajectories for macro analysis
    print("using all trajectories for macro analysis")
    all_results_pos = np.stack(
        [
            np.stack([result[0] for result in results])
            for results in all_trajectories.values()
        ],
        axis=0,
    )
    all_results_vel = np.stack(
        [
            np.stack([result[1] for result in results])
            for results in all_trajectories.values()
        ],
        axis=0,
    )
    all_results_mass = np.stack(
        [
            np.stack([result[2] for result in results])
            for results in all_trajectories.values()
        ],
        axis=0,
    )

    # Combine data for each dt
    combined_positions = np.stack([all_results_pos[i] for i in range(len(DTS))], axis=0)
    combined_velocities = np.stack(
        [all_results_vel[i] for i in range(len(DTS))], axis=0
    )

    plot_dir = f"{plot_dir}/macros"
    os.makedirs(plot_dir, exist_ok=True)
    plot_sticking_and_collision_distribution_multiplot(
        combined_positions, save_dir=plot_dir, title_suffixes=[f"dt={dt}" for dt in DTS]
    )
    plot_trajectories_static_multiplot(
        combined_positions, save_dir=plot_dir, title_suffixes=[f"dt={dt}" for dt in DTS]
    )
    plot_trajectories_static_3D_to_2D_multiplot(
        combined_positions, save_dir=plot_dir, title_suffixes=[f"dt={dt}" for dt in DTS]
    )
    plot_feature_distribution_multiplot(
        combined_positions,
        combined_velocities,
        save_dir=plot_dir,
        title_suffixes=[f"dt={dt}" for dt in DTS],
    )
    plot_differences_distribution_multiplot(
        combined_positions,
        combined_velocities,
        save_dir=plot_dir,
        title_suffixes=[f"dt={dt}" for dt in DTS],
    )
    plot_nbodies_leaving_area_multiplot(
        combined_positions,
        save_dir=plot_dir,
        title_suffixes=[f"dt={dt}" for dt in DTS],
    )
    plot_sharp_turns_distribution_multiplot(
        combined_velocities,
        save_dir=plot_dir,
        title_suffixes=[f"dt={dt}" for dt in DTS],
    )
    dataset = GravityDatasetOtf()

    energies_array = plot_energies_of_all_sims_multiplot(
        dataset,
        combined_positions,
        combined_velocities,
        save_dir=plot_dir,
        filename="energies.png",
        title_suffixes=[f"dt={dt}" for dt in DTS],
    )
    plot_energy_distributions_across_all_sims_multiplot(
        dataset,
        combined_positions,
        combined_velocities,
        save_dir=plot_dir,
        filename="energy_distributions.png",
        title_suffixes=[f"dt={dt}" for dt in DTS],
        energies_array=energies_array,
    )
    plot_energy_statistics_multiplot(
        dataset,
        combined_positions,
        combined_velocities,
        save_dir=plot_dir,
        filename="energy_statistics.png",
        title_suffixes=[f"dt={dt}" for dt in DTS],
        energies_array=energies_array,
    )

    print(f"saved to {plot_dir}")


if __name__ == "__main__":
    main()
