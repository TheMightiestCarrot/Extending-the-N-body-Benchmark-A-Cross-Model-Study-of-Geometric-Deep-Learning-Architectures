import inspect
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages

from datasets.nbody.dataset_gravity_otf import GravityDatasetOtf
from utils.nbody_utils import is_headless

FONT_SIZE = 20
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.titlesize"] = FONT_SIZE
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"] = FONT_SIZE
plt.rcParams["ytick.labelsize"] = FONT_SIZE
plt.rcParams["legend.fontsize"] = FONT_SIZE


@staticmethod
def capitalize_title(title):
    return " ".join(word.capitalize() for word in title.split())


def plot_sharp_turns_distribution_multiplot(
    vel,
    num_bins=60,
    save_dir=None,
    title_suffixes=None,
    angle_thresholds=[30, 45],
    filename_sharp_turns_pattern="sharp_turns_distribution_{}.png",
):
    """
    Counts sharp turns for every body in every simulation for multiple angle thresholds.
    Saves separate JSON files and plots for each angle threshold.
    """
    num_batches = vel.shape[0]
    current_time = datetime.now().isoformat()

    for angle_threshold in angle_thresholds:
        global_sharp_turns_counts = []
        for batch in range(num_batches):
            global_sharp_turns_counts.append(
                count_sharp_turns(vel[batch, ...], angle_threshold)
            )

        min_sharp_turns = min([min(counts) for counts in global_sharp_turns_counts])
        max_sharp_turns = max([max(counts) for counts in global_sharp_turns_counts])
        bin_edges_sharp_turns = np.linspace(
            min_sharp_turns, max_sharp_turns, num_bins + 1
        )

        fig_sharp_turns, axs_sharp_turns = plt.subplots(
            num_batches, 1, figsize=(10, 6 * num_batches), sharex=True, sharey=True
        )
        if num_batches == 1:
            axs_sharp_turns = [
                axs_sharp_turns
            ]  # Ensure axs is iterable for a single batch

        sharp_turn_data = {}
        for batch, ax in enumerate(axs_sharp_turns):
            counts, _, _ = ax.hist(
                global_sharp_turns_counts[batch],
                bins=bin_edges_sharp_turns,
                alpha=0.7,
                color="pink",
                edgecolor="black",
            )
            ax.set_xlabel("Sharp Turns Count")
            ax.set_ylabel("Frequency")
            suffix = (
                title_suffixes[batch]
                if title_suffixes is not None
                else f"batch_{batch}"
            )
            ax.set_title(f"Sharp Turn Distribution {angle_threshold}° {suffix}".title())

            # Prepare data for JSON
            sharp_turn_data[suffix] = {
                "timestamp": current_time,
                f"sharp_turn_count_{angle_threshold}": global_sharp_turns_counts[
                    batch
                ].tolist(),
            }

        plt.tight_layout()
        if save_dir:
            filename_sharp_turns = filename_sharp_turns_pattern.format(angle_threshold)
            plt.savefig(os.path.join(save_dir, filename_sharp_turns))
        plt.close(fig_sharp_turns)  # Close the figure to free memory

        # Save the sharp_turn_data to a JSON file for this angle threshold
        sharp_turns_file = os.path.join(
            save_dir, f"sharp_turn_{angle_threshold}_distribution.json"
        )
        with open(sharp_turns_file, "w") as file:
            json.dump(sharp_turn_data, file, indent=4)


def plot_nbodies_leaving_area_multiplot(
    loc,
    save_dir=None,
    title_suffixes=None,
    filename_leaving="leaving_distribution.png",
):
    num_batches = loc.shape[0]
    global_leaving_counts = []
    num_bins = loc.shape[3] + 1
    # Initialize global min and max for both leavings and collisions

    for batch in range(num_batches):
        leaving_counts = count_balls_leaving_defined_area(loc[batch, :, :, :])
        global_leaving_counts.append(leaving_counts)

    # Determine global min and max for leavings and collisions to ensure consistent bin width
    min_leaving = min([min(counts) for counts in global_leaving_counts])
    max_leaving = max([max(counts) for counts in global_leaving_counts])
    bin_edges_leaving = np.linspace(min_leaving, max_leaving, num_bins + 1)

    # Plotting for leavings
    fig_leaving, axs_leaving = plt.subplots(
        num_batches, 1, figsize=(10, 6 * num_batches), sharex=True, sharey=True
    )
    if num_batches == 1:
        axs_leaving = [axs_leaving]  # Ensure axs is iterable for a single batch

    leaving_histograms = []
    for batch, ax in enumerate(axs_leaving):
        counts, _, _ = ax.hist(
            global_leaving_counts[batch],
            bins=bin_edges_leaving,
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        leaving_histograms.append(global_leaving_counts[batch])
        ax.set_xlabel("Leaving Count")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Leaving Distribution {str(title_suffixes[batch]) if title_suffixes is not None else ''}".title()
        )

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{filename_leaving}")
    plt.close(fig_leaving)  # Close the figure to free memory

    leaving_file = os.path.join(save_dir, "leaving_distribution.json")

    current_time = datetime.now().isoformat()
    leaving_data = {}

    for batch, suffix in enumerate(title_suffixes):
        # leaving distribution
        leaving_data[suffix] = {
            "timestamp": current_time,
            "leaving_count": global_leaving_counts[batch].tolist(),
        }

    with open(leaving_file, "w") as file:
        json.dump(leaving_data, file, indent=4)


def plot_max_com_distance_multiplot(
    loc,
    num_bins=60,
    save_dir=None,
    title_suffixes=None,
    filename_com_distance="max_com_distance_distribution.png",
):
    num_batches = loc.shape[0]
    global_max_com_distances = []
    # Initialize global min and max for com distance

    for batch in range(num_batches):
        max_com_distance = get_max_distance_of_com_from_starting_position(
            loc[batch, :, :, :]
        )
        global_max_com_distances.append(max_com_distance)

    # Determine global min and max for com distance to ensure consistent bin width
    min_com_distance = min([min(counts) for counts in global_max_com_distances])
    max_com_distance = max([max(counts) for counts in global_max_com_distances])
    bin_edges = np.linspace(min_com_distance, max_com_distance, num_bins + 1)

    # Plotting for com
    fig, axs = plt.subplots(
        num_batches, 1, figsize=(10, 6 * num_batches), sharex=True, sharey=True
    )
    if num_batches == 1:
        axs = [axs]  # Ensure axs is iterable for a single batch

    com_distance_histogram = []
    for batch, ax in enumerate(axs):
        counts, _, _ = ax.hist(
            global_max_com_distances[batch],
            bins=bin_edges,
            alpha=0.7,
            color="black",
            edgecolor="black",
        )
        com_distance_histogram.append(global_max_com_distances[batch])
        ax.set_xlabel("Maximal Distance of Centre of Mass from Starting Position")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Distribution of Maximal distances of centre of mass from starting position {str(title_suffixes[batch]) if title_suffixes is not None else ''}".title()
        )

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{filename_com_distance}")
    plt.close(fig)  # Close the figure to free memory

    com_movement_file = os.path.join(save_dir, "max_com_distance_distribution.json")

    current_time = datetime.now().isoformat()
    com_distance_data = {}

    for batch, suffix in enumerate(title_suffixes):
        # com distance distribution
        com_distance_data[suffix] = {
            "timestamp": current_time,
            "com_movement": global_max_com_distances[batch].tolist(),
        }

    with open(com_movement_file, "w") as file:
        json.dump(com_distance_data, file, indent=4)


def plot_sticking_and_collision_distribution_multiplot(
    loc,
    num_bins=50,
    save_dir=None,
    title_suffixes=None,
    filename_sticking="sticking_distribution.png",
    filename_collision="collision_distribution.png",
):
    num_batches = loc.shape[0]

    # Initialize global min and max for both stickings and collisions
    global_sticking_counts = []
    global_collision_counts = []
    for batch in range(num_batches):
        sticking_counts, collision_counts = count_stickings_and_collisions(
            loc[batch, :, :, :]
        )
        global_sticking_counts.append(sticking_counts)
        global_collision_counts.append(collision_counts)

    # Determine global min and max for stickings and collisions to ensure consistent bin width
    min_sticking = min([min(counts) for counts in global_sticking_counts])
    max_sticking = max([max(counts) for counts in global_sticking_counts])
    bin_edges_sticking = np.linspace(min_sticking, max_sticking, num_bins + 1)

    min_collision = min([min(counts) for counts in global_collision_counts])
    max_collision = max([max(counts) for counts in global_collision_counts])
    bin_edges_collision = np.linspace(min_collision, max_collision, num_bins + 1)

    # Plotting for stickings
    fig_sticking, axs_sticking = plt.subplots(
        num_batches, 1, figsize=(10, 6 * num_batches), sharex=True, sharey=True
    )
    if num_batches == 1:
        axs_sticking = [axs_sticking]  # Ensure axs is iterable for a single batch

    sticking_histograms = []
    for batch, ax in enumerate(axs_sticking):
        counts, _, _ = ax.hist(
            global_sticking_counts[batch],
            bins=bin_edges_sticking,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        sticking_histograms.append(global_sticking_counts[batch])
        ax.set_xlabel("Sticking Count")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Sticking Distribution {str(title_suffixes[batch]) if title_suffixes is not None else ''}".title()
        )

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{filename_sticking}")
    plt.close(fig_sticking)  # Close the figure to free memory

    # Plotting for collisions
    fig_collision, axs_collision = plt.subplots(
        num_batches, 1, figsize=(10, 6 * num_batches), sharex=True, sharey=True
    )
    if num_batches == 1:
        axs_collision = [axs_collision]  # Ensure axs is iterable for a single batch

    collision_histograms = []
    for batch, ax in enumerate(axs_collision):
        counts, _, _ = ax.hist(
            global_collision_counts[batch],
            bins=bin_edges_collision,
            alpha=0.7,
            color="red",
            edgecolor="black",
        )
        collision_histograms.append(global_collision_counts[batch])
        ax.set_xlabel("Collision Count")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Collision Distribution {str(title_suffixes[batch]) if title_suffixes is not None else ''}".title()
        )

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{filename_collision}")
    plt.close(fig_collision)  # Close the figure to free memory

    # Save histograms to a file for later analysis
    sticking_distribution_file = os.path.join(save_dir, "sticking_distributions.json")
    collision_distribution_file = os.path.join(save_dir, "collision_distributions.json")

    current_time = datetime.now().isoformat()
    sticking_data = {}
    collision_data = {}

    for batch, suffix in enumerate(title_suffixes):
        # sticking distribution
        sticking_data[suffix] = {
            "timestamp": current_time,
            "sticking_histogram": sticking_histograms[batch].tolist(),
        }
        # collision distribution
        collision_data[suffix] = {
            "timestamp": current_time,
            "collision_histogram": collision_histograms[batch].tolist(),
        }

    with open(sticking_distribution_file, "w") as file:
        json.dump(sticking_data, file, indent=4)
    with open(collision_distribution_file, "w") as file:
        json.dump(collision_data, file, indent=4)


def plot_trajectories_static_multiplot(
    loc,
    opacity=0.4,
    max_sims=256,
    offline_plot=is_headless(),
    save_dir=None,
    filename="trajectories_static.pdf",
    title_suffixes=None,
):
    num_batches = loc.shape[0]
    num_sims = loc.shape[1]
    num_dims = loc.shape[4]
    n_balls = loc.shape[3]

    with PdfPages(f"{save_dir}/{filename}") as pdf:
        plt.figure(
            figsize=(10, 8)
        )  # Set the plot size outside the loop to ensure consistency across batches

        # Determine the global limits for all dimensions across all batches and simulations
        global_lim_x, global_lim_y, global_lim_z = get_loc_axes_limits(loc)

        for batch in range(num_batches):
            # If the number of simulations exceeds max_sims, sample max_sims randomly
            if num_sims > max_sims:
                selected_sims = np.random.choice(num_sims, max_sims, replace=False)
            else:
                selected_sims = np.arange(num_sims)

            if num_dims == 2:
                for sim in selected_sims:
                    for n in range(n_balls):
                        plt.plot(
                            loc[batch, sim, :, n, 0],
                            loc[batch, sim, :, n, 1],
                            alpha=opacity,
                            linewidth=0.5,
                        )
                plt.xlim(global_lim_x)
                plt.ylim(global_lim_y)

            elif num_dims == 3:
                ax = plt.axes(
                    projection="3d"
                )  # Adjusted to use plt.axes for consistency in 3D plotting
                for sim in selected_sims:
                    for n in range(n_balls):
                        ax.plot(
                            loc[batch, sim, :, n, 0],
                            loc[batch, sim, :, n, 1],
                            loc[batch, sim, :, n, 2],
                            alpha=opacity,
                            linewidth=0.5,
                        )
                ax.set_xlim(global_lim_x)
                ax.set_ylim(global_lim_y)
                ax.set_zlim(global_lim_z)
            else:
                raise ValueError("Dimensions not supported for plotting")
            plt.title(
                f"Trajectories of Particles Across Simulations {str(title_suffixes[batch]) if title_suffixes is not None else ''}".title()
            )
            plt.grid(True)
            pdf.savefig()  # Adjusted to save each figure to the PDF within the loop
            plt.clf()  # Clear the figure to prepare for the next batch's plot


def plot_trajectories_static_3D_to_2D_multiplot(
    loc,
    opacity=0.4,
    max_sims=256,
    offline_plot=is_headless(),
    save_dir=None,
    filename="trajectories_static_3D_to_2D.pdf",
    title_suffixes=None,
):
    num_batches = loc.shape[0]
    num_sims = loc.shape[1]
    n_balls = loc.shape[3]

    with PdfPages(f"{save_dir}/{filename}") as pdf:
        global_lim_x, global_lim_y, global_lim_z = get_loc_axes_limits(loc)

        for batch in range(num_batches):
            # If max_sims is specified and the number of simulations exceeds it, sample max_sims randomly
            if max_sims is not None and num_sims > max_sims:
                selected_sims = np.random.choice(num_sims, max_sims, replace=False)
            else:
                selected_sims = np.arange(num_sims)

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(
                f"Trajectories of Particles Across Simulations {str(title_suffixes[batch]) if title_suffixes is not None else ''}".title(),
                fontsize=FONT_SIZE,
            )

            for sim in selected_sims:
                for n in range(n_balls):
                    # XY plane
                    axs[0].plot(
                        loc[batch, sim, :, n, 0],
                        loc[batch, sim, :, n, 1],
                        alpha=opacity,
                    )
                    axs[0].set_xlabel("X Position")
                    axs[0].set_ylabel("Y Position")
                    axs[0].set_title("XY Plane")
                    axs[0].set_xlim(global_lim_x)
                    axs[0].set_ylim(global_lim_y)

                    # XZ plane
                    axs[1].plot(
                        loc[batch, sim, :, n, 0],
                        loc[batch, sim, :, n, 2],
                        alpha=opacity,
                    )
                    axs[1].set_xlabel("X Position")
                    axs[1].set_ylabel("Z Position")
                    axs[1].set_title("XZ Plane")
                    axs[1].set_xlim(global_lim_x)
                    axs[1].set_ylim(global_lim_z)

                    # YZ plane
                    axs[2].plot(
                        loc[batch, sim, :, n, 1],
                        loc[batch, sim, :, n, 2],
                        alpha=opacity,
                    )
                    axs[2].set_xlabel("Y Position")
                    axs[2].set_ylabel("Z Position")
                    axs[2].set_title("YZ Plane")
                    axs[2].set_xlim(global_lim_y)
                    axs[2].set_ylim(global_lim_z)

                    for ax in axs:
                        ax.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig()
            plt.clf()


def get_loc_axes_limits(loc):
    num_batches = loc.shape[0]
    num_sims = loc.shape[1]
    num_dims = loc.shape[4]
    all_x_locs = np.concatenate(
        [
            loc[batch, sim, step, :, 0]
            for batch in range(num_batches)
            for sim in range(num_sims)
            for step in range(loc.shape[2])
        ]
    )
    all_y_locs = np.concatenate(
        [
            loc[batch, sim, step, :, 1]
            for batch in range(num_batches)
            for sim in range(num_sims)
            for step in range(loc.shape[2])
        ]
    )
    all_z_locs = np.concatenate(
        [
            loc[batch, sim, step, :, 2]
            for batch in range(num_batches)
            for sim in range(num_sims)
            for step in range(loc.shape[2])
        ]
    )
    global_lim_x, global_lim_y, global_lim_z = (
        (min(all_x_locs), max(all_x_locs)),
        (min(all_y_locs), max(all_y_locs)),
        (min(all_z_locs), max(all_z_locs)),
    )
    return global_lim_x, global_lim_y, global_lim_z or None


def plot_feature_distribution_multiplot(
    loc,
    vel,
    force=None,
    bins=50,
    offline_plot=is_headless(),
    save_dir=None,
    filename="histograms.png",
    title_suffixes=None,
):
    num_batches = loc.shape[0]
    if title_suffixes is None:
        title_suffixes = ["" for _ in range(num_batches)]
    num_dims = loc.shape[
        4
    ]  # Reflect new indexing due to the additional 'bodies' dimension
    dim_labels = ["x", "y", "z"][:num_dims]  # Labels for dimensions
    colors = ["red", "green", "blue"][:num_dims]  # Color for each dimension

    distributions = {
        "position": [],
        "velocity": [],
        "force": [] if force is not None else None,
    }

    fig, axs = plt.subplots(
        num_batches, 3, figsize=(15, 5 * num_batches), sharex="col", sharey="col"
    )
    for batch in range(num_batches):
        for i, (color, label) in enumerate(zip(colors, dim_labels)):
            # Positions
            pos_data = loc[batch, :, :, :, i].flatten()
            axs[batch, 0].hist(
                pos_data, bins=bins, alpha=0.5, color=color, label=f"{label} position"
            )
            axs[batch, 0].set_title(f"Positions {title_suffixes[batch]}".title())
            axs[batch, 0].legend()
            distributions["position"].append(pos_data.tolist())

            # Velocities
            vel_data = vel[batch, :, :, :, i].flatten()
            axs[batch, 1].hist(
                vel_data, bins=bins, alpha=0.5, color=color, label=f"{label} velocity"
            )
            axs[batch, 1].set_title(f"Velocities {title_suffixes[batch]}".title())
            axs[batch, 1].legend()
            distributions["velocity"].append(vel_data.tolist())

            # Forces
            if force is not None:
                force_data = force[batch, :, :, :, i].flatten()
                axs[batch, 2].hist(
                    force_data,
                    bins=bins,
                    alpha=0.5,
                    color=color,
                    label=f"{label} force",
                )
                axs[batch, 2].legend()
                axs[batch, 2].set_title(f"Forces {title_suffixes[batch]}".title())
                distributions["force"].append(force_data.tolist())

    fig.tight_layout()
    fig.savefig(f"{save_dir}/{filename}")

    # Save distributions to a file for later analysis
    distribution_file = os.path.join(save_dir, "feature_distributions.json")
    data = {}

    for batch, suffix in enumerate(title_suffixes):
        data[suffix] = {
            "timestamp": datetime.now().isoformat(),
            "position": distributions["position"][batch],
            "velocity": distributions["velocity"][batch],
            "force": distributions["force"][batch] if force is not None else None,
        }

    with open(distribution_file, "w") as file:
        json.dump(data, file, indent=4)


def plot_differences_distribution_multiplot(
    loc,
    vel,
    step=1,
    bins=100,
    offline_plot=is_headless(),
    save_dir=None,
    filename="differences.png",
    title_suffixes=None,
):
    num_batches = loc.shape[0]
    if title_suffixes is None:
        title_suffixes = ["" for _ in range(num_batches)]
    num_dims = loc.shape[4]
    dim_labels = ["x", "y", "z"][:num_dims]
    colors = ["red", "green", "blue"][:num_dims]

    distributions = {
        "position_difference": [],
        "velocity_difference": [],
    }

    fig, axs = plt.subplots(
        num_batches, 2, figsize=(20, 5 * num_batches), sharex="col", sharey="col"
    )

    # Position Differences
    for batch in range(num_batches):
        for i, (color, label) in enumerate(zip(colors, dim_labels)):
            # Calculate differences along the steps dimension (axis=1)
            diffs = np.diff(loc[batch, :, :, :, i], axis=1, n=step).flatten()
            distributions["position_difference"].append(diffs.tolist())
            axs[batch, 0].hist(
                diffs,
                bins=bins,
                alpha=0.5,
                color=color,
                label=f"{label} position difference",
            )
            axs[batch, 0].set_title(
                f"Position Differences {title_suffixes[batch]}".title()
            )
            axs[batch, 0].legend()

            # Velocity Differences
            diffs = np.diff(vel[batch, :, :, :, i], axis=1, n=step).flatten()
            distributions["velocity_difference"].append(diffs.tolist())
            axs[batch, 1].hist(
                diffs,
                bins=bins,
                alpha=0.5,
                color=color,
                label=f"{label} velocity difference",
            )
            axs[batch, 1].set_title(
                f"Velocity Differences {title_suffixes[batch]}".title()
            )
            axs[batch, 1].legend()

    fig.tight_layout()

    plt.savefig(f"{save_dir}/{filename}")

    # Save distributions to a file for later analysis
    distribution_file = os.path.join(save_dir, "difference_distributions.json")
    data = {}

    for batch, suffix in enumerate(title_suffixes):
        data[suffix] = {
            "timestamp": datetime.now().isoformat(),
            "position_difference": distributions["position_difference"][batch],
            "velocity_difference": distributions["velocity_difference"][batch],
        }

    with open(distribution_file, "w") as file:
        json.dump(data, file)


def plot_energies_of_all_sims_multiplot(
    dataset, loc, vel, save_dir=None, filename="energies.png", title_suffixes=None
):
    num_batches = loc.shape[0]
    if title_suffixes is None:
        title_suffixes = ["" for _ in range(num_batches)]
    num_simulations = loc.shape[1]

    energies_array = []
    # TODO: gain from parallelizing this is minimal, as we are already parallelizing
    # over trajectories in get_energies_async
    for batch in range(num_batches):
        energies = dataset.get_energies_async(loc[batch, ...], vel[batch, ...])
        energies_array.append(energies)
    energies_array = np.stack(energies_array, axis=0)

    fig, axs = plt.subplots(
        num_batches, 1, figsize=(14, 8 * num_batches), sharex=True, sharey=True
    )

    colors = {
        "Kinetic Energy": "red",
        "Potential Energy": "blue",
        "Total Energy": "green",
    }
    legend_added = {
        "Kinetic Energy": False,
        "Potential Energy": False,
        "Total Energy": False,
    }

    for batch in range(num_batches):
        for sim in range(num_simulations):
            times = np.arange(energies_array[batch, sim].shape[0])

            # Kinetic Energy
            axs[batch].plot(
                times,
                energies_array[batch, sim, :, 0],
                alpha=0.3,
                color=colors["Kinetic Energy"],
                linestyle="--",
                label="Kinetic Energy" if not legend_added["Kinetic Energy"] else "",
            )
            legend_added["Kinetic Energy"] = True

            # Potential Energy
            axs[batch].plot(
                times,
                energies_array[batch, sim, :, 1],
                alpha=0.3,
                color=colors["Potential Energy"],
                linestyle=":",
                label=(
                    "Potential Energy" if not legend_added["Potential Energy"] else ""
                ),
            )
            legend_added["Potential Energy"] = True

            # Total Energy
            axs[batch].plot(
                times,
                energies_array[batch, sim, :, 2],
                alpha=0.3,
                color=colors["Total Energy"],
                label="Total Energy" if not legend_added["Total Energy"] else "",
            )
            legend_added["Total Energy"] = True

        # Reset legend_added flags for the next batch
        legend_added = {
            "Kinetic Energy": False,
            "Potential Energy": False,
            "Total Energy": False,
        }

        axs[batch].set_title(f"Energy {title_suffixes[batch]}".title())
        axs[batch].legend()

    # Set common labels and title
    fig.supxlabel("Time")  # Use supxlabel for a common x-axis label
    fig.supylabel("Energy")  # Use supylabel for a common y-axis label
    fig.suptitle("Energy vs Time for Multiple Simulations")

    # Only add legend to the last batch for clarity
    axs[-1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{save_dir}/{filename}")
    return energies_array


def plot_energy_distributions_across_all_sims_multiplot(
    dataset,
    loc,
    vel,
    mass=None,
    bins=50,
    offline_plot=is_headless(),
    save_dir=None,
    filename="energy_distributions.png",
    energies_array=None,
    title_suffixes=None,
):
    num_batches = loc.shape[0]
    if title_suffixes is None:
        title_suffixes = ["" for _ in range(num_batches)]
    if energies_array is None:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    dataset.get_energies_async, loc[batch, ...], vel[batch, ...]
                )
                for batch in range(num_batches)
            ]
            energies_list = [future.result() for future in futures]
        energies_array = np.array(energies_list)

    energy_types = ["Kinetic Energy", "Potential Energy", "Total Energy"]
    colors = ["red", "blue", "green"]

    # Determine the number of rows needed for the subplot, one row per batch
    fig, axs = plt.subplots(
        num_batches, 3, figsize=(18, 6 * num_batches), sharex="col", sharey="row"
    )
    fig.suptitle("Energy Distributions Across All Time Points and Simulations")

    for batch in range(num_batches):
        kinetic_energies = energies_array[batch, :, 0].flatten()
        potential_energies = energies_array[batch, :, 1].flatten()
        total_energies = energies_array[batch, :, 2].flatten()
        energies = [kinetic_energies, potential_energies, total_energies]

        for i, ax in enumerate(axs[batch]):
            ax.hist(energies[i], bins=bins, color=colors[i], alpha=0.7, density=True)
            ax.set_title(f"{energy_types[i]} {title_suffixes[batch]}".title())
            ax.set_xlabel("Energy")
            ax.set_ylabel("Density")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_dir}/{filename}")


def plot_momentum_statistics(
    vel, save_dir=None, filename="momentum_statistics.png", title_suffixes=None
):
    """
    Plot momentum statistics for inference batch.
    """
    num_batches = vel.shape[0]
    num_sims = vel.shape[1]
    if title_suffixes is None:
        title_suffixes = ["" for _ in range(num_batches)]

    fig, axs = plt.subplots(
        num_batches, 1, figsize=(14, 8 * num_batches), sharex=True, sharey=True
    )

    vector_vel_of_systems = np.sum(vel, axis=3)
    scalar_vel_of_systems = np.sqrt(np.sum(vector_vel_of_systems**2, axis=3))

    for batch in range(num_batches):
        scalar_vel_of_systems_batch = scalar_vel_of_systems[batch, ...]
        # if batch == 0:
        #     max_momentum_threshold = scalar_vel_of_system.max() * 100000000000000000000000
        # else:
        #     scalar_vel_of_system = np.clip(scalar_vel_of_system, scalar_vel_of_system.min(), max_momentum_threshold)
        for sim in range(num_sims):
            times = np.arange(scalar_vel_of_systems_batch[batch].shape[0])

            axs[batch].plot(
                times,
                scalar_vel_of_systems_batch[sim],
                alpha=0.3,
                color="black",
                linestyle=":",
                label=("Distribution of momentums in systems"),
            )
        axs[batch].set_title(f"Momentum {title_suffixes[batch]}".title())
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_dir}/{filename}")

    momentum_file = os.path.join(save_dir, "momentum_statistics.json")

    current_time = datetime.now().isoformat()
    momentum_data = {}

    for batch, suffix in enumerate(title_suffixes):
        # leaving distribution
        momentum_data[suffix] = {
            "timestamp": current_time,
            "momentum_statistics": np.mean(
                scalar_vel_of_systems[batch], axis=1
            ).tolist(),
        }

    with open(momentum_file, "w") as file:
        json.dump(momentum_data, file, indent=4)


def plot_energy_statistics_multiplot(
    dataset,
    loc,
    vel,
    mass=None,
    offline_plot=is_headless(),
    save_dir=None,
    filename="energy_statistics.png",
    energies_array=None,
    title_suffixes=None,
):
    num_batches = loc.shape[0]
    if title_suffixes is None:
        title_suffixes = ["" for _ in range(num_batches)]
    if energies_array is None:
        energies_list = []
        for batch in range(num_batches):
            energies = dataset.get_energies_async(loc[batch, ...], vel[batch, ...])
            energies_list.append(energies)
        energies_array = np.array(energies_list)

    colors = {
        "Kinetic Energy": "red",
        "Potential Energy": "blue",
        "Total Energy": "green",
    }
    energy_labels = ["Kinetic Energy", "Potential Energy", "Total Energy"]

    fig, axs = plt.subplots(
        num_batches,
        1,
        figsize=(14, 8 * num_batches),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.suptitle("Average Energy vs Time for Multiple Simulations with Std. Dev.")

    energy_statistics = {}

    for batch in range(num_batches):
        ax = axs[batch, 0]  # Get the current axis for the batch
        batch_statistics = []
        for i, energy_label in enumerate(energy_labels):
            energy_mean = energies_array[batch, :, :, i].mean(axis=0)
            energy_std = energies_array[batch, :, :, i].std(axis=0)
            times = np.arange(energy_mean.shape[0])

            # Plot mean
            ax.plot(times, energy_mean, color=colors[energy_label], label=energy_label)
            # Plot standard deviation range
            ax.fill_between(
                times,
                energy_mean - energy_std,
                energy_mean + energy_std,
                color=colors[energy_label],
                alpha=0.2,
            )
            batch_statistics.append(
                {
                    "time": times.tolist(),
                    "mean": energy_mean.tolist(),
                    "std_dev": energy_std.tolist(),
                    "label": energy_label,
                }
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")
        ax.set_title(f"{title_suffixes[batch]}".title())
        ax.grid(True)
        ax.legend()

        energy_statistics[title_suffixes[batch]] = batch_statistics

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_dir}/{filename}")

    # Save energy statistics to a JSON file for later analysis
    statistics_file = os.path.join(save_dir, "energy_statistics.json")
    data = {}

    for batch, suffix in enumerate(title_suffixes):
        data[suffix] = {
            "timestamp": datetime.now().isoformat(),
            "data": energy_statistics[suffix],
        }

    with open(statistics_file, "w") as file:
        json.dump(data, file, indent=4)


def count_stickings_and_collisions(loc, time_threshold=3, distance_threshold=0.5):
    num_sims = loc.shape[0]
    num_steps = loc.shape[1]
    n_balls = loc.shape[2]

    sticking_counts = np.zeros(num_sims)
    collision_counts = np.zeros(num_sims)

    for sim in range(num_sims):
        ongoing_contacts = np.zeros((n_balls, n_balls))
        for step in range(1, num_steps):
            for i in range(n_balls):
                for j in range(i + 1, n_balls):
                    # Calculate distance between particles i and j
                    distance = np.linalg.norm(
                        loc[sim, step, i, :] - loc[sim, step, j, :]
                    )
                    if distance <= distance_threshold:
                        ongoing_contacts[i, j] += 1
                        # Count as collision immediately upon contact
                        if ongoing_contacts[i, j] == 1:
                            collision_counts[sim] += 1
                        # Upgrade to sticking after reaching time threshold
                        if ongoing_contacts[i, j] == time_threshold:
                            sticking_counts[sim] += 1
                            # Remove one collision count as it becomes a sticking
                            collision_counts[sim] -= 1
                    else:
                        ongoing_contacts[i, j] = (
                            0  # Reset if particles move apart beyond threshold
                        )
    return sticking_counts, collision_counts


def calculate_euclidean_distance(a, b):
    """
    Calculates euclidean distance between two points in 3 dimensional space
    """
    x_diff = a[0] - b[0]
    y_diff = a[1] - b[1]
    z_diff = a[2] - b[2]
    return np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)


def calculate_centre_of_mass(n_bodies_pos):
    """
    Calculate centre of mass for n bodies.
    We assume that all bodies have equal mass.
    """
    return np.mean(n_bodies_pos, axis=0)


def count_balls_leaving_defined_area(loc, distance_threshold=15):
    """
    returns number of bodies that moved more than distance_threshold from centre of mass.
    """
    num_sims = loc.shape[0]
    num_steps = loc.shape[1]
    n_balls = loc.shape[2]

    bodies_left_count = np.zeros(num_sims)
    for sim_idx in range(num_sims):
        outside_of_area = np.zeros(n_balls)
        for step_idx in range(1, num_steps):
            b = calculate_centre_of_mass(loc[sim_idx, step_idx])
            for ball_idx in range(n_balls):
                if (
                    calculate_euclidean_distance(loc[sim_idx, step_idx, ball_idx], b)
                    > distance_threshold
                ):
                    outside_of_area[ball_idx] += 1
                else:
                    outside_of_area[ball_idx] = 0
        bodies_left_count[sim_idx] = len([i for i in outside_of_area if i > 10])
    return bodies_left_count


def get_max_distance_of_com_from_starting_position(loc):
    """
    returns maximal distance of center of mass from its starting position.
    """
    num_sims = loc.shape[0]
    num_steps = loc.shape[1]

    max_distances = np.zeros(num_sims)
    for sim_idx in range(num_sims):
        starting_com_position = calculate_centre_of_mass(loc[sim_idx, 0])
        for step_idx in range(1, num_steps):
            current_com_position = calculate_centre_of_mass(loc[sim_idx, step_idx])
            current_distance = calculate_euclidean_distance(
                starting_com_position, current_com_position
            )
            if current_distance > max_distances[sim_idx]:
                max_distances[sim_idx] = current_distance
    return max_distances


def calculate_angle(a, b):
    """
    Calucates angle between two vectors of speed.
    """

    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def count_sharp_turns(vel, angle_threshold=30):
    """
    returns number times that bodies changed their trajectory by more than
    'angle_threshold' angles.
    """
    num_sims = vel.shape[0]
    num_steps = vel.shape[1]
    n_balls = vel.shape[2]

    sharp_turns_count = np.zeros(num_sims)
    for sim_idx in range(num_sims):
        for step_idx in range(1, num_steps):
            for ball_idx in range(n_balls):
                if (
                    calculate_angle(
                        vel[sim_idx, step_idx, ball_idx],
                        vel[sim_idx, step_idx - 1, ball_idx],
                    )
                    > angle_threshold
                ):
                    sharp_turns_count[sim_idx] += 1
    return sharp_turns_count


def interactive_plotly_offline_plot_multi_trajectory(
    pos, labels, output_file="3D_offline_plot.html", duration=8, save_dir="."
):
    num_trajectories = pos.shape[0]
    if len(labels) != num_trajectories:
        raise ValueError(
            "The number of labels must match the number of position arrays."
        )

    # Define colors for each trajectory for better distinction
    colors = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "white",
    ]
    if num_trajectories > len(colors):
        raise ValueError(
            "Number of position arrays exceeds the number of predefined colors."
        )

    # Determine the number of particles and steps based on the first trajectory
    particles = pos.shape[2]
    steps = max(len(pos) for pos in pos)

    # Initialize the figure
    fig = go.Figure(
        layout=go.Layout(
            template="plotly_white",
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=1.05,
                    xanchor="right",
                    yanchor="top",
                    pad=dict(t=0, r=10),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": duration, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                    "transition": {"duration": duration},
                                },
                            ],
                        ),
                        dict(
                            label="Stop",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [f"frame {k}"],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": duration, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            label=f"{k}",
                        )
                        for k in range(steps)
                    ],
                    transition={"duration": duration},
                    x=0,
                    y=0,
                    currentvalue={
                        "font": {"size": 12},
                        "prefix": "Point: ",
                        "visible": True,
                    },
                    len=1.0,
                )
            ],
        )
    )

    # Add traces for each particle in each position array
    for batch in range(num_trajectories):
        for i in range(particles):
            # Initial lines for trajectory visualization with legend labels
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[batch, 0, i, 0]],
                    y=[pos[batch, 0, i, 1]],
                    z=[pos[batch, 0, i, 2]],
                    mode="markers+lines",
                    marker=dict(size=5, color=colors[batch]),
                    name=labels[batch],
                )
            )
            # Placeholder markers for the current step without adding to legend again
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[batch, 0, i, 0]],
                    y=[pos[batch, 0, i, 1]],
                    z=[pos[batch, 0, i, 2]],
                    mode="markers",
                    marker=dict(size=2, color=colors[batch]),
                    showlegend=True,
                )
            )

    # Create frames for each step
    frames = []
    for k in range(1, steps + 1):
        frame_data = []
        for batch in range(num_trajectories):
            start_idx = max(0, k - 15)
            for i in range(particles):
                # Determine last valid index for this trajectory
                last_idx = min(k, len(pos[batch])) - 1

                # Plot trajectory segment up to the last available step
                frame_data.append(
                    go.Scatter3d(
                        x=pos[batch, start_idx : last_idx + 1, i, 0],
                        y=pos[batch, start_idx : last_idx + 1, i, 1],
                        z=pos[batch, start_idx : last_idx + 1, i, 2],
                        mode="lines",
                        line=dict(width=5, color=colors[batch]),
                        opacity=0.3,
                        showlegend=True,
                    )
                )
                # Marker for the last available position
                frame_data.append(
                    go.Scatter3d(
                        x=[pos[batch, last_idx, i, 0]],
                        y=[pos[batch, last_idx, i, 1]],
                        z=[pos[batch, last_idx, i, 2]],
                        mode="markers",
                        marker=dict(size=2, color=colors[batch]),
                        name=f"{labels[batch]} Step {min(k, len(pos[batch]))}",
                        showlegend=False,
                    )
                )

        frames.append(go.Frame(data=frame_data, name=f"frame {k}"))

    fig.frames = frames

    # Calculate min and max for all dimensions across all trajectories
    all_pos = np.concatenate(pos, axis=1)
    x_min, x_max = np.min(all_pos[:, :, 0]), np.max(all_pos[:, :, 0])
    y_min, y_max = np.min(all_pos[:, :, 1]), np.max(all_pos[:, :, 1])
    z_min, z_max = np.min(all_pos[:, :, 2]), np.max(all_pos[:, :, 2])

    # Calculate ranges with some padding
    padding = 5
    x_range = [x_min - padding, x_max + padding]
    y_range = [y_min - padding, y_max + padding]
    z_range = [z_min - padding, z_max + padding]

    # Fix the axes ranges and ensure a fixed aspect ratio
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-25, 25], autorange=False),
            yaxis=dict(range=[-25, 25], autorange=False),
            zaxis=dict(range=[-25, 25], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode="cube",
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis",
        ),
        title="N-body Trajectories Comparison",
        showlegend=True,
    )

    # Save the plot as an HTML file
    os.makedirs(save_dir, exist_ok=True)
    fig.write_html(f"{save_dir}/{output_file}")


def pad_trajectories_to_same_length(pos_arrays):
    max_length = max(len(pos) for pos in pos_arrays)
    for i, pos in enumerate(pos_arrays):
        if len(pos) < max_length:
            last_pos = pos[-1]
            pos_arrays[i] = np.concatenate(
                [pos, np.tile(last_pos, (max_length - len(pos), 1, 1))], axis=0
            )
    return pos_arrays


def load_dataset_from_metadata_file(metadata_file_path, n_bodies=None):
    with open(metadata_file_path, "r") as f:
        metadata = json.load(f)

    # Get the parameters expected by the GravityDataset constructor
    params = inspect.signature(GravityDatasetOtf.__init__).parameters
    expected_keys = [p for p in params if p != "self"]

    # Filter metadata to include only the keys that are parameters of the GravityDataset constructor
    filtered_metadata = {key: metadata[key] for key in expected_keys if key in metadata}
    # change number of bodies in dataset
    if n_bodies is not None:
        filtered_metadata["num_nodes"] = n_bodies
    dataset = GravityDatasetOtf(**filtered_metadata)
    return dataset


def plot_group_collision_distribution_multiplot(
    loc,
    time_threshold=2,
    distance_threshold=2,
    num_bins=50,
    save_dir=None,
    filename="group_collision_distribution_multiplot.png",
    title_suffixes=None,
):
    """
    Counts the number of group collisions in each simulation where a pair and a triplet of bodies,
    each having been stuck together (within a distance_threshold for more than time_threshold steps),
    collide with each other (come within distance_threshold for at least one step).

    Parameters:
    - loc: numpy array of shape (num_batches, num_sims, num_steps, n_balls, num_dims)
    - time_threshold: int, number of consecutive steps bodies must be within distance_threshold to be considered stuck
    - distance_threshold: float, maximum distance between bodies to be considered in contact
    - num_bins: int, number of bins for histogram
    - save_dir: str, directory to save plots and JSON files
    - filename: str, filename for the plot image
    - title_suffixes: list of str, suffixes for titles in multiplot corresponding to each batch
    """
    import json
    import os
    from datetime import datetime
    from itertools import combinations

    import matplotlib.pyplot as plt
    import numpy as np

    num_batches = loc.shape[0]
    num_sims = loc.shape[1]
    num_steps = loc.shape[2]
    n_balls = loc.shape[3]

    group_collision_counts = []

    for batch in range(num_batches):
        batch_group_collision_counts = np.zeros(num_sims)

        for sim in range(num_sims):
            # Initialize data structures
            # For pairs
            ongoing_contacts_pairs = np.zeros((n_balls, n_balls), dtype=int)
            stuck_pairs = {}
            # For triplets
            triplet_indices = list(combinations(range(n_balls), 3))
            ongoing_contacts_triplets = {}
            stuck_triplets = {}
            # Initialize contact durations
            for triplet in triplet_indices:
                ongoing_contacts_triplets[triplet] = 0

            # For each time step, detect stuck pairs and triplets
            for step in range(num_steps):
                # Update pairs
                for i in range(n_balls):
                    for j in range(i + 1, n_balls):
                        distance = np.linalg.norm(
                            loc[batch, sim, step, i] - loc[batch, sim, step, j]
                        )
                        if distance <= distance_threshold:
                            ongoing_contacts_pairs[i, j] += 1
                            if ongoing_contacts_pairs[i, j] == time_threshold:
                                # Pair (i, j) became stuck at this step
                                t_start = step - time_threshold + 1
                                if (i, j) not in stuck_pairs:
                                    stuck_pairs[(i, j)] = []
                                stuck_pairs[(i, j)].append(
                                    [t_start, None]
                                )  # t_end to be updated later
                        else:
                            if ongoing_contacts_pairs[i, j] >= time_threshold:
                                # Pair (i, j) was stuck, now separated
                                t_end = step - 1
                                stuck_pairs[(i, j)][-1][1] = t_end
                            ongoing_contacts_pairs[i, j] = 0

                # Update triplets
                for triplet in triplet_indices:
                    i, j, k = triplet
                    distances = [
                        np.linalg.norm(
                            loc[batch, sim, step, i] - loc[batch, sim, step, j]
                        ),
                        np.linalg.norm(
                            loc[batch, sim, step, i] - loc[batch, sim, step, k]
                        ),
                        np.linalg.norm(
                            loc[batch, sim, step, j] - loc[batch, sim, step, k]
                        ),
                    ]
                    if all(d <= distance_threshold for d in distances):
                        ongoing_contacts_triplets[triplet] += 1
                        if ongoing_contacts_triplets[triplet] == time_threshold:
                            # Triplet became stuck at this step
                            t_start = step - time_threshold + 1
                            if triplet not in stuck_triplets:
                                stuck_triplets[triplet] = []
                            stuck_triplets[triplet].append(
                                [t_start, None]
                            )  # t_end to be updated later
                    else:
                        if ongoing_contacts_triplets[triplet] >= time_threshold:
                            # Triplet was stuck, now separated
                            t_end = step - 1
                            stuck_triplets[triplet][-1][1] = t_end
                        ongoing_contacts_triplets[triplet] = 0

            # Handle pairs and triplets that remained stuck until the last step
            for (i, j), intervals in stuck_pairs.items():
                if intervals[-1][1] is None:
                    intervals[-1][1] = num_steps - 1
            for triplet, intervals in stuck_triplets.items():
                if intervals[-1][1] is None:
                    intervals[-1][1] = num_steps - 1

            # Now, for each combination of stuck pair and stuck triplet, check for group collisions
            group_collisions = 0
            checked_combinations = set()  # To avoid double counting
            for pair, pair_intervals in stuck_pairs.items():
                for triplet, triplet_intervals in stuck_triplets.items():
                    # Ensure the pair and triplet are disjoint (no common bodies)
                    if set(pair).isdisjoint(triplet):
                        # Find overlapping intervals
                        for p_interval in pair_intervals:
                            p_start, p_end = p_interval
                            for t_interval in triplet_intervals:
                                t_start, t_end = t_interval
                                overlap_start = max(p_start, t_start)
                                overlap_end = min(p_end, t_end)
                                if overlap_start <= overlap_end:
                                    # Check for collision after both groups have formed
                                    for t in range(overlap_start, num_steps):
                                        # Check if any body in pair is close to any body in triplet
                                        collision_found = False
                                        for i in pair:
                                            for j in triplet:
                                                distance = np.linalg.norm(
                                                    loc[batch, sim, t, i]
                                                    - loc[batch, sim, t, j]
                                                )
                                                if distance <= distance_threshold:
                                                    group_collisions += 1
                                                    collision_found = True
                                                    break  # Exit inner loop
                                            if collision_found:
                                                break  # Exit middle loop
                                        if collision_found:
                                            break  # Exit outer loop
                        checked_combinations.add((pair, triplet))

            batch_group_collision_counts[sim] = group_collisions

        group_collision_counts.append(batch_group_collision_counts)

    # Determine global min and max across all batches for consistent binning
    all_counts = np.concatenate(group_collision_counts)
    global_min = all_counts.min()
    global_max = all_counts.max()
    bin_edges = np.linspace(global_min, global_max, num_bins + 1)

    # Plotting
    fig, axs = plt.subplots(
        num_batches, 1, figsize=(10, 6 * num_batches), sharex=True, sharey=True
    )
    if num_batches == 1:
        axs = [axs]  # Ensure axs is iterable for a single batch

    current_time = datetime.now().isoformat()
    group_collision_data = {}

    for batch, ax in enumerate(axs):
        counts, _, _ = ax.hist(
            group_collision_counts[batch],
            bins=bin_edges,
            alpha=0.7,
            color="violet",
            edgecolor="black",
        )
        ax.set_xlabel("Group Collision Count")
        ax.set_ylabel("Frequency")
        suffix = (
            title_suffixes[batch] if title_suffixes is not None else f"batch_{batch}"
        )
        ax.set_title(f"Group Collision Distribution {suffix}".title())

        # Prepare data for JSON
        group_collision_data[suffix] = {
            "timestamp": current_time,
            "group_collision_count": group_collision_counts[batch].tolist(),
        }

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, filename))
    plt.close(fig)  # Close the figure to free memory

    # Save the group_collision_data to a JSON file
    group_collision_file = os.path.join(save_dir, "group_collision_distribution.json")
    with open(group_collision_file, "w") as file:
        json.dump(group_collision_data, file, indent=4)
