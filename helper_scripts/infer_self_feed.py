import os
from datetime import datetime

import numpy as np
import torch
from torch_geometric.data import Data

from datasets.nbody.visualization_utils import load_dataset_from_metadata_file
from helper_scripts.plot_macros import plot_macros as plot_macros_f
from models.segnn.o3_building_blocks import O3Transform
from utils import segnn_utils
from utils.build_fully_connected_graph import build_graph_with_knn
from utils.nbody_utils import (
    get_dataset_metadata_path,
    get_device,
    load_model_for_inference,
)


@torch.no_grad()
def run_inference(
    model_type,
    dataloader,
    model_path=None,
    model=None,
    save_dir=None,
    print_step=True,
    n_bodies=None,
    plot_macros=False,
    num_neighbors=None,
    device=None,
    max_rollout_steps=None,
):

    torch.manual_seed(42)

    if device is None:
        device = get_device()

    dataset_metadata_path = get_dataset_metadata_path(model_path)
    dataset = load_dataset_from_metadata_file(dataset_metadata_path, n_bodies=n_bodies)
    if model is None:
        model = load_model_for_inference(model_path, model_type, device)

    if dataset.double_precision:
        model = model.double()
    else:
        model = model.float()

    batch_size = dataset.batch_size
    batch_data, _ = dataset.get_ground_truth_trajectories(batch_size=batch_size)
    loc_actual, vel_actual, force_actual, mass_actual = [
        torch.from_numpy(np.array(d)) for d in zip(*batch_data)
    ]

    output_dims = loc_actual.shape[-1]
    n_nodes = loc_actual.shape[-2]
    num_neighbors = num_neighbors if num_neighbors is not None else n_nodes - 1

    loc_initial = loc_actual[:, :1, :, :].reshape(batch_size, n_nodes, output_dims)
    vel_initial = vel_actual[:, :1, :, :].reshape(batch_size, n_nodes, output_dims)
    force_initial = force_actual[:, :1, :, :].reshape(batch_size, n_nodes, output_dims)
    mass_initial = mass_actual.reshape(batch_size, n_nodes, 1)

    predicted_loc = [loc_initial]
    predicted_vel = [vel_initial]
    predicted_force = [force_initial]
    predicted_mass = [mass_initial]

    num_steps_to_generate = loc_actual.shape[1]
    if max_rollout_steps is not None:
        try:
            max_rollout_steps = int(max_rollout_steps)
            if max_rollout_steps > 0:
                num_steps_to_generate = min(num_steps_to_generate, max_rollout_steps)
                # also truncate the ground truth so shapes match
                loc_actual = loc_actual[:, :num_steps_to_generate]
                vel_actual = vel_actual[:, :num_steps_to_generate]
        except Exception as e:
            print(e)
            pass
    print(f"Number of steps to generate: {num_steps_to_generate}")

    rows, cols = [], []
    n_nodes = loc_actual[0].size(1)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    # Create batch_size copies of the edges tensor
    batch_size = loc_actual.size(0)
    edges_cgenn = (
        torch.Tensor([rows, cols]).long().unsqueeze(0).repeat(batch_size, 1, 1)
    )

    edges_cgenn = edges_cgenn.to(device)

    for step in range(num_steps_to_generate - 1):
        if print_step:
            print(f"Predicting step {step} for {batch_size} simulations")

        data = [
            predicted_loc[-1].reshape(batch_size * n_nodes, output_dims),
            predicted_vel[-1].reshape(batch_size * n_nodes, output_dims),
            predicted_force[-1].reshape(batch_size * n_nodes, output_dims),
            predicted_mass[-1].reshape(batch_size * n_nodes, 1),
        ]

        batch = torch.arange(0, batch_size).repeat_interleave(n_nodes).to(device=device)

        data = [d.double() if dataset.double_precision else d.float() for d in data]
        data = [d.to(device=device) for d in data]

        if model_type == "segnn":
            loc, vel, force, mass = data
            graph = Data(pos=loc, vel=vel, force=force, mass=mass)
            batch = torch.arange(0, batch_size).to(device)
            graph.batch = batch.repeat_interleave(n_nodes).long()

            edge_index = build_graph_with_knn(
                loc, batch_size, dataset.num_nodes, device, num_neighbors
            )
            graph.edge_index = edge_index

            transform = O3Transform(dataset.lmax_attr)

            graph = transform(graph)  # Add O3 attributes
            graph = graph.to(device)
            pred = model(graph)
        elif model_type == "ponita":
            loc, vel, force, mass = data
            graph = Data(torch.hstack([mass]))
            graph.pos = loc
            graph.vec = vel
            graph.vec = graph.vec.reshape(graph.vec.shape[0], 1, graph.vec.shape[1])
            edge_index = build_graph_with_knn(
                loc, batch_size, dataset.num_nodes, device, num_neighbors
            )
            graph.edge_index = edge_index
            pos_send, pos_receive = (
                loc[edge_index[0]],
                loc[edge_index[1]],
            )  # [num_edges, 3]
            graph.rel_pos = pos_send - pos_receive
            graph.to(device)
            pred = model(graph)
        elif model_type == "cgenn":

            data, _ = dataloader.get_batch()
            data = dataloader.preprocess_batch(data[0], device)
            pred = model(data)

        elif model_type == "graph_transformer":
            # Build a minimal PyG graph and call the model with one argument
            loc, vel, force, mass = data
            graph = Data(pos=loc, vel=vel, force=force, mass=mass)
            graph.batch = batch.long()
            graph = graph.to(device)
            pred = model(graph)
        elif model_type == "egnn_mc":
            if dataloader is None:
                raise ValueError(
                    "EGNNMultiChannel inference requires an initialized dataloader."
                )
            loc, vel, force, mass = data
            graph = Data(pos=loc, vel=vel, force=force, mass=mass)
            graph.batch = batch.long()
            graph = dataloader.preprocess_batch(graph, device=device, training=False)
            pred = model(graph)
        elif model_type == "painn":
            loc, vel, force, mass = data
            graph = Data(pos=loc, vel=vel, force=force, mass=mass)
            graph.batch = batch.long()
            graph.edge_index = build_graph_with_knn(loc, batch_size, n_nodes, device, num_neighbors)
            graph = graph.to(device)
            pred = model(graph)
        else:
            data = (data[0], data[1], data[2], data[3], data[0])
            data = tuple(d for d in data)
            pred = model(data, batch)
        pred_loc = pred[..., :3].reshape(batch_size, n_nodes, output_dims)
        pred_vel = pred[..., 3:].reshape(batch_size, n_nodes, output_dims)

        if dataset.target == "pos_dt+vel":
            pred_loc = predicted_loc[-1] + pred_loc.cpu()
        predicted_loc.append(pred_loc.cpu())
        predicted_vel.append(pred_vel.cpu())
        predicted_force.append(
            torch.zeros_like(pred_loc.cpu())
        )  # Assuming force is not predicted
        predicted_mass.append(
            predicted_mass[-1].cpu()
        )  # Assuming mass remains constant

    print("Finished prediction for all simulations")

    predicted_loc = torch.stack(predicted_loc, dim=1)
    predicted_vel = torch.stack(predicted_vel, dim=1)
    predicted_force = torch.stack(predicted_force, dim=1)
    predicted_mass = torch.stack(predicted_mass, dim=1)

    steps_in_actual = loc_actual.shape[1]

    loc_actual = loc_actual.view(batch_size, steps_in_actual, n_nodes, output_dims)
    vel_actual = vel_actual.view(batch_size, steps_in_actual, n_nodes, output_dims)
    loc_pred = predicted_loc.numpy()
    vel_pred = predicted_vel.numpy()  # Saving predicted velocities

    combined_locations = np.stack([loc_actual, loc_pred], axis=0)
    combined_velocities = np.stack([vel_actual, vel_pred], axis=0)

    # Define the directory to save the generated trajectories for all simulations
    if not save_dir:
        save_dir = f"{os.path.dirname(model_path)}/generated_trajectories/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if plot_macros:
        plot_macros_f(
            combined_positions=np.asarray([loc_actual, loc_pred]),
            combined_velocities=np.asarray([vel_actual, vel_pred]),
            dataset=dataset,
            plot_dir=save_dir,
        )

    trajectories_save_dir = os.path.join(save_dir, "trajectories_data")
    if not os.path.exists(trajectories_save_dir):
        os.makedirs(trajectories_save_dir)

    # Save the actual and predicted locations as numpy arrays for all simulations
    for sim_index in range(batch_size):
        np.save(
            os.path.join(trajectories_save_dir, f"loc_actual_sim_{sim_index}.npy"),
            loc_actual[sim_index],
        )
        np.save(
            os.path.join(trajectories_save_dir, f"loc_pred_sim_{sim_index}.npy"),
            loc_pred[sim_index],
        )
        np.save(
            os.path.join(trajectories_save_dir, f"vel_actual_sim_{sim_index}.npy"),
            vel_actual[sim_index],
        )
        np.save(  # Saving predicted velocities for each simulation
            os.path.join(trajectories_save_dir, f"vel_pred_sim_{sim_index}.npy"),
            vel_pred[sim_index],
        )

    print(
        f"Saved actual and predicted trajectories for all simulations to {trajectories_save_dir}"
    )

    return trajectories_save_dir, combined_locations, combined_velocities


def main():
    parser = segnn_utils.create_argparser()
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["equiformer", "equiformer_v2", "segnn", "ponita"],
        default="equiformer",
        help="Type of model to use",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model",
    )
    parser.add_argument(
        "--n-bodies",
        type=int,
        default=None,
        help="Number of bodies in simulation",
    )
    parser.add_argument(
        "--plot-macros",
        action="store_true",
        help="Flag to indicate whether to plot macros",
    )
    args = parser.parse_args()

    run_inference(
        model_type=args.model_type,
        dataloader=None,
        model_path=args.model_path,
        n_bodies=args.n_bodies,
        plot_macros=args.plot_macros,
    )


if __name__ == "__main__":
    main()
