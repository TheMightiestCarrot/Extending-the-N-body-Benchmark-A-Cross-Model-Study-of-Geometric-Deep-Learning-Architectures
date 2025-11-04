import os
from datetime import datetime

import numpy as np
import torch

from helper_scripts.plot_macros import plot_macros as plot_macros_f


class Inferencer:
    def __init__(self, model, dataloader, args) -> None:
        self.model = model
        self.dataloader = dataloader
        self.args = args

    def _self_feed_loop(self, data, steps_to_generate):
        batch_size = self.dataloader.dataset.batch_size
        n_nodes = self.dataloader.args.num_atoms
        output_dims = data[0].shape[1]

        loc_initial, vel_initial, force_initial, mass_initial = data

        predicted_loc = [loc_initial.reshape(batch_size, n_nodes, output_dims).cpu()]
        predicted_vel = [vel_initial.reshape(batch_size, n_nodes, output_dims).cpu()]
        predicted_force = [
            force_initial.reshape(batch_size, n_nodes, output_dims).cpu()
        ]
        predicted_mass = [mass_initial.cpu()]

        for step in range(steps_to_generate - 1):
            print(f"Predicting step {step} for {batch_size} simulations")
            graph = self.dataloader.preprocess_batch(data, training=False)
            pred = self.model(graph)
            pred = self.dataloader.postprocess_batch(pred, self.dataloader.device)

            pred_loc = (
                (pred[..., :3] + graph.pos)
                .reshape(batch_size, n_nodes, output_dims)
                .detach()
            )
            pred_vel = pred[..., 3:].reshape(batch_size, n_nodes, output_dims).detach()

            # use predicted values as input for next step
            data = [
                pred_loc.reshape(batch_size * n_nodes, output_dims),
                pred_vel.reshape(batch_size * n_nodes, output_dims),
                torch.zeros_like(pred_loc).reshape(batch_size * n_nodes, output_dims),
                predicted_mass[-1].to(self.dataloader.device),
            ]

            predicted_loc.append(pred_loc.cpu())
            predicted_vel.append(pred_vel.cpu())
            predicted_force.append(
                torch.zeros_like(pred_loc).cpu()
            )  # Assuming force is not predicted
            predicted_mass.append(
                predicted_mass[-1].cpu()
            )  # Assuming mass remains constant

        print("Finished prediction for all simulations")
        predicted_loc = torch.stack(predicted_loc, dim=1)
        predicted_vel = torch.stack(predicted_vel, dim=1)
        predicted_force = torch.stack(predicted_force, dim=1)
        predicted_mass = torch.stack(predicted_mass, dim=1)

        return predicted_loc, predicted_vel

    def _combine_data(self, predicted_loc, predicted_vel, actual_loc, actual_vel):
        """ """
        batch_size = self.dataloader.dataset.batch_size
        n_nodes = self.dataloader.args.num_atoms
        output_dims = predicted_loc[0].shape[-1]
        steps_in_actual = actual_loc.shape[1]

        actual_loc = actual_loc.view(batch_size, steps_in_actual, n_nodes, output_dims)
        actual_vel = actual_vel.view(batch_size, steps_in_actual, n_nodes, output_dims)

        combined_locations = np.stack([actual_loc, predicted_loc.cpu().numpy()], axis=0)
        combined_velocities = np.stack(
            [actual_vel, predicted_vel.cpu().numpy()], axis=0
        )
        return combined_locations, combined_velocities

    def _create_subfolders(self, save_dir_path):
        # Define the directory to save the generated trajectories for all simulations
        save_dir_path = f"{save_dir_path}/generated_trajectories/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        trajectories_save_dir = os.path.join(save_dir_path, "trajectories_data")
        if not os.path.exists(trajectories_save_dir):
            os.makedirs(trajectories_save_dir)
        return save_dir_path

    def _save_data(
        self, loc_actual, loc_pred, vel_actual, vel_pred, trajectories_save_dir
    ):
        # Save the actual and predicted locations as numpy arrays for all simulations
        batch_size = self.dataloader.dataset.batch_size
        trajectories_save_dir = os.path.join(trajectories_save_dir, "trajectories_data")
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

    def run_inference(self, save_dir_path):
        batch_size = self.dataloader.dataset.batch_size
        batch_data, _ = self.dataloader.dataset.get_ground_truth_trajectories(
            batch_size=batch_size
        )
        actual_loc, actual_vel, force_actual, mass_actual = [
            torch.from_numpy(np.array(d)) for d in zip(*batch_data)
        ]

        output_dims = actual_loc.shape[-1]
        n_nodes = actual_loc.shape[-2]

        loc_initial = actual_loc[:, :1, :, :].reshape(batch_size, n_nodes, output_dims)
        vel_initial = actual_vel[:, :1, :, :].reshape(batch_size, n_nodes, output_dims)
        force_initial = force_actual[:, :1, :, :].reshape(
            batch_size, n_nodes, output_dims
        )
        mass_initial = mass_actual.reshape(batch_size, n_nodes, 1)

        predicted_loc = [loc_initial]
        predicted_vel = [vel_initial]
        predicted_force = [force_initial]
        predicted_mass = [mass_initial]

        num_steps_to_generate = actual_loc.shape[1]
        print(f"Number of steps to generate: {num_steps_to_generate}")

        data = [
            predicted_loc[-1].reshape(batch_size * n_nodes, output_dims),
            predicted_vel[-1].reshape(batch_size * n_nodes, output_dims),
            predicted_force[-1].reshape(batch_size * n_nodes, output_dims),
            predicted_mass[-1].reshape(batch_size * n_nodes, 1),
        ]

        data = [
            d.double() if self.dataloader.dataset.double_precision else d.float()
            for d in data
        ]
        data = [d.to(device=self.dataloader.device) for d in data]

        predicted_loc, predicted_vel = self._self_feed_loop(data, num_steps_to_generate)
        combined_locations, combined_velocities = self._combine_data(
            predicted_loc, predicted_vel, actual_loc, actual_vel
        )

        save_dir_path = self._create_subfolders(save_dir_path)

        plot_macros_f(
            combined_positions=combined_locations,
            combined_velocities=combined_velocities,
            dataset=self.dataloader.dataset,
            plot_dir=f"{save_dir_path}/plots",
        )

        self._save_data(
            actual_loc, predicted_loc, actual_vel, predicted_vel, save_dir_path
        )


