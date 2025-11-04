import os
from typing import Dict

import torch

from utils.utils_data import calculate_energies


class SimpleLoss(torch.nn.Module):
    def __init__(self, weight=1, args=None, name=None):
        super().__init__()
        self.name = name or "Simple loss"
        self.weight = weight
        self.criterion = torch.nn.MSELoss()
        # args parameter kept for API compatibility

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        return self.weight * loss


class TargetCommonLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "Total target loss"
        self.losses = []
        # this way we guarantee the same order of losses as in target assignment in dataset
        if "pos" in args.target or "pos_dt" in args.target:
            self.losses.append(
                SimpleLoss(args.position_loss_weight, name="Position loss")
            )
        if "vel" in args.target or "vel_dt" in args.target:
            self.losses.append(
                SimpleLoss(args.velocity_loss_weight, name="Velocity loss")
            )
        if "force" in args.target or "force_dt" in args.target:
            self.losses.append(SimpleLoss(args.force_loss_weight, name="Force loss"))

    def forward(self, pred, data):
        total_loss = 0
        for i, loss in enumerate(self.losses):
            target_quantity = data.y[..., 3 * i : 3 * (i + 1)]
            pred_quantity = pred[..., 3 * i : 3 * (i + 1)]
            total_loss += loss.forward(pred_quantity, target_quantity)
        return total_loss


class CentreOfMassLoss(torch.nn.Module):
    def __init__(self, weight=1, args=None):
        super().__init__()

        self.name = "Centre of mass loss"
        self.weight = weight
        self.criterion = torch.nn.MSELoss()

        self.batch_size = args.batch_size
        self.num_atoms = args.num_atoms

    def forward(self, pred, data):
        """
        Calculate centre of mass loss for both predicted and gt step,
        then calculate loss.
        """
        # Compute predicted and true positions
        pos_pred = data.pos + pred[..., :3]
        pos_true = data.pos + data.y[..., :3]

        com_pred = []
        com_true = []

        for sim_idx in range(self.batch_size):
            start_idx = sim_idx * self.num_atoms
            end_idx = (sim_idx + 1) * self.num_atoms

            com_pred.append(pos_pred[start_idx:end_idx].mean(dim=0))
            com_true.append(pos_true[start_idx:end_idx].mean(dim=0))

        com_pred = torch.stack(com_pred)
        com_true = torch.stack(com_true)

        # Compute loss
        loss_com = self.criterion(com_pred, com_true)

        return loss_com * self.weight


class EnergyLoss(torch.nn.Module):
    def __init__(self, weight=1, args=None):
        super().__init__()

        self.name = "Energy loss"
        self.weight = weight
        self.criterion = torch.nn.MSELoss()

        self.num_nodes = args.num_atoms
        self.interaction_strength = args.interaction_strength
        self.softening = args.softening

    def forward(self, pred, data):
        predicted_pos = ((data.pos + pred[..., :3]).detach().cpu().numpy(),)
        target_pos = ((data.pos + data.y[..., :3]).detach().cpu().numpy(),)
        predicted_vel = (pred[..., 3:].detach().cpu().numpy(),)
        target_vel = (data.y[..., 3:].detach().cpu().numpy(),)
        mass = (data.mass.detach().cpu().numpy(),)

        # Compute energies
        pred_energy = calculate_energies(
            predicted_pos,
            predicted_vel,
            mass,
            self.num_nodes,
            self.interaction_strength,
            self.softening,
        )
        target_energy = calculate_energies(
            target_pos,
            target_vel,
            mass,
            self.num_nodes,
            self.interaction_strength,
            self.softening,
        )

        loss_energy = self.criterion(
            torch.tensor(pred_energy), torch.tensor(target_energy)
        )

        return loss_energy * self.weight


class MomentumLoss(torch.nn.Module):
    """Momentum conservation loss for physics-informed neural networks.

    This loss encourages the model to respect conservation of linear momentum,
    a fundamental principle in physics stating that the total momentum of an
    isolated system remains constant.

    The loss is computed as the MSE between the total momentum of the predicted
    next state and the current state, enforcing frame-to-frame momentum conservation.

    Args:
        weight: Loss weight multiplier (default: 0.0001)
        args: Configuration object containing:
            - batch_size: Number of simulations in a batch
            - num_atoms: Number of atoms per simulation

    Note:
        - Assumes unit mass if mass data is not provided
        - Uses efficient parallel computation with scatter_add for batched data
        - Only penalizes predicted momentum deviation (not ground truth)

    """

    def __init__(self, weight=0.0001, args=None):
        super().__init__()

        self.name = "Momentum loss"
        self.weight = weight
        self.criterion = torch.nn.MSELoss()

        self.batch_size = args.batch_size
        self.num_atoms = args.num_atoms

    def forward(self, pred, data):
        """Calculate momentum conservation loss.

        Total momentum should be conserved: sum(m * v) should remain constant.
        This method computes the total momentum for current and predicted states
        and penalizes deviations from conservation.

        Args:
            pred: Model predictions tensor with shape (N, 6) or (N, 9) where:
                  - [..., :3] are position deltas (or positions)
                  - [..., 3:6] are velocities
                  - [..., 6:9] are forces (if present)
            data: Graph data object containing:
                  - vel or vec: current velocities
                  - mass: atom masses (optional, defaults to unit mass)
                  - batch: batch assignment for each atom
                  - y: target values with same structure as pred

        Returns:
            Weighted momentum conservation loss (scalar)
        """
        # Current velocities
        current_vel = data.vel if hasattr(data, "vel") else data.vec

        predicted_vel = pred[..., 3:]
        target_vel = data.y[..., 3:]

        # Get masses (assuming unit mass if not available)
        mass = (
            data.mass
            if hasattr(data, "mass")
            else torch.ones(
                current_vel.shape[0], device=current_vel.device, dtype=current_vel.dtype
            )
        )

        # Get batch assignments
        batch = (
            data.batch
            if hasattr(data, "batch")
            else torch.arange(
                current_vel.shape[0] // self.num_atoms, device=current_vel.device
            ).repeat_interleave(self.num_atoms)
        )

        # Ensure mass is properly shaped for broadcasting
        mass = mass.unsqueeze(-1)  # (total_atoms, 1)

        # Calculate momentum for all atoms: mass * velocity
        momentum_current_all = mass * current_vel  # (total_atoms, 3)
        momentum_pred_all = mass * predicted_vel  # (total_atoms, 3)
        momentum_target_all = mass * target_vel  # (total_atoms, 3)

        # Use scatter_add to sum momentum for each batch in parallel
        num_batches = batch.max().item() + 1
        momentum_current = torch.zeros(
            num_batches, 3, device=pred.device, dtype=pred.dtype
        )
        momentum_pred = torch.zeros(
            num_batches, 3, device=pred.device, dtype=pred.dtype
        )
        momentum_target = torch.zeros(
            num_batches, 3, device=pred.device, dtype=pred.dtype
        )

        # Expand batch indices for 3D momentum vectors
        batch_expanded = batch.unsqueeze(-1).expand(-1, 3)  # (total_atoms, 3)

        # Parallel summation using scatter_add
        momentum_current.scatter_add_(0, batch_expanded, momentum_current_all)
        momentum_pred.scatter_add_(0, batch_expanded, momentum_pred_all)
        momentum_target.scatter_add_(0, batch_expanded, momentum_target_all)

        # Penalize deviation from momentum conservation
        # Only focus on predicted momentum conservation relative to current state
        loss_pred = self.criterion(momentum_pred, momentum_current)

        return loss_pred * self.weight
