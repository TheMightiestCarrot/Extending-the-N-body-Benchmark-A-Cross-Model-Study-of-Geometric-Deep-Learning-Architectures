import torch
import torch.nn as nn

class DynamicWeightedLoss(nn.Module):
    def __init__(self):
        super(DynamicWeightedLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.register_buffer('running_loss_pos', torch.tensor(1.0))
        self.register_buffer('running_loss_vel', torch.tensor(1.0))
        self.alpha = 0.99  # Smoothing factor for running averages

    def forward(self, pred, target):
        pred_pos, pred_vel = pred[..., :3], pred[..., 3:]
        target_pos, target_vel = target[..., :3], target[..., 3:]

        loss_pos = self.mse_loss(pred_pos, target_pos).mean()
        loss_vel = self.mse_loss(pred_vel, target_vel).mean()

        # Update running averages
        with torch.no_grad():
            self.running_loss_pos = self.alpha * self.running_loss_pos + (1 - self.alpha) * loss_pos
            self.running_loss_vel = self.alpha * self.running_loss_vel + (1 - self.alpha) * loss_vel

        # Calculate dynamic weights
        total_running_loss = self.running_loss_pos + self.running_loss_vel
        weight_pos = self.running_loss_vel / total_running_loss
        weight_vel = self.running_loss_pos / total_running_loss

        # Apply dynamic weights
        total_loss = (weight_pos * loss_pos) + (weight_vel * loss_vel)
        return total_loss
