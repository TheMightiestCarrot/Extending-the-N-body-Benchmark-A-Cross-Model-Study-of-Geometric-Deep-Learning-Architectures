import torch

from dataloaders.n_body_dataloader import NBodyDataLoader
from utils.build_fully_connected_graph import build_graph_with_knn


class EgnnMcNBodyDataLoader(NBodyDataLoader):
    def preprocess_batch(self, data, device, training=True):
        batch = data.to(device)

        batch_size = self.args.batch_size
        num_nodes = self.dataset.num_nodes
        num_neighbors = getattr(self.args, "num_neighbors", None)
        if (
            num_neighbors is None
            or num_neighbors <= 0
            or num_neighbors >= num_nodes
        ):
            num_neighbors = num_nodes - 1

        edge_index = build_graph_with_knn(
            batch.pos,
            batch_size,
            num_nodes,
            device,
            num_neighbors,
        )
        batch.edge_index = edge_index

        if not hasattr(batch, "batch") or batch.batch is None:
            batch.batch = (
                torch.arange(batch_size, device=device)
                .repeat_interleave(num_nodes)
                .long()
            )

        vel = batch.vel
        pos_dtype = batch.pos.dtype
        mass = batch.mass if hasattr(batch, "mass") else torch.ones(
            vel.size(0), 1, device=device, dtype=vel.dtype
        )
        mass = mass.to(device=device, dtype=pos_dtype)

        speed = torch.norm(vel, dim=-1, keepdim=True)
        node_features = torch.cat([speed, mass], dim=-1).to(dtype=pos_dtype)
        batch.x = node_features

        row, col = edge_index
        edge_vec = batch.pos[row] - batch.pos[col]
        dist_sq = (edge_vec**2).sum(dim=-1, keepdim=True)
        dist = dist_sq.sqrt().clamp_min(1e-12)
        direction = edge_vec / dist

        proj_row = (vel[row] * direction).sum(dim=-1, keepdim=True)
        proj_col = (vel[col] * direction).sum(dim=-1, keepdim=True)
        mass_prod = mass[row] * mass[col]

        edge_attr = torch.cat([mass_prod, proj_row, proj_col, dist_sq], dim=-1)
        batch.edge_attr = edge_attr.to(dtype=pos_dtype)

        return batch

    def postprocess_batch(self, predictions, device):
        if isinstance(predictions, torch.Tensor):
            return predictions.to(device)
        return predictions
