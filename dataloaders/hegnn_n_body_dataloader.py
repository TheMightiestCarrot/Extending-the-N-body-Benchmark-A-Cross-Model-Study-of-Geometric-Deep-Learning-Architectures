import torch

from dataloaders.n_body_dataloader import NBodyDataLoader
from utils.build_fully_connected_graph import build_graph_with_knn


class HEGNNNBodyDataLoader(NBodyDataLoader):
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

        pos = batch.pos
        vel = batch.vel
        dtype = pos.dtype
        mass = batch.mass if hasattr(batch, "mass") else torch.ones(
            pos.size(0), 1, device=device, dtype=dtype
        )
        mass = mass.to(device=device, dtype=dtype)
        vel = vel.to(dtype=dtype)

        speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
        kinetic = 0.5 * mass * speed.square()
        ones = torch.ones_like(speed)
        node_feat = torch.cat([mass, speed, kinetic, ones], dim=-1)
        batch.node_feat = node_feat
        batch.x = node_feat

        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        batch.rel_pos = rel_pos
        edge_length = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
        batch.edge_length = edge_length

        return batch

    def postprocess_batch(self, predictions, device):
        if isinstance(predictions, dict):
            targets = getattr(self.args, "target", "pos_dt+vel")
            ordered = []
            for target in targets.split("+"):
                if target not in predictions:
                    raise KeyError(
                        f"Prediction for target '{target}' missing in HEGNN outputs."
                    )
                ordered.append(predictions[target])
            predictions = torch.cat(ordered, dim=-1)

        if isinstance(predictions, torch.Tensor):
            return predictions.to(device)

        raise TypeError(
            "Dataloader received unsupported prediction type from model."
        )
