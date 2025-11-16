from dataloaders.n_body_dataloader import NBodyDataLoader
import torch

from utils.build_fully_connected_graph import build_graph_with_knn


class PaiNNNBodyDataLoader(NBodyDataLoader):
    """Dataloader wrapper that builds neighbour graphs for PaiNN."""

    def preprocess_batch(self, data, device, training: bool = True):
        batch = data.to(device)

        batch_size = self.args.batch_size
        num_nodes = self.dataset.num_nodes
        edge_index = build_graph_with_knn(
            batch.pos,
            batch_size,
            num_nodes,
            device,
            getattr(self.args, "num_neighbors", None),
        )
        batch.edge_index = edge_index

        if not hasattr(batch, "batch") or batch.batch is None:
            batch.batch = (
                torch.arange(batch_size, device=device)
                .repeat_interleave(num_nodes)
                .long()
            )

        return batch

    def postprocess_batch(self, predictions, device):
        if isinstance(predictions, torch.Tensor):
            return predictions.to(device)
        return predictions
