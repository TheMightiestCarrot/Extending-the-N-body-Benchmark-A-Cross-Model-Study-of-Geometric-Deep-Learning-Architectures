"""DataLoader for Graph Transformer full-attention N-body simulations (Torch model).

Builds a PyG Data with pos, vel, force, mass and batch indices.
Edges are not needed by the model.
"""

from dataloaders.n_body_dataloader import NBodyDataLoader
from torch_geometric.data import Data
import torch


class GraphTransformerNBodyDataLoader(NBodyDataLoader):
    def preprocess_batch(self, data, device, training=True):
        # data comes as a batched Data from parent get_batch(); keep it simple
        # Recreate minimal fields to ensure shapes and device placement
        batch = data.to(device)

        # Ensure batch vector exists (Batch.from_data_list sets it)
        if not hasattr(batch, "batch") or batch.batch is None:
            batch_size = self.args.batch_size
            n_nodes = self.dataset.num_nodes
            batch.batch = torch.arange(0, batch_size, device=device).repeat_interleave(n_nodes).long()

        return batch

    def postprocess_batch(self, predictions, device):
        if isinstance(predictions, torch.Tensor):
            return predictions.to(device)
        return predictions

