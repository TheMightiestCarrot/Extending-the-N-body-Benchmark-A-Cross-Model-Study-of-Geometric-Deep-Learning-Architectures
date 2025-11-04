from dataloaders.n_body_dataloader import NBodyDataLoader
from torch_geometric.data import Data
import torch
from utils.build_fully_connected_graph import build_graph_with_knn
from models.segnn.o3_building_blocks import O3Transform


class SegnnNBodyDataLoader(NBodyDataLoader):
    def preprocess_batch(self, data, device, training=True):
        # data is already a batched Data object from get_batch
        batch = data
        
        # Move to device
        batch = batch.to(device)
        
        # Build edges for the batched graph
        batch_size = self.args.batch_size
        n_nodes = self.dataset.num_nodes
        
        edge_index = build_graph_with_knn(
            batch.pos,
            batch_size,
            n_nodes,
            device,
            self.args.num_neighbors,
        )
        batch.edge_index = edge_index

        # Apply O3 transform
        transform = O3Transform(self.args.lmax_attr)
        batch = transform(batch)  # Add O3 attributes
        
        return batch

    def postprocess_batch(self, predictions, device):
        """Return predictions converted to device."""
        if isinstance(predictions, torch.Tensor):
            return predictions.to(device)
        return predictions
