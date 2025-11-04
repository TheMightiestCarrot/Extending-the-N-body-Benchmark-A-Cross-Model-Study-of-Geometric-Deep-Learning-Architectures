from dataloaders.n_body_dataloader import NBodyDataLoader
from torch_geometric.data import Data
import torch
from utils.build_fully_connected_graph import build_graph_with_knn


class PonitaNBodyDataLoader(NBodyDataLoader):
    def preprocess_batch(self, data, device, training=True):
        # data is already a batched Data object from get_batch
        batch = data
        
        # Move to device
        batch = batch.to(device)
        
        # Reshape vec for PONITA
        batch.vec = batch.vel.reshape(batch.vel.shape[0], 1, batch.vel.shape[1])
        
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
        
        # Compute relative positions for PONITA
        row, col = edge_index
        batch.rel_pos = batch.pos[row] - batch.pos[col]
        
        # Add node features (mass)
        batch.x = batch.mass
        
        return batch
    
    def postprocess_batch(self, predictions, device):
        """Return predictions converted to device."""
        if isinstance(predictions, torch.Tensor):
            return predictions.to(device)
        return predictions
