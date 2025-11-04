from dataloaders.n_body_dataloader import NBodyDataLoader
from torch_geometric.data import Data
import torch
from utils.build_fully_connected_graph import build_graph_with_knn


class EquiformerV2NBodyDataLoader(NBodyDataLoader):
    def preprocess_batch(self, data, device, training=True):
        # data is already a batched Data object from get_batch
        batch = data
        
        # Move to device
        batch = batch.to(device)
        
        # Get batch info
        batch_size = self.args.batch_size
        n_nodes = self.dataset.num_nodes
        
        # Use KNN graph construction instead of radius graph
        # This avoids the memory issues with large radius values
        max_neighbors = self.args.max_neighbors if hasattr(self.args, 'max_neighbors') else min(n_nodes - 1, 32)
        
        # Build KNN graph directly (more stable than radius graph)
        edge_index = build_graph_with_knn(
            batch.pos,
            batch_size,
            n_nodes,
            device,
            min(max_neighbors, n_nodes - 1),
        )
        
        batch.edge_index = edge_index
        
        # Add node features required by Equiformer V2
        # Node type (all atoms are the same type in N-body)
        batch.node_type = torch.zeros(batch_size * n_nodes, dtype=torch.long, device=device)
        
        # Edge attributes (distances)
        row, col = edge_index
        edge_vec = batch.pos[row] - batch.pos[col]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        batch.edge_attr = edge_dist
        
        # Add velocities as node features
        batch.node_attr = batch.vel
        
        # Keep mass information
        batch.x = batch.mass
        
        return batch
    
    def postprocess_batch(self, predictions, device):
        """Convert model predictions back to dataset format"""
        # Equiformer V2 returns predictions in the same format as input
        # No postprocessing needed for n-body simulations
        if isinstance(predictions, torch.Tensor):
            return predictions.to(device)
        return predictions