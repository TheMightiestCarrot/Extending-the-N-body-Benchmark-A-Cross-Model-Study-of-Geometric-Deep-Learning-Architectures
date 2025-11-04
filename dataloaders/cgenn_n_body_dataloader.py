import torch
from torch_geometric.data import Data

from dataloaders.n_body_dataloader import NBodyDataLoader


class CgennNBodyDataLoader(NBodyDataLoader):
    def preprocess_batch(self, data, device, training=True):
        # data is already a batched Data object from get_batch
        batch = data

        # Move to device
        batch = batch.to(device)
        self.device = device

        batch_size = self.dataset.batch_size
        n_nodes = self.dataset.num_nodes

        # Extract data from batch
        loc = batch.pos
        vel = batch.vel
        mass = batch.mass
        if hasattr(batch, "y"):
            y = batch.y

        output_dims = loc.shape[-1]
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        # Create batch_size copies of the edges tensor
        edges_cgenn = (
            torch.Tensor([rows, cols]).long().unsqueeze(0).repeat(batch_size, 1, 1)
        )

        edges_cgenn = edges_cgenn.to(self.device)

        # Reshape loc and vel to (batch_size, n_nodes, output_dims)
        loc = loc.view(batch_size, n_nodes, output_dims)
        vel = vel.view(batch_size, n_nodes, output_dims)
        mass = mass.view(batch_size, n_nodes, 1)  # Assuming mass is 1D

        # Prepare edge attributes (None in this case)
        edge_attr = None

        # Charges (mass) and placeholder for unused variable
        charges = mass
        _ = None

        # Prepare data tuple as expected by the model
        data = Data(tuple=(loc, vel, edge_attr, charges, _, edges_cgenn))
        data.pos = loc.view(batch_size * n_nodes, output_dims)
        data.vel = vel.view(batch_size * n_nodes, output_dims)
        data.mass = mass.view(batch_size * n_nodes, 1)
        if hasattr(batch, "y"):
            data.y = y
        return data

    def postprocess_batch(self, predictions, device):
        """Convert model predictions back to dataset format"""
        # CGENN returns predictions in the same format as input
        # No postprocessing needed for n-body simulations
        return predictions
