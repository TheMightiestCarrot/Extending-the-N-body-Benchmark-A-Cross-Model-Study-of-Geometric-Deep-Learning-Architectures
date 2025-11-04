import torch
from torch_geometric.loader import DataLoader

from dataloaders.base_dataloader import BaseDataLoader
from datasets.nbody_offline.dataset import NBodySystemDataset
from models.segnn.o3_building_blocks import O3Transform


class SegnnNbodyOfflineDataloader(BaseDataLoader):
    """Offline N-body dataloader for SEGNN.
    """

    def __init__(self, args, partition: str = "train"):
        self.partition = partition
        super().__init__(args)

        shuffle = partition == "train"
        # Drop incomplete batches during training for stability 
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            drop_last=(partition == "train"),
        )
        self.dataloader_iter = iter(self.dataloader)

        # Prepare transform identical to online SEGNN n-body
        self.transform = O3Transform(self.args.lmax_attr)

    def __len__(self):
        return len(self.dataloader)

    def create_dataset(self):
        return NBodySystemDataset(
            dataset_name=self.args.dataset_name,
            data_dir=self.args.data_directory,
            virtual_channels=self.args.virtual_channels,
            partition=self.partition,
            max_samples=self.args.max_samples,
            frame_0=self.args.frame_0,
            frame_T=self.args.frame_T,
            cutoff_rate=self.args.cutoff_rate,
            device=self.device,
        )

    def get_batch(self):
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)
        # Mirror MD-style return: list of one batched graph
        return [batch], None

    def preprocess_batch(self, data, device, training=True):
        # Map offline fields to the ones expected by the SEGNN transform and trainer
        # Positions/velocities at t0
        data.pos = data.loc_0
        data.vel = data.vel_0

        # Targets follow the same convention as the OTF n-body dataset
        target = getattr(self.args, "target", "pos_dt+vel")
        if target == "pos":
            data.y = data.loc_t
        elif target == "pos_dt":
            data.y = data.loc_t - data.loc_0
        elif target == "pos_dt+vel":
            data.y = torch.cat((data.loc_t - data.loc_0, data.vel_t), dim=1)
        elif target == "pos+vel":
            data.y = torch.cat((data.loc_t, data.vel_t), dim=1)
        else:
            raise ValueError(f"Unsupported target for SEGNN offline dataloader: {target}")

        # Provide mass and force placeholders for O3Transform
        # - Use charges as mass proxies 
        # - No force available in offline dataset
        num_nodes = data.pos.shape[0]
        if not hasattr(data, "mass"):
            if hasattr(data, "node_attr"):
                # ensure (N,1)
                mass = data.node_attr
                if mass.dim() == 1:
                    mass = mass.unsqueeze(-1)
                data.mass = mass
            else:
                data.mass = torch.ones((num_nodes, 1), device=data.pos.device, dtype=data.pos.dtype)
        if not hasattr(data, "force"):
            data.force = torch.zeros_like(data.pos)

        # Keep provided graph connectivity; O3Transform will compute edge_attr (SH)
        graph = data.to(device)
        graph = self.transform(graph)

        return graph

    def postprocess_batch(self, predictions, device):
        if isinstance(predictions, torch.Tensor):
            return predictions.to(device)
        return predictions
