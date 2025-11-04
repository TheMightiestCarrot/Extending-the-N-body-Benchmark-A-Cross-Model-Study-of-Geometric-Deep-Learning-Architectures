from abc import abstractmethod

from torch_geometric.data import Data

from dataloaders.base_dataloader import BaseDataLoader
from datasets.nbody.dataset_gravity_otf import GravityDatasetOtf


class NBodyDataLoader(BaseDataLoader):
    def __init__(self, args, partition="train"):
        # N-body datasets generate data on-the-fly, so partition is ignored
        super().__init__(args)

    def create_dataset(self):
        return GravityDatasetOtf(
            dataset_name=self.args.dataset_name,
            num_nodes=self.args.num_atoms,
            target=self.args.target,
            sample_freq=self.args.sample_freq,
            batch_size=self.args.batch_size,
            double_precision=self.args.precision_mode == "double",
            center_of_mass=self.args.center_of_mass,
            use_cached=self.args.model_path is None,
            cache_data=True,
        )

    def get_batch(self):
        from torch_geometric.data import Batch

        batch_data = next(self.dataset_iter)

        # Reshape and convert precision
        data = [d.view(-1, d.size(2)) for d in batch_data]
        data = [
            d.double() if self.args.precision_mode == "double" else d.float()
            for d in data
        ]
        data = [d.to(self.device) for d in data]

        # Unpack the tensors
        loc, vel, force, mass, y = data

        # Create a list of Data objects (one per sample in batch)
        batch_size = self.args.batch_size
        n_nodes = self.args.num_atoms

        data_list = []

        for i in range(batch_size):
            start_idx = i * n_nodes
            end_idx = (i + 1) * n_nodes

            # Create a Data object for each sample
            sample = Data(
                pos=loc[start_idx:end_idx],
                vel=vel[start_idx:end_idx],
                force=force[start_idx:end_idx],
                mass=mass[start_idx:end_idx],
                y=y[start_idx:end_idx],  # This is the target
            )
            data_list.append(sample)

        # Convert list to a single batched graph
        batched_data = Batch.from_data_list(data_list)

        # Wrap in a list to match MD's pattern (trainer uses data[0])
        return [batched_data], None

    @abstractmethod
    def preprocess_batch(self, data, training=True):
        pass

    def get_num_nodes(self):
        return self.args.num_atoms
