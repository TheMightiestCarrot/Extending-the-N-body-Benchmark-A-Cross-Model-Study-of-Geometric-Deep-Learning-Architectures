import torch

from dataloaders.n_body_dataloader import NBodyDataLoader


class PlatonicTransformerNBodyDataLoader(NBodyDataLoader):
    """Prepare N-body OTF batches for the PlatonicTransformer model."""

    def preprocess_batch(self, data, device, training=True):
        batch = data.to(device)

        batch_size = self.args.batch_size
        num_nodes = self.dataset.num_nodes

        if not hasattr(batch, "batch") or batch.batch is None:
            batch.batch = (
                torch.arange(batch_size, device=device)
                .repeat_interleave(num_nodes)
                .long()
            )

        # Scalar node features: default to masses when available, otherwise ones.
        if not hasattr(batch, "x") or batch.x is None:
            if hasattr(batch, "mass"):
                batch.x = batch.mass
            else:
                batch.x = torch.ones(
                    batch.pos.size(0), 1, device=device, dtype=batch.pos.dtype
                )

        # Vector node features: velocities are the natural choice for dynamics.
        if not hasattr(batch, "vec") or batch.vec is None:
            if hasattr(batch, "vel"):
                batch.vec = batch.vel.unsqueeze(1)
            else:
                batch.vec = torch.zeros(
                    batch.pos.size(0), 1, 3, device=device, dtype=batch.pos.dtype
                )

        return batch

    def postprocess_batch(self, predictions, device):
        if isinstance(predictions, torch.Tensor):
            return predictions.to(device)
        return predictions
