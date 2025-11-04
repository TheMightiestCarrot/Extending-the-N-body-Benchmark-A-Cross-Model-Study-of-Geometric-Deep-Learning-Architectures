from abc import ABC, abstractmethod

from utils.get_device import get_device


class BaseDataLoader(ABC):
    def __init__(self, args):
        self.args = args
        # Determine the device as early as possible so subclasses can
        # reference ``self.device`` during dataset creation.
        self.device = get_device(self.args.gpu_id)
        self.dataset = self.create_dataset()
        self.dataset_iter = iter(self.dataset)

    @abstractmethod
    def get_batch(self):
        pass

    @abstractmethod
    def preprocess_batch(self, data, device, training=True):
        pass

    @abstractmethod
    def create_dataset(self):
        pass

    @abstractmethod
    def postprocess_batch(self, predictions, device):
        """Convert model predictions back to dataset format"""
        pass
