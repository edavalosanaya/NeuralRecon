
from . import transforms
from . import sampler
from .demo import DemoDataset
from .scannet import ScanNetDataset

# import importlib

# # find the dataset definition by name, for example ScanNetDataset (scannet.py)
def find_dataset_def(dataset_name):
    if dataset_name == 'scannet':
        return ScanNetDataset
    elif dataset_name == 'demo':
        return DemoDataset
