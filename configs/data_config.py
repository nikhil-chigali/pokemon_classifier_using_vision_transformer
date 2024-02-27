import torch
from ml_collections import ConfigDict


def get_data_config():
    config = ConfigDict()
    config.img_size = (224, 224)
    config.mean = torch.Tensor([0.5972, 0.5810, 0.5484])
    config.std = torch.Tensor([0.2495, 0.2412, 0.2481])

    return config
