import torch
from ml_collections import ConfigDict


def get_data_config():
    config = ConfigDict()
    config.img_size = (224, 224)
    config.mean = torch.Tensor([0.6053, 0.5874, 0.5538])
    config.std = torch.Tensor([0.2468, 0.2372, 0.2453])

    return config
