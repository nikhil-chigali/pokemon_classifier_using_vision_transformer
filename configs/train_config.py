import torch
from ml_collections import ConfigDict
import yaml
import os


def get_train_config():
    with open(os.path.join(os.path.dirname(__file__), "train_config.yaml"), "r") as f:
        cfg_dict = yaml.safe_load(f.read())

    config = ConfigDict(cfg_dict["experiment"])
    if config.training.device == "cuda" and not torch.cuda.is_available():
        config.training.device = "cpu"

    return config
