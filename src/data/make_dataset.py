import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

import json
from configs import path_config, data_config, train_config


class ToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, tensor):
        return tensor.to(self.device)


def get_transforms():
    pipeline = v2.Compose(
        [
            v2.PILToTensor(),
            v2.Resize(data_config.img_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(data_config.mean, data_config.std),
            ToDevice(train_config.training.device),
        ]
    )
    return pipeline


def make_dataset():
    dataset = ImageFolder(
        root=path_config.dataset_path,
    )

    data_config.class_to_idx = dataset.class_to_idx
    data_config.idx_to_class = dataset.classes

    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.1,
        random_state=train_config.training.seed,
        shuffle=True,
        stratify=dataset.targets,
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.1,
        random_state=train_config.training.seed,
        shuffle=True,
        stratify=np.array(dataset.targets)[train_idx],
    )

    indices = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    with open(path_config.indices_file, "w") as f:
        json.dump(indices, f, indent=4)

    classes = {
        "class_to_idx": dataset.class_to_idx,
        "idx_to_class": dataset.classes,
    }
    with open(path_config.classes_file, "w") as f:
        json.dump(classes, f, indent=4)


def get_dataset(train=True):
    make_dataset()
    dataset = ImageFolder(
        root=path_config.dataset_path,
        transform=get_transforms(),
    )
    with open(path_config.indices_file, "r") as f:
        indices = json.load(f)
    if train:
        train_set = Subset(dataset, indices["train"])
        val_set = Subset(dataset, indices["val"])
        return train_set, val_set
    else:
        test_set = Subset(dataset, indices["test"])
        return test_set


def get_dataloader(dataset, train=True):
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.training.batch_size,
        shuffle=train,
    )
    return dataloader