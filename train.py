"""
 TODO:: Implement logger [Loguru]
 TODO:: Incorporate PyTorch TQDM
 TODO:: Test the code
 TODO:: Integrate WandB
 TODO:: Build data pipelines to Dagster augment dataset
 TODO:: Integrate MLFlow
 TODO:: README
 TODO:: Github actions to deploy on Azure
 TODO:: Train on Azure
"""

from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.optim import Adam

from src.models import ViT, Trainer
from src.data import get_dataset, get_dataloader
from configs import data_config, train_config, path_config


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        action="store",
        default="dryrun",
        help="Experiment name for the run (default=dryrun)",
    )
    args = parser.parse_args()
    train_cfg = train_config.experiments[args.exp_name]
    if train_cfg.training.device == "cuda" and not torch.cuda.is_available():
        train_cfg.training.device = "cpu"

    trainset, valset = get_dataset(train_cfg, train=True)
    testset = get_dataset(train_cfg, train=False)

    trainloader = get_dataloader(trainset, train_config, train=True)
    valloader = get_dataloader(valset, train_config, train=True)
    testloader = get_dataloader(testset, train_config, train=False)

    model = ViT(data_cfg=data_config, train_cfg=train_cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        params=model.parameters(),
        lr=train_cfg.training.learning_rate,
        weight_decay=train_cfg.training.weight_decay,
    )
    trainer = Trainer(
        trainloader,
        valloader,
        testloader,
        model,
        train_cfg.training.epochs,
        criterion,
        optimizer,
        train_cfg.training.learning_rate,
        path_config.model_ckpt_path,
        train_cfg.training.device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
