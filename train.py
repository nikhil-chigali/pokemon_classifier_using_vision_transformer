"""
    DONE:: Implement logger [Loguru]
    DONE:: Incorporate PyTorch TQDM
    DONE:: Test the code
    TODO:: Migrate to PyTorch Lightning
    TODO:: Integrate WandB
    TODO:: Build data pipelines to Dagster augment dataset
    TODO:: Integrate MLFlow
    TODO:: README
    TODO:: Github actions to deploy on Azure
    TODO:: Train on Azure
"""

import sys
import warnings
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from src.models import ViT, Trainer
from src.data import get_dataset, get_dataloader
from configs import data_config, train_config, path_config


@logger.catch
def main():
    warnings.filterwarnings("ignore")
    parser = ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        action="store",
        default="dryrun",
        help="Experiment name for the run (default='dryrun')",
    )
    args = parser.parse_args()
    assert (
        args.exp_name in train_config.experiments.keys()
    ), f"Experiment {args.exp_name} not found in {path_config.experiments_file}"
    logger.remove()
    logger.add(
        "logs\\exp_" + args.exp_name + "_{time:YYYY-MM-DD_HH-mm-ss}.log",
        format="[{time:YYYY-MM-DD_HH:mm:ss}] | {level} | <lvl>{message}</lvl>",
        rotation="5 MB",
    )
    logger.add(
        sys.stderr,
        format="[{time:YYYY-MM-DD_HH:mm:ss}] || <bold><magenta>EXP: "
        + args.exp_name
        + "</></bold> | {level} | <lvl>{message}</lvl>",
    )
    train_cfg = train_config.experiments[args.exp_name]
    if train_cfg.training.device == "cuda" and not torch.cuda.is_available():
        train_cfg.training.device = "cpu"

    logger.debug("Running on device: {device}", device=train_cfg.training.device)
    trainset, valset = get_dataset(train_cfg, train=True)
    testset = get_dataset(train_cfg, train=False)

    trainloader = get_dataloader(trainset, train_cfg, train=True)
    valloader = get_dataloader(valset, train_cfg, train=True)
    testloader = get_dataloader(testset, train_cfg, train=False)

    logger.info(f"Trainset: {len(trainset.indices)}")
    logger.info(f"Valset: {len(valset.indices)}")
    logger.info(f"Testset: {len(testset)}")
    logger.success("Data loaders instantiated")
    model = ViT(data_cfg=data_config, train_cfg=train_cfg)
    total_params = sum(param.numel() for param in model.parameters())
    logger.info(
        f"Total Parameters: {total_params*1e-6:.2f} M",
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=train_cfg.training.learning_rate,
        weight_decay=train_cfg.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=3, min_lr=1e-6
    )
    logger.success("Model, Optimizer, and LR_Scheduler instantiated")
    trainer = Trainer(
        trainloader,
        valloader,
        testloader,
        model,
        train_cfg.training.epochs,
        criterion,
        optimizer,
        scheduler,
        path_config.model_ckpt_path,
        logger,
        train_cfg.training.device,
    )
    logger.debug(
        f"Hyperparameters:: epochs: {train_cfg.training.epochs}, lr: {train_cfg.training.learning_rate}, weight decay: {train_cfg.training.weight_decay}, batch size: {train_cfg.training.batch_size}, dropout: {train_cfg.training.dropout}"
    )

    logger.info("Model Training Started")
    trainer.train()
    logger.success("Model Training Finished")


if __name__ == "__main__":
    main()
