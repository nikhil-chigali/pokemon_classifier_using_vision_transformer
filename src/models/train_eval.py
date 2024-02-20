import os
import numpy as np

import torch
from torchmetrics.classification import MulticlassAccuracy


class Trainer:
    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        model,
        epochs,
        criterion,
        optimizer,
        learning_rate,
        ckpt_path,
        device="cpu",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = len(self.train_loader.dataset.dataset.classes)
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.ckpt_path = ckpt_path

        self.criterion = criterion
        self.mca = MulticlassAccuracy(
            num_classes=self.num_classes,
            average=None,
        )
        self.mca.to(device)

        self.optimizer = optimizer
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def classwise_acc(self, loader):
        total_acc = torch.zeros((self.num_classes,))
        self.model.eval()
        n = 0.0
        for batch in loader:
            images, targets = batch
            n += images.size(0)
            with torch.no_grad():
                logits = self.model(images)

                total_acc += self.mca(logits, targets)
        total_acc /= n
        return total_acc

    def train_step(self):
        epoch_loss = 0.0
        epoch_acc = 0.0
        n = 0.0
        self.model.train()

        for batch in self.train_loader:
            images, targets = batch
            n += images.size(0)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            epoch_acc += self.mca(logits, targets).sum().item()

        epoch_acc /= n
        epoch_loss /= n
        self.history["train_acc"].append(epoch_acc)
        self.history["train_loss"].append(epoch_loss)

    def eval_step(self, val=True):
        epoch_loss = 0.0
        epoch_acc = 0.0
        n = 0.0
        self.model.eval()

        if val:
            loader = self.val_loader
        else:
            loader = self.test_loader

        for batch in loader:
            images, targets = batch
            n += images.size(0)
            with torch.no_grad():
                logits = self.model(images)
                loss = self.criterion(logits, targets)

                epoch_loss += loss.item() * images.size(0)
                epoch_acc += self.mca(logits, targets).sum().item()

        epoch_acc /= n
        epoch_loss /= n

        if val:
            self.history["val_acc"].append(epoch_acc)
            self.history["val_loss"].append(epoch_loss)
        else:
            return epoch_loss, epoch_acc

    def train(self):
        best_train_loss = np.inf
        best_val_loss = np.inf
        best_model_wts = None

        self.history["train_acc"] = []
        self.history["train_loss"] = []
        self.history["val_acc"] = []
        self.history["val_loss"] = []
        for _ in range(self.epochs):
            self.train_step()
            self.eval_step(val=True)
            if self.history["val_loss"][-1] < best_val_loss:
                best_train_loss = self.history["train_loss"][-1]
                best_val_loss = self.history["val_loss"][-1]
                best_model_wts = self.model.state_dict()

        ckpt_file = os.path.join(
            self.ckpt_path,
            "ckpt-trainloss_{:0.02f}-valloss_{:0.02f}.pt".format(
                best_train_loss, best_val_loss
            ),
        )
        torch.save(ckpt_file, best_model_wts)
        self.model.load_state_dict(torch.load(ckpt_file))
        train_acc = self.classwise_acc(self.train_loader)
        val_acc = self.classwise_acc(self.val_loader)
        return train_acc, val_acc

    def test(self):
        test_loss, test_acc = self.eval_step(val=False)
        classwise_acc = self.classwise_acc(self.test_loader)
        return test_loss, test_acc, classwise_acc
