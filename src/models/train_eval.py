import os
import numpy as np

import torch
from torchmetrics.classification import MulticlassAccuracy
from torch.hub import tqdm


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
        scheduler,
        ckpt_path,
        logger,
        device="cpu",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = len(self.train_loader.dataset.dataset.classes)
        self.model = model
        self.epochs = epochs
        self.scheduler = scheduler
        self.ckpt_path = ckpt_path
        self.logger = logger

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

    def train_step(self, current_epoch):
        epoch_loss = 0.0
        epoch_acc = 0.0
        self.model.train()

        tk = tqdm(
            self.train_loader, desc=f"EPOCH[TRAIN] {current_epoch+1}/{self.epochs}"
        )

        for t, batch in enumerate(tk):
            images, targets = batch

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            epoch_acc += self.mca(logits, targets).mean().item()

            tk.set_postfix(
                {
                    "Loss": f"{epoch_loss / (t+1):0.02f}",
                    "Accuracy": f"{epoch_acc / (t+1):.04f}",
                }
            )

        self.logger.debug(tk.desc)
        epoch_acc /= len(self.train_loader)
        epoch_loss /= len(self.train_loader)
        self.history["train_acc"].append(epoch_acc)
        self.history["train_loss"].append(epoch_loss)
        self.logger.debug(f"Loss: {epoch_loss:0.02f} | Accuracy: {epoch_acc:0.04f}")

    def eval_step(self, current_epoch, val=True):
        epoch_loss = 0.0
        epoch_acc = 0.0
        self.model.eval()

        if val:
            tk = tqdm(
                self.val_loader, desc=f"EPOCH[VAL] {current_epoch+1}/{self.epochs}"
            )
        else:
            tk = tqdm(self.test_loader, desc="EPOCH[TEST]")

        for t, batch in enumerate(tk):
            images, targets = batch
            with torch.no_grad():
                logits = self.model(images)
                loss = self.criterion(logits, targets)

                epoch_loss += loss.item() * images.size(0)
                epoch_acc += self.mca(logits, targets).mean().item()
                tk.set_postfix(
                    {
                        "Loss": f"{epoch_loss / (t + 1):0.02f}",
                        "Accuracy": f"{epoch_acc / (t+1):.04f}",
                    }
                )

        self.logger.debug(tk.desc)
        if val:
            epoch_acc /= len(self.val_loader)
            epoch_loss /= len(self.val_loader)
            self.history["val_acc"].append(epoch_acc)
            self.history["val_loss"].append(epoch_loss)
            self.logger.debug(f"Loss: {epoch_loss:0.02f} | Accuracy: {epoch_acc:0.04f}")
        else:
            epoch_acc /= len(self.test_loader)
            epoch_loss /= len(self.test_loader)
            self.logger.debug(f"Loss: {epoch_loss:0.02f} | Accuracy: {epoch_acc:0.04f}")
            return epoch_loss, epoch_acc

    def train(self):
        best_train_loss = np.inf
        best_val_loss = np.inf
        best_model_wts = None

        self.history["train_acc"] = []
        self.history["train_loss"] = []
        self.history["val_acc"] = []
        self.history["val_loss"] = []
        for e in range(self.epochs):
            self.train_step(e)
            self.eval_step(e, val=True)
            self.scheduler.step(self.history["val_loss"][-1])
            if self.history["val_loss"][-1] < best_val_loss:
                best_train_loss = self.history["train_loss"][-1]
                best_val_loss = self.history["val_loss"][-1]
                best_model_wts = self.model.state_dict()

        ckpt_file = os.path.join(
            self.ckpt_path,
            f"ckpt-trainloss_{best_train_loss:0.02f}-valloss_{best_val_loss:0.02f}.pt",
        )
        torch.save(ckpt_file, best_model_wts)

        self.logger.info(
            "Saved Trained Model Checkpoint at: {ckpt_file}", ckpt_file=ckpt_file
        )

        self.model.load_state_dict(torch.load(ckpt_file))

        train_acc = self.classwise_acc(self.train_loader)
        val_acc = self.classwise_acc(self.val_loader)

        return train_acc, val_acc

    def test(self):

        test_loss, test_acc = self.eval_step(-1, val=False)
        classwise_acc = self.classwise_acc(self.test_loader)
        return test_loss, test_acc, classwise_acc
