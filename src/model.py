import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torchmetrics
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection, PeakSignalNoiseRatio

from .loss import LabelSmoothingLoss
from .network import MobileNetBase, VITBase


class TIMMModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.criterion = instantiate(self.config.loss)
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.model = instantiate(self.config.arch)

    def configure_optimizers(self):
        optimizer = instantiate(self.config.optimizer, params=self.model.parameters())
        scheduler = instantiate(self.config.lr_scheduler, optimizer=optimizer)
        if "ReduceLROnPlateau" in self.config.lr_scheduler._target_:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.config.monitor,
            }
        # return [optimizer], [scheduler]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": self.config.monitor,
            },
        }

    def forward(self, x):

        x = self.model(x)
        return x

    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx) -> None:
        loss, preds, targets = self.step(batch)
        acc = self.train_acc(preds, targets)

        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.sync_dist,
        )
        self.log(
            "train/acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.sync_dist,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> None:
        val_loss, val_preds, val_targets = self.step(batch)
        print(val_preds)
        val_acc = self.val_acc(val_preds, val_targets)

        self.log(
            "val/loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.sync_dist,
        )
        self.log(
            "val/acc",
            val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.sync_dist,
        )
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx) -> None:
        raise NotImplementedError

    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()