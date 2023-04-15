import logging
from pathlib import Path
import random
import numpy as np
import hydra
import pytorch_lightning as pl
import torch
import wandb
import random
import numpy as np

### seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from omegaconf import DictConfig, OmegaConf
# from pytorch_lightning.callbacks import Callback, ModelCheckpoint, RichProgressBar, LearningRateMonitor
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

from src.dataset import ChestXrayDatamodule
from src.model import TIMMModel

log = logging.getLogger(__name__)


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


def train(config):

    model = TIMMModel(config.model)
    datamodule = ChestXrayDatamodule(config.dataset)
    trainer = pl.Trainer(
        **config.trainer,
    )

    trainer.fit(model, datamodule=datamodule)


@hydra.main(config_path="configs", config_name="default")
def main(config: DictConfig) -> None:

    log.info("MLops-Project - Chest Xray Classification")
    log.info(f"Current working directory : {Path.cwd()}")

    train(config)


if __name__ == "__main__":
    main()