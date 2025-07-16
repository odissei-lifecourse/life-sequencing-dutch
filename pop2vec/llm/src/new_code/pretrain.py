import argparse
import logging
import os
import pickle
import re
import sys
import shutil

import numpy as np

from pathlib import Path
import pandas as pd
import torch
from pytorch_lightning import Trainer

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from datetime import timedelta

import pop2vec.llm.src.transformer


from pop2vec.llm.src.new_code.load_data import CustomLazyHDF5Dataset
from pop2vec.llm.src.new_code.load_data import CustomIterableDataset
from pop2vec.llm.src.new_code.utils import read_json
from pop2vec.llm.src.new_code.utils import read_hparams
from pop2vec.llm.src.transformer.models import TransformerEncoder





logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s: %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
  level=logging.INFO
)
logger = logging.getLogger(__name__)


DEFAULT_VALS = {
    'accumulate_grad_batches': 1,
    'resume_from_checkpoint': None,
    'val_check_interval': 1.0,
    'gradient_clip_val': 1,
    'training_precision': "32-true",
    'load_model_weights_only': False,
}

# required keys 
REQ_KEYS = [
    'checkpoint_dir', 
    'mlm_path', 
    'num_val_items', 
    'batch_size',
    'epochs',
    'vocab_path'
]


def hparams_integrity_check(hparams):
    missing = [k for k in REQ_KEYS if k not in hparams]
    if missing:
        raise ValueError(f"Missing required hparams: {', '.join(missing)}")


def update_hparams_with_defaults(hparams):
    for key, val in DEFAULT_VALS.items():
        hparams.setdefault(key, val)
    return hparams 



def get_callbacks(ckpoint_dir):
  os.makedirs(ckpoint_dir, exist_ok=True)
  callbacks = [
    ModelCheckpoint(
      dirpath=ckpoint_dir,
      filename="model-{epoch:02d}-{step}-{val_loss_track:.2f}",
      monitor="val_loss_track",
      save_top_k=2,
      save_last=False,
      mode='min',
      save_weights_only=False,
      verbose=True,
    ),
    ModelCheckpoint(
      dirpath=os.path.join(ckpoint_dir, "time_ckpts"),
      filename="model-{epoch:02d}-{step}",
      train_time_interval=timedelta(hours=1),
      monitor=None,
      save_top_k=1,
      save_last=False,
      save_weights_only=False,
      verbose=True,
    ),
  ]
  return callbacks

def get_vocab_size(path):
    return len(pd.read_csv(path))

# Helper: Load and update hyperparameters.
def load_hparams(hparams_path: str):
    hparams = read_hparams(hparams_path)
    hparams_integrity_check(hparams)
    hparams['vocab_size'] = get_vocab_size(hparams['vocab_path'])
    return update_hparams_with_defaults(hparams)

# Helper: Create training and validation dataloaders.
def get_dataloaders(mlm_path, num_val_items, batch_size):
    val_dataset = CustomLazyHDF5Dataset(
        mlm_path,
        validation=True,
        num_val_items=num_val_items
    )
    train_dataset = CustomLazyHDF5Dataset(
        mlm_path,
        validation=False,
        num_val_items=num_val_items
    )
    num_workers = len(os.sched_getaffinity(0)) - 3
    logging.info(f"num of workers for train dataloader = {num_workers}")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True,
        shuffle=True
    )
    return train_dataloader, val_dataloader


# Helper: Determine the DDP strategy.
def get_ddp_strategy():
    if DDP_STRATEGY == "auto":
        return "auto"
    elif DDP_STRATEGY == "ddp":
        return DDPStrategy()
    elif DDP_STRATEGY == "ddp_mpi":
        return DDPStrategy(process_group_backend="mpi")
    elif DDP_STRATEGY == "gloo":
        return DDPStrategy(process_group_backend="gloo")
    else:
        raise ValueError(f"Unsupported DDP_STRATEGY: {DDP_STRATEGY}")

# Main training function.
def pretrain(hparams):
    ckpt_dir = hparams["checkpoint_dir"]
    mlm_path = hparams["mlm_path"]

    logger.debug(f"hparams --\n{hparams}")
    # Determine batch size and validation interval.
    num_val_items = hparams['num_val_items']
    batch_size = hparams['batch_size'] 
    # Create dataloaders.
    logger.info("loading dataloaders")
    train_dataloader, val_dataloader = get_dataloaders(mlm_path, num_val_items, batch_size)
    accumulate_grad_batches = hparams['accumulate_grad_batches']
    hparams['steps_per_epoch'] = (
        int(len(train_dataloader) / (N_DEVICES*accumulate_grad_batches))+2
    )
    logger.info("dataloaders loaded")
    # Set up CSV logger.
    csv_logger = CSVLogger(save_dir=hparams['checkpoint_dir'])

    # Create callbacks.
    callbacks = get_callbacks(ckpt_dir)

    # Decide on the distributed strategy.
    strategy = get_ddp_strategy()

    checkpoint_path = hparams.get("resume_from_checkpoint")
    load_weights_only = hparams.get("load_model_weights_only")

    if checkpoint_path and load_weights_only:
        # ── WEIGHTS-ONLY ───────────────────────────────
        model = TransformerEncoder(hparams)          # fresh LightningModule
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        logger.info(f"Weights loaded. missing={missing}, unexpected={unexpected}")
        fit_ckpt_path = None                         # do NOT let Trainer resume
    else:
        # ── NORMAL RESUME or FRESH START ──────────────
        model = TransformerEncoder(hparams)
        fit_ckpt_path = checkpoint_path 
    # Create Trainer instance. The resume_from_checkpoint argument ensures that
    # model state, optimizer, scheduler (e.g., OneCycleLR), global step, and epoch are resumed.
    trainer = Trainer(
        strategy=strategy,
        default_root_dir=ckpt_dir,
        callbacks=callbacks,
        max_epochs=hparams['epochs'],
        val_check_interval=hparams['val_check_interval'],
        accelerator=ACCELERATOR,
        devices=N_DEVICES,
        logger=csv_logger,
        precision=hparams['training_precision'],
        log_every_n_steps=1000,
        gradient_clip_val=hparams['gradient_clip_val'],
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=accumulate_grad_batches
    )

    logger.info("Starting Trainer.fit(...)")
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=fit_ckpt_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", required=True, type=str, help="Path to hyperparameters file (REQUIRED).")
    parser.add_argument("--accelerator", default="gpu", help="Choose an accelerator that connects a Lightning Trainer to arbitrary hardware (CPUs, GPUs, TPUs, HPUs, MPS, ...)")
    parser.add_argument("--ddpstrategy", default="auto", help="pick ddp strategy (auto,gloo,mpi,...)")
    parser.add_argument("--devices", default=1, type=int, help="Number of devices")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    ACCELERATOR=args.accelerator
    N_DEVICES=args.devices
    DDP_STRATEGY=args.ddpstrategy # strategy for pl.Trainer
    HPARAMS=args.hparams
    
    assert DDP_STRATEGY in ["auto", "ddp_mpi", "ddp", "gloo"]

    # torch.set_float32_matmul_precision("medium")

    pretrain(load_hparams(HPARAMS))
    
