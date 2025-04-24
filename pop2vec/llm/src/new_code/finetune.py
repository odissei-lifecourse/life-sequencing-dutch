# In case of conflicts (overlapping parameters with different values)
# cfg has the highest priority
# then parameters passed through CLI
# then hparams 

# finetune.py
import argparse
import logging
import os
import sys

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

# Example imports; adapt to your actual structure:
# from pop2vec.llm.src.transformer.models import FinetuneTransformer
# from pop2vec.llm.src.new_code.finetune_dataset import FineTuneInMemoryDataset
from pop2vec.llm.src.new_code.utils import read_json, read_hparams
from pop2vec.llm.src.new_code.load_data import FineTuneInMemoryDataset
from pop2vec.llm.src.transformer.cls_model import Transformer_CLS
    
logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)
print(f"logger level = {logging.getLevelName(logger.getEffectiveLevel())}")
for handler in logger.handlers:
    handler.setLevel(logging.DEBUG)
print(f"logger level = {logging.getLevelName(logger.getEffectiveLevel())}")


ACCELERATOR = None
DDP_STRATEGY = None
N_DEVICES = 1
PRECISION = "32-true"

def get_callbacks(ckpoint_dir): 
    """
    Create model checkpoint callbacks, etc.
    """
    os.makedirs(ckpoint_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpoint_dir,
            filename="finetune-{epoch:02d}-{step}-{val-binary-f1_80:.2f}",
            monitor="val-binary-f1_80",
            save_top_k=2,
            save_last=False,
            mode='max',
            save_weights_only=False,
            verbose=True,
        )
    ]
    return callbacks

def get_ddp_strategy():
    """
    Return the requested DDP strategy or 'auto'.
    """
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

def load_hparams(cfg, hparams_path=None):
    """
    Load hyperparameters. Possibly read a separate file if needed.
    The config should contain info about:
      - PRETRAINED_CHECKPOINT_PATH: path to the LLM checkpoint
      - PREDICTION_LAYER_TYPE: single-layer-NN, two-layer-NN, or attention-agg-NN
      - etc.
    """
    if hparams_path is None:
        hparams_path = cfg["HPARAMS_PATH"]  # or None if not provided
    hparams = read_hparams(hparams_path)
    # Merge hparams with config
    hparams.update(cfg)
    return hparams

def get_dataloaders(cfg, hparams):
    """
    Create train/val/test dataloaders for fine-tuning using your FineTuneInMemoryDataset.
    Assume the config includes paths like:
      - FINE_TUNE_DATA_H5
      - FINE_TUNE_LABEL_FILE
      - BATCH_SIZE
      - ...
    """
    data_h5 = cfg["SEQ_ENCODING"]
    label_file = cfg["FINE_TUNE_LABEL_FILE"]
    batch_size = hparams.get("batch_size", 32)
    num_workers = max(len(os.sched_getaffinity(0)) - 2, 1)

    # Create train/val/test dataset objects (adjust as needed)
    train_dataset = FineTuneInMemoryDataset(
        h5_file_path=data_h5,
        train_file_path=label_file,
        phase='train',
        task_type=hparams.get("task_type", "classification"),  # classification or regression
        target_col=hparams.get("target_col", "target_label"),
        # num_val_items=cfg.get("NUM_VAL_ITEMS", 5000),
        val_split=cfg.get("VAL_SPLIT", 0.1),
        return_sequence_id=True,
        assign_weights=True,
    )
    val_dataset = FineTuneInMemoryDataset(
        h5_file_path=data_h5,
        train_file_path=label_file,
        phase='validation',
        task_type=hparams.get("task_type", "classification"),
        target_col=hparams.get("target_col", "target_label"),
        # num_val_items=cfg.get("NUM_VAL_ITEMS", 100000),
        val_split=cfg.get("VAL_SPLIT", 0.1),
        return_sequence_id=True
    )
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=num_workers,
        sampler=train_dataset.sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    for b in val_loader:
        logger.info(b)
        break
    logger.info(
        f"batch size = {batch_size}\n"
        f"train batches = {len(train_loader)}\n"
        f"validation batches = {len(val_loader)}\n"
    )
    return train_loader, val_loader

def fine_tune(cfg, batch_size=None, hparams_path=None):
    """
    Main fine-tuning entry point. Mirrors the style of your pretraining function.
    """
    # 1) Load hyperparams
    hparams = load_hparams(cfg, hparams_path)
    logger.info(f"initial hparams:\n{hparams}")
    if batch_size is not None:
        hparams["batch_size"] = batch_size

    if "batch_size" in cfg:
        hparams['batch_size'] = cfg['batch_size']

    # 2) Create dataloaders
    logger.info("loading dataloaders")
    train_loader, val_loader = get_dataloaders(cfg, hparams)

    logger.info("dataloaders loaded")
    # 3) Setup logging
    ckpt_dir = cfg["FINETUNE_CHECKPOINT_DIR"]
    csv_logger = CSVLogger(save_dir=ckpt_dir)
    callbacks = get_callbacks(ckpt_dir)

    # 4) DDP strategy
    strategy = get_ddp_strategy()

    hparams_encoder = read_hparams(hparams['pretrained_model_hparams'])
    hparams.update(
        {k:v for k, v in hparams_encoder.items() if k not in hparams}
    )
    logger.debug(f"final hparams:\n{hparams}") 
    # 5) The model (excluded in this snippet).
    logger.debug("loading model")
    model = Transformer_CLS(hparams)
    logger.debug("model loaded")
    # 6) Build the Trainer

    trainer = Trainer(
        strategy=strategy,
        default_root_dir=ckpt_dir,
        callbacks=callbacks,
        max_epochs=hparams["epochs"],
        val_check_interval=hparams.get("val_check_interval", 1.0),
        accelerator=ACCELERATOR,
        devices=int(N_DEVICES),
        logger=csv_logger,
        precision=PRECISION,
        log_every_n_steps=50,
    )

    logger.info("Starting Trainer.fit(...) for fine-tuning.")
    trainer.fit(model, train_loader, val_loader)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default="gpu",
                        help="Choose an accelerator (cpu, gpu, tpu, etc.)")
    parser.add_argument("--ddpstrategy", default="auto",
                        help="DDP strategy (auto,gloo,mpi,...)")
    parser.add_argument("--devices", default=1,
                        help="Number of devices")
    parser.add_argument("--batch", default=None, type=int,
                        help="Batch size override")
    parser.add_argument("--hparams", default=None, type=str,
                        help="Path to hyperparameters file.")
    parser.add_argument("--config", required=True, type=str,
                        help=".json config")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ACCELERATOR = args.accelerator
    N_DEVICES = args.devices
    DDP_STRATEGY = args.ddpstrategy
    BATCH_SIZE = args.batch
    HPARAMS_PATH = args.hparams
    CFG_PATH = args.config

    torch.set_float32_matmul_precision("medium")

    cfg = read_json(CFG_PATH)
    fine_tune(cfg, batch_size=BATCH_SIZE, hparams_path=HPARAMS_PATH)
