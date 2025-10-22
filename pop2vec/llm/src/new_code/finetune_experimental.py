# finetune.py
# ---------------------------------------------------------------------
# Fine‑tune a pretrained encoder‑only LM for downstream classification/
# regression on 1–N GPUs (single node) in a style consistent with
# pretrain.py
# ---------------------------------------------------------------------
import argparse
import logging
import os
from datetime import timedelta
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from pop2vec.llm.src.new_code.load_data import FineTuneLazyDataset
from pop2vec.llm.src.new_code.utils import read_hparams, read_json
from pop2vec.llm.src.new_code.finetune_model import TransformerFT

# ─────────────────────────────── logging ──────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────── defaults ─────────────────────────────
DEFAULT_VALS = {
    "accumulate_grad_batches": 1,
    "finetune_resume_from_checkpoint": None,
    "val_check_interval": 1.0,
    "gradient_clip_val": 1,
    "training_precision": "32-true",
    "load_model_weights_only": False,
    "val_split": 0.1,
    "task_type": "classification",
    "target_col": "target_label",
    "freeze_positions": False,
    "pooled": True,
    "loss_type": "entropy",
    "class-balanced-loss": False,
    "weight_decay_enc": 1e-2,
    "layer_lr_decay": 0.95,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-6,
    "optimizer_type": "adamw",
    "lr_scheduler": "exp",
    "oversample": "False",
    "EARLY_STOP_PATIENCE": 10,  
    "MAX_EPOCHS": 50,

}


REQ_KEYS = [
    "finetune_checkpoint_dir",
    "sequence_encoded",
    "label_file",
    "pretrained_model_hparams",
    "pretrained_model_path",
    "batch_size",
    "epochs",
    "num_targets", # must be 1 for regression tasks at the moment
    "learning_rate",
]

# ──────────────────── helper: hparam integrity / update ───────────────
def _integrity_check(hp):
    missing = [k for k in REQ_KEYS if k not in hp]
    if missing:
        raise ValueError(f"Missing required hparams: {', '.join(missing)}")
    if hp['num_targets'] != 1 and hp['task_type'] == 'regression':
        raise ValueError(f"num_targets = {hp['num_targets']} is not okay for regression tasks. It must be 1")
    if hp['task_type'] not in ['regression', 'classification']:
        raise ValueError(f"task_type = {hp['task_type']} is not supported. Must be either regression or classification")

def _with_defaults(hp):
    for k, v in DEFAULT_VALS.items():
        hp.setdefault(k, v)
    return hp


def load_hparams(path: str):
    hp = read_hparams(path)
    _integrity_check(hp)
    hp = _with_defaults(hp)
    # merge encoder hparams → fine‑tune hparams take precedence
    enc_hp = read_hparams(hp["pretrained_model_hparams"])
    enc_hp.update(hp)  # fine‑tune values override encoder values
    return enc_hp


# ───────────────────────── callbacks (ckpt) ───────────────────────────
def get_callbacks(ckpt_dir: str, hp):
    os.makedirs(ckpt_dir, exist_ok=True)
    monitor = "val_binary-f1-best_epoch"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"finetune-{{epoch:02d}}-{{step}}-{{{monitor}:.2f}}",
            monitor=monitor,
            save_top_k=2,
            save_last=False,
            mode="max",
            verbose=True,
        ),
        ModelCheckpoint(  # time‑based safety checkpoint
            dirpath=os.path.join(ckpt_dir, "time_ckpts"),
            filename="finetune-{epoch:02d}-{step}",
            train_time_interval=timedelta(hours=1),
            monitor=None,
            save_top_k=1,
            save_last=False,
            verbose=True,
        ),
    ]
    if hp['target_type'] == 'binary':
        monitor = 'val_f1-best_epoch'
    elif hp['target_type'] == 'categorical':
        monitor = 'val_auc_epoch'
    elif hp['target_type'] == 'numeric':
        monitor = 'val_r2_epoch' 
    callbacks.append(EarlyStopping(
        patience=EARLY_STOP_PATIENCE, 
        mode='max', 
        monitor=monitor
    ))
    return callbacks

# ───────────────────────── data loaders ───────────────────────────────
def get_dataloaders(hp):
    train_ds = FineTuneLazyDataset(
        h5_file_path=hp["sequence_encoded"],
        train_file_path=hp["label_file"],
        phase="train",
        task_type=hp["task_type"],
        target_col=hp["target_col"],
        val_split=hp["val_split"],
        return_sequence_id=True,
        assign_weights=hp['oversample'],
    )
    val_ds = FineTuneLazyDataset(
        h5_file_path=hp["sequence_encoded"],
        train_file_path=hp["label_file"],
        phase="validation",
        task_type=hp["task_type"],
        target_col=hp["target_col"],
        val_split=hp["val_split"],
        return_sequence_id=True,
    )

    # worker heuristic: leave 2 for validaiton, 1 core for OS / lightning
    num_train_workers = max(len(os.sched_getaffinity(0)) - 3, 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=hp["batch_size"],
        shuffle=not hp['oversample'],
        sampler=train_ds.sampler if hp['oversample'] else None,
        num_workers=num_train_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=min(num_train_workers, 2),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    logger.info(
        f"Dataloader stats - batches: train={len(train_loader)}, val={len(val_loader)}, "
        f"batch_size={hp['batch_size']}, train_workers={num_train_workers}"
    )
    return train_loader, val_loader


# ────────────────────────── ddp strategy pick ─────────────────────────
def get_ddp_strategy():
    if DDP_STRATEGY == "auto":
        return "auto"
    if DDP_STRATEGY == "ddp":
        return DDPStrategy()
    if DDP_STRATEGY == "ddp_mpi":
        return DDPStrategy(process_group_backend="mpi")
    if DDP_STRATEGY == "gloo":
        return DDPStrategy(process_group_backend="gloo")
    raise ValueError(f"Unknown DDP strategy: {DDP_STRATEGY}")


# ───────────────────────────── trainer loop ───────────────────────────
def finetune(hp):
    ckpt_dir = hp["finetune_checkpoint_dir"]
    logger.debug(f"Hyperparameters\n{hp}")

    train_loader, val_loader = get_dataloaders(hp)
    acc_grad = hp["accumulate_grad_batches"]
    hp["steps_per_epoch"] = (
        int(len(train_loader) / (N_DEVICES * acc_grad)) + 2
    )

    csv_logger = CSVLogger(save_dir=ckpt_dir)
    callbacks = get_callbacks(ckpt_dir, hp)
    strategy = get_ddp_strategy()

    # ── weight loading logic ──────────────────────────────────────────
    ckpt_path = hp.get("finetune_resume_from_checkpoint")
    load_weights_only = hp.get("load_model_weights_only")

    if ckpt_path and load_weights_only:
        model = TransformerFT(hp)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(
            ckpt["state_dict"], strict=False
        )
        logger.info(
            f"Weights loaded from {ckpt_path}. "
            f"missing={missing}, unexpected={unexpected}"
        )
        fit_ckpt_path = None
    else:
        model = TransformerFT(hp)
        fit_ckpt_path = ckpt_path  # resume full training state if provided

    trainer = Trainer(
        strategy=strategy,
        default_root_dir=ckpt_dir,
        callbacks=callbacks,
        max_epochs=hp["MAX_EPOCHS"],
        val_check_interval=hp["val_check_interval"],
        accelerator=ACCELERATOR,
        devices=N_DEVICES,
        logger=csv_logger,
        precision=hp["training_precision"],
        log_every_n_steps=250,
        gradient_clip_val=hp["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=acc_grad,
    )

    logger.info("Starting fine‑tuning ...")
    trainer.fit(model, train_loader, val_loader, ckpt_path=fit_ckpt_path)


# ──────────────────────────── CLI / entry ─────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fine‑tune TransformerFT")
    p.add_argument("--hparams", required=True, help="*.yaml|json fine‑tune hparams")
    p.add_argument("--accelerator", default="gpu", help="cpu|gpu|tpu|mps ...")
    p.add_argument("--ddpstrategy", default="auto", help="auto|ddp|ddp_mpi|gloo")
    p.add_argument("--devices", default=1, type=int, help="#GPUs (single node)")
    p.add_argument(
        "--seed", default=42, type=int, help="Seed for reproducibility"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ACCELERATOR = args.accelerator
    DDP_STRATEGY = args.ddpstrategy
    N_DEVICES = args.devices

    # seed_everything(args.seed, workers=True)
    # torch.set_float32_matmul_precision("medium")

    finetune(load_hparams(args.hparams))
