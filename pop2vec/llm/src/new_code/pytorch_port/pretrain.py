#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone entry-point that replaces the Lightning Trainer.

Key features
------------
* DDP support via :pymod:`torch.distributed`
* Step- and epoch-level CSV logging, averaged across GPUs,
  emitted only on rank 0
* ModelCheckpoint (``top_k=2``) and optional EarlyStopping
* Epoch-wise reseeding (`seed + epoch`) identical to Lightning
* Hyper-parameters stored inside every checkpoint
* `torch.set_float32_matmul_precision("medium")` retained
* ≤ 80-character lines, Google Python Style
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from pop2vec.llm.src.new_code.load_data import CustomInMemoryDataset
from pop2vec.llm.src.new_code.utils import read_hparams
from pop2vec.llm.src.new_code.pytorch_port.transformer_encoder import TransformerEncoder

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
_LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def _get_vocab_size(csv_path: str) -> int:
    """Returns number of rows in a token vocabulary CSV file."""
    import pandas as pd

    return len(pd.read_csv(csv_path))


def _load_hparams(cfg: dict, override: Optional[str]) -> dict:
    """Loads, augments, and returns hyper-parameters."""
    path = override or cfg["HPARAMS_PATH"]
    hp = read_hparams(path)
    hp["vocab_size"] = _get_vocab_size(cfg["VOCAB_PATH"])
    hp.update(cfg)
    return hp


def _make_dataloaders(
    mlm_path: str,
    n_val: int,
    batch_size: int,
    ddp: bool,
) -> Tuple[DataLoader, DataLoader]:
    """Creates train/val dataloaders with optional distributed samplers."""
    val_ds = CustomInMemoryDataset(
        mlm_path, validation=True, num_val_items=n_val
    )
    train_ds = CustomInMemoryDataset(
        mlm_path, validation=False, num_val_items=n_val
    )
    num_workers = max(len(os.sched_getaffinity(0)) - 2, 1)

    if ddp:
        rank, world = dist.get_rank(), dist.get_world_size()
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world, rank=rank, shuffle=False
        )
    else:
        train_sampler = val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def _sync_average(tensor: torch.Tensor, world: int) -> torch.Tensor:
    """All-reduces and averages *tensor* across ranks."""
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world
    return tensor

# --- checked till here


# --------------------------------------------------------------------------- #
# Checkpoint & Early-stopping helpers
# --------------------------------------------------------------------------- #
class Checkpointer:
    """Simple implementation of Lightning's ``ModelCheckpoint``."""

    def __init__(
        self, directory: Path, mode: str = 'min', save_last: bool = False, top_k: int = 2
    ) -> None:
        self._mode = mode
        if self._mode not in ['min', 'max']:
            raise ValueError(
                f"Unsupported mode: {mode!r}. Expected 'min' or 'max'."
            )
    
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._save_last = save_last
        self._top_k = top_k
        self._buffer: list[Tuple[float, Path]] = []
    

    def _name(self, epoch: int, step: int, loss: float) -> Path:
        return self._dir / f"model-{epoch:02d}-{step}-{loss:.2f}.ckpt"

    def maybe_save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        step: int,
        loss: float,
        hparams: dict,
    ) -> None:
        path = self._name(epoch, step, loss)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "step": step,
                "val_loss_track": loss,
                "hparams": hparams,
            },
            path,
        )

        if self._save_last:
            shutil.copy(path, self._dir / "last.ckpt")

        self._buffer.append((loss, path))
        if self._mode == 'min':
            # smaller is better → ascending sort
            self._buffer.sort(key=lambda x: x[0])
        elif self._mode == 'max':
            # larger is better → descending sort
            self._buffer.sort(key=lambda x: x[0], reverse=True)
        while len(self._buffer) > self._top_k:
            _, loser = self._buffer.pop()
            loser.unlink(missing_ok=True)
        _LOGGER.info("Checkpoint saved to %s", path)


class EarlyStopper:
    """Faithful re-implementation of Lightning's EarlyStopping callback."""

    def __init__(self, patience: int, min_delta: float, mode: str = 'min') -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._mode = mode
        self._best: Optional[float] = None
        self._bad_epochs = 0
        self.should_stop = False

    def step(self, metric: float) -> None:
        if (
            self._best is None or 
            (self._mode == 'min' and metric < self._best - self._min_delta) or 
            (self._mode == 'max' and metric > self._best + self._min_delta)
        ):
            self._best = metric
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
            if self._bad_epochs >= self._patience:
                self.should_stop = True


# --------------------------------------------------------------------------- #
# Train / validation epoch runners
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Validation helper  (NEW)
# --------------------------------------------------------------------------- #
def _validation_epoch(
    model: TransformerEncoder,
    loader: DataLoader,
    device: torch.device,
    world: int,
    epoch: int,
    csv_step_path: Path,
    global_step: int,
) -> Tuple[float, float, float]:
    """Runs one full validation pass and logs to the step-level CSV."""
    model.eval()
    tot = torch.zeros(3, device=device)  # combined, mlm, cls
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.no_grad():
            loss, mlm, cls = model.validation_step(batch)
        tot += torch.tensor([loss.item(), mlm.item(), cls.item()],
                            device=device)

    tot = _sync_average(tot / len(loader), world)
    is_main = True if world == 1 else dist.get_rank() == 0
    if is_main:
        with csv_step_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},val,{global_step},{tot[0]:.6f},"
                    f"{tot[1]:.6f},{tot[2]:.6f},0.0\n")
    model.on_validation_epoch_end()
    return tot.tolist()  # combined, mlm, cls

# --------------------------------------------------------------------------- #
# Train epoch runner (UPDATED)
# --------------------------------------------------------------------------- #
def _train_epoch(
    model: TransformerEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    log_every: int,
    world: int,
    epoch: int,
    csv_step_path: Path,
    global_step: int,
    val_loader: Optional[DataLoader],
    val_every: int,
    ckptr: Checkpointer,
    stopper: Optional[EarlyStopper],
    hparams: dict,
) -> Tuple[float, float, float, int]:
    """Runs one training epoch with optional mid-epoch validation."""
    model.train()
    tot = torch.zeros(3, device=device)
    is_main = True if world == 1 else dist.get_rank() == 0
    iterator = (
        tqdm(loader, desc=f"train-e{epoch}", leave=False)
        if is_main else loader
    )

    for b_idx, batch in enumerate(iterator, start=1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        loss, mlm, cls = model.training_step(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.global_step += 1
        global_step = model.global_step

        tot += torch.tensor([loss.item(), mlm.item(), cls.item()],
                            device=device)

        # step-level train logging
        if b_idx % log_every == 0 and is_main:
            with csv_step_path.open("a", encoding="utf-8") as f:
                f.write(f"{epoch},train,{global_step},{loss.item():.6f},"
                        f"{mlm.item():.6f},{cls.item():.6f},"
                        f"{scheduler.get_last_lr()[0]:.6e}\n")

        # mid-epoch validation
        if val_loader is not None and val_every and b_idx % val_every == 0:
            v_c, v_m, v_cls = _validation_epoch(
                model, val_loader, device, world, epoch,
                csv_step_path, global_step,
            )
            if is_main:
                ckptr.maybe_save(model, optimizer, scheduler,
                                 epoch, global_step, v_c, hparams)
                if stopper:
                    stopper.step(v_c)
                    if stopper.should_stop:
                        break

    model.on_train_epoch_end()
    tot = _sync_average(tot / len(loader), world)
    return tot[0].item(), tot[1].item(), tot[2].item(), global_step


# --------------------------------------------------------------------- #
# Training orchestration
# --------------------------------------------------------------------- #
def _init_distributed(strategy: str) -> Tuple[bool, int, int]:
    """Initialises torch.distributed if requested."""
    ddp = strategy not in ("auto",)
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        world, rank = dist.get_world_size(), dist.get_rank()
    else:
        world, rank = 1, 0
    return ddp, world, rank

def _write_csv_headers(rank: int, csv_epoch: Path, csv_step: Path) -> None:
    if rank != 0:
        return
    if not csv_epoch.exists():
        csv_epoch.write_text(
            "epoch,train_loss,val_loss,train_mlm,val_mlm,"
            "train_cls,val_cls\n"
        )
    if not csv_step.exists():
        csv_step.write_text(
            "epoch,stage,global_step,loss,mlm_loss,cls_loss,lr\n"
        )


def _resume(
    cfg: dict,
    model: TransformerEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> Tuple[int, int]:
    """Loads a checkpoint if `RESUME_FROM_CHECKPOINT` is set."""
    ckpt_path = cfg.get("RESUME_FROM_CHECKPOINT")
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        model.global_step = ckpt["step"]
        _LOGGER.info("Resumed from %s", ckpt_path)
        return ckpt["epoch"] + 1, ckpt["step"]
    return 0, 0


# --------------------------------------------------------------------------- #
# Training orchestration (UPDATED)
# --------------------------------------------------------------------------- #
def _fit(
    config: str,
    hparams_path: Optional[str],
    ddpstrategy: str,
    accelerator: str,
    devices: int,
    batch: Optional[int],
    log_every: int,
    val_check_interval: Optional[float],
    save_last: bool,
    early_stop: bool,
    early_patience: int,
    early_min_delta: float,
) -> None:
    """High-level training routine."""
    torch.set_float32_matmul_precision("medium")

    cfg = json.load(open(config, encoding="utf-8"))
    hparams = _load_hparams(cfg, hparams_path) # Renamed to avoid conflict with hparams parameter

    ddp, world, rank = _init_distributed(ddpstrategy)
    device = torch.device("cuda" if accelerator == "gpu" else "cpu")

    batch_size = batch or hparams["batch_size"]
    train_dl, val_dl = _make_dataloaders(
        cfg["MLM_PATH"], cfg.get("NUM_VAL_ITEMS", 100_000),
        batch_size, ddp
    )
    hparams["steps_per_epoch"] = len(train_dl)

    # derive mid-epoch validation frequency
    if val_check_interval is None:
        val_every = 0
    elif val_check_interval < 1:
        val_every = max(1, int(val_check_interval * len(train_dl)))
    else:
        val_every = int(val_check_interval)

    model = TransformerEncoder(hparams).to(device)
    optimizer, scheduler = model.configure_optimizers()

    ckpt_dir = Path(cfg["CHECKPOINT_DIR"])
    csv_epoch = ckpt_dir / "metrics_epoch.csv"
    csv_step = ckpt_dir / "metrics_step.csv"
    _write_csv_headers(rank, csv_epoch, csv_step)

    ckptr = Checkpointer(ckpt_dir, save_last=save_last)
    stopper = (
        EarlyStopper(early_patience, early_min_delta)
        if early_stop else None
    )

    start_epoch, global_step = _resume(cfg, model, optimizer, scheduler)

    for epoch in range(start_epoch, hparams["epochs"]):
        # model.train_epoch_start(epoch)

        tr_loss, tr_mlm, tr_cls, global_step = _train_epoch(
            model, train_dl, optimizer, scheduler, device,
            log_every=log_every, world=world, epoch=epoch,
            csv_step_path=csv_step, global_step=global_step,
            val_loader=val_dl, val_every=val_every,
            ckptr=ckptr, stopper=stopper, hparams=hparams,
        )

        # final validation for the epoch (guaranteed)
        val_loss, val_mlm, val_cls = _validation_epoch(
            model, val_dl, device, world, epoch,
            csv_step, global_step,
        )

        if rank == 0:
            with csv_epoch.open("a", encoding="utf-8") as f:
                f.write(f"{epoch},{tr_loss:.6f},{val_loss:.6f},"
                        f"{tr_mlm:.6f},{val_mlm:.6f},"
                        f"{tr_cls:.6f},{val_cls:.6f}\n")
            ckptr.maybe_save(model, optimizer, scheduler, epoch,
                             global_step, val_loss, hparams)
            _LOGGER.info(
                "Epoch %d | train %.4f | val %.4f (MLM %.4f, CLS %.4f)",
                epoch, tr_loss, val_loss, val_mlm, val_cls
            )
            if stopper:
                stopper.step(val_loss)

        if stopper and stopper.should_stop:
            if rank == 0:
                _LOGGER.info("Early stopping triggered.")
            break

    if ddp:
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    """Parses command-line flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--ddpstrategy", default="auto")
    parser.add_argument("--devices", type=int, default=1) # TODO unused.
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--hparams_path", type=str, default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log_every", type=int, default=1_000)

    # mid-epoch validation frequency
    parser.add_argument(
        "--val_check_interval",
        type=float,
        help=("If <1 → fraction of an epoch; "
              "if ≥1 → validate every N train batches."),
        default=None,
    )

    parser.add_argument("--save_last", action="store_true", help="Save the most recent checkpoint to last.ckpt")

    # Early-stopping flags (kept unchanged)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--early_min_delta", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    _fit(**vars(_parse_args()))
