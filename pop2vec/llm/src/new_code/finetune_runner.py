"""
finetune_runner.py  - -  Multi‑GPU ready
======================================
Launch fine‑tuning runs (one per target column) and log results
to exactly the same CSV schema as 'train_simple.py'.

New keys understood in the JSON config (all optional):
    devices                : int   (default 1)
    accelerator            : str   (default "gpu")
    ddpstrategy            : str   (default "auto" -> same as Lightning)
    accumulate_grad_batches: int   (default 1)
    gradient_clip_val      : float (default 1)
    training_precision     : str   (default "32-true")
    val_check_interval     : float (default 1.0)
"""
from __future__ import annotations
import json, os, sys, logging, csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# ——— import the existing fine‑tune helpers ————————————————
import pop2vec.llm.src.new_code.finetune_new as finetune
from pop2vec.llm.src.new_code.utils import read_hparams
from pop2vec.llm.src.new_code.load_data import FineTuneLazyDataset
from pop2vec.llm.src.new_code.finetune_model import TransformerFT
from pop2vec.evaluation.prediction_settings.train_simple import _en_weights
from pytorch_lightning.strategies import DDPStrategy

import copy
import torch.distributed as distributed
import gc

import os


# ─────────────────── Config helpers ───────────────────────────────────
DEFAULTS = {
    "EARLY_STOP_PATIENCE": 3,
    "MAX_EPOCHS": 2000,
    "LR": 1e-6,
    "BATCH_SIZE": 32,
    "balance_dataset": False,
    "test_only": False,
    #  NEW multi‑GPU / trainer defaults 
    "devices": 4,
    "freeze_positions": False,
    "pooled": True,
    "class-balanced-loss": False,
    "accelerator": "gpu",
    "ddpstrategy": "auto",
    "accumulate_grad_batches": 1,
    "gradient_clip_val": 1,
    "training_precision": "bf16",
    "val_check_interval": 1.0,
    "weight_decay_enc": 1e-2,
    "layer_lr_decay": 0.95,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-6,
    "optimizer_type": "adamw",
    "lr_scheduler": "onecycle",
    "binary_threshold": 0.5,    
}

ALWAYS_REQUIRED = [
    "sequence_encoded",
    "result_dir",
    "task_file",
    "model_name",       
    "target_column"
]
REQUIRED_TRAIN = ["train_path", "val_path", "model_save_dir", "pretrained_model_path", "pretrained_model_hparams",]

REQUIRED_TEST =  ["ids_path", "checkpoint_path",]

def _with_defaults(cfg: Dict) -> Dict:
    out = cfg.copy()
    for k, v in DEFAULTS.items():
        out.setdefault(k, v)
    out["result_path"] = str(Path(out["result_dir"], f"{out['task_file']}_{out['target_column']}.csv"))
    return out

def _integrity_check(cfg: Dict):
    missing = [k for k in ALWAYS_REQUIRED if k not in cfg]
    if cfg.get("test_only"):
        missing += [k for k in REQUIRED_TEST if k not in cfg]
    else:
        missing += [k for k in REQUIRED_TRAIN if k not in cfg]
    
    if missing:
        raise ValueError("Missing required keys: " + ", ".join(missing))

def _fmt(metrics: Dict, key: str):
    v = metrics.get(key, None)
    if v is None:
        return ""
    if isinstance(v, torch.Tensor):
        v = v.item()
    return f"{v:.4f}"

def _write_row(result_path: str, header: List[str], row: List):
    need_hdr = not os.path.exists(result_path) or os.path.getsize(result_path) == 0
    with open(result_path, "a", newline="") as f:   
        w = csv.writer(f)
        if need_hdr:
            w.writerow(header)
        w.writerow(row)

# ─────────────────── Lightning helpers ────────────────────────────────
def _build_hparams(cfg: Dict, target_col: str,
                   target_type: str, num_outputs: int) -> Dict:
    ft_hp = {
        "finetune_checkpoint_dir": str(Path(cfg["model_save_dir"]) / target_col),
        # "sequence_encoded": cfg["sequence_encoded"],
        "train_label_file": cfg["train_path"],
        "val_label_file":   cfg["val_path"],
        # "pretrained_model_path":     cfg["pretrained_model_path"],
        # "pretrained_model_hparams":  cfg["pretrained_model_hparams"],
        "batch_size":  cfg["BATCH_SIZE"],
        "epochs":      cfg["MAX_EPOCHS"],
        "learning_rate": cfg["LR"],
        "num_targets": num_outputs,
        "target_col":  target_col,
        "oversample":  cfg["balance_dataset"],
        "val_split":   0.0,   # external val file
        # propagate trainer‑level knobs so they survive enc_hp merge
        # "accumulate_grad_batches": cfg["accumulate_grad_batches"],
        # "gradient_clip_val":       cfg["gradient_clip_val"],
        # "training_precision":      cfg["training_precision"],
        # "val_check_interval":      cfg["val_check_interval"],
        "task_type": target_type,
    }
    for k, v in cfg.items():
        ft_hp[k] = v
    enc_hp = read_hparams(cfg["pretrained_model_hparams"])
    enc_hp.update(ft_hp)
    return enc_hp

def _monitor_and_mode(task_type: str) -> Tuple[str, str]:
    if task_type == "numeric":
        return "val_r2_epoch", "max"
    return "val_auc_epoch", "max"      # binary + categorical

def _get_ddp_strategy(name: str):
    if name == "auto":
        return "auto"
    if name == "ddp":
        return DDPStrategy()
    if name == "ddp_mpi":
        return DDPStrategy(process_group_backend="mpi")
    if name == "gloo":
        return DDPStrategy(process_group_backend="gloo")
    raise ValueError(f"Unknown ddpstrategy '{name}'")

def transform_categorical_labels(loader, num_outputs):
    labels_tensor = loader.dataset.labels_tensor
    if bool(1 <= torch.min(labels_tensor)) and bool(torch.max(labels_tensor) <= num_outputs):
        labels_tensor -= 1
        loader.dataset.labels_tensor = labels_tensor
        return loader
    else:
        raise ValueError(
            f"labels must be between 1 and num_ouputs = {num_outputs}, found min = {np.min(_labels)} max = {np.max(_labels)}"
        )

def _train_one_target(cfg, target_col, target_type, num_outputs):
    hp = _build_hparams(cfg, target_col, target_type, num_outputs)
    monitor, mode = _monitor_and_mode(target_type)

    os.makedirs(hp["finetune_checkpoint_dir"], exist_ok=True)

    # — dataloaders —
    train_loader, val_loader = finetune.get_dataloaders(hp)
    if target_type == 'categorical':
        train_loader = transform_categorical_labels(train_loader, num_outputs)
        val_loader = transform_categorical_labels(val_loader, num_outputs)
    if cfg['class-balanced-loss']:
        _labels = train_loader.dataset._labels
        counts = np.bincount(
            np.array(_labels, dtype='long'), 
            minlength=num_outputs
        ).tolist()
        weights  = _en_weights(counts)
        logging.info(f"counts = {counts}")
        logging.info(f"weights = {weights}")
        hp['loss_weights'] = weights

    if target_type == 'numeric':
        hp['sigma'] = train_loader.dataset.labels_tensor.std(unbiased=False)
        hp['mu'] = train_loader.dataset.labels_tensor.mean()


    # keep lr‑scheduler logic in FT model happy
    acc_grad = hp["accumulate_grad_batches"]
    hp["steps_per_epoch"] = int(len(train_loader) /
                                (cfg["devices"] * acc_grad)) + 2

    model = TransformerFT(hp)

    # — callbacks —
    ckpt_cb = ModelCheckpoint(
        dirpath=hp["finetune_checkpoint_dir"],
        filename=f"finetune-{{epoch:02d}}-{{step}}-{{{monitor}:.2f}}",
        monitor=monitor, mode=mode, save_top_k=1, verbose=True,
    )
    early_cb = EarlyStopping(
        monitor=monitor, mode=mode,
        patience=cfg["EARLY_STOP_PATIENCE"], verbose=True, min_delta=0.001
    )
    logger = CSVLogger(save_dir=hp["finetune_checkpoint_dir"])

    strategy = _get_ddp_strategy(cfg["ddpstrategy"])

    trainer = Trainer(
        strategy=strategy,
        default_root_dir=hp["finetune_checkpoint_dir"],
        accelerator=cfg["accelerator"],
        devices=cfg["devices"],
        max_epochs=hp["epochs"],
        callbacks=[early_cb, ckpt_cb],
        logger=logger,
        precision=hp["training_precision"],
        log_every_n_steps=250,
        gradient_clip_val=hp["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=hp["accumulate_grad_batches"],
        val_check_interval=hp["val_check_interval"],
    )

    trainer.fit(model, train_loader, val_loader)

    best_metrics = trainer.validate(
        model=model, dataloaders=val_loader,
        ckpt_path="best", verbose=False,
    )[0]
    
    trainer.strategy.teardown()
    if distributed.is_initialized():
        distributed.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_metrics, ckpt_cb.best_model_path

def _run_test(c: Dict, tgt: str, ttype: str, k_out: int):
    # hp = _build_hparams(c, tgt, ttype, k_out)

    ds = FineTuneLazyDataset(
        h5_file_path=c['sequence_encoded'],
        train_file_path=c['ids_path'],          # only RINPERSOON column required
        phase="test",                           # activates prediction mode
        return_sequence_id=True,                # optional
    )
    nw = len(os.sched_getaffinity(0))-1
    logging.info(f"num_workers = {nw}")
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=c["BATCH_SIZE"],
        shuffle=False,
        num_workers=nw
    )

    model = TransformerFT.load_from_checkpoint(
        c["checkpoint_path"], task_type=ttype, pretrained_model_path='RANDOM'
    )

    strategy = _get_ddp_strategy(c["ddpstrategy"])

    trainer = Trainer(
        strategy=strategy,
        accelerator=c["accelerator"],
        devices=c["devices"],
    )
    outputs = trainer.predict(model, dl, return_predictions=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    local_preds = torch.cat([out["preds"]   for out in outputs]).to(device)
    local_ids   = torch.cat([out["RINPERSOON"] for out in outputs]).to(device)
    
    preds = (trainer.strategy.all_gather(local_preds)).cpu().numpy()
    ids = (trainer.strategy.all_gather(local_ids)).cpu().numpy()

    preds = preds.reshape(-1, *preds.shape[2:])
    ids = ids.reshape(-1, *ids.shape[2:])

    if not trainer.is_global_zero:
        return 
    
    if ttype == "binary":
        probs = (torch.softmax(torch.tensor(preds), dim=1)[:, 1]).numpy()   # P(class=1)
        lbls = (probs >= c["binary_threshold"]).astype(int)
        arr  = np.c_[ids, probs, lbls]
        hdr  = "RINPERSOON,probability,prediction"
    else:
        if ttype == "categorical":
            preds = preds.argmax(1).numpy() + 1   # 1-based
        else:
            preds = preds.numpy()
        arr = np.c_[ids, preds]
        hdr = "RINPERSOON,prediction"

    out = Path(c["result_dir"],
               f"{c['task_file']}_{tgt}.csv")
    np.savetxt(out, arr, delimiter=",",
               header=hdr, comments="", fmt="%s")


def should_work(path):
  if not Path(path).is_file():
    return True              
  with open(path) as f:
    return sum(1 for _ in f) < 2 


# ─────────────────── main loop ────────────────────────────────────────
def main(cfg_path: str):
    cfg = _with_defaults(json.load(open(cfg_path)))
    _integrity_check(cfg)
    if not should_work(cfg['result_path']):
        logging.info("work was already done. Exiting without doing anything.")
        return
    header = [
        "mode", "task_file", "model_name", "target", "type", "model_path",
        "val_acc", "val_f1", "val_auc", "val_mcc", "val_mae", "val_r2",
        "test_acc", "test_f1", "test_auc", "test_mcc", "test_mae", "test_r2",
        "LR", "BATCH-SIZE"
    ]
    os.makedirs(cfg["result_dir"], exist_ok=True)

    for tgt_col, (tgt_type, k_out) in cfg["target_column"].items():
        logging.info(f"=== Fine‑tune '{tgt_col}' ({tgt_type}) ===")
        cfg_copy = copy.deepcopy(cfg)
        if cfg_copy["test_only"]:
            _run_test(cfg_copy, tgt_col, tgt_type, k_out)
            continue
        val_metrics, model_path = _train_one_target(
            cfg_copy, tgt_col, tgt_type, k_out
        )

        row = [
            "train", cfg_copy["task_file"], cfg_copy["model_name"], tgt_col, tgt_type,
            model_path,
            _fmt(val_metrics, "val_acc_epoch"),
            _fmt(val_metrics, "val_f1_epoch"),
            _fmt(val_metrics, "val_auc_epoch"),
            _fmt(val_metrics, "val_mcc_epoch"),
            _fmt(val_metrics, "val_mae_epoch"),
            _fmt(val_metrics, "val_r2_epoch"),
            "", "", "", "", "", "",
            cfg_copy["LR"], cfg_copy["BATCH_SIZE"],
        ]
        if os.getenv("SLURM_PROCID", "0") == "0":
            _write_row(cfg_copy["result_path"], header, row)
        logging.info("RESULT_ROW  " + ", ".join(map(str, row)))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    if len(sys.argv) != 2:
        logging.error("Usage: python -m finetune_runner CONFIG.json")
        sys.exit(1)
    main(sys.argv[1])
