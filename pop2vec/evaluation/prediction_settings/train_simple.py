#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a 2‑layer MLP on pre‑split data ( *either* Train+Val **or** Test‑only ).

Updates over previous version
-----------------------------
1. **MCC metric** (Matthews correlation coefficient) added for all classification tasks.
2. New config option **`test_only: true`** with required `load_model_path`.
   * When `test_only` is **true**  -> model is loaded and **only** the test split is evaluated.
   * When `test_only` is **false** -> standard Train+Val pipeline; **no** test evaluation.
   * Config integrity check enforces mutual exclusivity.
3. Results CSV now contains **val_mcc / test_mcc** columns.

All other behaviour (SimpleMLP, EN‑weighting, early‑stopping, etc.) remains unchanged.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import sys
from typing import Dict, List, Tuple
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path


from pop2vec.evaluation.prediction_settings.simple_mlp import SimpleMLP

PRIMARY_KEY = "RINPERSOON"
PARTNER_KEY = None  # optional couple key, overwritten by cfg if present

# -----------------------------------------------------------------------------
# Config schema ----------------------------------------------------------------
# -----------------------------------------------------------------------------
DEFAULTS = {
    "EARLY_STOP_PATIENCE": 10,
    "MAX_EPOCHS": 2000,
    "DROPOUT_RATE": 0.00,
    "LR": 1e-6,
    "BATCH_SIZE": 32,
    "DRY_RUN": False,
    "balance_dataset": False,
    "num_layers": 2,  # fixed internally
    "test_only": False,
}

# Keys *always* required
ALWAYS_REQUIRED = [
    "emb_path",
    "target_column", 
    "model_name",
    "result_dir",
    "task_file",
]

# Additional keys depending on mode
REQUIRED_TRAIN = ["train_path", "val_path", "model_save_dir"]
REQUIRED_TEST = ["test_path", "load_model_path"]


# -----------------------------------------------------------------------------
# Config helpers ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def _with_defaults(cfg: Dict) -> Dict:
    out = cfg.copy()
    for k, v in DEFAULTS.items():
        out.setdefault(k, v)
    out['result_path'] = str(
        Path(out['result_dir'], f"{out['task_file']}.csv")
    ) 
    return out


def _integrity_check(cfg: Dict):
    missing = [k for k in ALWAYS_REQUIRED if k not in cfg]
    if cfg.get("test_only", False):
        missing += [k for k in REQUIRED_TEST if k not in cfg]
        forbidden = [k for k in REQUIRED_TRAIN if k in cfg]
        if forbidden:
            raise ValueError(
                "Config is in test‑only mode but has train/val keys: " + ", ".join(forbidden)
            )
    else:
        missing += [k for k in REQUIRED_TRAIN if k not in cfg]
        forbidden = [k for k in REQUIRED_TEST if k in cfg]
        if forbidden:
            raise ValueError(
                "Config is in train mode but has test‑only keys: " + ", ".join(forbidden)
            )
    if missing:
        raise ValueError("Missing required keys: " + ", ".join(missing))


# -----------------------------------------------------------------------------
# I/O helpers ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _read_any(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension in {path}")


def _rename_primary(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"rinpersoon_id": PRIMARY_KEY, "RINPERSOON": PRIMARY_KEY})


# -----------------------------------------------------------------------------
# Split‑loading and embedding merge -------------------------------------------
# -----------------------------------------------------------------------------

def report_stat(data, target_type, stage, num_out):
    logging.info(f"stage = {stage}, target is of type = {target_type}")
    if target_type == 'numeric':
        logging.info(f"min = {np.min(data)}, median = {np.median(data)}, max = {np.max(data)}, mean = {np.mean(data)}, std = {np.std(data)}")
    elif target_type == 'binary':
        pos_count, neg_count = (data==1).sum(), (data==0).sum()
        logging.info(f"positive = {pos_count}, negative = {neg_count}")
        logging.info(f"pos % = {pos_count/len(data)*100}, neg % = {neg_count/len(data)*100}")
    else:
        counts = np.bincount(data.values.astype(int), minlength=num_out).tolist()
        for i, count in enumerate(counts):
            logging.info(f'category {i}, total # = {count}, {count/len(data)*100}%')

def transform_categorical_labels(df, target_col, num_out):
    labels_unique_sorted = sorted(list(df[target_col].unique()))
    expected = list(range(num_out))
    if np.max(labels_unique_sorted) == num_out and np.min(labels_unique_sorted) == 1:
        df[target_col] -= 1
        logging.info("subtracting 1 from all labels")

    # if len(labels_unique_sorted) > num_out:
    #     logging.error(f"Found labels = \n {labels_unique_sorted}")
    #     raise ValueError(f"number of unique labels ({len(labels_unique_sorted)}) must be equal to or less than num_out = {num_out}.")
    # if labels_unique_sorted != expected:
    #     map_label = {}
    #     logging.info("-"*20)
    #     for i, label in enumerate(labels_unique_sorted):
    #         map_label[label] = i
    #         if i != label:
    #             logging.info(f"mapping label {label} to {i}")
        
    #     logging.info("-"*20)
    #     df[target_col] = df[target_col].apply(lambda x: map_label[x])

    return df

def _load_split(
    path: str, 
    emb_df: pd.DataFrame, 
    target_col: str, 
    couple: bool, 
    target_type: str, 
    stage: str,
    num_out: int
) -> pd.DataFrame:
    data_df = _rename_primary(_read_any(path))
    if couple and PARTNER_KEY not in data_df.columns:
        raise ValueError(f"Couple mode requires column '{PARTNER_KEY}' in data file {path}")
    logging.info(f"dtypes in data_df are {data_df.dtypes}")
    logging.info(f"Embedding dimension (individual) = {len(emb_df.columns)-1}")
    logging.info(f"initial df size {len(data_df)}")
    if couple:
        mask = (
            data_df[PRIMARY_KEY].isin(emb_df[PRIMARY_KEY]) &
            data_df[PARTNER_KEY].isin(emb_df[PRIMARY_KEY])
        )
        data_df = data_df[mask]

        df = data_df[[PRIMARY_KEY, PARTNER_KEY, target_col]].copy()
        df = df.merge(emb_df, on=PRIMARY_KEY, how="inner")
        df = df.merge(
            emb_df,
            left_on=PARTNER_KEY,
            right_on=PRIMARY_KEY,
            how="inner",
            suffixes=("", "_partner"),
        ).drop(columns=[f"{PRIMARY_KEY}_partner"])
    else:
        df = data_df[[PRIMARY_KEY, target_col]].merge(emb_df, on=PRIMARY_KEY, how="inner")

    logging.info(f"After merging with emb_df, df size {len(df)}")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    logging.info(f"After dropping na values from {target_col}, size of df = {len(df)}")
    if target_type not in ['binary', 'numeric']:
        df = transform_categorical_labels(df, target_col, num_out)
    report_stat(df[target_col], target_type, stage, num_out)
    return df

# -----------------------------------------------------------------------------
# Label helpers ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def _verify_label_set(arrays: List[np.ndarray | None],
                      num_out: int,
                      target_type: str):
    """
    * binary  (num_out == 1) -> expected labels {0,1}
    * categorical            -> expected labels {0 ... num_out‑1}
    """
    expected = {0, 1} if target_type == "binary" and num_out == 1 \
                        else set(range(num_out))

    for split_name, y in zip(["train", "val", "test"], arrays):
        if y is None:
            continue
        labels = set(np.unique(y))
        if labels != expected:
            logging.info(
                f"{split_name} labels {sorted(labels)} "
                f"do not match expected {sorted(expected)}"
            )



def _one_hot(y: np.ndarray, k: int) -> np.ndarray:
    return np.eye(k, dtype=np.float32)[y]


# -----------------------------------------------------------------------------
# Effective‑Number weighting ---------------------------------------------------
# -----------------------------------------------------------------------------

def _en_weights(counts, max_clip_ratio: float = 5.0) -> torch.Tensor:
    n = sum(counts)
    beta = (n - 1) / n
    eff = [(1 - beta ** c) for c in counts]
    raw = [(1 - beta) / e if e != 0 else 0 for e in eff]
    mean_raw = np.mean(raw)
    clipped = [min(w, mean_raw * max_clip_ratio) for w in raw]
    mean_clip = np.mean(clipped)
    normed = [w / mean_clip for w in clipped]
    return torch.tensor(normed, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Model factory ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def _make_mlp(input_dim: int, out_dim: int, dropout: float) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SimpleMLP(
        input_dim,
        out_dim,
        num_layers=2,
        activation_fn="LeakyReLU",
        dropout_rate=dropout,
    ).to(device)




# -----------------------------------------------------------------------------
# Metrics ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _best_f1_threshold_torch(p: torch.Tensor, y: torch.Tensor):
        """Return (thr*, f1*) using the exact sweep."""
        # sort descending by probability
        p_sorted, idx = torch.sort(p, descending=True)
        y_sorted = y[idx]

        tp = torch.cumsum(y_sorted == 1, 0)
        fp = torch.cumsum(y_sorted == 0, 0)
        fn = tp[-1] - tp

        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1        = 2 * precision * recall / (precision + recall + 1e-12)

        best_idx = torch.argmax(f1)
        return p_sorted[best_idx], f1[best_idx]



def _classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    if logits.ndim == 1:  # binary
        probas = 1 / (1 + np.exp(-logits))
        best_thr, best_f1 = _best_f1_threshold_torch(
            torch.from_numpy(probas), torch.from_numpy(y_true)
        )        
        preds = (probas >= float(best_thr)).astype(int)
        auc = roc_auc_score(y_true, probas)
        f1 = f1_score(y_true, preds, average="binary")
        # logging.info(f"f1 check: {best_f1}, {f1}")
        if abs(best_f1 - f1) > 1e-4:
            logging.error(f"best_f1 = {best_f1}, f1 = {f1}")
    else:  # multi‑class
        probas = softmax(logits)
        preds = np.argmax(probas, axis=1)
        logging.info(probas)
        auc = roc_auc_score(y_true, probas, average="macro", multi_class="ovr")
        f1 = f1_score(y_true, preds, average="macro")
    mcc = matthews_corrcoef(y_true, preds)
    acc = accuracy_score(y_true, preds)
    return {"acc": acc, "f1": f1, "auc": auc, "mcc": mcc}


def _regression_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "mae": np.mean(np.abs(preds - y_true)),
        "r2": r2_score(y_true, preds),
        "mse": mean_squared_error(y_true, preds),
    }


# -----------------------------------------------------------------------------
# Train + validate -------------------------------------------------------------
# -----------------------------------------------------------------------------


def softmax(logits, axis=-1, temperature=1.0):
    z = logits / temperature
    z -= np.max(z, axis=axis, keepdims=True)   # subtract max for stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def _train_target(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    target_type: str,
    num_out: int,
    cfg: Dict,
) -> Tuple[Dict, str]:
    """Train on train_df, early‑stop on val_df. Return val_metrics and model_path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _split_xy(df_: pd.DataFrame):
        drop_cols = [PRIMARY_KEY, target_col]
        if PARTNER_KEY in df_.columns:
            drop_cols.append(PARTNER_KEY)
        X_ = df_.drop(columns=drop_cols).values.astype(np.float32)
        y_ = df_[target_col].values
        logging.info(f"X shape = {X_.shape}, y shape = {y_.shape}")
        return X_, y_

    X_tr, y_tr = _split_xy(train_df)
    X_val, y_val = _split_xy(val_df)

    if target_type != "numeric":
        y_tr, y_val = y_tr.astype(int), y_val.astype(int)


    if target_type != "numeric": 
        _verify_label_set([y_tr, y_val, None], num_out, target_type) 

    
    # Torch tensors
    X_tr_t = torch.tensor(X_tr, device=device)
    X_val_t = torch.tensor(X_val, device=device)
    y_tr_t = torch.tensor(
        y_tr, device=device,
        dtype=torch.long if target_type == "categorical" else torch.float32
    )
    y_val_t = torch.tensor(
        y_val, device=device,
        dtype=torch.long if target_type == "categorical" else torch.float32
    )

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=cfg["BATCH_SIZE"], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=cfg["BATCH_SIZE"], shuffle=False
    )

    # Model & loss
    model = _make_mlp(X_tr.shape[1], num_out, cfg["DROPOUT_RATE"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["LR"])

    if target_type == "numeric":
        criterion = nn.MSELoss()
    else:
        counts = np.bincount(y_tr, minlength=num_out).tolist()

        weights  = _en_weights(counts).to(device)
        logging.info(f"counts = {counts}")
        logging.info(f"weights = {weights}")
        if target_type == "binary":          # <-- new branch
            # counts of negatives / positives
            # n_neg = (y_tr == 0).sum()
            # n_pos = (y_tr == 1).sum()
            # pos_w  = torch.tensor(n_neg / max(1, n_pos), dtype=torch.float32,
            #                       device=device)
    
            pos_w = weights[1] / weights[0] # for binary, only pos label's weight is needed
            logging.info(f"loss pos_weight = {pos_w}")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        else:  
            # categorical (multi‑class >1)
            criterion = nn.CrossEntropyLoss(weight=weights)
            logging.info(f"loss weights = {weights}")

    # Early‑stopping params
    best_state = None
    best_metric = -np.inf
    no_improve = 0
    patience = cfg["EARLY_STOP_PATIENCE"]

    def _monitor(y_true_np: np.ndarray, logits_np: np.ndarray):
        if target_type == "numeric":
            return r2_score(y_true_np, logits_np)
        elif target_type == "binary":
            # logits_np shape (N,) -> probabilities
            probas = 1 / (1 + np.exp(-logits_np))
            thr, best_f1 = _best_f1_threshold_torch(
                torch.from_numpy(probas), torch.from_numpy(y_true_np)
            )
            preds = (probas >= float(thr)).astype(int)
            f1 = f1_score(y_true_np, preds, average="binary")
            # logging.info(f"f1 check: {best_f1}, {ret}")
            if abs(best_f1 - f1) > 1e-4:
                logging.error(f"best_f1 = {best_f1}, f1 = {f1}")
            
            return f1 
        else:
            # multi‑class
            preds = np.argmax(logits_np, axis=1)
            return f1_score(y_true_np, preds, average="macro")
    
    # ─── Training loop (epoch progress bar) ─────────────────────────────────
    epoch_bar = tqdm(range(cfg["MAX_EPOCHS"]),
                     desc=f"{target_col} epochs", leave=False)

    for epoch in epoch_bar:
        model.train()
        batch_losses = []
        epoch_logits = []
        epoch_targets = []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits.squeeze() if target_type != "categorical" else logits, yb)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

            # ---- save batch outputs for epoch‑level train metric ----
            if target_type in ("numeric", "binary"):
                epoch_logits.append(logits.detach().cpu().numpy().squeeze())  # (B,)
            else:  # categorical
                epoch_logits.append(logits.detach().cpu().numpy())            # (B,K)
            epoch_targets.append(yb.detach().cpu().numpy())
                
        # ---- compute epoch training monitor from accumulated batches ----
        train_logits_np = np.concatenate(epoch_logits, axis=0)
        train_y_np = np.concatenate(epoch_targets, axis=0)
        train_metric = _monitor(train_y_np, train_logits_np)

        # ---- validation pass ----
        model.eval()
        with torch.no_grad():
            val_logits_np = model(X_val_t).cpu().numpy()
            if target_type == "numeric" or (target_type == "binary"):
                val_logits_np = val_logits_np.squeeze()
        val_metric = _monitor(y_val, val_logits_np)

        epoch_bar.set_postfix(
            loss=f"{np.mean(batch_losses):.4f}",
            train=f"{train_metric:.4f}",
            val=f"{val_metric:.4f}",
        )

        if val_metric > best_metric:
            best_metric, best_state, no_improve = val_metric, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if no_improve >= patience:
            logging.info("Early stopping.")
            break
    # ---- final val metrics ----
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_all = model(X_val_t).cpu().numpy()
        if target_type == "numeric" or (target_type == "binary"):
            logits_all = logits_all.squeeze()
    
    val_metrics = (
        _regression_metrics(y_val, logits_all)
        if target_type == "numeric"
        else _classification_metrics(y_val, logits_all)
    )

    # save model
    os.makedirs(cfg["model_save_dir"], exist_ok=True)
    mpath = os.path.join(
        cfg["model_save_dir"], 
        f"{cfg['model_name']}_{cfg['task_file']}_{target_col}.pt"
    )
    torch.save(best_state, mpath)
    return val_metrics, mpath


# -----------------------------------------------------------------------------
# Test‑only evaluation ---------------------------------------------------------
# -----------------------------------------------------------------------------

def _eval_test(
    test_df: pd.DataFrame,
    target_col: str,
    target_type: str,
    num_out: int,
    cfg: Dict,
    load_path: str,
) -> Dict:
    """Load model and evaluate on test_df only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    drop_cols = [PRIMARY_KEY, target_col]
    if PARTNER_KEY in test_df.columns:
        drop_cols.append(PARTNER_KEY)
    X = test_df.drop(columns=drop_cols).values.astype(np.float32)
    y = test_df[target_col].values

    if target_type != "numeric":
        _verify_label_set([None, None, y], num_out, target_type)    
    
    X_t = torch.tensor(X, device=device)

    # Build model and load weights
    model = _make_mlp(X.shape[1], num_out, cfg["DROPOUT_RATE"])
    state = torch.load(load_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(X_t).cpu().numpy()
        if target_type in ["numeric", "binary"]:
            logits = logits.squeeze()

    return (
        _regression_metrics(y, logits)
        if target_type == "numeric"
        else _classification_metrics(y, logits)
    )


# -----------------------------------------------------------------------------
# Results CSV ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _write_row(result_path: str, header: List[str], row: List):
    need_header = not os.path.exists(result_path) or os.path.getsize(result_path) == 0
    with open(result_path, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)


def _fmt(d: Dict, key: str):
    return "" if key not in d else f"{d[key]:.4f}"


# -----------------------------------------------------------------------------
# Main -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main(cfg_path: str):
    cfg = _with_defaults(json.load(open(cfg_path)))
    _integrity_check(cfg)

    global PARTNER_KEY, PRIMARY_KEY
    PARTNER_KEY = cfg.get("PARTNER_KEY", None)
    PRIMARY_KEY = cfg.get("PRIMARY_KEY", PRIMARY_KEY)   

    emb_df = _rename_primary(_read_any(cfg["emb_path"]))
    couple_mode = PARTNER_KEY is not None

    header = [
        "mode", "task_file", "model_name", "target", "type", "model_path",
        "val_acc", "val_f1", "val_auc", "val_mcc", "val_mae", "val_r2",
        "test_acc", "test_f1", "test_auc", "test_mcc", "test_mae", "test_r2",
        "LR", "BATCH-SIZE"
    ]

    os.makedirs(cfg["result_dir"], exist_ok=True)

    for target_col, (target_type, num_outputs) in cfg["target_column"].items():
        if cfg["test_only"]:
            logging.info(f"--- Test‑only evaluation for '{target_col}' ---")
            task_file = cfg["task_file"]
            test_df = _load_split(cfg["test_path"], emb_df, target_col, couple_mode, target_type, 'test', num_outputs)
            test_metrics = _eval_test(
                test_df, target_col, target_type, num_outputs, cfg, cfg["load_model_path"]
            )
            row = [
                "test", task_file, cfg["model_name"], target_col, target_type, cfg["load_model_path"],
                "", "", "", "", "", "",  # val *blank*
                _fmt(test_metrics, "acc"), _fmt(test_metrics, "f1"), _fmt(test_metrics, "auc"), _fmt(test_metrics, "mcc"), _fmt(test_metrics, "mae"), _fmt(test_metrics, "r2"),
                cfg['LR'], cfg['BATCH_SIZE'],
            ]
        else:
            logging.info(f"--- Train+Val for '{target_col}' ---")
            task_file = cfg["task_file"]

            train_df = _load_split(cfg["train_path"], emb_df, target_col, couple_mode, target_type, 'train', num_outputs)
            val_df = _load_split(cfg["val_path"], emb_df, target_col, couple_mode, target_type, 'val', num_outputs)
            val_metrics, model_path = _train_target(train_df, val_df, target_col, target_type, num_outputs, cfg)
            row = [
                "train", task_file, cfg["model_name"], target_col, target_type, model_path,
                _fmt(val_metrics, "acc"), _fmt(val_metrics, "f1"), _fmt(val_metrics, "auc"), _fmt(val_metrics, "mcc"), _fmt(val_metrics, "mae"), _fmt(val_metrics, "r2"),
                "", "", "", "", "", "", cfg['LR'], cfg['BATCH_SIZE'], # test *blank*
            ]
        
        _write_row(cfg["result_path"], header, row)
        logging.info(header)
        logging.info("RESULT_ROW  " + ", ".join(map(str, row)))



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S", 
        level=logging.DEBUG
    )
    if len(sys.argv) != 2:
        logging.error("Usage: python -m predict_no_cv CONFIG.json")
        sys.exit(1)
    main(sys.argv[1])
