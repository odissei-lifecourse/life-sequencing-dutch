#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    python -m predict_and_eval predict_settings.cfg

Trains one or more models (numeric, binary, or categorical) using 5-fold CV,
and saves a single model plus a row of results in CSV.

- Uses MSE as the training loss for numeric tasks.
- Reports average MSE, MAE, and R2 for numeric tasks.
- For classification (binary/categorical), reports accuracy, F1, and now ROC-AUC (weighted, OVR).
"""

import copy
import sys
import json
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from pop2vec.evaluation.prediction_settings.simple_mlp import SimpleMLP
from pop2vec.evaluation.prediction_settings.attn_agg import AttentionMLP

PRIMARY_KEY = 'RINPERSOON'
EARLY_STOP_PATIENCE = None
MAX_EPOCHS = None
DROPOUT_RATE = None
BATCH_SIZE = None
LR = None
DRY_RUN = True
PARTNER_KEY = None


def load_config(cfg_path):
    """Loads JSON config from a file."""
    with open(cfg_path, 'r') as f:
        return json.load(f)


def sample_df(df, dry_run, n=100000):
    """Optionally subsample the dataframe if DRY_RUN is True."""
    if dry_run:
        return df.sample(n=min(len(df), n))
    return df


def load_data_and_emb_df(cfg):
    """Loads main data and embeddings into DataFrames."""
    data_path = cfg['data_path']
    emb_path = cfg['emb_path']

    if data_path.endswith('.csv'):
        data_df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        data_df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"data_path must be csv or parquet, got {data_path}")

    data_df = sample_df(data_df, DRY_RUN)
    emb_df = pd.read_parquet(emb_path)

    # Unify primary key column name
    for df in [data_df, emb_df]:
        df.rename(columns={'rinpersoon_id': PRIMARY_KEY}, inplace=True)
    return data_df, emb_df


def load_and_merge_single_data(cfg):
    """
    Loads data and embeddings, merges on PRIMARY_KEY.
    Returns: (data_df, emb_df)
    """
    data_df, emb_df = load_data_and_emb_df(cfg)
    target_cols = list(cfg['target_column'].keys())
    keep_cols = [PRIMARY_KEY] + target_cols
    data_df = data_df[keep_cols]

    # Merge only on common IDs
    common_ids = set(data_df[PRIMARY_KEY]).intersection(set(emb_df[PRIMARY_KEY]))
    data_df = data_df[data_df[PRIMARY_KEY].isin(common_ids)].reset_index(drop=True)
    emb_df = emb_df[emb_df[PRIMARY_KEY].isin(common_ids)].reset_index(drop=True)

    return data_df, emb_df


def load_and_merge_couple_data(cfg):
    """
    Loads "couple" dataset and merges each couple's embeddings into a single row.
    """
    data_df, emb_df = load_data_and_emb_df(cfg)
    target_cols = list(cfg['target_column'].keys())
    data_cols_to_keep = [PRIMARY_KEY, PARTNER_KEY] + target_cols
    data_df = data_df[data_cols_to_keep]

    # Remove duplicate pairs (A,B) and (B,A)
    pair_col = f"{PRIMARY_KEY}_{PARTNER_KEY}"
    data_df[pair_col] = data_df.apply(
        lambda row: frozenset([row[PRIMARY_KEY], row[PARTNER_KEY]]), axis=1
    )
    data_df.drop_duplicates(subset=[pair_col], inplace=True)
    data_df.drop(columns=[pair_col], inplace=True)

    # Only keep rows whose keys are in emb_df
    data_df = data_df[
        data_df[PRIMARY_KEY].isin(emb_df[PRIMARY_KEY]) &
        data_df[PARTNER_KEY].isin(emb_df[PRIMARY_KEY])
    ].reset_index(drop=True)

    # Limit emb_df to needed keys
    emb_df = emb_df[
        emb_df[PRIMARY_KEY].isin(data_df[PRIMARY_KEY]) |
        emb_df[PRIMARY_KEY].isin(data_df[PARTNER_KEY])
    ].reset_index(drop=True)

    return data_df, emb_df


def load_data(cfg):
    """
    Wrapper that decides which data loader to use (single vs couple).
    Returns: (data_df, emb_df)
    """
    if 'PARTNER_KEY' in cfg:
        return load_and_merge_couple_data(cfg)
    return load_and_merge_single_data(cfg)


def prepare_data_for_target_single_emb(data_df, emb_df, target_col):
    """
    Merges single-person embeddings with target data on PRIMARY_KEY.
    """
    df = data_df[[PRIMARY_KEY, target_col]].copy()
    # Avoid accidental column name collisions
    if target_col in emb_df.columns:
        emb_df = emb_df.rename(columns={target_col: f'input_{target_col}'})
    df = df.merge(emb_df, on=PRIMARY_KEY, how='inner').dropna(subset=[target_col])
    return df.reset_index(drop=True)


def prepare_data_for_target_double_emb(data_df, emb_df, target_col):
    """
    Merges couple embeddings: each row will have A's embeddings + B's embeddings.
    """
    df = data_df[[PRIMARY_KEY, PARTNER_KEY, target_col]].copy()
    if target_col in emb_df.columns:
        emb_df = emb_df.rename(columns={target_col: f'input_{target_col}'})

    df = df.merge(emb_df, on=PRIMARY_KEY, how='inner')
    df = df.merge(
        emb_df, left_on=PARTNER_KEY, right_on=PRIMARY_KEY, how='inner',
        suffixes=[None, "_partner"]
    )
    df.drop(columns=[PRIMARY_KEY+"_partner"], inplace=True)
    df.dropna(subset=[target_col], inplace=True)
    return df.reset_index(drop=True)


def prepare_data_for_target(data_df, emb_df, target_col, cfg):
    """
    Returns a merged DataFrame for one target. Handles single vs couple logic.
    """
    if 'PARTNER_KEY' in cfg:
        return prepare_data_for_target_double_emb(data_df, emb_df, target_col)
    return prepare_data_for_target_single_emb(data_df, emb_df, target_col)

def balance_dataset(X, y, unique_vals):
    idx0, idx1 = np.where(y==unique_vals[0])[0], np.where(y==unique_vals[1])[0]
    keep = min(len(idx0), len(idx1))
    idx = np.concatenate([
            np.random.choice(idx0, keep, replace=False), 
            np.random.choice(idx1, keep, replace=False)
    ])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def encode_target(X, y, target_type, cfg):
    """
    Encodes y for numeric/binary/categorical tasks. Also handles 'special_value' filtering.
    Returns (X, y, output_dim).
    """
    global F1_AGG
    # Remove special values if specified
    if 'special_value' in cfg:
        for v in cfg['special_value']:
            mask = (y != v)
            X, y = X[mask], y[mask]

    print("--------------- Data Statistics -------------------")
    if target_type == 'numeric':
        _print_numeric_stats(y, label="(Before Transformation)")
        if 'transformation' in cfg:
            X, y = _apply_numeric_transformation(X, y, cfg['transformation'])
        _print_numeric_stats(y, label="(After Transformation)")
        return X, y, 1

    if target_type == 'binary':
        F1_AGG = 'binary'
        if 'transformation' in cfg and cfg['transformation'] == 'MEDIAN-BOUNDARY':
            y = (y > np.median(y)).astype(int)
        unique_vals = sorted(set(y))
        if len(unique_vals) != 2:
            raise ValueError(
                f"Binary target must have 2 unique values, found {len(unique_vals)}: {unique_vals}"
            )
        if cfg.get('balance_dataset', False):
            X, y = balance_dataset(X, y, unique_vals)
        for val in unique_vals:
            print(f"Count/class {val}: {np.sum(y == val)} ({np.sum(y==val)/len(y)*100:.1f}%)")

        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        y_encoded = np.array([mapping[val] for val in y])
        return X, y_encoded, 1

    if target_type == 'categorical':
        F1_AGG = 'weighted'
        unique_vals = sorted(set(y))
        class_to_idx = {val: i for i, val in enumerate(unique_vals)}
        for val in unique_vals:
            print(f"Count/class {val}: {np.sum(y == val)} ({np.sum(y==val)/len(y)*100:.1f}%)")

        y_encoded = np.array([class_to_idx[val] for val in y])
        return X, y_encoded, len(unique_vals)

    raise ValueError("Invalid target_type. Must be 'numeric', 'binary', or 'categorical'.")


def _apply_numeric_transformation(X, y, transform_type):
    """Applies optional transformations for numeric targets."""
    if transform_type == 'LOG':
        mask = y > 1
        return X[mask], np.log(y[mask])
    if transform_type == 'STANDARDIZE':
        scaler = StandardScaler()
        return X, scaler.fit_transform(y.reshape(-1, 1)).flatten()
    if transform_type == 'MIN-MAX-SCALING':
        scaler = MinMaxScaler(feature_range=(0, 1))
        return X, scaler.fit_transform(y.reshape(-1, 1)).flatten()
    return X, y


def _print_numeric_stats(arr, label=""):
    """Utility to print min/median/max/mean/std of numeric data."""
    print(f"---- {label} ----")
    print(f"min={np.min(arr):.4f}, median={np.median(arr):.4f}, max={np.max(arr):.4f}, "
          f"mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, size={len(arr)}")


def create_model(input_dim, output_dim, cfg):
    """
    Creates and returns either a SimpleMLP or AttentionMLP, on the correct device.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = cfg.get('model_type', 'mlp')
    if model_type == 'mlp':
        model = SimpleMLP(
            input_dim, output_dim,
            num_layers=cfg['num_layers'],
            activation_fn=cfg['activation_fn'],
            dropout_rate=cfg['DROPOUT_RATE']
        )
    elif model_type == 'attn_mlp':
        model = AttentionMLP(input_dim, output_dim, cfg)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model.to(device)


def select_criterion(target_type, y):
    """Selects the appropriate loss function."""
    if target_type == 'numeric':
        return nn.MSELoss()
    elif target_type == 'categorical':
        return nn.CrossEntropyLoss()
    else:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y==0).sum()/(2*(y==1).sum())))


def run_epoch(model, data_loader, criterion, target_type, optimizer=None):
    """
    Runs one epoch. If optimizer is provided, it's training mode; otherwise validation.
    
    Returns:
       (avg_loss, metric1, metric2, roc_auc)
         For numeric: (MSE, MAE, R2, None)
         For classification: (Loss, Accuracy, F1, ROC-AUC)
    """
    is_training = (optimizer is not None)
    model.train() if is_training else model.eval()

    losses = []
    all_preds, all_trues = [], []
    all_probas = []     # <-- Store raw probabilities (for binary) or probability vectors (for multiclass)
    abs_errors = []

    # For better progress tracking in training mode
    pbar = tqdm(data_loader, desc="Training" if is_training else "Validating", unit=" batch")
    
    for xb, yb in pbar:
        if is_training:
            optimizer.zero_grad()

        preds = model(xb)
        if target_type == 'categorical':
            loss = criterion(preds, yb)
        else:
            loss = criterion(preds.squeeze(), yb.float())

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        # Collect predictions for metrics
        with torch.no_grad():
            preds_cpu = preds.detach().cpu()
            yb_cpu = yb.detach().cpu()

            if target_type == 'numeric':
                p = preds_cpu.numpy().ravel()
                t = yb_cpu.numpy().ravel()
                abs_errors.append(np.abs(p - t))
                all_preds.extend(p)
                all_trues.extend(t)

                # Show some stats in training loop
                if is_training:
                    pbar.set_postfix({
                        "BatchLoss": f"{loss.item():.4f}",
                        "PredMean": f"{p.mean():.4f}"
                    })
            else:
                # classification
                t = yb_cpu.numpy().ravel()
                all_trues.extend(t)

                if target_type == 'binary':
                    # For ROC-AUC, we need probabilities of the positive class
                    probas = torch.sigmoid(preds_cpu).numpy().ravel()
                    all_probas.extend(probas)
                    # Hard predictions for accuracy/f1
                    p = (probas > 0.5).astype(int)
                    all_preds.extend(p)     
                else:
                    # multi-class => use softmax for probabilities
                    probas = torch.softmax(preds_cpu, dim=1).numpy()
                    all_probas.extend(probas)
                    # Hard predictions
                    p = np.argmax(probas, axis=1)
                    all_preds.extend(p)

                # Show some stats in training loop
                if is_training:
                    pbar.set_postfix({"BatchLoss": f"{loss.item():.4f}"})
    pbar.close()

    return _aggregate_epoch_metrics(losses, all_preds, all_trues, all_probas, abs_errors, target_type)


def _aggregate_epoch_metrics(losses, all_preds, all_trues, all_probas, abs_errors, target_type):
    """
    Aggregates final metrics after one full epoch.
    Returns:
       (avg_loss, metric1, metric2, roc_auc)

       numeric => (MSE, MAE, R2, None)
       classification => (Loss, Acc, F1, ROC-AUC)
    """
    if target_type == 'binary':
        print(f"precision, recall, f1, support = {precision_recall_fscore_support(all_trues, all_preds, average='binary')}")
    
    avg_loss = np.mean(losses)

    if target_type == 'numeric':
        # MSE is average loss, also compute MAE, R2
        mae = np.mean(np.concatenate(abs_errors)) if abs_errors else 0.0
        r2_val = r2_score(all_trues, all_preds) if len(all_preds) > 0 else 0.0
        return avg_loss, mae, r2_val, None

    # classification
    # Accuracy & F1
    acc = accuracy_score(all_trues, all_preds)
    f1 = f1_score(all_trues, all_preds, average=F1_AGG)

    # Compute ROC-AUC only if we have at least two classes present
    # (roc_auc_score will fail if there's only one class)
    unique_labels = np.unique(all_trues)
    if len(unique_labels) < 2:
        roc_auc = float('nan')
    else:
        # For binary: y_proba is shape (n_samples,) or (n_samples,1)
        # For multiclass: y_proba is shape (n_samples, n_classes)
        try:
            if all_probas and isinstance(all_probas[0], float):
                # Binary
                roc_auc = roc_auc_score(all_trues, all_probas)
            else:
                # Multi-class
                all_probas_arr = np.array(all_probas)
                roc_auc = roc_auc_score(
                    all_trues, 
                    all_probas_arr, 
                    average='weighted', 
                    multi_class='ovr'
                )
        except ValueError:
            # If something goes wrong (e.g. single label in the entire set)
            roc_auc = float('nan')

    return avg_loss, acc, f1, roc_auc


def train_single_fold(model, dataset, train_indices, val_indices, target_type, criterion):
    """
    Clones model for this fold, trains with early stopping, returns best state & final metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_model = copy.deepcopy(model)
    optimizer = optim.Adam(fold_model.parameters(), lr=LR)

    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_indices), batch_size=BATCH_SIZE)
    val_loader = DataLoader(dataset, sampler=SubsetRandomSampler(val_indices), batch_size=BATCH_SIZE)

    best_val_m2 = float('-inf')
    no_improve_count = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        train_loss, train_m1, train_m2, train_roc = run_epoch(fold_model, train_loader, criterion, target_type, optimizer)
        val_loss, val_m1, val_m2, val_roc = run_epoch(fold_model, val_loader, criterion, target_type, optimizer=None)

        if target_type == 'numeric':
            # train_m1=MAE, train_m2=R2
            print(f"[Epoch {epoch+1}] "
                  f"Train(MSE={train_loss:.4f}, MAE={train_m1:.4f}, R2={train_m2:.4f}) | "
                  f"Val(MSE={val_loss:.4f}, MAE={val_m1:.4f}, R2={val_m2:.4f})")
        else:
            # train_m1=Acc, train_m2=F1, plus train_roc
            print(f"[Epoch {epoch+1}] "
                  f"Train(Loss={train_loss:.4f}, Acc={train_m1:.3f}, F1={train_m2:.3f}, ROC-AUC={train_roc:.3f}) | "
                  f"Val(Loss={val_loss:.4f}, Acc={val_m1:.3f}, F1={val_m2:.3f}, ROC-AUC={val_roc:.3f})")
            
        # Early stopping logic on val_loss
        if val_m2 > best_val_m2:
            best_val_m2 = val_m2
            best_state = copy.deepcopy(fold_model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            print(f"best_val_m2 (F1 or R2[linear]) = {best_val_m2}")
            break

    # Final fold metrics from best model
    fold_model.load_state_dict(best_state)
    val_x, val_y = dataset[val_indices]
    with torch.no_grad():
        preds = fold_model(val_x).squeeze().cpu()

    if target_type == 'numeric':
        preds_np = preds.numpy()
        val_y_np = val_y.cpu().numpy()
        mae = np.mean(np.abs(preds_np - val_y_np))
        mse = mean_squared_error(val_y_np, preds_np)
        r2_ = r2_score(val_y_np, preds_np)
        fold_metrics = (mae, mse, r2_, None)  # numeric => (MAE, MSE, R2, None for ROC)
    else:
        val_y_np = val_y.cpu().numpy()
        if target_type == 'binary':
            # Probability of positive class
            probas = torch.sigmoid(preds).cpu().numpy().ravel()
            preds_label = (probas > 0.5).astype(int)
            # ROC-AUC
            try:
                roc_val = roc_auc_score(val_y_np, probas)
            except ValueError:
                roc_val = float('nan')
            print(f"precision, recall, f1, support = {precision_recall_fscore_support(val_y_np, preds_label, average='binary')}")
        else:
            # Multiclass
            preds_2d = fold_model(val_x).cpu()
            probas = torch.softmax(preds_2d, dim=1).numpy()
            preds_label = np.argmax(probas, axis=1)
            unique_labels = np.unique(val_y_np)
            if len(unique_labels) < 2:
                roc_val = float('nan')
            else:
                roc_val = roc_auc_score(
                    val_y_np, 
                    probas, 
                    average='weighted', 
                    multi_class='ovr'
                )

        acc = accuracy_score(val_y_np, preds_label)
        f1 = f1_score(val_y_np, preds_label, average=F1_AGG)
        fold_metrics = (acc, f1, roc_val)
        print(f"final fold f1 = {f1}, during training best val f1 = {best_val_m2}")
    return best_state, best_val_m2, fold_metrics


def train_and_evaluate(data_df, emb_df, target_col, target_type, cfg):
    """
    Runs 5-Fold CV for one target. Returns final metrics and saved model path.
    """
    # Prepare data
    df = prepare_data_for_target(data_df, emb_df, target_col, cfg)
    cols_to_drop = [PRIMARY_KEY, target_col]
    if 'PARTNER_KEY' in cfg:
        cols_to_drop.append(PARTNER_KEY)
    X = df.drop(columns=cols_to_drop).values
    y = df[target_col].values

    # Encode target
    X, y, output_dim = encode_target(X, y, target_type, cfg)

    # Create dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_torch = torch.tensor(X, dtype=torch.float, device=device)
    y_torch = torch.tensor(
        y, 
        dtype=torch.long if target_type != 'numeric' else torch.float, 
        device=device
    )
    dataset = TensorDataset(X_torch, y_torch)

    # Build base model
    input_dim = X.shape[1]
    base_model = create_model(input_dim, output_dim, cfg)
    criterion = select_criterion(target_type, y)
    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    best_fold_state = None
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_torch)):
        print(f"=== Fold {fold_idx+1} / 5 for target '{target_col}' ({target_type}) ===")
        state, _, fold_metrics = train_single_fold(base_model, dataset, train_idx, val_idx, target_type, criterion)
        fold_results.append(fold_metrics)
        best_fold_state = state  # keep the latest best state

    # Summaries
    if target_type == 'numeric':
        # fold_metrics => (MAE, MSE, R2, None)
        avg_mae = np.mean([m[0] for m in fold_results])
        avg_mse = np.mean([m[1] for m in fold_results])
        avg_r2 = np.mean([m[2] for m in fold_results])
        avg_acc, avg_f1, avg_rocauc = None, None, None
        print(f"Completed 5-Fold CV for {target_col} - numeric. "
              f"Avg MAE={avg_mae:.4f}, Avg MSE={avg_mse:.4f}, Avg R2={avg_r2:.4f}")
    else:
        # For classification:
        # binary => (acc, f1, roc)
        # multiclass => (acc, f1, roc)
        print(f"len(fold_results) = {len(fold_results)}")
        print(f"fold f1s = {[m[1] for m in fold_results]}")
        avg_acc = np.mean([m[0] for m in fold_results])
        avg_f1 = np.mean([m[1] for m in fold_results])
        # For numeric, third is None, for class = roc
        avg_rocauc = np.mean([m[2] for m in fold_results])
        avg_mae, avg_mse, avg_r2 = None, None, None
        print(f"Completed 5-Fold CV for {target_col} - {target_type}. "
              f"Avg Acc={avg_acc:.3f}, Avg F1={avg_f1:.3f}, Avg ROC-AUC={avg_rocauc:.3f}")

    # Save final model
    base_model.load_state_dict(best_fold_state)

    os.makedirs(cfg['model_save_dir'], exist_ok=True)
    model_path = f"{cfg['model_save_dir']}/model_{target_col}_{target_type}.pt"
    torch.save(base_model.state_dict(), model_path)

    return avg_mae, avg_mse, avg_r2, avg_acc, avg_f1, avg_rocauc, model_path


def eval_and_write(cfg, data_df, emb_df):
    """
    Iterates over targets, trains models, writes results to CSV.
    """
    # Open CSV in append mode; write header if needed
    with open(cfg['result_path'], 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Updated header to include 'avg_roc_auc'
        header = [
            'target', 'type', 'num_layers', 'activation_fn',
            'model_path', 'avg_mse', 'avg_mae', 'avg_r2', 'avg_acc', 
            'avg_f1', 'avg_roc_auc'
        ]
        # You may want to check if the file is empty to write a header only once
        writer.writerow(header)

        for tgt, tgt_type in cfg['target_column'].items():
            (avg_mae, avg_mse, avg_r2,
             avg_acc, avg_f1, avg_rocauc, model_path) = train_and_evaluate(
                data_df, emb_df, tgt, tgt_type, cfg
            )

            writer.writerow([
                tgt, 
                tgt_type, 
                cfg['num_layers'], 
                cfg['activation_fn'],
                model_path, 
                avg_mse if avg_mse is not None else "",
                avg_mae if avg_mae is not None else "",
                avg_r2 if avg_r2 is not None else "",
                avg_acc if avg_acc is not None else "",
                avg_f1 if avg_f1 is not None else "",
                avg_rocauc if avg_rocauc is not None else ""
            ])


def load_global_settings(config):
    """
    Load global hyperparameters from config, set module-level constants.
    """
    global EARLY_STOP_PATIENCE, MAX_EPOCHS, DROPOUT_RATE, BATCH_SIZE, LR, DRY_RUN, PARTNER_KEY
    EARLY_STOP_PATIENCE = config.get('EARLY_STOP_PATIENCE', 3)
    MAX_EPOCHS = config.get('MAX_EPOCHS', 50)
    DROPOUT_RATE = config.get('DROPOUT_RATE', 0.1)
    BATCH_SIZE = config.get('BATCH_SIZE', 128)
    LR = config.get('LR', 1e-3)
    DRY_RUN = config.get('DRY_RUN', False)
    PARTNER_KEY = config.get('PARTNER_KEY', None)


def main():
    """Main entry point."""
    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)
    load_global_settings(cfg)

    data_df, emb_df = load_data(cfg)
    eval_and_write(cfg, data_df, emb_df)


if __name__ == '__main__':
    main()
