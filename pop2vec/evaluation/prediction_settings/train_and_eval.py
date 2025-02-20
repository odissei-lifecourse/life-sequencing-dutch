#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
	python -m predict_and_eval predict_settings.cfg

Trains one or more models (numeric, binary, or categorical) using 5-fold CV,
and saves a single model plus a row of results in CSV.

This version:
- Uses MSE as the training loss for numeric tasks.
- Reports average MSE, MAE, and R2 (Avg_R2) as validation metrics for numeric tasks.
- For classification (binary/categorical), reports accuracy and F1.

Follows Google's Python style guide as much as possible.
"""

import sys
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score
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



def load_config(cfg_path):
	"""Loads JSON config from a file.

	Args:
		cfg_path: str, path to the JSON config file.

	Returns:
		dict, the loaded configuration.
	"""
	with open(cfg_path, 'r') as f:
		cfg = json.load(f)
	return cfg


def load_and_merge_data(cfg):
	"""Loads data and embeddings, merges on PRIMARY_KEY, returns merged DataFrames.

	Args:
		cfg: dict, containing keys 'data_path', 'emb_path', and 'target_column'.

	Returns:
		Tuple[pd.DataFrame, pd.DataFrame]:
			- data_df with [PRIMARY_KEY, target_columns]
			- emb_df with [PRIMARY_KEY, embedding features]
	"""
	if cfg['data_path'].endswith('.csv'):
		data_df = pd.read_csv(cfg['data_path'])
	elif cfg['data_path'].endswith('.parquet'):
		data_df = pd.read_parquet(cfg['data_path'])
	else:
		raise ValueError(f"data_path must be csv or parquet, got {cfg['data_path']}")
	emb_df = pd.read_parquet(cfg['emb_path'])
	if DRY_RUN:
		emb_df = emb_df.sample(n=min(len(emb_df), 100000))
	for df in [data_df, emb_df]:
		df.rename(columns={'rinpersoon_id': PRIMARY_KEY}, inplace=True)
	# Keep only the primary key and target columns
	target_cols = list(cfg['target_column'].keys())
	data_cols_to_keep = [PRIMARY_KEY] + target_cols
	data_df = data_df[data_cols_to_keep]

	# Merge on the intersection of primary keys
	common_ids = set(data_df[PRIMARY_KEY]).intersection(set(emb_df[PRIMARY_KEY]))
	data_df = data_df[data_df[PRIMARY_KEY].isin(common_ids)].reset_index(drop=True)
	emb_df = emb_df[emb_df[PRIMARY_KEY].isin(common_ids)].reset_index(drop=True)

	return data_df, emb_df


def prepare_data_for_target(data_df, emb_df, target_col):
	"""Combines data and embeddings for a specific target, dropping NaNs.

	Args:
		data_df: pd.DataFrame containing [PRIMARY_KEY, target_col].
		emb_df: pd.DataFrame containing [PRIMARY_KEY] and embedding features.
		target_col: str, the name of the target column.

	Returns:
		pd.DataFrame, combined DataFrame with no NaNs for the target column.
	"""
	df = data_df[[PRIMARY_KEY, target_col]] 
	if target_col in emb_df.columns:
		emb_df.rename(columns={target_col: f'input_{target_col}'}, inplace=True)

	df = df.merge(emb_df, on=PRIMARY_KEY, how='inner')
	df = df.dropna(subset=[target_col]).reset_index(drop=True)
	return df


def encode_target(X, y, target_type, cfg):
	"""Encodes targets for numeric, binary, or categorical tasks.

	For numeric, the original y is returned as is.
	For binary, we handle cases with any two distinct labels by mapping
	them to 0 and 1. For categorical, we perform label encoding on however
	many distinct values are present.

	Args:
		y: np.ndarray, the original target values.
		target_type: str, one of ['numeric', 'binary', 'categorical'].

	Returns:
		(np.ndarray, int): A tuple of (encoded_targets, output_dim).
			- encoded_targets: np.ndarray of encoded y values.
			- output_dim: int, the dimension of the output layer needed for the model.

	Raises:
		ValueError: If the target_type is not recognized or if a binary target
			does not have exactly 2 unique values.
	"""
	if 'special_value' in cfg:
		sp_values = cfg['special_value']
		for v in sp_values:
			mask = y!=v
			X, y = X[mask], y[mask]

	print("--------------- Data Statistics -------------------")
	if target_type == 'numeric':
		# Numeric targets remain unchanged (e.g., continuous values).
		print(f"min = {np.min(y)}, median = {np.median(y)}, max = {np.max(y)}, mean = {np.mean(y)}, std = {np.std(y)}")
		print(f"dataset size = {len(y)}")
		if 'transformation' in cfg:
			if cfg['transformation'] == 'LOG':
				mask = y > 1
				X, y = X[mask], y[mask]
				y = np.log(y)
			elif cfg['transformation'] == 'STANDARDIZE':
				scaler = StandardScaler()
				y = scaler.fit_transform(y.reshape(-1, 1)).flatten()
			elif cfg['transformation'] == 'MIN-MAX-SCALING':
				scaler = MinMaxScaler(feature_range=(0, 1))
				y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

			print("--------------- (After Transformation) Data Statistics -------------------")
			print(f"min = {np.min(y)}, median = {np.median(y)}, max = {np.max(y)}, mean = {np.mean(y)}, std = {np.std(y)}")  
			print(f"dataset size = {len(y)}")
		return X, y, 1

	
	if target_type == 'binary':
		# For binary, map any two distinct labels to 0 and 1.
		if 'transformation' in cfg:
			if cfg['transformation'] == 'MEDIAN-BOUNDARY':
				y = (y > np.median(y)).astype(int)
		unique_vals = sorted(set(y))
		if len(unique_vals) != 2:
			raise ValueError(
					f"Binary target must have exactly 2 unique values, "
					f"but got {len(unique_vals)} ({unique_vals})."
			)
		for i in range(2):
			print(f"Count      -- class {unique_vals[i]}: {np.sum(y==unique_vals[i])}")
			print(f"Percentage -- class {unique_vals[i]}: {np.sum(y==unique_vals[i])/len(y)*100}")
			
		mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
		y_encoded = np.array([mapping[val] for val in y])
		return X, y_encoded, 1

	if target_type == 'categorical':
		unique_vals = sorted(set(y))
		# For categorical, label encode the targets.
		class_to_idx = {val: i for i, val in enumerate(unique_vals)}
		for i in range(len(unique_vals)):
			print(f"Count      -- class {unique_vals[i]}: {np.sum(y==unique_vals[i])}")
			print(f"Percentage -- class {unique_vals[i]}: {np.sum(y==unique_vals[i])/len(y)*100}")

		y_encoded = np.array([class_to_idx[val] for val in y])
		return X, y_encoded, len(unique_vals)

	raise ValueError(
			f"Invalid target_type: '{target_type}'. Must be one of "
			"['numeric', 'binary', 'categorical']."
	)


def create_model(input_dim, output_dim, cfg):
	"""Creates a SimpleMLP model and moves it to device.

	Args:
		input_dim: int, number of input features.
		output_dim: int, dimension of the output layer.
		cfg: dict, containing 'num_layers' and 'activation_fn'.

	Returns:
		nn.Module, the initialized model on the correct device.
	"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_type = cfg.get('model_type', 'mlp')
	if model_type == 'mlp':
		model = SimpleMLP(input_dim, output_dim, cfg['num_layers'], cfg['activation_fn'], cfg['DROPOUT_RATE'])
	elif model_type == 'attn_mlp':
		model = AttentionMLP(input_dim, output_dim, cfg)
	else:
		raise ValueError(f"model_type = {model_type} is not supported!")
	model.to(device)
	return model


def select_criterion(model, target_type):
	"""Selects the appropriate loss function and initializes the optimizer.

	For numeric targets, we use MSELoss for backprop. We will track MSE, MAE,
	and R2 as metrics, but the criterion is MSE.

	Args:
		model: nn.Module, the model to optimize.
		target_type: str, one of ['numeric', 'binary', 'categorical'].

	Returns:
		tuple of (criterion, optimizer).
	"""
	if target_type == 'numeric':
		criterion = nn.MSELoss()
	elif target_type == 'categorical':
		criterion = nn.CrossEntropyLoss()
	else:  # 'binary'
		criterion = nn.BCEWithLogitsLoss()
	return criterion


def train_one_epoch(model, train_loader, criterion, optimizer, target_type):
	"""Runs one epoch of training.

	Args:
		model: nn.Module, the model to train.
		train_loader: DataLoader for the training data.
		criterion: loss function (MSE for numeric, CE/BCE for classification).
		optimizer: optimization algorithm (e.g. Adam).
		target_type: str, one of ['numeric', 'binary', 'categorical'].

	Returns:
		tuple of:
			- float, average train loss (MSE for numeric, CE/BCE for classification)
			- float or None, second metric (MAE for numeric, Accuracy for classification)
			- float or None, third metric (R2 for numeric, F1 for classification)
	"""
	model.train()
	train_losses = []

	# For numeric tasks, track MSE-based loss but also compute MAE, R2
	abs_errors = []
	preds_list, trues_list = [], []
	with tqdm(total=len(train_loader), unit=" batch") as tepoch:
		tepoch.set_description("Training")
		for batch_idx, (xb, yb) in enumerate(train_loader):
			optimizer.zero_grad()
			preds = model(xb)

			if target_type == 'categorical':
				loss = criterion(preds.squeeze(), yb)
			else:
				loss = criterion(preds.squeeze(), yb.float())

			loss.backward()
			optimizer.step()
			train_losses.append(loss.item())

			# Collect additional metrics
			if target_type == 'numeric':
				preds_np = preds.squeeze().detach().cpu().numpy()
				targets_np = yb.detach().cpu().numpy()
				preds_list.extend(preds_np if preds_np.ndim == 1 else preds_np.flatten())
				trues_list.extend(targets_np if targets_np.ndim == 1 else targets_np.flatten())
				abs_errors.append(np.abs(preds_np - targets_np))
				tepoch.set_postfix(
					{
						"Batch": batch_idx + 1,
						"Loss": f"{loss.item():.4f}",
						"Min": np.min(preds_np),
						"Median": np.median(preds_np),
						"Max": np.max(preds_np),
						"Mean": np.mean(preds_np),
					}
				)
			else:
				# Classification
				tqdm_dict = {
						"Batch": batch_idx + 1,
						"Loss": f"{loss.item():.4f}",
				}
				if target_type == 'binary':
					p = torch.sigmoid(preds).detach().cpu().numpy()
					p = (p > 0.5).astype(int).ravel()
					for i in range(2):
						tqdm_dict[f"class_{i}_%"] = np.sum(p == i) / len(p) * 100
				else:  # 'categorical'
					p = torch.argmax(preds, dim=1).detach().cpu().numpy()
					for i in range(len(preds[0])):
						tqdm_dict[f"class_{i}_%"] = np.sum(p == i) / len(p) * 100
				tepoch.set_postfix(tqdm_dict)
				preds_list.extend(p)
				trues_list.extend(yb.cpu().numpy())

			# This ensures each batch increments the progress bar,
			# enabling proper time estimation.
			tepoch.update(1)

	tepoch.close()
	train_loss_avg = np.mean(train_losses)

	if target_type == 'numeric':
		mae = np.mean(np.concatenate(abs_errors)) if abs_errors else 0.0
		# R2 for train set
		if len(preds_list) > 0:
			r2_val = r2_score(trues_list, preds_list)
		else:
			r2_val = 0.0
		return train_loss_avg, mae, r2_val

	# For classification
	acc = accuracy_score(trues_list, preds_list)
	f1 = f1_score(trues_list, preds_list, average='weighted')
	return train_loss_avg, acc, f1


def validate_one_epoch(model, val_loader, criterion, target_type):
	"""Runs one epoch of validation.

	Args:
		model: nn.Module, the model to validate.
		val_loader: DataLoader for validation data.
		criterion: loss function (MSE for numeric, CE/BCE for classification).
		target_type: str, one of ['numeric', 'binary', 'categorical'].

	Returns:
		tuple of:
			- float, average validation loss (MSE for numeric, CE/BCE for classification)
			- float or None, second metric (MAE for numeric, Accuracy for classification)
			- float or None, third metric (R2 for numeric, F1 for classification)
	"""
	model.eval()
	val_losses = []

	abs_errors = []
	preds_list, trues_list = [], []

	with torch.no_grad():
		for xb, yb in val_loader:
			preds = model(xb)
			if target_type == 'categorical':
				loss = criterion(preds.squeeze(), yb)
			else:
				loss = criterion(preds.squeeze(), yb.float())      
			
			val_losses.append(loss.item())

			# Collect metrics
			if target_type == 'numeric':
				preds_np = preds.squeeze().cpu().numpy()
				targets_np = yb.cpu().numpy()
				preds_list.extend(preds_np if preds_np.ndim == 1 else preds_np.flatten())
				trues_list.extend(targets_np if targets_np.ndim == 1 else targets_np.flatten())
				abs_errors.append(np.abs(preds_np - targets_np))
			else:
				if target_type == 'binary':
					p = torch.sigmoid(preds).detach().cpu().numpy()
					p = (p > 0.5).astype(int).ravel()
				else:  # 'categorical'
					p = torch.argmax(preds, dim=1).detach().cpu().numpy()
				preds_list.extend(p)
				trues_list.extend(yb.cpu().numpy())

	val_loss_avg = np.mean(val_losses)

	if target_type == 'numeric':
		mae = np.mean(np.concatenate(abs_errors)) if abs_errors else 0.0
		# R2 on the validation set
		r2_val = r2_score(trues_list, preds_list) if len(preds_list) > 0 else 0.0
		return val_loss_avg, mae, r2_val

	# Classification
	acc = accuracy_score(trues_list, preds_list)
	f1 = f1_score(trues_list, preds_list, average='weighted')
	return val_loss_avg, acc, f1


def train_single_fold(
		model, dataset, train_indices, val_indices, target_type, criterion
):
	"""Trains a new model instance on a single fold.

	Args:
		model: nn.Module, a template model whose state dict will be copied.
		dataset: TensorDataset for all samples.
		train_indices: indices for the training set in this fold.
		val_indices: indices for the validation set in this fold.
		target_type: str, one of ['numeric', 'binary', 'categorical'].
		criterion: loss function for the task (MSE for numeric, CE/BCE otherwise).

	Returns:
		tuple of:
			- dict, best state_dict from this fold
			- float, best validation loss (for early stopping)
			- object, final fold metric(s). For numeric: (mae, mse, r2).
				For classification: (acc, f1).
	"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Reconstruct model with the same structure
	if hasattr(model, 'lin1'):  # 2-layer net
		input_dim = model.lin1.in_features
		output_dim = model.lin2.out_features
		num_layers = model.num_layers
		activation_name = type(model.activation).__name__
	else:  # 1-layer net
		input_dim = model.net.in_features
		output_dim = model.net.out_features
		num_layers = 1
		activation_name = type(model.activation).__name__


	fold_model = copy.deepcopy(model)


	optimizer = optim.Adam(fold_model.parameters(), lr=LR)

	train_sampler = SubsetRandomSampler(train_indices)
	val_sampler = SubsetRandomSampler(val_indices)

	train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
	val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=BATCH_SIZE)

	best_val_loss = float('inf')
	no_improve_count = 0
	best_state = None

	for epoch in range(MAX_EPOCHS):
		train_loss, train_metric_1, train_metric_2 = train_one_epoch(
				fold_model, train_loader, criterion, optimizer, target_type
		)
		val_loss, val_metric_1, val_metric_2 = validate_one_epoch(
				fold_model, val_loader, criterion, target_type
		)

		if target_type == 'numeric':
			# train_metric_1 = MAE, train_metric_2 = R2
			# val_metric_1 = MAE, val_metric_2 = R2
			print(
					f'[Epoch {epoch + 1}] '
					f'Train Loss (MSE): {train_loss:.4f}, Train MAE: {train_metric_1:.4f}, Train R2: {train_metric_2:.4f} | '
					f'Val Loss (MSE): {val_loss:.4f}, Val MAE: {val_metric_1:.4f}, Val R2: {val_metric_2:.4f}'
			)
		else:
			# train_metric_1 = Acc, train_metric_2 = F1
			# val_metric_1 = Acc, val_metric_2 = F1
			print(
					f'[Epoch {epoch + 1}] '
					f'Train Loss: {train_loss:.4f}, Train Acc: {train_metric_1:.3f}, Train F1: {train_metric_2:.3f} | '
					f'Val Loss: {val_loss:.4f}, Val Acc: {val_metric_1:.3f}, Val F1: {val_metric_2:.3f}'
			)

		# Early stopping logic uses val_loss (MSE or CE/BCE).
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_state = fold_model.state_dict()
			no_improve_count = 0
		else:
			no_improve_count += 1

		if no_improve_count >= EARLY_STOP_PATIENCE:
			break

	# Compute final metrics for this fold using the best model state
	fold_model.load_state_dict(best_state)
	fold_model.eval()

	x_tensors, y_tensors = dataset[:]
	val_x = x_tensors[val_indices]
	val_y = y_tensors[val_indices].cpu().numpy()

	with torch.no_grad():
		pred_vals = fold_model(val_x).squeeze().cpu().numpy()

	if target_type == 'numeric':
		# For numeric: final fold metric = (MAE, MSE, R2)
		fold_mae = np.mean(np.abs(pred_vals - val_y))
		fold_mse = mean_squared_error(val_y, pred_vals)
		fold_r2 = r2_score(val_y, pred_vals)
		fold_metric = (fold_mae, fold_mse, fold_r2)
	else:
		if target_type == 'binary':
			pred_bools = (torch.sigmoid(torch.from_numpy(pred_vals)).numpy() > 0.5).astype(int)
		else:  # 'categorical'
			# If pred_vals isn't an integer array yet, we need to do argmax carefully
			# Because we called squeeze(), shape might be smaller
			# We'll replicate the same logic as in the loop
			logits_tensor = torch.from_numpy(pred_vals)
			pred_bools = torch.argmax(logits_tensor, dim=1).numpy()

		acc = accuracy_score(val_y, pred_bools)
		f1 = f1_score(val_y, pred_bools, average='weighted')
		fold_metric = (acc, f1)

	return best_state, best_val_loss, fold_metric


def train_and_evaluate(data_df, emb_df, target_col, target_type, cfg):
	"""5-Fold CV training routine. Saves model, returns average metrics.

	For numeric:
		- We use MSE as training loss.
		- We compute MSE, MAE, and R2 on validation sets.
		- The final fold metric is (MAE, MSE, R2), and we average them across folds.
	For binary/categorical:
		- We compute accuracy and weighted-F1 on validation sets.

	Args:
		data_df: pd.DataFrame with columns [PRIMARY_KEY, target_col].
		emb_df: pd.DataFrame with embeddings and [PRIMARY_KEY].
		target_col: str, name of the target column.
		target_type: str, one of ['numeric', 'binary', 'categorical'].
		cfg: dict, containing model config and file paths.

	Returns:
		tuple with final metrics and model_path:

		(avg_mae, avg_mse, avg_r2, avg_acc, avg_f1, model_path)

		- For numeric tasks: all numeric metrics are filled (MAE, MSE, R2), and
			(avg_acc, avg_f1) will be None.
		- For classification tasks: (avg_acc, avg_f1) are filled, and
			(avg_mae, avg_mse, avg_r2) will be None.
	"""
	# Prepare data
	df = prepare_data_for_target(data_df, emb_df, target_col)
	X = df.drop(columns=[PRIMARY_KEY, target_col]).values
	y = df[target_col].values

	# Encode target if needed
	X, y, output_dim = encode_target(X, y, target_type, cfg)

	# Move data to torch
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	X_torch = torch.from_numpy(X).float().to(device)
	if target_type == 'numeric':
		y_torch = torch.from_numpy(y).float().to(device)
	else:
		y_torch = torch.from_numpy(y).long().to(device)

	dataset = TensorDataset(X_torch, y_torch)

	# Build base model
	input_dim = X.shape[1]
	base_model = create_model(input_dim, output_dim, cfg)
	criterion = select_criterion(base_model, target_type)

	# 5-fold CV
	kf = KFold(n_splits=5, shuffle=True, random_state=42)
	all_fold_results = []

	for fold_idx, (train_indices, val_indices) in enumerate(kf.split(X_torch)):
		print(f'=== Fold {fold_idx + 1} / 5 ===')
		best_state, _, fold_metric = train_single_fold(
				base_model, dataset, train_indices, val_indices, target_type, criterion
		)
		all_fold_results.append(fold_metric)

	# Summaries
	if target_type == 'numeric':
		# each fold_metric = (MAE, MSE, R2)
		avg_mae = np.mean([m[0] for m in all_fold_results])
		avg_mse = np.mean([m[1] for m in all_fold_results])
		avg_r2 = np.mean([m[2] for m in all_fold_results])
		avg_acc, avg_f1 = None, None
		summary_str = (f'Avg MAE: {avg_mae:.4f}, '
									 f'Avg MSE: {avg_mse:.4f}, '
									 f'Avg R2: {avg_r2:.4f}')
	else:
		# each fold_metric = (acc, f1)
		avg_acc = np.mean([m[0] for m in all_fold_results])
		avg_f1 = np.mean([m[1] for m in all_fold_results])
		avg_mae, avg_mse, avg_r2 = None, None, None
		summary_str = f'Avg Acc: {avg_acc:.3f}, Avg F1: {avg_f1:.3f}'

	print(f'Completed 5-Fold CV for {target_col} ({target_type}). {summary_str}')

	# Reuse last fold's best_state 
	base_model.load_state_dict(best_state)
	save_name = f"{cfg['model_save_dir']}/model_{target_col}_{target_type}.pt"
	torch.save(base_model.state_dict(), save_name)

	return avg_mae, avg_mse, avg_r2, avg_acc, avg_f1, save_name


def load_global_settings(config):
	global EARLY_STOP_PATIENCE, MAX_EPOCHS, DROPOUT_RATE, BATCH_SIZE, LR, DRY_RUN
	EARLY_STOP_PATIENCE = config.get('EARLY_STOP_PATIENCE', 3)
	MAX_EPOCHS = config.get('MAX_EPOCHS', 50)
	DROPOUT_RATE = config.get('DROPOUT_RATE', 0.1)
	BATCH_SIZE = config.get('BATCH_SIZE', 128)
	LR = config.get('LR', 1e-3)
	DRY_RUN = config.get('DRY_RUN', False)


def main():
	"""Main entry point for script execution."""
	cfg_path = sys.argv[1]
	cfg = load_config(cfg_path)
	load_global_settings(cfg)
	data_df, emb_df = load_and_merge_data(cfg)

	# Prepare CSV for results
	with open(cfg['result_path'], 'a', newline='') as f:
		writer = csv.writer(f)
		# We'll have columns for both numeric and classification tasks.
		header = [
				'target', 'type', 'num_layers', 'activation_fn',
				'model_path', 'avg_mse', 'avg_mae', 'avg_r2',
				'avg_acc', 'avg_f1'
		]
		writer.writerow(header)

		for tgt, tgt_type in cfg['target_column'].items():
			(avg_mae, avg_mse, avg_r2,
			 avg_acc, avg_f1, model_path) = train_and_evaluate(
					data_df, emb_df, tgt, tgt_type, cfg
			)

			# For numeric, (avg_acc, avg_f1) will be None. For classification,
			# (avg_mae, avg_mse, avg_r2) will be None.
			writer.writerow([
					tgt,
					tgt_type,
					cfg['num_layers'],
					cfg['activation_fn'],
					model_path,
					avg_mse,
					avg_mae,
					avg_r2,
					avg_acc,
					avg_f1
			])


if __name__ == '__main__':
	main()
