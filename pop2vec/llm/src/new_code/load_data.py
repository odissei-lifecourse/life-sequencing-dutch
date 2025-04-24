from __future__ import annotations

import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter

import h5py
from torch.utils.data import IterableDataset
from pathlib import Path

import os
import numpy as np

import pandas as pd
from typing import Optional
import logging



logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

class CustomIterableDataset(IterableDataset):
    def __init__(self, file_path, validation, num_val_items=None, val_split=0.1, mlm_encoded=True, inference=False):
        self.file_path = file_path
        self.validation = validation
        self.num_val_items = num_val_items
        self.val_split = val_split
        self.inference = inference
        self.set_mlm_encoded(mlm_encoded)

    def set_mlm_encoded(self, mlm_encoded, return_index=None):
        self.mlm_encoded = mlm_encoded
        if return_index is None:
          self.return_index = not self.mlm_encoded
        else:
          self.return_index = return_index


    def __len__(self):
        with h5py.File(self.file_path, 'r') as hdf5:
            return hdf5['input_ids'].shape[0] 


    def __iter__(self):
        with h5py.File(self.file_path, 'r') as hdf5:
            num_val_items = self.num_val_items
            if num_val_items is None:
              num_val_items = int(hdf5['input_ids'].shape[0] * self.val_split)
            
            n_items = hdf5['input_ids'].shape[0]
            num_train_items = n_items - num_val_items
            rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))    
            if self.validation:
                per_worker = num_val_items // world_size
                start_index = rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else num_val_items
            elif self.inference:
                per_worker = n_items // world_size
                start_index = rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else n_items
            else:
                per_worker = num_train_items // world_size
                start_index = num_val_items + rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else num_train_items + num_val_items


            for index in range(start_index, end_index):
                ret_dict = {
                    "input_ids": hdf5['input_ids'][index],
                    "padding_mask": hdf5['padding_mask'][index],
                }

                if self.mlm_encoded:
                    neg_one_index = np.where(hdf5['target_tokens'][index] == -1)[0]
                    target_tokens = hdf5['target_tokens'][index][:neg_one_index[0] if neg_one_index.size > 0 else None]
                    target_pos = hdf5['target_pos'][index][:neg_one_index[0] if neg_one_index.size > 0 else None]
                    
                    ret_dict.update({
                        "original_sequence": hdf5['original_sequence'][index],
                        "target_tokens": target_tokens,
                        "target_pos": target_pos,
                        "target_cls": hdf5['target_cls'][index],
                    })

                if self.return_index:
                    ret_dict["sequence_id"] = hdf5['sequence_id'][index]

                yield ret_dict

class CustomDataset(Dataset):
    def __init__(self, file_path, validation, num_val_items=None, val_split=0.1, mlm_encoded=True, inference=False):
        self.file_path = file_path
        self.validation = validation
        self.num_val_items = num_val_items
        self.val_split = val_split
        self.inference = inference
        self.set_mlm_encoded(mlm_encoded)
        self.load_data()

    def set_mlm_encoded(self, mlm_encoded, return_index=None):
        self.mlm_encoded = mlm_encoded
        if return_index is None:
            self.return_index = not self.mlm_encoded
        else:
            self.return_index = return_index

    def load_data(self):
        with h5py.File(self.file_path, 'r') as hdf5:
            n_items = hdf5['input_ids'].shape[0]
            num_val_items = self.num_val_items
            if num_val_items is None:
                num_val_items = int(n_items * self.val_split)
            num_train_items = n_items - num_val_items

            # Get rank and world_size from environment variables
            rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            if self.validation:
                per_worker = num_val_items // world_size
                start_index = rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else num_val_items
                indices = np.arange(start_index, end_index)
            elif self.inference:
                per_worker = n_items // world_size
                start_index = rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else n_items
                indices = np.arange(start_index, end_index)
            else:
                per_worker = num_train_items // world_size
                start_index = num_val_items + rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else n_items
                indices = np.arange(start_index, end_index)

            # Load datasets into memory
            self.input_ids = hdf5['input_ids'][indices]
            self.padding_mask = hdf5['padding_mask'][indices]

            if self.mlm_encoded:
                self.original_sequence = hdf5['original_sequence'][indices]
                self.target_tokens = hdf5['target_tokens'][indices]
                self.target_pos = hdf5['target_pos'][indices]
                self.target_cls = hdf5['target_cls'][indices]

            if self.return_index:
                self.sequence_id = hdf5['sequence_id'][indices]

            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ret_dict = {
            "input_ids": self.input_ids[idx],
            "padding_mask": self.padding_mask[idx],
        }

        if self.mlm_encoded:
            # Process 'target_tokens' and 'target_pos' as in the original code
            neg_one_index = np.where(self.target_tokens[idx] == -1)[0]
            end_slice = neg_one_index[0] if neg_one_index.size > 0 else None
            target_tokens = self.target_tokens[idx][:end_slice]
            target_pos = self.target_pos[idx][:end_slice]

            ret_dict.update({
                "original_sequence": self.original_sequence[idx],
                "target_tokens": target_tokens,
                "target_pos": target_pos,
                "target_cls": self.target_cls[idx],
            })

        if self.return_index:
            ret_dict["sequence_id"] = self.sequence_id[idx]

        return ret_dict

class CustomInMemoryDataset(Dataset):
    def __init__(self, file_path, validation=False, num_val_items=None, val_split=None, mlm_encoded=True, inference=False):
        if num_val_items is None and val_split is None:
            raise ValueError("Both num_val_items and val_split cannot be None. One of them must have a value")
        self.file_path = file_path
        self.validation = validation
        self.inference = inference
        self.mlm_encoded = mlm_encoded
        self.set_mlm_encoded(mlm_encoded)
        self.loaded = False
        self.num_val_items, self.length = self._fix_num_val_items(num_val_items, val_split)
        
    
    def _fix_num_val_items(self, num_val_items, val_split):
        with h5py.File(self.file_path, 'r') as hdf5:
            n_items = hdf5['input_ids'].shape[0]
            if num_val_items is None: 
                num_val_items = int(n_items * val_split)
                
            num_train_items = n_items - num_val_items

            if self.validation:
                length = num_val_items 
            elif self.inference:
                length = n_items 
            else:
                length = num_train_items
            return num_val_items, length


    def set_mlm_encoded(self, mlm_encoded, return_index=None):
        self.mlm_encoded = mlm_encoded
        if return_index is None:
            self.return_index = not self.mlm_encoded
        else:
            self.return_index = return_index

    def load_data(self):
        with h5py.File(self.file_path, 'r') as hdf5:
            n_items = hdf5['input_ids'].shape[0]

            if self.validation:
                indices = np.arange(0, self.num_val_items)
            elif self.inference:
                indices = np.arange(0, n_items)
            else:
                indices = np.arange(self.num_val_items, n_items)

            # Load datasets into memory and move to shared memory
            self.input_ids = torch.from_numpy(hdf5['input_ids'][indices])#.share_memory_()
            self.padding_mask = torch.from_numpy(hdf5['padding_mask'][indices])#.share_memory_()

            if self.mlm_encoded:
                self.original_sequence = torch.from_numpy(hdf5['original_sequence'][indices])#.share_memory_()
                self.target_tokens = torch.from_numpy(hdf5['target_tokens'][indices])#.share_memory_()
                self.target_pos = torch.from_numpy(hdf5['target_pos'][indices])#.share_memory_()
                self.target_cls = torch.from_numpy(hdf5['target_cls'][indices])#.share_memory_()

            if self.return_index:
                self.sequence_id = torch.from_numpy(hdf5['sequence_id'][indices])#.share_memory_()

            self.indices = indices

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.loaded == False:
            self.load_data()
            self.loaded = True
        ret_dict = {
            "input_ids": self.input_ids[idx],
            "padding_mask": self.padding_mask[idx],
        }

        if self.mlm_encoded:
            # Handle variable-length target sequences
            target_tokens = self.target_tokens[idx]
            target_pos = self.target_pos[idx]
            neg_one_index = (target_tokens == -1).nonzero(as_tuple=True)[0]
            end_slice = neg_one_index[0] if len(neg_one_index) > 0 else target_tokens.size(0)
            target_tokens = target_tokens[:end_slice]
            target_pos = target_pos[:end_slice]

            ret_dict.update({
                "original_sequence": self.original_sequence[idx],
                "target_tokens": target_tokens,
                "target_pos": target_pos,
                "target_cls": self.target_cls[idx],
            })

        if self.return_index:
            ret_dict["sequence_id"] = self.sequence_id[idx]

        return ret_dict


class FineTuneInMemoryDataset(Dataset):
    """
    Dataset class for fine-tuning.
    Merges sequences.h5 data with a CSV or Parquet file
    that contains labels (or other targets).
    """
    def __init__(
        self,
        h5_file_path,
        train_file_path,
        target_col='target_label',  # name of the column containing the label
        phase='train',
        num_val_items=None,
        val_split=0.1,
        return_sequence_id=False,  # Whether to return the sequence_id
        task_type='classification',  # 'classification' or 'regression'
        assign_weights=False,
    ):
        """
        :param h5_file_path: Path to the .h5 file, e.g. 'sequences.h5'
        :param train_file_path: Path to the train file, e.g. 'train.csv' or 'train.parquet'
        :param target_col: Column name in the train file containing the target label
        :param phase: one of "train", "validation", "test"
        :param num_val_items: Optional fixed number of validation items
        :param val_split: If num_val_items is None, use this fraction for validation
        :param return_sequence_id: If True, return the sequence_id in __getitem__
        :param task_type: 'classification' or 'regression'; used to set label dtype
        """
        super().__init__()

        self.h5_file_path = h5_file_path
        self.train_file_path = train_file_path
        self.target_col = target_col
        self.phase = phase
        self.num_val_items = num_val_items
        self.val_split = val_split
        self.return_sequence_id = return_sequence_id
        self.task_type = task_type
        self.assign_weights = assign_weights
        # 1) Read the label file (CSV or Parquet)
        self.label_df = self._load_label_file(self.train_file_path)

        # 2) Load data from the HDF5 and intersect with label_df
        self._load_h5_and_intersect()
    
    def _load_label_file(self, path):
        """
        Load the CSV or Parquet into a DataFrame with columns:
          RINPERSOON  target_label
        (Your actual column names may differ, but must contain a unique ID
         that matches 'sequence_id' in the HDF5.)
        """
        ext = os.path.splitext(path)[1]
        if ext == '.csv':
            df = pd.read_csv(path)
        elif ext == '.parquet':
            df = pd.read_parquet(path)
        else:
            raise ValueError(f'Unsupported training file extension: {ext}')

        if 'RINPERSOON' not in df.columns:
            raise ValueError("Train file must have a 'RINPERSOON' column")

        logger.info(f'Before dropping nan values, len(df) = {len(df)}')
        df.dropna(subset=[self.target_col, 'RINPERSOON'], inplace=True)
        logger.info(f'After dropping nan values, len(df) = {len(df)}')
        df.set_index('RINPERSOON', inplace=True)

        return df

    def _load_h5_and_intersect(self):
        """
        Reads sequences.h5 and retains only those IDs that also exist in the label file.
        Splits data for train/validation/test as requested.
        """

        with h5py.File(self.h5_file_path, 'r') as hdf5:
            # 1) Get sequence IDs
            n_items = hdf5['input_ids'].shape[0]
            self.num_val_items = self.num_val_items if self.num_val_items is not None else int(n_items * self.val_split)
            logger.info(f"1 done, num_val_items = {self.num_val_items}, n_items = {n_items}")
            # 2) Figure out split
            full_indices = np.arange(n_items)
            if self.phase == 'test':
                indices = full_indices
            elif self.phase == 'validation':
                print("inside validation")
                indices = full_indices[:self.num_val_items]
                print(indices[:10])
            else:  # 'train'
                print("inside train")
                indices = full_indices[self.num_val_items:]
                print(indices[:7])
            print("x"*10)
            print(f"phase = {self.phase}, {indices[:4]}")
            self.input_ids = torch.from_numpy(hdf5['input_ids'][indices])#.share_memory_()
            self.padding_mask = torch.from_numpy(hdf5['padding_mask'][indices])#.share_memory_()
            self.sequence_id = torch.from_numpy(hdf5['sequence_id'][indices])#.share_memory_()

            logger.info(f"2 done, {self.input_ids.shape, self.padding_mask.shape, self.sequence_id.shape}")

            # 3) Create a set for fast membership checks
            label_id_set = set(self.label_df.index)
            logger.info(f"3 done, {len(label_id_set)}")
            
            # 4) Identify intersection: which IDs appear in both?
            mask = np.array([sid.item() in label_id_set for sid in self.sequence_id])
            logger.info(f"{self.label_df.dtypes}, {self.label_df.index.dtype}, {self.label_df.index.inferred_type}, {self.sequence_id.dtype}")
            logger.info(f"4 done, {mask.sum()}, {len(mask)}")
            for i, x in enumerate(label_id_set):
                if i == 4:
                    break
                print(x)
                print(self.sequence_id[i])
                print("*"*10)

            # 5) Build final subset of indices & sequence IDs
            self.input_ids = self.input_ids[mask]
            self.padding_mask = self.padding_mask[mask]
            self.sequence_id = self.sequence_id[mask]
            logger.info(f"5 done, {self.input_ids.shape, self.padding_mask.shape, self.sequence_id.shape}")
            
            
        # 7) Gather the labels in the correct order
        #    We know each self.sequence_id is in label_id_set, so no NaN.
        raw_labels = self.label_df.loc[self.sequence_id.numpy(), self.target_col].values
        logger.info(f"6 done, {len(raw_labels)}")
            
        if self.task_type == 'classification':
            self.labels = torch.tensor(raw_labels, dtype=torch.long)
            if self.assign_weights:
                labels = self.labels.tolist()
                class_counts = Counter(labels)
                weights = [1.0 / class_counts[label] for label in labels]
                self.sampler = WeightedRandomSampler(
                    weights, num_samples=len(labels), replacement=True
                )
        elif self.task_type == 'regression':
            self.labels = torch.tensor(raw_labels, dtype=torch.float)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        logger.info(f"7 done")
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):

        ret_dict = {
            "input_ids": self.input_ids[idx],
            "padding_mask": self.padding_mask[idx],
            "target": self.labels[idx],
        }
        if self.return_sequence_id:
            ret_dict["sequence_id"] = self.sequence_id[idx]
        return ret_dict


class CustomLazyHDF5Dataset(Dataset):
    """
    Lazy HDF5 reader that mirrors the behaviour of `CustomInMemoryDataset`
    without loading the entire dataset into RAM.

    Parameters
    ----------
    file_path : str | Path
        Path to the `.h5` file.
    validation : bool, default False
        If True, yield the first *val_split* fraction of samples.
    num_val_items : int | None, default None
        Exact number of validation items.  Overrides *val_split*.
    val_split : float, default 0.10
        Fraction of the file that belongs to the validation split.
    mlm_encoded : bool, default True
        Whether the file contains masked-LM target columns.
    return_index : bool | None, default None
        If None, the index is returned only when *mlm_encoded* is False.
        Otherwise forces returning the `"sequence_id"` column.
    inference : bool, default False
        If True, return the *whole* dataset irrespective of the split.
    """

    # ------------------------------------------------------------------ #
    # constructor / helpers                                              #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        file_path: str | Path,
        validation: bool = False,
        num_val_items: Optional[int] = None,
        val_split: float = 0.10,
        mlm_encoded: bool = True,
        return_index: Optional[bool] = None,
        inference: bool = False,
    ) -> None:
        super().__init__()

        self.file_path = Path(file_path)
        self.validation = validation
        self.val_split = val_split
        self.inference = inference

        # masked-LM columns
        self.set_mlm_encoded(mlm_encoded, return_index)

        # Compute split indices once (cheap, uses only the header)
        with h5py.File(self.file_path, "r") as f:
            n_items = len(f["input_ids"])
        n_val = num_val_items or int(n_items * val_split)

        if self.inference:
            self._indices = np.arange(0, n_items, dtype=np.int64)
        elif self.validation:
            self._indices = np.arange(0, n_val, dtype=np.int64)
        else:
            self._indices = np.arange(n_val, n_items, dtype=np.int64)

        # This attribute will hold the *worker-local* handle
        self._h5: Optional[h5py.File] = None

    def set_mlm_encoded(
        self,
        mlm_encoded: bool,
        return_index: Optional[bool] = None,
    ) -> None:
        self.mlm_encoded = mlm_encoded
        self.return_index = (not mlm_encoded) if return_index is None else return_index

    # ------------------------------------------------------------------ #
    # Dataset API                                                        #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 1) ensure the HDF5 handle is initialised for *this* worker
        h5 = self._get_handle()

        # 2) translate `idx` (0 … len(self)-1) into the row inside the file
        row = self._indices[idx]

        # 3) mandatory columns
        sample: dict[str, torch.Tensor] = {
            "input_ids": torch.as_tensor(h5["input_ids"][row]),
            "padding_mask": torch.as_tensor(h5["padding_mask"][row]),
        }

        # 4) MLM-specific columns (optional)
        if self.mlm_encoded:
            tok = torch.as_tensor(h5["target_tokens"][row])
            pos = torch.as_tensor(h5["target_pos"][row])

            # cut at first -1 sentinel (variable-length targets)
            try:
                nz = (tok == -1).nonzero(as_tuple=False)
                end = nz[0, 0].item() if nz.numel() else tok.size(0)
                tok = tok[:end]
                pos = pos[:end]
            except IndexError:  # no -1 found
                pass

            sample.update(
                original_sequence=torch.as_tensor(
                    h5["original_sequence"][row]
                ),
                target_tokens=tok,
                target_pos=pos,
                target_cls=torch.as_tensor(h5["target_cls"][row]),
            )

        # 5) optionally return the sequence id
        if self.return_index:
            sample["sequence_id"] = torch.as_tensor(h5["sequence_id"][row])

        return sample

    # ------------------------------------------------------------------ #
    # private utilities                                                  #
    # ------------------------------------------------------------------ #
    def _get_handle(self) -> h5py.File:
        """
        Open the HDF5 file *once per worker* (lazily).
        Lightning / DataLoader forks workers, so every fork gets its own
        copy of 'self._h5' -  this function ensures we open it lazily
        instead of at import-time (which would break after fork/spawn).
        """
        if self._h5 is None:
            # read-only, enable SWMR so many readers may coexist
            self._h5 = h5py.File(
                self.file_path,
                mode="r",
                libver="latest",
                swmr=True,
            )
        return self._h5

    # ------------------------------------------------------------------ #
    # multiprocessing-pickling safety                                    #
    # ------------------------------------------------------------------ #
    def __getstate__(self):
        """Drop the file handle so torch.multiprocessing can pickle us."""
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        """Close the HDF5 file when the worker exits."""
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
