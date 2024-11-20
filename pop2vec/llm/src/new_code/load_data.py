import json
import torch
from torch.utils.data import Dataset, DataLoader

import h5py
from torch.utils.data import IterableDataset
import os
import numpy as np

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
    def __init__(self, file_path, validation=False, num_val_items=None, val_split=0.1, mlm_encoded=True, inference=False):
        self.file_path = file_path
        self.validation = validation
        self.num_val_items = num_val_items
        self.val_split = val_split
        self.inference = inference
        self.mlm_encoded = mlm_encoded
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

            if self.validation:
                indices = np.arange(0, num_val_items)
            elif self.inference:
                indices = np.arange(0, n_items)
            else:
                indices = np.arange(num_val_items, n_items)

            # Load datasets into memory and move to shared memory
            self.input_ids = torch.from_numpy(hdf5['input_ids'][indices]).share_memory_()
            self.padding_mask = torch.from_numpy(hdf5['padding_mask'][indices]).share_memory_()

            if self.mlm_encoded:
                self.original_sequence = torch.from_numpy(hdf5['original_sequence'][indices]).share_memory_()
                self.target_tokens = torch.from_numpy(hdf5['target_tokens'][indices]).share_memory_()
                self.target_pos = torch.from_numpy(hdf5['target_pos'][indices]).share_memory_()
                self.target_cls = torch.from_numpy(hdf5['target_cls'][indices]).share_memory_()

            if self.return_index:
                self.sequence_id = torch.from_numpy(hdf5['sequence_id'][indices]).share_memory_()

            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
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

# class CustomDataset(Dataset):
#     def __init__(self, data, mlm_encoded=True):
#       self.data = data
#       self.set_mlm_encoded(mlm_encoded)

#     def set_mlm_encoded(self, mlm_encoded):
#       self.mlm_encoded = mlm_encoded
#       self.return_index = not self.mlm_encoded
    
#     def __len__(self):
#         return self.data["input_ids"].shape[0]
#     def __reduce__(self):
#         return (self.__class__, (self.data,))

#     def __getitem__(self, index):
#         ret_dict = {            
#             "input_ids": self.data["input_ids"][index],
#             "padding_mask": self.data["padding_mask"][index],
#         }

#         if self.mlm_encoded:
#           ret_dict.update(
#             {
#               "original_sequence": self.data["original_sequence"][index],
#               "target_tokens": self.data["target_tokens"][index],
#               "target_pos": self.data["target_pos"][index],
#               "target_cls": self.data["target_cls"][index],
#             }
#           )

#         if self.return_index:
#           ret_dict["sequence_id"] = self.data["sequence_id"][index]

#         return ret_dict


