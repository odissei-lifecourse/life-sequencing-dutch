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

