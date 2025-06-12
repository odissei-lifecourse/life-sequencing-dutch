from datetime import datetime
import random
from typing import Optional
import unittest
import torch
import pandas as pd
import json
from pathlib import Path
import shutil
import argparse
import pickle
import numpy as np
import sys
from pytorch_lightning import seed_everything
from pop2vec.llm.src.new_code.utils import read_hparams

# The two entry points to compare
from pop2vec.llm.src.new_code.pretrain import main as original_pretrain_entry_point
from pop2vec.llm.src.new_code.pytorch_port.pretrain import _fit as refactored_pretrain_entry_point

def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class TestPretrainEquivalence(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).resolve().parent
        
        self.output_dir = self.test_dir / "test_outputs" / datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.output_dir.mkdir(parents=True)

        # Shared hparams file path
        self.hparams_path = self.test_dir / "pretrain_pt_gpu_1_test.txt"
        hparams = read_hparams(str(self.hparams_path))
        self.epochs = hparams["epochs"]
        
        # Config file paths
        self.original_config_path = self.output_dir / "pretrain_pt_gpu_1_test_ORIGINAL.cfg"
        self.refactored_config_path = self.output_dir / "pretrain_pt_gpu_1_test_REFACTORED.cfg"

        # Checkpoint directories
        self.original_checkpoint_dir = self.output_dir / "original_checkpoints"
        self.original_checkpoint_dir.mkdir()
        self.refactored_checkpoint_dir = self.output_dir / "refactored_checkpoints"
        self.refactored_checkpoint_dir.mkdir()

        # Generate the config files
        base_config = {
            "HPARAMS_PATH": str(self.hparams_path),
            "MLM_PATH": str(self.test_dir / "encoded_dryrun.h5"),
            "VOCAB_PATH": str(self.test_dir / "vocab_v0.csv"),
            "NUM_VAL_ITEMS": 1000
        }
        with open(self.original_config_path, 'w') as f:
            json.dump(base_config | {"CHECKPOINT_DIR": str(self.original_checkpoint_dir)}, f, indent=2)
        with open(self.refactored_config_path, 'w') as f:
            json.dump(base_config | {"CHECKPOINT_DIR": str(self.refactored_checkpoint_dir)}, f, indent=2)

        seed_everything(42)
        set_deterministic(42)

    def tearDown(self):
        # shutil.rmtree(self.output_dir)
        return

    def test_model_equivalence(self):
        # --- Run Original Pretrainer (new_code/pretrain.py) ---
        print(f"Running original pretrain")
        
        original_pretrain_entry_point(
            accelerator="gpu",
            devices=1,
            ddpstrategy="auto",
            batch=None,
            hparams=str(self.hparams_path) if self.hparams_path else None,
            config=str(self.original_config_path),
            save_last=True
        )
        
        original_checkpoint_path = self.original_checkpoint_dir / "last.ckpt"
        if not original_checkpoint_path.exists():
            self.fail("Original pretrainer did not produce a checkpoint.")

        print(f"Running refactored pretrain")
        refactored_pretrain_entry_point(
            config=str(self.refactored_config_path),
            hparams_path=str(self.hparams_path),
            ddpstrategy="auto", 
            accelerator="gpu",
            devices=1, 
            batch=None,
            log_every=1,
            val_check_interval=0.5,
            save_last=True,
            early_stop=False, 
            early_patience=3, 
            early_min_delta=0.0 
        )

        refactored_checkpoint_path =  self.refactored_checkpoint_dir / "last.ckpt"
        if not refactored_checkpoint_path.exists():
            self.fail("Refactored pretrainer did not produce a checkpoint.")

        # Load original model parameters
        ckpt_original_data = torch.load(original_checkpoint_path, map_location='cpu')
        # print(f"{ckpt_original_data.keys()=}")
        print("Original checkpoint keys and types:")
        for key in ckpt_original_data:
            print(f"Key: {key}, Type: {type(ckpt_original_data[key])}")
            if type(ckpt_original_data[key]) is dict:
                for subkey in ckpt_original_data[key]:
                    print(f"  Subkey: {subkey}, Type: {type(ckpt_original_data[key][subkey])}")
        original_sd = ckpt_original_data['state_dict']

        # Load refactored model parameters
        ckpt_refactored_data = torch.load(refactored_checkpoint_path, map_location='cpu')
        print("Refactored checkpoint keys and types:")
        for key in ckpt_refactored_data:
            print(f"Key: {key}, Type: {type(ckpt_refactored_data[key])}")
            if type(ckpt_refactored_data[key]) is dict:
                for subkey in ckpt_refactored_data[key]:
                    print(f"  Subkey: {subkey}, Type: {type(ckpt_refactored_data[key][subkey])}")
        refactored_sd = ckpt_refactored_data['model']

        # Compare state dicts
        self.assertEqual(set(original_sd.keys()), set(refactored_sd.keys()), "Model state_dict keys do not match.")

        atol = 1e-6
        mismatched_keys = []
        for key in original_sd:
            tensor_orig = original_sd[key]
            tensor_refact = refactored_sd[key]
            # self.assertTrue(
            #     torch.allclose(tensor_orig, tensor_refact, atol=atol),
            #     f"Tensor mismatch for key '{key}'. Max diff: {(tensor_orig - tensor_refact).abs().max()}"
            # )
            if not torch.allclose(tensor_orig, tensor_refact, atol=atol):
                mismatched_keys.append(key)
        if mismatched_keys:
            for key in sorted(original_sd):
                if key not in mismatched_keys:
                    print(f"Key '{key}' matches")
                else:
                    if original_sd[key].shape != refactored_sd[key].shape:
                        original_shape = original_sd[key].shape
                        refactored_shape = refactored_sd[key].shape
                        print(f"Key '{key}' has different shapes: {original_shape=} vs {refactored_shape=}")
                    else:
                        abs_diffs = (original_sd[key] - refactored_sd[key]).abs()
                        rel_diffs = abs_diffs / (original_sd[key].abs().clamp(min=1e-12))
                        print(f"Key '{key}' has different values:")
                        print(f"\t{abs_diffs.mean()=}")
                        print(f"\t{abs_diffs.median()=}")
                        print(f"\t{abs_diffs.max()=}")
                        print(f"\t{rel_diffs.mean()=}")
                        print(f"\t{rel_diffs.median()=}")
                        print(f"\t{rel_diffs.max()=}")
            print(f"Total mismatched keys: {len(mismatched_keys)} of {len(original_sd)}")
            self.fail("Model tensors do not match between original and refactored pretrain.")
        else:
            print("All model tensors match between original and refactored pretrain.")
if __name__ == "__main__":
    unittest.main()
