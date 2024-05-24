import re
import src.transformer
from src.transformer.models import TransformerEncoder
# from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
import sys
from pathlib import Path
import logging
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import torch
from src.new_code.load_data import CustomIterableDataset
from src.new_code.utils import read_json, print_now
import os
from pytorch_lightning.strategies import DDPStrategy


def is_float(string):
    try:
      float(string)
      return True
    except ValueError:
      return False

# Read hparams from the text file
def read_hparams_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        hparams = {}
        for line in lines:
            if len(line) < 2 or line.startswith("#"):
              continue
            #print(line)

            line = line.strip().split('#')[0]

            key, value = line.strip().split(': ')
            value = value.replace('"','')
            if value in ['True', 'False']:
              if value == 'True':
                value = True
              else:
                value = False
            elif value.isdigit():
              value = int(value)
            elif is_float(value):
              value = float(value)
            hparams[key] = value # float(value) if value.isdigit() else value
            #print(key, value)
        return hparams

def get_callbacks(ckpoint_dir):
  if os.path.exists(ckpoint_dir) is False:
    os.mkdir(ckpoint_dir)
  callbacks = [
    ModelCheckpoint(
      dirpath=ckpoint_dir,#'projects/baseball/models/2010',
      filename='model-{epoch:02d}',
      monitor='val_loss_combined',
      save_top_k=3,
    )
  ]
  return callbacks

def pretrain(cfg):
  hparams_path = cfg['HPARAMS_PATH']#'src/new_code/regular_hparams.txt'
  ckpoint_dir = cfg['CHECKPOINT_DIR']
  mlm_path = cfg['MLM_PATH']
  hparams = read_hparams_from_file(hparams_path)
  if 'RESUME_FROM_CHECKPOINT' in cfg:
    print_now(f"resuming training from checkpoint {cfg['RESUME_FROM_CHECKPOINT']}")
    model = TransformerEncoder.load_from_checkpoint(
      cfg['RESUME_FROM_CHECKPOINT'], 
      hparams=hparams
    )
  else:
    model = TransformerEncoder(hparams)
  
  batch_size = cfg['BATCH_SIZE']
  callbacks = get_callbacks(ckpoint_dir)
  ddp = DDPStrategy(process_group_backend="mpi")
  trainer = Trainer(
    #strategy=ddp,
    default_root_dir=ckpoint_dir,
    callbacks=callbacks,
    max_epochs=cfg['MAX_EPOCHS'],
    accelerator='gpu',
    devices=1
  )
  val_dataset = CustomIterableDataset(mlm_path, validation=True)
  train_dataset = CustomIterableDataset(mlm_path, validation=False)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size) 
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
  
  print_now("training and validation dataloaders are created")
  trainer.fit(model, train_dataloader, val_dataloader)
  
if __name__ == "__main__":
  torch.set_float32_matmul_precision("medium")
  CFG_PATH = sys.argv[1]
  print_now(CFG_PATH)
  cfg = read_json(CFG_PATH)
  pretrain(cfg)