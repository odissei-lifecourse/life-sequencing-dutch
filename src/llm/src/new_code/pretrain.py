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
from pytorch_lightning.loggers import CSVLogger


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

def get_callbacks(ckpoint_dir, val_check_interval):
  if os.path.exists(ckpoint_dir) is False:
    os.mkdir(ckpoint_dir)
  callbacks = [
    ModelCheckpoint(
      dirpath=ckpoint_dir,#'projects/baseball/models/2010',
      filename='model-{epoch:02d}-{step}-{val_loss:.2f}',
      monitor='val_loss_combined',
      save_top_k=3,
      save_last=False,
      save_weights_only=False,
      every_n_train_steps=val_check_interval+1,
      verbose=True,
    )
  ]
  return callbacks

<<<<<<< HEAD:src/llm/src/new_code/pretrain.py
=======
def get_train_val_dataloaders(dataset, batch_size, train_split=0.8, shuffle=True):
  total_samples = len(dataset)
  train_size = int(train_split * total_samples)
  val_size = total_samples - train_size

  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  return (
    DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=71),
    DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=71)
  )

def subset(data, lim):
  for key in data:
    data[key] = data[key][:lim]
  # data["input_ids"] = data["input_ids"][:lim]
  # data["padding_mask"] = data["padding_mask"][:lim]
  # data["original_sequence"] = data["original_sequence"][:lim]
  # data["target_tokens"] = data["target_tokens"][:lim]
  # data["target_pos"] = data["target_pos"][:lim]
  # data["target_cls"] = data["target_cls"][:lim]
  return data

>>>>>>> e9cd94c (subsets dataset):code/llm/src/new_code/pretrain.py
def pretrain(cfg):
  hparams_path = cfg['HPARAMS_PATH']#'src/new_code/regular_hparams.txt'
  ckpoint_dir = cfg['CHECKPOINT_DIR']
  mlm_path = cfg['MLM_PATH']
  hparams = read_hparams_from_file(hparams_path)
  batch_size = cfg['BATCH_SIZE']
  num_val_items = cfg.get('NUM_VAL_ITEMS', 100000)
  val_check_interval = cfg.get(
    'VAL_CHECK_INTERVAL', 
    int(num_val_items*5/batch_size)
  )
  
  if 'RESUME_FROM_CHECKPOINT' in cfg:
    print_now(f"resuming training from checkpoint {cfg['RESUME_FROM_CHECKPOINT']}")
    model = TransformerEncoder.load_from_checkpoint(
      cfg['RESUME_FROM_CHECKPOINT'], 
      hparams=hparams
    )
  else:
    model = TransformerEncoder(hparams)
  
<<<<<<< HEAD:src/llm/src/new_code/pretrain.py
  callbacks = get_callbacks(ckpoint_dir, val_check_interval+1)
  #ddp = DDPStrategy(process_group_backend="mpi")
  logger = CSVLogger(ckpoint_dir)
  trainer = Trainer(
    #strategy=ddp,
    default_root_dir=ckpoint_dir,
    callbacks=callbacks,
    max_epochs=cfg['MAX_EPOCHS'],
    val_check_interval=val_check_interval,
    accelerator='gpu',
    devices=1,
    logger=logger,
    precision=16
  )
  val_dataset = CustomIterableDataset(
    mlm_path, 
    validation=True, 
    num_val_items=num_val_items
  )
  train_dataset = CustomIterableDataset(
    mlm_path, 
    validation=False, 
    num_val_items=num_val_items
  )
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size) 
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
  
  print_now("training and validation dataloaders are created")
  trainer.fit(model, train_dataloader, val_dataloader)
  
=======
  mlm_paths = []
  for root, dirs, files in os.walk(mlm_dir):
    for file_path in files:
      mlm_paths.append(os.path.join(root, file_path))

  LIM = 1000
  val_dataloader = None
  batch_size = cfg['BATCH_SIZE']
  for epoch in range(cfg['MAX_EPOCHS']):
    for counter, mlm_path in enumerate(mlm_paths):
      if counter == 0:
        if val_dataloader is None:
          with open(mlm_path, 'rb') as f:
            dataset = pickle.load(f)
            dataset.data = subset(dataset.data, LIM)
          val_dataloader = DataLoader(dataset, batch_size=batch_size)
        continue
      with open(mlm_path, 'rb') as f:
        dataset = pickle.load(f)
        dataset.data = subset(dataset.data, LIM)
      callbacks = get_callbacks(ckpoint_dir, counter)
      trainer = Trainer(
        default_root_dir=ckpoint_dir,
        callbacks=callbacks,
        max_epochs=1,
        accelerator='gpu',
        devices=1
      )
      # Create a data loader
      # train_dataloader, val_dataloader = get_train_val_dataloaders(
      #   dataset=dataset,
      #   batch_size=batch_size,
      #   train_split=hparams['train_split']
      # )
      train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

      print_now("training and validation dataloaders are created")
      print_now(f"# of batches in training: {len(train_dataloader)}")
      print_now(f"# of batches in validation: {len(val_dataloader)}")
      trainer.fit(model, train_dataloader)
      trainer.validate(model, val_dataloader)

>>>>>>> e9cd94c (subsets dataset):code/llm/src/new_code/pretrain.py
if __name__ == "__main__":
  torch.set_float32_matmul_precision("medium")
  CFG_PATH = sys.argv[1]
  print_now(CFG_PATH)
  cfg = read_json(CFG_PATH)
  pretrain(cfg)