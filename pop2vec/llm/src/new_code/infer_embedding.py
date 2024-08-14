import numpy as np
import torch
import json
import sys
import logging 
from tqdm import tqdm

from src.transformer.models import TransformerEncoder
from src.new_code.load_data import CustomIterableDataset
from src.new_code.pretrain import read_hparams_from_file
from src.new_code.pipeline import write_to_hdf5
from src.new_code.utils import read_json, print_now
from torch.utils.data import DataLoader



def load_model(checkpoint_path, hparams):
  model = TransformerEncoder.load_from_checkpoint(checkpoint_path, hparams=hparams)
  model = model.transformer
  model.eval()
  device = str(next(model.parameters()).device)
  print_now(f"Model is on {device}")
  return model

def print_now_dataset_stuff(dataset):
  print_now(f"length of dataset {len(dataset)}")
  print_now(f"type of dataset {type(dataset)}")
  print_now(f"input_ids shape = {dataset.data['input_ids'].shape}")
  if "original_sequence" in dataset.data:
    print_now(f"original sequence shape = {dataset.data['original_sequence'].shape}")

def dump_embeddings(path, embeddings_dict):
  with open(path, 'w') as json_file:
    json.dump(embeddings_dict, json_file)


def inference(cfg):
  hparams_path = cfg['HPARAMS_PATH']
  hparams = read_hparams_from_file(hparams_path)
  checkpoint_path = cfg['CHECKPOINT_PATH']
  write_path = cfg['EMB_WRITE_PATH']
  tokenized_path = cfg['TOKENIZED_PATH']
  model = load_model(checkpoint_path, hparams)

  logging.info("Reading from tokenzied path: %s", tokenized_path)
  dataset = CustomIterableDataset(
    tokenized_path, 
    validation=False,
    inference=True
  )
  dataset.set_mlm_encoded(False)

  if 'BATCH_SIZE' in cfg:
    batch_size = cfg['BATCH_SIZE']
  else:
    batch_size = 512
  dataloader = DataLoader(dataset, batch_size=batch_size)

  for i, batch in enumerate(tqdm(dataloader, desc="Inferring by batch")):
      if torch.cuda.is_available():
          batch['input_ids'] = batch['input_ids'].to('cuda')
          batch['padding_mask'] = batch['padding_mask'].to('cuda')
      # Pass the batch through the model
      with torch.no_grad():
        outputs = model(
          x=batch["input_ids"].long(),
          padding_mask=batch["padding_mask"].long(),
        )
      if i%100 == 0:
        print_now(f"printing for batch {i}:")
        print_now(f"len(outputs) = {len(outputs)}")
        print_now(f"batch length = {len(batch['sequence_id'])}")

      sequence_id = [x.decode() for x in batch["sequence_id"]]
      cls_emb = outputs[:, 0, :].cpu()
      mean_emb = torch.mean(outputs, axis=1).cpu()
      data_dict = {
         "sequence_id": sequence_id,
         "cls_emb": cls_emb,
         "mean_emb": mean_emb
      }

      write_to_hdf5(write_path, data_dict, np.float16)
        


if __name__ == "__main__":
  logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
  )
  CFG_PATH = sys.argv[1]
  print_now(CFG_PATH)
  cfg = read_json(CFG_PATH)
  inference(cfg)  


