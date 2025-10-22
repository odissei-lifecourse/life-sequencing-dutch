import json
import logging
import sys
import os
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pop2vec.llm.src.new_code.load_data import CustomLazyHDF5Dataset
from pop2vec.llm.src.new_code.pipeline import write_to_hdf5
from pop2vec.llm.src.new_code.utils import read_json
from pop2vec.llm.src.transformer.models import TransformerEncoder
from pop2vec.utils.convert_hdf5_to_parquet import h5_array_to_pq

DTYPE = np.float64

REQ_KEYS = [
    "emb_write_path",
    "tokenized_path",
    "checkpoint_path",
]

DEFAULT_VALS = {
    "save_token_embs": False,
    "batch_size": 512,
    "needed_ids_path": None,
}

# ──────────────────── helper: hparam integrity / update ───────────────
def _integrity_check(cfg):
    missing = [k for k in REQ_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required arguments in cfg: {', '.join(missing)}")

def _with_defaults(cfg):
    for k, v in DEFAULT_VALS.items():
        cfg.setdefault(k, v)
    return cfg



def load_model(checkpoint_path):
    model = TransformerEncoder.load_from_checkpoint(
        checkpoint_path, 
        # assuming you trained with a recent version of PyTorch-Lightning and 
        # you called self.save_hyperparameters(hparams) in your __init__ 
        # (which you did), Lightning will store all of your hparams in the 
        # checkpoint and automatically pass them back into your constructor 
        # when you call load_from_checkpoint. So no need to pass hparams separately

        # hparams=read_hparams(hparams_path) 
    )
    model = model.transformer
    model.eval()
    default_device = str(next(model.parameters()).device)
    logging.info(f"Model is on {default_device} by default")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logging.info(f"Moved model to {device} by force!")
    
    return model


def log_dataset_stuff(dataset):
    logging.info(f"length of dataset {len(dataset)}")
    logging.info(f"type of dataset {type(dataset)}")
    sample0 = dataset[0]
    logging.info(f"input_ids shape = {sample0['input_ids'].shape}")
    if 'original_sequence' in sample0:
        logging.info(f"original sequence shape = {sample0['original_sequence'].shape}")

    

def dump_embeddings(path, embeddings_dict):
    with open(path, "w") as json_file:
        json.dump(embeddings_dict, json_file)


def inference(cfg, transform_to_parquet=True):
    """Run inference on trained model.

    Args:
        cfg (dict): configuration.
        transform_to_parquet (bool): If true (the default), the stored embeddings
        are copied from hdf5 into a parquet file.

    Notes:
       Embeddings are always stored in hdf5. If a file with the same name exists already, it
       is replaced. If parquet files are created, they are stored in a new folder with the
       name from `cfg["emb_write_path"]` (without the suffix). Moreover, storing
       in parquet requires loading the full set of embeddings into memory, which can
       require a lot of memory. In some situations, it might thus be better to
       do the transformation to parquet in a separate step -- for instance when multiple
       inferences are running on the same node and memory is relatively scarce.
    """
    write_path = cfg["emb_write_path"]
    tokenized_path = cfg["tokenized_path"]
    model = load_model(cfg['checkpoint_path'])
    save_token_embs = cfg['save_token_embs']
    logging.info("Reading from tokenized path: %s", tokenized_path)

    if cfg['needed_ids_path']:
        needed_id_set = set(
            pd.read_parquet(cfg['needed_ids_path'])['RINPERSOON'].tolist()
        )
    else:
        needed_id_set = None

    dataset = CustomLazyHDF5Dataset(
        tokenized_path,
        validation=False,
        inference=True,
        mlm_encoded=False,              
        num_val_items=0,
        needed_id_set=needed_id_set
    )
    log_dataset_stuff(dataset)
    # dataset.set_mlm_encoded(False)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['batch_size'], 
        num_workers=max(1, len(os.sched_getaffinity(0)) - 1)
    )

    for i, batch in enumerate(tqdm(dataloader, desc="Inferring by batch")):
        if torch.cuda.is_available():
            batch["input_ids"] = batch["input_ids"].to("cuda")
            batch["padding_mask"] = batch["padding_mask"].to("cuda")
        # Pass the batch through the model
        with torch.no_grad():
            outputs = model(
                x=batch["input_ids"].long(),
                padding_mask=batch["padding_mask"].long(),
            )
        if i % 100 == 0:
            logging.info(f"printing for batch {i}:")
            logging.info(f"len(outputs) = {len(outputs)}")
            logging.info(f"batch length = {len(batch['sequence_id'])}")

        sequence_id = batch["sequence_id"]
        # cls_emb = outputs[:, 0, :].cpu()
        
        padding_mask = batch["padding_mask"].bool()  # Convert to boolean mask
        valid_token_counts = padding_mask.sum(dim=1, keepdim=True)  # Count non-padding tokens
        valid_token_counts = valid_token_counts.clamp(min=1)  # Avoid division by zero
        mean_emb = (outputs * padding_mask.unsqueeze(-1)).sum(dim=1) / valid_token_counts
        mean_emb = mean_emb.cpu()

        data_dict = {"sequence_id": sequence_id, "mean_emb": mean_emb}
        if save_token_embs:
            data_dict['token_embs'] = outputs.cpu()
            data_dict['padding_mask'] = batch['padding_mask'].cpu()

        if i == 0 and Path(write_path).is_file():
            logging.info(f"Replacing file {write_path} with new embeddings.")
            Path(write_path).unlink()

        write_to_hdf5(
            write_path=write_path, 
            data_dict=data_dict, 
            dtype=DTYPE
        )

    if transform_to_parquet:
        write_path = Path(write_path)
        for emb_type in ["mean_emb"]:
            h5_array_to_pq(
                input_path=write_path.parent,
                output_path=write_path.parent,
                emb_filename=write_path.stem,
                emb_type=emb_type,
                id_array="sequence_id",
            )

def load_cfg(cfg_path):
    cfg = read_json(cfg_path)
    _integrity_check(cfg)
    return _with_defaults(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG
    )
    CFG_PATH = sys.argv[1]
    logging.info(CFG_PATH)
    cfg = load_cfg(CFG_PATH)
    os.makedirs(os.path.dirname(cfg['emb_write_path']), exist_ok=False)

    inference(cfg)
