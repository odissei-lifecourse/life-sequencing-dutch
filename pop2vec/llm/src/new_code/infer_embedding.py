import json
import logging
import sys
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pop2vec.llm.src.new_code.load_data import CustomInMemoryDataset
from pop2vec.llm.src.new_code.pipeline import write_to_hdf5
from pop2vec.llm.src.new_code.utils import read_hparams
from pop2vec.llm.src.new_code.utils import print_now
from pop2vec.llm.src.new_code.utils import read_json
from pop2vec.llm.src.transformer.models import TransformerEncoder
from pop2vec.utils.convert_hdf5_to_parquet import h5_array_to_pq

DTYPE = np.float64

def load_model(checkpoint_path, hparams_path):
    model = TransformerEncoder.load_from_checkpoint(
        checkpoint_path, 
        hparams=read_hparams(hparams_path) 
    )
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
       name from `cfg["EMB_WRITE_PATH"]` (without the suffix). Moreover, storing
       in parquet requires loading the full set of embeddings into memory, which can
       require a lot of memory. In some situations, it might thus be better to
       do the transformation to parquet in a separate step -- for instance when multiple
       inferences are running on the same node and memory is relatively scarce.
    """
    write_path = cfg["EMB_WRITE_PATH"]
    tokenized_path = cfg["TOKENIZED_PATH"]
    model = load_model(cfg['CHECKPOINT_PATH'], cfg['HPARAMS_PATH'])
    save_token_embs = cfg.get('save_token_embs', False)
    logging.info("Reading from tokenized path: %s", tokenized_path)
    dataset = CustomInMemoryDataset(
        tokenized_path,
        validation=False,
        inference=True,
        mlm_encoded=False
    )
    # dataset.set_mlm_encoded(False)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.get('BATCH_SIZE', 512), 
        num_workers=max(1, len(os.sched_getaffinity(0)) - 2)
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
            print_now(f"printing for batch {i}:")
            print_now(f"len(outputs) = {len(outputs)}")
            print_now(f"batch length = {len(batch['sequence_id'])}")

        sequence_id = batch["sequence_id"]
        cls_emb = outputs[:, 0, :].cpu()

        padding_mask = batch["padding_mask"].bool()  # Convert to boolean mask
        valid_token_counts = padding_mask.sum(dim=1, keepdim=True)  # Count non-padding tokens
        valid_token_counts = valid_token_counts.clamp(min=1)  # Avoid division by zero
        mean_emb = (outputs * padding_mask.unsqueeze(-1)).sum(dim=1) / valid_token_counts
        mean_emb = mean_emb.cpu()

        data_dict = {"sequence_id": sequence_id, "cls_emb": cls_emb, "mean_emb": mean_emb}
        if save_token_embs:
            data_dict['token_embs'] = outputs.cpu()
            data_dict['padding_mask'] = batch['padding_mask'].cpu()

        if i == 0 and Path(write_path).is_file():
            print_now(f"Replacing file {write_path} with new embeddings.")
            Path(write_path).unlink()

        write_to_hdf5(
            write_path=write_path, 
            data_dict=data_dict, 
            dtype=DTYPE
        )

    if transform_to_parquet:
        write_path = Path(write_path)
        for emb_type in ["cls_emb", "mean_emb"]:
            h5_array_to_pq(
                input_path=write_path.parent,
                output_path=write_path.parent,
                emb_filename=write_path.stem,
                emb_type=emb_type,
                id_array="sequence_id",
            )

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG
    )
    CFG_PATH = sys.argv[1]
    print_now(CFG_PATH)
    cfg = read_json(CFG_PATH)
    os.makedirs(os.path.dirname(cfg['EMB_WRITE_PATH']), exist_ok=False)

    inference(cfg)
