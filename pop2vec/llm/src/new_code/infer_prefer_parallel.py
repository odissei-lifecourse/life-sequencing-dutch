# This does not work on OSSC due to PL parallel and OSSC security not agreeing with eachother

import json
import logging
import sys
import os
import csv
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pop2vec.llm.src.new_code.load_data import CustomInMemoryDataset
from pop2vec.llm.src.new_code.utils import read_json
from pop2vec.llm.src.transformer.cls_model import Transformer_CLS
from pop2vec.llm.src.new_code.utils import read_hparams
from pop2vec.llm.src.new_code.utils import print_now


# from pop2vec.llm.src.new_code.utils import print_now  # Uncomment if you want to use a custom logger

def load_model(checkpoint_path, hparams_path):
    """Load and return a finetuned binary classification model.
    
    """
    model = Transformer_CLS.load_from_checkpoint(
        checkpoint_path, 
        # hparams=read_hparams(hparams_path) 
    )
    # model = model.transformer
    model.eval()
    device = str(next(model.parameters()).device)
    print_now(f"Model is on {device}")
    logging.info(f"Embedding sample after final load: {model.transformer.embedding.token.weight[1, 0].detach()}, {model.transformer.embedding.token.weight[198, 0].detach()}")
    return model

def inference(cfg):
    """Run inference on a finetuned binary classification model.

    The model is expected to output two logits (for classes [0, 1]). 
    We apply softmax to get probabilities and write the probability for 
    the positive class (index 1) to a CSV file.

    Args:
        cfg (dict): Configuration containing:
            - CHECKPOINT_PATH (str): Path to the model checkpoint.
            - HPARAMS_PATH (str): Path to hyperparameters file.
            - CSV_WRITE_PATH (str): Where the CSV results should be written.
            - TOKENIZED_PATH (str): Path to the dataset file (HDF5 or similar).
            - BATCH_SIZE (int, optional): Batch size for DataLoader.
    """
    # 1. Prepare to write CSV
    csv_write_path = cfg["CSV_WRITE_PATH"]
    os.makedirs(os.path.dirname(csv_write_path), exist_ok=True)

    # 2. Load model
    model = load_model(
        checkpoint_path=cfg['CHECKPOINT_PATH'], 
        hparams_path=cfg['HPARAMS_PATH']
    )
    model.eval()

    #  Wrap the model in DataParallel if multiple GPUs are available
    #  and you want to run on all of them.
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference!")
        model = torch.nn.DataParallel(model)  # uses all available GPUs by default
    
    if torch.cuda.is_available():
        model.to("cuda")

    # 3. Read dataset
    dataset = CustomInMemoryDataset(
        cfg["TOKENIZED_PATH"],
        validation=False,
        inference=True,
        mlm_encoded=False,              
        num_val_items=0,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.get('BATCH_SIZE', 512), 
        num_workers=0  # Adjust as needed
    )

    # 4. Collect results: sequence_id (RINPERSOON) and Probability
    results = []

    # 5. Inference loop
    for counter, batch in tqdm(
        enumerate(dataloader), desc="Inferring by batch"
    ):
        # Move to CUDA if available
        if torch.cuda.is_available():
            batch["input_ids"] = batch["input_ids"].to("cuda").long()
            batch["padding_mask"] = batch["padding_mask"].to("cuda").long()
            model.to("cuda")

        with torch.no_grad():
            # Model is expected to output two logits: shape [batch_size, 2]
            logits = model(batch)
            # Compute probabilities for the positive class (index=1)
            probs = torch.softmax(logits, dim=1)[:, 1]

        sequence_ids = batch["sequence_id"].tolist()
        probs = probs.cpu().tolist()

        # Store results
        for seq_id, prob in zip(sequence_ids, probs):
            results.append((seq_id, prob))

        if counter % 60 == 0:
            with open(csv_write_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                # Write header
                if counter == 0:
                    writer.writerow(["RINPERSOON", "probability"])
                # Write rows
                for seq_id, prob in results:
                    writer.writerow([seq_id, prob])
                results = []
    # 6. Write results to CSV


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG
    )
    CFG_PATH = sys.argv[1]
    cfg = read_json(CFG_PATH)
    inference(cfg)
