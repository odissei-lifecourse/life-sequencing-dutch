import logging
from pathlib import Path
import torch
from pop2vec.llm.src.new_code.infer_embedding import load_model
from pop2vec.llm.src.new_code.pretrain import read_hparams_from_file
from pop2vec.llm.src.new_code.utils import read_json

cfg_root = "pop2vec/llm/projects/dutch_real/"

cfg_paths = {
    "small": Path(cfg_root, "infer_cfg_small.json"),
    "medium": Path(cfg_root, "infer_cfg_medium.json"),
    "medium2x": Path(cfg_root, "infer_cfg_medium2x.json"),
    "large": Path(cfg_root, "infer_cfg_large.json"),
}


def check_model(cfg):
    """Load a model and count fraction of NaNs in each layer."""
    hparams_path = cfg["HPARAMS_PATH"]
    hparams = read_hparams_from_file(hparams_path)
    checkpoint_path = cfg["CHECKPOINT_PATH"]
    model = load_model(checkpoint_path, hparams)
    model_state_dict = model.state_dict()
    data = {}
    for layer_name, value in model_state_dict.items():
        mean_nan = torch.isnan(value).numpy().mean()
        data[layer_name] = mean_nan
    return data


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG
    )
    summary_stats = {}
    for model_name, cfg_path in cfg_paths.items():
        cfg = read_json(cfg_path)
        summary_stats[model_name] = check_model(cfg)
