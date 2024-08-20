"""Create hdf5 files with fake embeddings.

Generate HDF5 files containing fake embeddings for different configurations.
Each generated file has one or several arrays of embeddings, and one array of
person identifiers. The person identifers are all stored in string type.

Command-line arguments:
- --dry-run: If provided, runs the script with a reduced dataset size for testing.
- --cfg: Path to the JSON configuration file that specifies input and output paths.

Constants:
- SEED: The random seed used for generating embeddings.
- RINPERS_ID: The identifier of persons.
- LLM_EMBEDDINGS: List of embedding names used in the configuration.
- SAMPLE_SIZE_DRY_RUN: The sample size used when running in dry-run mode.
- EMB_CONFIG: A dictionary containing the embedding configurations, specifying dimensions
  and names for different embedding types.


Configuration File (`cfg` JSON):
- DATA_ROOT: Root directory for data storage.
- INPA_PATH: Relative path to the input persons data.
- SAVE_PATH: Dictionary containing relative paths to save
  network and llm embeddings, respectively.

"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from pop2vec.fake_data.create_eval_pkl_files import load_persons
from pop2vec.llm.src.new_code.pipeline import write_to_hdf5
from pop2vec.llm.src.new_code.utils import read_json

SEED = 52754352
RINPERS_ID = "sequence_id"
LLM_EMBEDDINGS = ["mean_emb", "cls_emb"]
SAMPLE_SIZE_DRY_RUN = 1000
EMB_DIM = 8


EMB_CONFIG = {
    "2017_large":
        {"dim": EMB_DIM, "names": LLM_EMBEDDINGS},
    "2017_medium2x":
        {"dim": EMB_DIM, "names": LLM_EMBEDDINGS},
    "2017_medium":
        {"dim": EMB_DIM, "names": LLM_EMBEDDINGS},
    "2017_small":
        {"dim": EMB_DIM, "names": LLM_EMBEDDINGS},
    "2017_large_v0.0.1":
        {"dim": EMB_DIM, "names": LLM_EMBEDDINGS},
    "2017_medium2x_v0.0.1":
        {"dim": EMB_DIM, "names": LLM_EMBEDDINGS},
    "2017_medium_v0.0.1":
        {"dim": EMB_DIM, "names": LLM_EMBEDDINGS},
    "2017_small_v0.0.1":
        {"dim": EMB_DIM, "names": LLM_EMBEDDINGS},
    "lr_steve_full_network_2010_30":
        {"dim": EMB_DIM, "names": ["embeddings"]},
    "lr_steve_full_network_2020":
        {"dim": EMB_DIM, "names": ["embeddings"]},
}


rng = np.random.default_rng(SEED)


def parse_args(): # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",
                        dest="dry_run",
                        help="If given, runs a test with a small output data size.",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--cfg",
                        type=str,
                        help="Path to config file")

    return parser.parse_args()


def main(cfg, n_obs):
    """Main function.
   
    Args:
        n_obs (int, optional): If provided, this uses only as many person observations
        and saves the embeddings with a "_dry" suffix.
    
    """
    cfg = read_json(cfg)
    data_root = cfg["DATA_ROOT"]
    inpa_path = cfg["INPA_PATH"]
    save_path = cfg["SAVE_PATH"]

    persons = load_persons(root=str(Path(data_root, inpa_path)), sample_size=n_obs)
    n_obs = persons.shape[0]

    for filename, config in tqdm(EMB_CONFIG.items(), desc="Creating embeddings"):
        dim = config["dim"]
        emb_names = config["names"]
        embeddings = {k: rng.random((n_obs, dim)).astype(np.float16) for k in emb_names}

        embeddings["sequence_id"] = persons

        emb_path = save_path["network"] if "network" in filename else save_path["llm"]
        
        h5file = filename + "_dry" if n_obs else filename 
        h5file = Path(data_root, emb_path, h5file).with_suffix(".h5")
        if not h5file.parent.is_dir():
            h5file.parent.mkdir(parents=True)
        write_to_hdf5(str(h5file), embeddings, mode="w")


if __name__ == "__main__":
    args = parse_args()
    SAMPLE_SIZE = SAMPLE_SIZE_DRY_RUN if args.dry_run else None

    main(args.cfg, SAMPLE_SIZE)

