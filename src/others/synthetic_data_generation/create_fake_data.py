"""
Data Generation Script

This script generates fake data based on summary statistics provided in a configuration file. 

Usage:
    Run the script from the command line with the necessary arguments for configuration file path and optional dry run mode.

Args:
    --cfg (str): Path to the configuration file (required).
    --dry-run (bool): If specified, runs a test with a small output data size. 

Example:
    python create_fake_data.py --cfg path/to/config.json --dry-run

Configuration File:
    The configuration file should be a JSON file containing:
        - "SUMMARY_STAT_DIR" (str): Directory containing summary statistics of multiple data files. 
        - "ORIGINAL_ROOT" (str): Path to the root directory where the original data are stored.
        - "NEW_ROOT" (str): Path to the root directory where the fake data are stored.
        
Constants:
    SAMPLE_SIZE_DRY_RUN (int): Default sample size for dry run mode.
    RANDOM_SEED (int): Seed for random number generation.
"""


import argparse
import numpy as np 
import os
from tqdm import tqdm

from src.others.synthetic_data_generation.fake_data_generator import FakeDataGenerator
from src.others.synthetic_data_generation.utils import get_unique_source_files
from src.llm.src.new_code.utils import read_json


SAMPLE_SIZE_DRY_RUN = 1_000
RANDOM_SEED = 935723583


def main(cfg, n_observations=None):
    """Generate data
    
    Args
        cfg (dict): configuration file
        n_observations (int, optional): size of the data to generate. If `None` (the default), size is taken from
        the summary statistics.
    """
    cfg = read_json(cfg)

    rng = np.random.default_rng(seed=RANDOM_SEED)
    src_files = get_unique_source_files(cfg["SUMMARY_STAT_DIR"])

    generator = FakeDataGenerator()
    for src_file in tqdm(src_files):
        url, filename = os.path.split(src_file)
        generator.load_metadata(url=url, filename=filename)
        generator.fit()
        data = generator.generate(rng=rng, size=n_observations)
        generator.save(
            original_root=cfg["ORIGINAL_ROOT"],
            new_root=cfg["NEW_ROOT"], 
            data=data
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", 
                        dest="dry_run", 
                        help="If given, runs a test with a small output data size.",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--cfg",
                        type=str,
                        help="Path to config file")

    args = parser.parse_args()

    n_obs = None
    if args.dry_run:
        n_obs = SAMPLE_SIZE_DRY_RUN
    else:
        raise NotImplementedError

    main(cfg=args.cfg, n_observations=n_obs)


    
    

