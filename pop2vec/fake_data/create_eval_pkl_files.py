"""Create pickle files for use in the evaluation pipeline.

This script generates and saves various data structures related to person demographics,
marriages, and municipalities. The data is artificial and used for evaluation purposes.

Constants:
    STANDARDIZE (bool): Whether to standardize categorical values for birth year and municipality IDs.
    USE_INT_IDS (bool): Whether to use integer person identifiers.
    N_MUNICIPALITIES (int): Number of municipalities to generate.
    RANDOM_SEED (int): Seed for the random number generator to ensure reproducibility.
    SAMPLE_SIZE_DRY_RUN (int): Number of samples to use when performing a dry run.

Key Functionalities:
    - Loads unique person identifiers from SPSS files.
    - Generates fake demographic data (gender, birth year, birth municipality).
    - Creates synthetic marriage data across multiple years.
    - Saves generated data as pickle files for further use.

Usage:
    Run the script with appropriate command-line arguments:
    --cfg: Path to the configuration file (required)
    --dry-run: Optional flag to run a test with a small output data size
    --debug: Optional flag to set logging level to DEBUG

Output:
    Several pickle files containing generated data, saved in the directory specified in the config file.

Docstring generated with the help of Claude.ai.
"""


import argparse
import logging
import os
from tqdm import tqdm
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from pop2vec.fake_data.fake_data_generator import draw_categorical
from pop2vec.fake_data.utils import batched
from pop2vec.llm.src.new_code.utils import read_json

N_MUNICIPALITIES = 342
RANDOM_SEED = 593586236
STANDARDIZE = True
USE_INT_IDS = False
SAMPLE_SIZE_DRY_RUN = 1000


def load_persons(root, id_col="RINPERSOON", sample_size=None, use_ints=False, single_file=True):
    """Load unique person identifiers from a set of sav files.

    Args:
        root (str or Path): directory containing the sav files.
        id_col (str): column name with person identifiers.
        sample_size (int, optional): if given, samples the first `sample_size`
        persons from the first sav file.
        use_ints (bool, optional): if True, convert person identifiers to integer.
        single_file (bool, optional): If True, only loads persons from the first file.
        Currently, reading only one file is enough because the set of person IDs
        does not change over time. But this may change if we extend the
        fake data.

    """
    files = os.listdir(root)
    if single_file:
        files = [files[0]]
    all_persons = []
    if isinstance(root, str):
        root = Path(root)
    for f in tqdm(files, desc="loading persons"):
        filename = root.joinpath(f)
        person_df = pd.read_spss(filename, usecols=[id_col])
        persons = np.array(person_df[id_col])
        if use_ints:
            persons = np.int64(persons)
        if sample_size:
            persons = persons[:sample_size]
        all_persons.append(persons)
    all_persons = np.concatenate(all_persons)
    return np.unique(all_persons)


def power_law_probs(n, rng, alpha=3):
    """Create choice probabilies according to a power law."""
    probs = rng.power(alpha, n)
    return probs / probs.sum()


def create_marriage_data(persons, years, person_gender_dict, rng, use_ints=False):
    """Create marriage data that are consistent with each other.

    Args:
        persons (np.ndarray): population from which to draw the marriage pairs from.
        years (range): years of marriage data to generate.
        person_gender_dict (dict): mapping of person identifier to gender.
        rng (np.random.default_rng): random number generator.
        use_ints (bool, optional): if True, store person identifiers as np.int64.

    Returns:
        tuple of (
            marriages_by_year: dict,
            id_to_gender_map: dict,
            full_female_list: list,
            full_male_list: list
        )
        marriages_by_year[year] is a dictionary where each married person has a key, with the value
            being the identifier of the partner.
        id_to_gender_map: maps the identifier of the married person to the gender of the partner.
        full_female_list, full_male_list: list of all men and women that got married at some point.
    """
    n_marriages_per_year = rng.choice(range(12_000, 18_000), len(years))
    marriages_by_year = {}
    for myear, n in zip(list(years), n_marriages_per_year):
        current_persons = np.int64(persons).copy()
        rng.shuffle(current_persons)
        marriages = list(batched(current_persons, 2))
        marriages = marriages[:n]
        yearly_dict = {}
        for m in marriages:
            person, partner = m
            if not use_ints:
                person = str(person)
                partner = str(partner)
            yearly_dict[person] = partner
            yearly_dict[partner] = person
        marriages_by_year[myear] = yearly_dict


    full_male_list = set() # 0
    full_female_list = set() # 1
    id_to_gender_map = {}
    for marriages in marriages_by_year.values():
        for person in marriages:
            # since there is an entry for both person and partner,
            # we only add the information for the person in one iteration
            person_gender = person_gender_dict[person]
            assert person_gender in [0, 1], "gender has unexpected values" # noqa: S101
            if person_gender == 0:
                full_male_list.add(person)
            else:
                full_female_list.add(person)

            id_to_gender_map[person] = person_gender

    full_female_list = list(full_female_list)
    full_male_list = list(full_male_list)

    return marriages_by_year, id_to_gender_map, full_female_list, full_male_list


def parse_args(): # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",
                        dest="dry_run",
                        help="If given, runs a test with a small output data size.",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--cfg",
                        type=str,
                        help="Path to config file")
    parser.add_argument("--debug",
                        action=argparse.BooleanOptionalAction)

    return parser.parse_args()


def main(cfg, n_obs=None):# noqa: D103

    cfg = read_json(cfg)

    start, end = tuple(cfg["YEARS"])
    years = range(start, end)

    rng = np.random.default_rng(RANDOM_SEED)
    persons = load_persons(root=cfg["INPATAB_DIR"], sample_size=n_obs, use_ints=USE_INT_IDS)

    person_gender = draw_categorical(persons=persons, rng=rng, values=[0, 1], dtype="float64")

    set_of_munipalities = np.arange(N_MUNICIPALITIES)
    person_birth_municipality = draw_categorical(
        persons=persons,
        values=set_of_munipalities,
        probs=power_law_probs(N_MUNICIPALITIES, rng),
        rng=rng,
        dtype="float64",
        standardize=STANDARDIZE)

    set_of_birth_years = np.arange(1940, 2010)
    person_birth_year = draw_categorical(
        persons=persons,
        values=set_of_birth_years,
        rng=rng,
        dtype="float64",
        standardize=STANDARDIZE
    )

    marriage_data = create_marriage_data(
        persons=persons, years=years, person_gender_dict=person_gender, rng=rng, use_ints=USE_INT_IDS
    )
    marriages_by_year, id_to_gender_map, full_female_list, full_male_list = marriage_data

    saving_map = {
        "marriages_by_year": marriages_by_year,
        "id_to_gender_map": id_to_gender_map,
        "full_female_list": full_female_list,
        "full_male_list": full_male_list,
        "person_birth_year": person_birth_year,
        "person_birth_municipality": person_birth_municipality,
        "person_gender": person_gender
    }


    save_path = Path(cfg["SAVE_DIR"])
    if not save_path.exists():
        save_path.mkdir(parents=True)

    for f, data in saving_map.items():
        filename = save_path.joinpath(f).with_suffix(".pkl")
        with filename.open("wb") as pkl_file:
            pickle.dump(data, pkl_file)



if __name__ == "__main__":

    args = parse_args()
    SAMPLE_SIZE = SAMPLE_SIZE_DRY_RUN if args.dry_run else None

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level
    )

    main(args.cfg, SAMPLE_SIZE)


