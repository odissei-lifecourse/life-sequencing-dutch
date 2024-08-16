"""Create some pickle files used in the evaluation pipeline

Constants
    - STANDARDIZE: whether the categorical values for birth year and municipality ids should be standardized
    - USE_INT_IDS: whether the person identifiers should be used as integers
    - N_MUNICIPALITIES: number of municipalities
    - RANDOM_SEED: seed for random number generator
    - SAMPLE_SIZE_DRY_RUN: number of samples when doing a dry run
"""

import argparse
import logging
import pandas as pd 
from pathlib import Path
import numpy as np 
from pop2vec.fake_data.utils import batched
from pop2vec.fake_data.fake_data_generator import draw_categorical
import os
import pickle
from pop2vec.llm.src.new_code.utils import read_json


N_MUNICIPALITIES = 342
RANDOM_SEED = 593586236
STANDARDIZE = True
USE_INT_IDS = False
SAMPLE_SIZE_DRY_RUN = 1000


def load_persons(root, id_col="RINPERSOON", sample_size=None, use_ints=False):
    "Load unique person identifiers from a set of sav files"

    files = os.listdir(root)
    all_persons = []
    for f in files:
        filename = Path(root + f)
        df = pd.read_spss(filename, usecols=[id_col])
        persons = np.array(df[id_col])
        if use_ints:
            persons = np.int64(persons)
        if sample_size:
            persons = persons[:sample_size]
        all_persons.append(persons)
    all_persons = np.concatenate(all_persons)
    return np.unique(all_persons)


def power_law_probs(n, rng, alpha=3):
    probs = rng.power(alpha, n)
    probs = probs / probs.sum()
    return probs


def create_marriage_data(persons, years, person_gender_dict, rng, use_ints=False):
    """Create marriage data that are consistent with each other.
    
    Args:
        persons (np.ndarray): population from which to draw the marriage pairs from.
        years (range): years of marriage data to generate
        person_gender_dict: mapping of person identifier to gender

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
        for person, partner in marriages.items():
            # since there is an entry for both person and partner, 
            # we only add the information for the person in one iteration
            person_gender = person_gender_dict[person]
            assert person_gender in [0, 1], "gender has unexpected values"
            if person_gender == 0:
                full_male_list.add(person)
            else:
                full_female_list.add(person)
            
            id_to_gender_map[person] = person_gender

    full_female_list = list(full_female_list)
    full_male_list = list(full_male_list)

    return marriages_by_year, id_to_gender_map, full_female_list, full_male_list


def parse_args():
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

    args = parser.parse_args()
    return args 


def main(cfg, n_obs=None):

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
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for f, data in saving_map.items():
        f += ".pkl"
        filename = save_path.joinpath(f)
        with filename.open("wb") as pkl_file:
            pickle.dump(data, pkl_file)



if __name__ == "__main__":

    args = parse_args()
    SAMPLE_SIZE = SAMPLE_SIZE_DRY_RUN if args.dry_run else None

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging_level
    )

    main(args.cfg, SAMPLE_SIZE)


