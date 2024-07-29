import pickle
import os
import numpy as np
import h5py
import logging
import argparse

from tqdm import tqdm
from report_utils import load_hdf5, select_from_indexed_array

RINPERS_ID = "sequence_id"
EMB_TYPES_LLM = ["cls_emb", "mean_emb"]
EMB_TYPES_NETWORK = ['embeddings']

NOBS_DRY_RUN = 30
EMB_SIZE_DRY_RUN = 5
EMB_SUBSET_SAVEDIR = "embedding_subsets/"

data_root = "data/processed/"

input_map = {
    "llm_new": {
        "emb_url": "/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/gen_data/embeddings/",
        "emb_types": EMB_TYPES_LLM
    },
    "llm_old": {
        "emb_url": "/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/gen_data/old_embeddings/",
        "emb_types": EMB_TYPES_LLM
    },
    "network": {
        "emb_url": "/gpfs/ostor/ossc9424/homedir/Dakota_network/embeddings/",
        "emb_types": EMB_TYPES_NETWORK
    }
}


def load_and_subset_embeddings(
        embedding_file_url: str,
        id_subsets: dict,
        dry_run: bool,
        embedding_types: list = EMB_TYPES_LLM
):
    """Load full-population embeddings and extract embeddings for different subsets of the population.

    Args:
      embedding_file_url: hdf5 file with embeddings.
      id_subsets: dictionary where the values are the set of person identifiers to be extracted and the 
        keys are the name of this set.
      dry_run: If True, restricts the extracted subsets.
      embedding_types: Which embeddings to extract. Correponds to group names in the embedding file.   
    """
    embedding_data = {}
    n_obs = NOBS_DRY_RUN if dry_run else -1
    emb_size = EMB_SIZE_DRY_RUN if dry_run else -1
    for emb_type in embedding_types:
        ids, embs = load_hdf5(
            emb_url=embedding_file_url,
            id_key=RINPERS_ID, 
            value_key=emb_type, 
            sample_size=n_obs, 
            embedding_size=emb_size)
        if RINPERS_ID not in embedding_data.keys():
            embedding_data[RINPERS_ID] = ids.astype(np.int64)
        embedding_data[emb_type] = embs

    data_dict = {}
    for subset_type, subset_ids in id_subsets.items():
        emb_ids = embedding_data[RINPERS_ID]

        data = {}
        for emb_type in embedding_types:
            ids, embs = select_from_indexed_array(
                ids=emb_ids, 
                array=embedding_data[emb_type], 
                id_subset=subset_ids)
            if RINPERS_ID not in data:
                data[RINPERS_ID] = ids
            data[emb_type] = embs
        data_dict[subset_type] = data

    return data_dict


def save_nested_dict_to_h5(data_dict: dict, filename: str):
    with h5py.File(filename, "w") as h5file:
        for subset_type, subdict in data_dict.items():
            group = h5file.create_group(subset_type)
            for subkey, array in subdict.items():
                group.create_dataset(subkey, data=array)


def main(emb_url: str, emb_types: list, dry_run: bool):
    file_map = {
        "income_eval": "income_eval_subset.pkl",
        "income_model": "income_model_subset.pkl",
        "marriage_eval": "marriage_eval_subset.pkl",
        "marriage_model": "marriage_model_subset.pkl",
        "marriage_rank": "marriage_rank_subset.pkl"
    }

    id_subsets = {}
    for subset_type, filename in file_map.items():
        with open(os.path.join(data_root, filename), "rb") as pkl_file:
            id_subsets[subset_type] = np.array(list(pickle.load(pkl_file)))

    all_files = os.listdir(emb_url)
    filter_func = lambda x: os.path.splitext(x)[1] == ".h5"
    for embedding_file in tqdm(filter(filter_func, all_files)):
        logging.info("Processing %s" % embedding_file)

        embedding_file_url = os.path.join(emb_url, embedding_file)
        data = load_and_subset_embeddings(
            embedding_file_url,
            id_subsets,
            dry_run,
            emb_types
        )

        save_dir = os.path.join(data_root, EMB_SUBSET_SAVEDIR)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if dry_run:
            file_splitted = os.path.splitext(embedding_file)
            embedding_file = file_splitted[0] + "_dry" + file_splitted[1]

        h5_filename = os.path.join(save_dir, embedding_file)
        save_nested_dict_to_h5(data, h5_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", dest="dry_run", action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
    parser.add_argument("--model", 
                        choices=["llm_new", "llm_old", "network"],
                        type=str, 
                        help="Which model embeddings to use")

    args = parser.parse_args()
    logging_level = logging.DEBUG if args.debug else logging.INFO

    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging_level
    )

    emb_url = input_map[args.model]["emb_url"]
    emb_types = input_map[args.model]["emb_types"]
    main(emb_url, emb_types, args.dry_run)