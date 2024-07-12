import pickle 
import os 
import numpy as np 
import h5py 
import logging 
import argparse 

from tqdm import tqdm
from report_utils import load_hdf5

RINPERS_ID = ["sequence_id"]
EMB_TYPES = ["cls_emb", "mean_emb"]

NOBS_DRY_RUN = 30 
EMB_SIZE_DRY_RUN = 5
EMB_SUBSET_SAVEDIR = "embedding_subsets/"

data_root = "data/processed/"
emb_url = "/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/gen_data/embeddings/"

def load_and_subset_embeddings(
        embedding_file_url: str, 
        id_subsets: dict,
        dry_run: bool, 
        embedding_types: list=EMB_TYPES
): 
    embedding_data = {}
    n_obs = NOBS_DRY_RUN if dry_run else -1 
    emb_size = EMB_SIZE_DRY_RUN if dry_run else -1 
    for emb_type in embedding_types: 
        ids, embs = load_hdf5(embedding_file_url, RINPERS_ID, emb_type, n_obs, emb_size)
        if RINPERS_ID not in embedding_data.keys():
            embedding_data[RINPERS_ID] = ids.astype(np.int64)
        embedding_data[emb_type] = embs

    data_dict = {}
    for subset_type, subset_ids in id_subsets.items():
        emb_ids = embedding_data[RINPERS_ID]
        selector = np.where(np.isin(emb_ids, subset_ids))[0]
        data = {}
        data[RINPERS_ID] = subset_ids
        for emb_type in embedding_types:
            embs = embedding_data[emb_type][selector]
            data[emb_type] = embs 
        data_dict[subset_type] = data 
    
    return data


def save_nested_dict_to_h5(data_dict: dict, filename: str):
    with h5py.File(filename, "w") as h5file:
        for subset_type, subdict in data_dict.items():
            group = h5file.create_group(subset_type)
            for subkey, array in subdict.items():
                group.create_dataset(subkey, data=array)


def main(dry_run: bool):
    file_map = {
        "income": "income_subset.pkl",
        "marriage": "marriage_subset.pkl"
    }

    id_subsets = {}
    for subset_type, filename in file_map.items():
        with open(os.path.join(data_root, filename), "rb") as pkl_file:
            id_subsets[subset_type] = np.array(list(pickle.load(pkl_file)))

    for embedding_file in tqdm(os.listdir(emb_url)):
        logging.info("Processing %s" % embedding_file)

        embedding_file_url = os.path.join(emb_url, embedding_file)
        data = load_and_subset_embeddings(
            embedding_file_url,
            id_subsets,
            dry_run
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

    args = parser.parse_args()
    logging_level = logging.DEBUG if args.debug else logging.INFO 

    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging_level
    )

    main(args.dry_run)