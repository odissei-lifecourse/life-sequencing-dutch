"""Convert embeddings from old training runs into hdf5
"""

import json 
import os 
import h5py 
import numpy as np 

RINPERS_ID = "sequence_id"
llm_root = "/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/gen_data/"

def convert_file(data_root: str, filename_out: str):

    emb_file_map = {
        "mean_emb": "mean_embedding_2017_64.json",
        "cls_emb": "cls_embedding_2017_64.json"
    }

    original_embeddings = {}
    for emb_type, emb_file in emb_file_map.items():
        filename = os.path.join(data_root, emb_file)
        with open(filename, "r") as f:
            embs = json.load(f)
        original_embeddings[emb_type] = embs

    mean_keys = list(original_embeddings["mean_emb"].keys())
    cls_keys = list(original_embeddings["cls_emb"].keys())
    assert mean_keys == cls_keys, "embeddings are not ordered in the same way"

    int_keys = [np.int64(x) for x in mean_keys]

    data_out = {}
    data_out[RINPERS_ID] = np.array(int_keys)

    for emb_type, emb_dict in original_embeddings.items():
        embeddings = list(emb_dict.values())
        embeddings = np.array(embeddings)
        embeddings = np.float16(embeddings)
        data_out[emb_type] = embeddings


    filename = os.path.join(data_root, filename_out)
    with h5py.File(filename, "w") as f:
        for dataname, datavalues in data_out.items():
            f.create_dataset(dataname, data=datavalues)


if __name__ == "__main__":
    outfile = "old_embeddings/embedding_2017_64.h5"
    convert_file(llm_root, outfile)