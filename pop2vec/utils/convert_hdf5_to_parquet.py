import re
from pathlib import Path
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from pop2vec.utils.constants import OSSC_ROOT

DRY_RUN = False
model = "LLM"


if model == "LLM":
    data_origin = "data/llm/embeddings/inference_v0.1"
    data_destination = "data/llm/embeddings/inference_v0.1"

    model_emb_map = [
        ("small", "cls_emb"),
        ("small", "mean_emb"),
        ("medium2x", "cls_emb"),
        ("medium2x", "mean_emb"),
        ("medium", "cls_emb"),
        ("medium", "mean_emb"),
        ("large", "cls_emb"),
        ("large", "mean_emb"),
    ]
elif model == "network":
    data_origin = "Dakota_network/embeddings/"
    data_destination = "data/graph/embeddings/"
    model_emb_map = [("lr_steve_full_network_2010_30", "embeddings"), ("lr_steve_full_network_2020", "embeddings")]


def h5_array_to_pq(input_path, output_path, emb_filename, emb_type="cls_emb", id_array="sequence_id"):
    """Convert hdf5 file into parquet.

    Args:
        input_path (str): Path to the hdf5 file.
        output_path (str): Path to store the parquet file.
        emb_filename (str): Name embedding file, without file suffix.
        emb_type (str): Embedding type. For LLM, "cls_emb" or "mean_emb", for
        network "embeddings".
        id_array (str): Key of the hdf5 array that identifies the unique persons.


    """
    input_path = input_path.joinpath(Path(emb_filename))
    h5file = h5py.File(input_path.with_suffix(".h5"))
    if DRY_RUN:
        identifiers = h5file[id_array][:10_000]
        cls_emb = h5file[emb_type][:10_000]
    else:
        identifiers = h5file[id_array][:]
        cls_emb = h5file[emb_type][:]

    data_dict = {}
    data_dict["rinpersoon_id"] = identifiers.astype(np.int64)

    for idx in range(cls_emb.shape[1]):
        temp_col = pa.array(cls_emb[:, idx].astype(np.float32))
        data_dict[f"emb{idx}"] = temp_col

    data = pa.table(data_dict)
    emb_type_savename = re.sub("_emb", "", emb_type)
    output_path = output_path.joinpath(Path(emb_filename, emb_type_savename))
    output_file = output_path.with_suffix(".parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, output_file)


def main():
    """Iterate over inputs and convert hdf5 embeddings to parquet."""
    source_path = Path(OSSC_ROOT, data_origin)
    write_path = Path(OSSC_ROOT, data_destination)

    for pair in tqdm(model_emb_map):
        model, emb_name = pair
        h5_array_to_pq(source_path, write_path, model, emb_name, "sequence_id")


if __name__ == "__main__":
    main()
