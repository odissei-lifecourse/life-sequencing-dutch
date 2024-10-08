# Untested

import csv
from pathlib import Path
from pop2vec.evaluation.report_utils import load_embeddings_from_hdf5

DRY_RUN = False


def write_dict_to_csv(data_dict, file_name):
    """Write dictionary to csv."""
    num_columns = len(next(iter(data_dict.values())))
    header = ["RINPERSOON"] + [f"dim_{i+1}" for i in range(num_columns)]
    with Path.open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(header)
        for person_id, values in data_dict.items():
            writer.writerow([person_id, *values])


def convert_emb_to_csv(filename, emb_name, path_in, path_out):
    """Convert embeddings from hdf5 to csv."""
    sample_size = 4 if DRY_RUN else -1
    embedding_size = 5 if DRY_RUN else -1
    query_keys = ["income_eval"]
    hdf5_file = path_in + filename + ".h5"
    data = load_embeddings_from_hdf5(
        hdf5_file, emb_name, sample_size=sample_size, embedding_size=embedding_size, nested_query_keys=query_keys
    )

    if not Path.exists(path_out):
        Path(path_out).mkdir(parents=True)

    file_out = path_out + filename + ".csv"
    write_dict_to_csv(data, file_out)


def main():
    """Read hdf5 embeddings and save to csv."""
    emb_path = "data/processed/embedding_subsets/"
    save_dir = emb_path + "csvs/"

    emb_files = {"2017_medium2x": "cls_emb", "lr_steve_full_network_2010_30": "embeddings"}
    for f, emb_name in emb_files.items():
        convert_emb_to_csv(f, emb_name, emb_path, save_dir)


if __name__ == "__main__":
    main()
