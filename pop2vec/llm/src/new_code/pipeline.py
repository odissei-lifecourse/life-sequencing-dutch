import csv
import fnmatch
import json
import logging
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
from functools import partial
from multiprocessing import Pool
import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from pop2vec.llm.src.data_new.types import Background
from pop2vec.llm.src.data_new.types import PersonDocument
from pop2vec.llm.src.new_code.constants import DAYS_SINCE_FIRST
from pop2vec.llm.src.new_code.constants import INF
from pop2vec.llm.src.new_code.custom_vocab import CustomVocabulary
from pop2vec.llm.src.new_code.custom_vocab import DataFile
from pop2vec.llm.src.new_code.utils import get_column_names
from pop2vec.llm.src.new_code.utils import print_now
from pop2vec.llm.src.new_code.utils import read_json
from pop2vec.llm.src.new_code.utils import shuffle_json
from pop2vec.llm.src.tasks.mlm import MLM
from pop2vec.utils.merge_hdf5 import merge_hdf5_files

"""
  The pipeline is like the following:
  1. Create life_sequence Parquet files (data is stored in Parquet format)
  2. Create vocab. Vocab must be created using the same data files as the ones
     used for creating life sequence Parquet files.
     TODO: Add functionality for loading vocab from directory.
  3. Read rows from the Parquet file and run MLM to get mlmencoded documents
"""


PRIMARY_KEY = "PRIMARY_KEY"
DATA_PATH = "DATA_PATH"
SEQUENCE_PATH = "SEQUENCE_PATH"
VOCAB_NAME = "VOCAB_NAME"
VOCAB_PATH = "VOCAB_PATH"
ENCODING_WRITE_PATH = "ENCODING_WRITE_PATH"
TIME_RANGE_START = "TIME_RANGE_START"
TIME_RANGE_END = "TIME_RANGE_END"

MIN_EVENT_THRESHOLD = 5  # 12
LOG_THRESHOLD = 100

logging.basicConfig(level=logging.DEBUG)


# Global variables for worker processes
global_custom_vocab = None
global_mlm = None


def worker_initializer(custom_vocab, time_range, max_seq_len):
    global global_custom_vocab
    global global_mlm
    global_custom_vocab = custom_vocab
    global_mlm = MLM("dutch_v0", max_seq_len)
    global_mlm.set_vocabulary(global_custom_vocab)
    if time_range:
        global_mlm.set_time_range(time_range)


def get_raw_file_name(path):
    return os.path.basename(path).split(".")[0]


def create_vocab(
    vocab_write_path,
    data_file_paths,
    vocab_name,
    primary_key,
    num_processes=1,
):
    logging.debug("Starting create_vocab function")
    data_files = []
    for path in data_file_paths:
        data_files.append(
            DataFile(
                path=path,
                name=get_raw_file_name(path),
                primary_key=primary_key,
            )
        )

    custom_vocab = CustomVocabulary(name=vocab_name, data_files=data_files)
    custom_vocab.save_vocab(vocab_write_path, num_processes)
    logging.debug("Finished create_vocab function")
    return custom_vocab


def get_ids(path):
    with open(path) as f:
        ids = json.load(f)
    return set(ids)


def init_data_dict(do_mlm):
    data_dict = {
        "input_ids": [],
        "padding_mask": [],
        "original_sequence": [],
        "sequence_id": [],
    }
    if do_mlm:
        data_dict.update(
            {
                "target_tokens": [],
                "target_pos": [],
                "target_cls": [],
            }
        )
    return data_dict


def update_data_dict(data_dict, output, do_mlm):
    data_dict["sequence_id"].append(output.sequence_id)
    data_dict["original_sequence"].append(output.original_sequence)
    data_dict["input_ids"].append(output.input_ids)
    data_dict["padding_mask"].append(output.padding_mask)
    if do_mlm:
        data_dict["target_tokens"].append(output.target_tokens)
        data_dict["target_pos"].append(output.target_pos)
        data_dict["target_cls"].append(output.target_cls)


def convert_to_numpy(data_dict):
    if len(data_dict["original_sequence"]) == 0:
        return
    context_len = data_dict["original_sequence"][0].shape[0]

    for key, value in data_dict.items():
        if key == "sequence_id":
            data_dict[key] = np.array(value)
            continue
        if key in ["target_tokens", "target_pos"]:
            np_value = np.full((len(value), context_len), -1)
            for i, row in enumerate(value):
                np_value[i][: len(row)] = row
            data_dict[key] = np_value
        else:
            data_dict[key] = np.stack(value)


def encode_documents(row_group_id, parquet_file_path, primary_key, write_path_prefix, needed_ids, do_mlm):
    logging.info(f"Chunk {row_group_id} starting")
    global global_custom_vocab
    global global_mlm  # Use the global MLM object

    data_dict = init_data_dict(do_mlm)

    # Use PyArrow to read only the required rows
    parquet_file = pq.ParquetFile(parquet_file_path)
    columns = [primary_key, "sentence", "abspos", "age", "segment", "background"]

    # Read the specified rows
    table = parquet_file.read_row_group(row_group_id, use_threads=False, columns=columns)
    df = table.to_pandas()
    done_counter = 0
    for i, row in enumerate(df.itertuples()):
        person_id = getattr(row, primary_key)
        if needed_ids is not None and person_id not in needed_ids:
            continue

        sentences = row.sentence
        if len(sentences) < MIN_EVENT_THRESHOLD:
            continue

        person_document = PersonDocument(
            person_id=person_id,
            sentences=[x.tolist() for x in sentences],
            abspos=[int(float(x)) for x in row.abspos],
            age=[int(float(x)) for x in row.age],
            segment=[int(x) for x in row.segment],
            background=Background(**row.background),
        )

        output = global_mlm.encode_document(
            person_document,
            do_mlm=do_mlm,
            do_print=(done_counter % LOG_THRESHOLD == 0),
        )
        if output is None:
            continue
        update_data_dict(data_dict, output, do_mlm)
        done_counter += 1
        if done_counter % LOG_THRESHOLD == 1:
            logging.debug(
                f"""Chunk {row_group_id} -->
            done: {i+1},remaining: {len(df)-i-1}, done% = {(i+1)*100/len(df)}
            created: {done_counter}, created% = {done_counter/(i+1) * 100}"""
            )

    convert_to_numpy(data_dict)
    write_path = f"{write_path_prefix}chunk_{row_group_id}"
    if not do_mlm:
        write_path += "_no_mlm"

    write_path += ".h5"

    if os.path.exists(write_path):
        logging.info("Deleting existing file %s", write_path)
        os.remove(write_path)
    write_to_hdf5(write_path, data_dict, dtype=np.int64)


def init_hdf5_datasets(h5f, data_dict, dtype="i4"):
    """Initialize HDF5 datasets when they do not exist."""
    for key in data_dict:
        if key == "sequence_id":
            h5f.create_dataset(
                "sequence_id",
                data=data_dict[key],
                maxshape=(None,),
                dtype=np.int64,  # h5py.special_dtype(vlen=str),
                chunks=True,
                compression="gzip",
            )
        else:
            h5f.create_dataset(
                key,
                data=data_dict[key],
                maxshape=(None,) + data_dict[key].shape[1:],
                dtype=dtype,
                chunks=True,
                compression="gzip",
            )


def debug_log_hdf5(data_dict, h5f):
    logging.debug("data dict shape printing")
    for key, val in data_dict.items():
        if key == "sequence_id":
            logging.debug("%s, %s", key, len(val))
        else:
            logging.debug("%s, %s", key, val.shape)

    logging.debug("After resize, h5f shape printing")
    for key, val in h5f.items():
        if key == "sequence_id":
            logging.debug("%s, %s", key, len(val))
        else:
            logging.debug("%s, %s", key, val.shape)


def write_to_hdf5(write_path, data_dict, dtype="i4", mode="a"):
    """Write processed data to an HDF5 file.

    Args:
    dtype: data types for arrays except the `sequence_id` array.

    """
    if len(data_dict["sequence_id"]) == 0:
        return
    with h5py.File(write_path, mode) as h5f:
        if "sequence_id" not in h5f:
            init_hdf5_datasets(h5f, data_dict, dtype)

        else:
            current_size = h5f["sequence_id"].shape[0]
            new_size = current_size + len(data_dict["sequence_id"])
            # debug_log_hdf5(data_dict, h5f)
            for key in h5f:
                h5f[key].resize(new_size, axis=0)
                h5f[key][current_size:new_size] = data_dict[key]


def generate_encoded_data(
    custom_vocab,
    sequence_path,
    write_path_prefix,
    primary_key,
    time_range=None,
    do_mlm=True,
    needed_ids_path=None,
    shuffle=False,
    parallel=True,
    max_seq_len=512,
):
    logging.debug("Starting generate_encoded_data function")
    if needed_ids_path:
        needed_ids = get_ids(needed_ids_path)
        logging.info("needed ids # = %s", len(needed_ids))
        random_id = list(needed_ids)[0]
        logging.info("a random id is %s, type is %s", random_id, type(random_id))
    else:
        needed_ids = None

    parquet_file = pq.ParquetFile(sequence_path)
    total_docs = parquet_file.metadata.num_rows

    # TODO: add shuffling option

    # if shuffle:
    #     indices = np.random.permutation(total_docs)
    # else:
    #     indices = np.arange(total_docs)

    if parallel:
        num_processes = len(os.sched_getaffinity(0)) - 2
    else:
        num_processes = 1
    logging.info(f"# of processes = {num_processes}")


    num_row_groups = parquet_file.num_row_groups
    num_workers = min(num_processes, num_row_groups)

    row_group_ids = list(range(num_row_groups))

    helper_encode_documents = partial(
        encode_documents,
        parquet_file_path=sequence_path,
        primary_key=primary_key,
        write_path_prefix=write_path_prefix,
        needed_ids=needed_ids,
        do_mlm=do_mlm,
    )

    logging.info("Starting multiprocessing")
    with Pool(
        processes=num_workers,
        initializer=worker_initializer,
        initargs=(custom_vocab, time_range, max_seq_len),
    ) as pool:
        _ = list(
                tqdm(pool.imap(helper_encode_documents, row_group_ids),
                     total=num_row_groups,
                     desc="Processing row groups",
                     unit="group")
                )

    logging.debug("Finished generate_encoded_data function")


def get_data_files_from_directory(directory, primary_key):
    data_files = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, "*.parquet"):
            current_file_path = os.path.join(root, filename)
            columns = get_column_names(current_file_path)
            if primary_key in columns and ("background" in filename or DAYS_SINCE_FIRST in columns):
                data_files.append(current_file_path)
    return data_files


def get_time_range(cfg):
    time_range = -INF, +INF
    if TIME_RANGE_START in cfg:
        time_range = (cfg[TIME_RANGE_START], time_range[1])
    if TIME_RANGE_END in cfg:
        time_range = (time_range[0], cfg[TIME_RANGE_END])
    return time_range


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
    )
    CFG_PATH = sys.argv[1]
    cfg = read_json(CFG_PATH)

    primary_key = cfg[PRIMARY_KEY]
    vocab_path = cfg[VOCAB_PATH]
    vocab_name = cfg[VOCAB_NAME]
    data_file_paths = get_data_files_from_directory(cfg[DATA_PATH], primary_key)
    logging.info("# of data_files_paths = %s", len(data_file_paths))

    if cfg.get("LOAD_VOCAB", False):
        logging.info(f"Loading Vocab from {vocab_path}")
        custom_vocab = CustomVocabulary(name=vocab_name)
        custom_vocab.load_vocab(vocab_path)
    else:
        logging.info(f"Creating Vocab and saving at {vocab_path}")
        custom_vocab = create_vocab(
            vocab_write_path=vocab_path,
            data_file_paths=data_file_paths,
            vocab_name=vocab_name,
            primary_key=primary_key,
            num_processes=min(65, mp.cpu_count() - 5),
        )
    logging.info("Vocab is ready")
    write_path_prefix = cfg[ENCODING_WRITE_PATH]
    chunk_dir = f"{write_path_prefix}chunks/"
    if not os.path.exists(chunk_dir):
        os.mkdir(chunk_dir)

    generate_encoded_data(
        custom_vocab=custom_vocab,
        sequence_path=cfg[SEQUENCE_PATH],
        write_path_prefix=chunk_dir,
        primary_key=primary_key,
        time_range=get_time_range(cfg),
        do_mlm=cfg["DO_MLM"],
        needed_ids_path=cfg.get("NEEDED_IDS_PATH", None),
        shuffle=cfg.get("SHUFFLE", False),
        parallel=cfg.get("PARALLEL", True),
        max_seq_len=cfg.get("MAX_SEQ_LEN", 512),
    )

    logging.info("Chunks are ready, merging them and deleting")
    chunk_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)]
    mlm = "mlm" if cfg["DO_MLM"] else "no_mlm"
    file_name = f"{mlm}_encoded.h5"
    if "_dryrun" in cfg[SEQUENCE_PATH]:
        file_name = f"{mlm}_encoded_dryrun.h5"
    merge_hdf5_files(chunk_files, f"{write_path_prefix}{file_name}")
    [os.remove(f) for f in chunk_files]
    logging.info("All done.")
