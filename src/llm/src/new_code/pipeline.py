import sys

from src.new_code.custom_vocab import CustomVocabulary
from src.new_code.custom_vocab import DataFile
from src.tasks.mlm import MLM
from src.data_new.types import PersonDocument, Background
from src.new_code.utils import get_column_names, print_now, read_json, shuffle_json
from src.new_code.constants import DAYS_SINCE_FIRST, INF

import os
import json
import torch
import numpy as np
import pickle 
import fnmatch
import csv
import time
import logging
import h5py
import subprocess
import multiprocessing as mp
from functools import partial

from multiprocessing import Pool
from tqdm import tqdm

'''
  The pipeline is like the following: 
  1. create life_sequence json files (which should have been already done)
  2. create vocab. Vocab must be created using the same data files as the ones
     used for creating life sequence jsons. 
     TODO: Add a functionality for loading vocab from directory. So we will
     create vocab in create_life_seq_jsons.py ensuring the data file consistency 
  3. read lines one by one and run MLM to get mlmencoded documents
'''

PRIMARY_KEY = "primary_key"
DATA_DIRECTORY_PATH = 'data_directory_path'
VOCAB_NAME = 'vocab_name'
VOCAB_WRITE_PATH = 'vocab_write_path'
TIME_KEY = "TIME_KEY"
SEQUENCE_PATH = "SEQUENCE_PATH"
MLM_WRITE_PATH = "MLM_WRITE_PATH"
TIME_RANGE_START = "TIME_RANGE_START"
TIME_RANGE_END = "TIME_RANGE_END"

min_event_threshold = 5

def read_jsonl_file_in_chunks(file_path, chunk_size, needed_ids, do_mlm):
  """Generator that yields chunks of JSON objects from a JSONL file."""
  with open(file_path, 'r') as file:
    chunk = []
    counter = 0 
    for line in file:
      chunk.append(line)
      if len(chunk) == chunk_size:
        yield chunk, counter
        chunk = []
        counter += 1
    if chunk:  # Yield any remaining JSON objects
        yield chunk, counter

def get_raw_file_name(path):
  return path.split("/")[-1].split(".")[0]

def create_vocab(vocab_write_path, data_file_paths, vocab_name, primary_key):
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
  # uncomment when you are finally saving and loading vocabs.
  # vocab = custom_vocab.vocab()
  # with open(vocab_write_path, 'w') as f:
  #   json.dump(custom_vocab.token2index, f)
  custom_vocab.save_vocab(vocab_write_path)
  logging.debug("Finished create_vocab function")
  return custom_vocab

def get_ids(path):
  with open(path, 'r') as f:
    ids = json.load(f)
  ret_ids = [str(int_id) for int_id in ids]
  return set(ret_ids)

def print_info(start_time, i, total, sequence_id):
  elapsed_time = time.time() - start_time
  done_fraction = (i+1)/total
  print_now(f"time elapsed: {elapsed_time}, ETA: {elapsed_time/(done_fraction) - elapsed_time}")
  print_now(f"done: {i}")
  print_now(f"done%: {done_fraction*100}")
  print_now(f"included: {len(sequence_id)}")
  print_now(f"included%: {len(sequence_id)/(i+1)*100}")

def count_lines(file_path):
  start = time.time()
  result = subprocess.run(
    ['wc', '-l', file_path], 
    text=True, 
    capture_output=True
  )
  line_count = int(result.stdout.split()[0])
  end = time.time()
  logging.info(f"Time needed to wc -l {file_path}: {end-start} seconds")
  return line_count

def load_json_obj(document):
  person_dict = None
  try:
    person_dict = json.loads(document)
  except json.JSONDecodeError:
    logging.error(f"Failed to decode JSON.\n document = {document}")
  except Exception as e:
    logging.error(f"{str(e)}.\n document = {document}")
  return person_dict

def init_data_dict(do_mlm):
  data_dict = {
    'input_ids':[],
    'padding_mask':[],
    'original_sequence':[],
    'sequence_id':[],
  }
  if do_mlm:
    data_dict.update(
      {
        'target_tokens':[],
        'target_pos':[],
        'target_cls':[],
      }
    )
  return data_dict

def update_data_dict(data_dict, output, do_mlm):
  data_dict['sequence_id'].append(str(output.sequence_id))
  data_dict['original_sequence'].append(output.original_sequence)
  data_dict['input_ids'].append(output.input_ids)
  data_dict['padding_mask'].append(output.padding_mask)
  if do_mlm:
    data_dict['target_tokens'].append(output.target_tokens)
    data_dict['target_pos'].append(output.target_pos)
    data_dict['target_cls'].append(output.target_cls)

def convert_to_numpy(data_dict):
  # check if data_dict is empty
  if len(data_dict) == 0:
    return 
  context_len = len(data_dict['original_sequence'][0])

  for key, value in data_dict.items():
    if key == 'sequence_id':
      continue
    if key in ['target_tokens', 'target_pos']:
      np_value = np.full((len(value), context_len), -1)
      for i, row in enumerate(value):
        np_value[i][:len(row)] = row
      value = np_value
    data_dict[key] = np.array(value)

def encode_documents(docs_with_counter, write_path_prefix, needed_ids, do_mlm, mlm):
  docs, counter = docs_with_counter
  print(f"Starting to encode document {counter}",flush=True)
  data_dict = init_data_dict(do_mlm)
  for document in docs:
    person_dict = load_json_obj(document)
    if (
      person_dict is None or len(person_dict['sentence']) < min_event_threshold
    ):
      continue
    person_id = person_dict['person_id']
    if needed_ids is not None and person_id not in needed_ids:
        continue
    person_document = PersonDocument(
      person_id=person_dict['person_id'],
      sentences=person_dict['sentence'],
      abspos=[int(float(x)) for x in person_dict['abspos']],  
      age=[int(float(x)) for x in person_dict['age']],
      segment=person_dict['segment'],
      background=Background(**person_dict['background']),
    )
    output = mlm.encode_document(
      person_document,
      do_mlm=do_mlm,
    )
    if output is None:
      continue
    update_data_dict(data_dict, output, do_mlm)
    
  convert_to_numpy(data_dict)
  write_path = f"{write_path_prefix}.h5" 
  write_to_hdf5(write_path, data_dict)

def init_hdf5_datasets(h5f, data_dict, dtype='i4'):
  """Initialize HDF5 datasets when they do not exist."""
  for key in data_dict:
    if key == 'sequence_id':
      h5f.create_dataset(
        "sequence_id", 
        shape=(0, ), 
        maxshape=(None,), 
        dtype=h5py.special_dtype(vlen=str), 
        chunks=True, 
        compression="gzip"
      )
    else:
      final_shape = list(data_dict[key].shape)
      final_shape[0] = 0
      final_shape = tuple(final_shape)
      
      maxshape = list(data_dict[key].shape)
      maxshape[0] = None
      maxshape = tuple(maxshape)

      if len(data_dict[key].shape) > 1:
        chunks = list(data_dict[key].shape)
        chunks[0] = 1
        chunks = tuple(chunks)
      else:
        chunks=True
      h5f.create_dataset(
        key, 
        shape=final_shape,
        maxshape=maxshape, 
        dtype=dtype, 
        chunks=chunks, 
        compression="gzip"
      )


def debug_log_hdf5(data_dict, h5f):
  logging.debug("data dict shape printing")
  for key, val in data_dict.items():
    if key == 'sequence_id':
      logging.debug("%s, %s", key, len(val))
    else:
      logging.debug("%s, %s", key, val.shape)

  logging.debug("After resize, h5f shape printing")
  for key, val in h5f.items():
    if key == 'sequence_id':
      logging.debug("%s, %s", key, len(val))
    else:
      logging.debug("%s, %s", key, val.shape)

def write_to_hdf5(write_path, data_dict, dtype='i4'):
  """Write processed data to an HDF5 file.

  Args:
  dtype: data types for arrays except the `sequence_id` array. 
   
  """
  if len(data_dict) == 0:
    return
  with h5py.File(write_path, 'a') as h5f:
    if 'sequence_id' not in h5f:
      init_hdf5_datasets(h5f, data_dict, dtype)

    current_size = h5f['sequence_id'].shape[0]
    new_size = current_size + len(data_dict['sequence_id'])
    debug_log_hdf5(data_dict, h5f)
    for key in h5f:
      h5f[key].resize(new_size, axis=0)
      h5f[key][current_size:new_size] = data_dict[key]

def generate_encoded_data(
  custom_vocab,
  sequence_path,
  write_path_prefix,
  time_range=None,
  do_mlm=True,
  needed_ids_path=None,
  shuffle=False,
  chunk_size=5000,
  parallel=True
):
  logging.debug("Starting generate_encoded_data function")
  if needed_ids_path:
    needed_ids = get_ids(needed_ids_path)
    logging.info('needed ids # = %s', len(needed_ids))
    random_id = list(needed_ids)[0]
    logging.info('a random id is %s, type is %s', random_id, type(random_id))
  else:
    needed_ids = None

  if shuffle:
    new_seq_path = sequence_path[:-5] + "_shuffled_work.json"
    if not os.path.isfile(new_seq_path):
      shuffle_json(sequence_path, new_seq_path)
      logging.info("Shuffled json file created")
    else:
      logging.info("Using existing shuffled json file at %s", new_seq_path)
    sequence_path = new_seq_path

  total_docs = count_lines(sequence_path)
  mlm = MLM('dutch_v0', 64)
  mlm.set_vocabulary(custom_vocab)
  if time_range:
    mlm.set_time_range(time_range)
  
  num_processes = max(1, mp.cpu_count()-5)
  logging.info(f"# of processes = {num_processes}")
  chunks = read_jsonl_file_in_chunks(
    sequence_path, 
    chunk_size, 
    needed_ids,
    do_mlm
  )
  progress_bar = tqdm(total=total_docs, desc="Encoding documents", unit="doc")

  helper_encode_documents = partial(
    encode_documents, 
    write_path_prefix=write_path_prefix,
    needed_ids=needed_ids, 
    do_mlm=do_mlm,
    mlm=mlm,
  )

  if parallel:
    logging.info("Starting multiprocessing")
    with Pool(processes=num_processes) as pool:
      for _ in pool.imap_unordered(helper_encode_documents, chunks):
        progress_bar.update(chunk_size)
  else: 
    for chunk in chunks:
      helper_encode_documents(chunk)
      progress_bar.update(chunk_size)
  
  progress_bar.close()
  logging.debug("Finished generate_encoded_data function")


def get_data_files_from_directory(directory, primary_key):
  data_files = []
  for root, dirs, files in os.walk(directory):
    for filename in fnmatch.filter(files, '*.csv'):
      current_file_path = os.path.join(root, filename)
      columns = get_column_names(current_file_path)
      if (
        primary_key in columns and
        ("background" in filename or DAYS_SINCE_FIRST in columns)  
      ): 
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
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
  )
  CFG_PATH = sys.argv[1]
  cfg = read_json(CFG_PATH)

  primary_key = cfg[PRIMARY_KEY]
  sequence_path = cfg[SEQUENCE_PATH]
  vocab_write_path = cfg[VOCAB_WRITE_PATH]
  vocab_name = cfg[VOCAB_NAME]
  time_key = cfg[TIME_KEY]
  mlm_write_path = cfg[MLM_WRITE_PATH]
  data_file_paths = get_data_files_from_directory(
    cfg[DATA_DIRECTORY_PATH], primary_key
  )
  logging.info("# of data_files_paths = %s", len(data_file_paths))

  custom_vocab = create_vocab(
    data_file_paths=data_file_paths,
    vocab_write_path=vocab_write_path,
    vocab_name=vocab_name,
    primary_key=primary_key,
  )

  generate_encoded_data(
    custom_vocab=custom_vocab, 
    sequence_path=sequence_path, 
    write_path_prefix=mlm_write_path,
    time_range=get_time_range(cfg),
    do_mlm=cfg['DO_MLM'],
    needed_ids_path=cfg.get('NEEDED_IDS_PATH', None),
    shuffle=cfg.get('SHUFFLE', False),
    chunk_size=cfg.get('CHUNK_SIZE', 500),
    parallel=cfg.get("PARALLEL", True)
  )
