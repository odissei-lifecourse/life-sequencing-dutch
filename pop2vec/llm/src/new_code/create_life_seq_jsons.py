from src.new_code.create_person_dict import CreatePersonDict
from src.new_code.utils import get_column_names, print_now, read_json
from src.new_code.constants import DAYS_SINCE_FIRST, INF

import os
import json
import torch
import numpy as np
import fnmatch
import csv
import sys
import time

PRIMARY_KEY = "primary_key"
DATA_DIRECTORY_PATH = 'data_directory_path'
TIME_KEY = "TIME_KEY"
SEQUENCE_WRITE_PATH = "SEQUENCE_WRITE_PATH"


def create_person_sequence(file_paths, custom_vocab, write_path, primary_key):
  #create person json files
  creator = CreatePersonDict(
    file_paths=file_paths, 
    primary_key=primary_key, 
    vocab=custom_vocab,
  )
  creator.generate_people_data(write_path)

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

if __name__ == "__main__":
  CFG_PATH = sys.argv[1]
  cfg = read_json(CFG_PATH)
  
  data_file_paths = get_data_files_from_directory(
    cfg[DATA_DIRECTORY_PATH], 
    cfg[PRIMARY_KEY]
  )
  print_now(f"# of data_files_paths = {len(data_file_paths)}")

  create_person_sequence(
    file_paths=data_file_paths, 
    custom_vocab=None, #custom_vocab, 
    write_path=cfg[SEQUENCE_WRITE_PATH],
    primary_key=cfg[PRIMARY_KEY],
  )