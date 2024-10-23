import fnmatch
import json
import math
import os
import shutil
import sys
from functools import partial
import logging
from typing import Dict, Union, List


import numpy as np
import pandas as pd

from pop2vec.llm.src.new_code.constants import (
    AGE,
    DAYS_SINCE_FIRST,
    INF,
    MISSING,
)
from pop2vec.llm.src.new_code.utils import (
  get_column_names, 
  print_now, 
  load_parquet_with_metadata,
  load_csv_and_create_metadata,
  transform_to_percentiles,
  replace_less_frequent
)

# Set up logging configuration
logging.basicConfig(
  level=logging.INFO, 
  format='%(asctime)s - %(levelname)s - %(message)s'
)

SOURCE = 'SOURCE'
DEST = 'DEST'
IMMUTABLE_COLS = 'IMMUTABLE_COLS'
PRIMARY_KEY = 'PRIMARY_KEY'
RESTRICTED_SUBSTRINGS = 'RESTRICTED_SUBSTRINGS'
DELIMITER = 'DELIMITER'
CATEGORICAL_THRESHOLD = 'CATEGORICAL_THRESHOLD'

# Global variables
all_cols = {}
all_primaries = set()

def read_cfg(path):
  """Reads a JSON configuration file.

  Args:
    path (str): Path to the JSON configuration file.

  Returns:
    dict: Configuration dictionary loaded from the JSON file.
  """
  with open(path, 'r') as file:
    cfg = json.load(file)
  return cfg

def process_and_write(
    df,
    meta_dict,
    write_path,
    primary_key,
    immutable_cols,
    restricted_substrings,
):
  """Processes a CSV file and writes the modified data.

  Args:
    df: dataframe containing the main data
    meta_dict: dict containing metadata
    write_path: where df will be written
    primary_key (str): Name of the primary key column, e.g., RINPERSOON in 
      dutch data
    immutable_cols (list): These columns cannot be modified.
    restricted_substrings (list): Any column containing any of the substrings 
      will be dropped
  """
  
  global all_primaries
  
  if 'background' in write_path:
    logging.info(f"background file {write_path} is written as it is.")
    df.to_parquet(write_path)
    return
  
  logging.info(f"preparing {write_path}")
  logging.info("*" * 100)
  
  all_primaries = all_primaries | set(df[primary_key].unique())
  
  #drop any row that has nan in time columns
  df.dropna(subset=[DAYS_SINCE_FIRST, AGE], inplace=True)
  
  # convert time columns to int
  df[AGE] = df[AGE].round().astype(int)
  df[DAYS_SINCE_FIRST] = df[DAYS_SINCE_FIRST].round().astype(int)
  
  # fill all missing values (nan) by MISSING
  df.fillna("MISSING", inplace=True)

  for column in df.columns:
    if column == primary_key or column in immutable_cols:
      continue
    elif any(item in column for item in restricted_substrings):
      logging.info(f"dropping column = {column}; contains restriced substring")
      df.drop(columns=[column], inplace=True)
    elif meta_dict[column] == 'Numeric':
      logging.info(f"transforming {column} column to percentiles")
      df[column] = transform_to_percentiles(df[column])
    elif meta_dict[column] == 'String':
      logging.info(
          f"keeping categorical column {column} with "
          f"{len(df[column].unique())} items"
      )
      df[column] = replace_less_frequent(
        df[column], 
        keep_top_n = 100, 
        name_others = "Others",
        ignore_list = ['MISSING'],
      )
    else:
      logging.error(f"column {column} with write_path = {write_path} has malformed meta_dict {meta_dict} ")
  
  for col in df.columns:
    if col not in all_cols:
      all_cols[col] = []
    all_cols[col].append(write_path)
  
  df.to_parquet(write_path, index=False)

def get_csv_data(
  input_directory: str,
  output_directory: str, 
  delimiter: str, 
  categorical_threshold: int
) -> List[Dict[str, Union[pd.DataFrame, Dict, str]]]:
  """Process all CSV files in a given directory.
  
  Args:
    input_directory: The directory to search for CSV files.
    output_directory: where processed parquet files will be written.
    delimiter: delimiter for csv files
    categorical_threshold: numerical columns having unique values less than
      the threshold will be treated as categorical columns.
  """
  ret = []
  for root, _, files in os.walk(input_directory):
    for file in files:
      if file.endswith(".csv"):
        csv_path = os.path.join(root, file)
        df, meta = load_csv_and_create_metadata(
          csv_path, 
          delimiter, 
          categorical_threshold
        )
        ret.append({
            'dataframe': df, 
            'meta_dict': meta,
            'write_path': os.path.join(
              output_directory, 
              file.replace('.csv', '.parquet')
            )
        })
  
  return ret

def _get_data_and_metadata_from_files(files: List[str], root: str):
  # Raise an error if the number of files is not exactly 2
  if len(files) != 2:
    raise ParquetFileError(
        f"Directory '{root}' contains {len(files)} files. "
        f"Expected exactly 2 parquet files."
    )
  data_file = None
  metadata_file = None
  for file in files:
    if file.endswith("_meta.parquet"):
      metadata_file = file
    elif file.endswith(".parquet"):
      data_file = file
    else:
      raise ValueError(f"{root} contains {file} that is not a parquet file.")

  if data_file is None or metadata_file is None:
    raise ValueError(
      f"""Directory '{root}' is missing either a parquet or metadata file.
      List of files found: {files}"""
    )

  if (
      os.path.splitext(parquet_file)[0] == 
      os.path.splitext(metadata_file)[0].replace("_meta", "")
  ) is False:  
    raise ValueError(
      f"Directory '{root}' contains parquet files but their names do not match the required pattern."
    )
  
  return data_file, metadata_file

def get_parquet_data(
  input_directory: str,
  output_directory: str,
) -> List[Dict[str, Union[pd.DataFrame, Dict, str]]]:
  ret = []
  for root, _, files in os.walk(input_directory):
    if root == input_directory:
      continue
    data_file, metadata_file = _get_data_and_metadata_from_files(files, root)
    df, meta = load_parquet_with_metadata(data_file, metadata_file)
    ret.append(
      {
        'dataframe': df, 
        'meta_dict': meta,
        'write_path': os.path.join(
          output_directory, 
          os.path.basename(data_file)
        )
      }
    )
  
  return ret

def main():
  cfg_path = sys.argv[1]
  logging.info(f"cfg_path = {cfg_path}")
  cfg = read_cfg(cfg_path)

  source_dir = cfg[SOURCE]
  destination_dir = cfg[DEST]
  immutable_cols = cfg.get(IMMUTABLE_COLS, [])
  primary_key = cfg[PRIMARY_KEY]
  restricted_substrings = cfg.get(RESTRICTED_SUBSTRINGS, [])
  categorical_threshold = cfg.get(CATEGORICAL_THRESHOLD, 100)
  delimiter = cfg.get(DELIMITER, ',')

  if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)  

  data_dicts = get_csv_data(
    source_dir, 
    destination_dir,
    delimiter,
    categorical_threshold,
  )
  data_dicts.extend(get_parquet_data(source_dir, destination_dir))
  
  for data_dict in data_dicts:
    logging.info(f"preparing {data_dict['write_path']}")
    process_and_write(
      df=data_dict['dataframe'],
      meta_dict=data_dict['meta_dict'],
      write_path=data_dict['write_path'],
      primary_key=primary_key,
      immutable_cols=immutable_cols,
      restricted_substrings=restricted_substrings,
    )
    logging.info(f"{data_dict['write_path']} is written")

  for col in all_cols:
    logging.info(
      f"""Column {col} is found in {len(all_cols[col])} files."
      the files are:  {all_cols[col]}"""
    )

  logging.info(f"# of people: {len(all_primaries)}")

if __name__ == "__main__":
  main()
