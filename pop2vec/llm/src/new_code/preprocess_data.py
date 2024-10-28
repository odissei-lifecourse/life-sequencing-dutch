import fnmatch
import json
import math
import os
import shutil
import sys
from functools import partial
import logging
from typing import Dict, Union, List
import gc

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
REPLACE_OLD = 'REPLACE_OLD'
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
  """Processes a DataFrame and writes the modified data in parquet.

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
    df.to_parquet(write_path, index=False)
    return
  
  
  #drop any row that has nan in time columns
  df.dropna(subset=[DAYS_SINCE_FIRST, AGE], inplace=True)
  
  # convert time columns to int
  # df[AGE] = df[AGE].round().astype(int)
  # df[DAYS_SINCE_FIRST] = df[DAYS_SINCE_FIRST].round().astype(int)
  
  # faster conversion below
  df[AGE] = pd.to_numeric(df[AGE], errors='coerce').round(0).astype(int, copy=False)
  df[DAYS_SINCE_FIRST] = pd.to_numeric(df[DAYS_SINCE_FIRST], errors='coerce').round(0).astype(int, copy=False)

  # fill all missing values (nan) by MISSING
  df.fillna(MISSING, inplace=True)

  for column in df.columns:
    if (
      column == primary_key or 
      column in immutable_cols or 
      column in [AGE, DAYS_SINCE_FIRST]
    ):
      continue
    elif any(item in column for item in restricted_substrings):
      logging.info(f"dropping column = {column}; contains restricted substring")
      df.drop(columns=[column], inplace=True)
    elif meta_dict[column] == 'Numeric':
      logging.info(
        f"transforming numeric column {column} with {df[column].nunique()} unique values to percentiles"
      )
      transform_to_percentiles(df, column, True)
      logging.info(
        f"{column} has {df[column].nunique()} unique values after transformation"
      )
    elif meta_dict[column] == 'String':
      logging.info(
        f"categorical column {column} has {len(df[column].unique())} unique values initially\n"
      )
      df[column] = replace_less_frequent(
        df[column], 
        keep_top_n = 100, 
        name_others = "Others",
        ignore_list = [MISSING],
      )
      logging.info(
        f"keeping the top {len(df[column].unique())} unique values"
      )
    else:
      logging.error(f"column {column} with write_path = {write_path} has malformed meta_dict {meta_dict} ")
  
  for col in df.columns:
    if col not in all_cols:
      all_cols[col] = []
    all_cols[col].append(write_path)
    if col not in [primary_key, AGE, DAYS_SINCE_FIRST]:
      df[col] = df[col].astype(str)
  
  all_primaries = all_primaries | set(df[primary_key].unique())
  df.to_parquet(write_path, index=False)
  del df
  gc.collect()

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
        ret.append({
            'input_csv_path': csv_path, 
            'write_path': os.path.join(
              output_directory, 
              file.replace('.csv', '.parquet')
            ),
            'delimiter': delimiter,
            'categorical_threshold': categorical_threshold,
            'type': 'csv',
        })
  
  return ret

# def _get_data_and_metadata_from_files(files: List[str], root: str):
#   """Retrieves data and metadata file paths from a list of files.

#   Args:
#       files (List[str]): List of file names under root.
#       root (str): Root directory path.

#   Returns:
#       Tuple[str, str]: Tuple containing data file path and metadata file path.

#   Raises:
#       ValueError: If the directory does not contain exactly two required files.
#   """
#   # Raise an error if the number of files is not exactly 2
#   if len(files) != 2:
#     raise ValueError(
#         f"Directory '{root}' contains {len(files)} files. "
#         f"Expected exactly 2 parquet files."
#     )
#   data_file = None
#   metadata_file = None
#   for file in files:
#     if file.endswith("_meta.parquet"):
#       metadata_file = file
#     elif file.endswith(".parquet"):
#       data_file = file
#     else:
#       raise ValueError(f"{root} contains {file} that is not a parquet file.")

#   if data_file is None or metadata_file is None:
#     raise ValueError(
#       f"""Directory '{root}' is missing either a parquet or metadata file.
#       List of files found: {files}"""
#     )

#   if (
#       os.path.splitext(data_file)[0] != 
#       os.path.splitext(metadata_file)[0].replace("_meta", "")
#   ):  
#     raise ValueError(
#       f"Directory '{root}' contains parquet files but their names do not match the required pattern."
#     )
  
#   return data_file, metadata_file

def get_parquet_data(
  input_directory: str,
  output_directory: str,
) -> List[Dict[str, Union[pd.DataFrame, Dict, str]]]:
  """Processes all Parquet files in a given directory.

  Args:
      input_directory (str): The directory to search for Parquet files.
      output_directory (str): Where processed Parquet files will be written.

  Returns:
      A list of dictionaries containing file paths and types.
  """
  ret = []
  for root, _, files in os.walk(input_directory):
    for f1 in files:
      if f1.endswith('_meta.parquet'):
        metadata_file = f1
        data_file = None
        for f2 in files:
          if f2 == f1.replace('_meta.parquet', '.parquet'):
            data_file = f2
            break
        if data_file is None:
          logging.info(
            f"Found metadata file {metadata_file} but no corresponding data file found"
          )
        else:
          ret.append(
            {
              'input_data_parquet_path': os.path.join(root, data_file),
              'input_meta_parquet_path': os.path.join(root, metadata_file), 
              'write_path': os.path.join(
                output_directory, 
                os.path.basename(data_file)
              ),
              'type': 'parquet'
            }
          )

    # if root.endswith('_parquet'):
    #   data_file, metadata_file = _get_data_and_metadata_from_files(files, root)
    #   ret.append(
    #     {
    #       'input_data_parquet_path': os.path.join(root, data_file),
    #       'input_meta_parquet_path': os.path.join(root, metadata_file), 
    #       'write_path': os.path.join(
    #         output_directory, 
    #         os.path.basename(data_file)
    #       ),
    #       'type': 'parquet'
    #     }
    #   )
  
  return ret

def load_data(path_dict, primary_key):
  try:
    if path_dict['type'] == 'csv':
      logging.info(f"reading {path_dict['input_csv_path']}")
      df, meta = load_csv_and_create_metadata(
        path_dict['input_csv_path'], 
        path_dict['delimiter'], 
        path_dict['categorical_threshold'],
        primary_key,
      )
    elif path_dict['type'] == 'parquet':
      logging.info(f"reading {path_dict['input_data_parquet_path']}")
      df, meta = load_parquet_with_metadata(
        path_dict['input_data_parquet_path'], 
        path_dict['input_meta_parquet_path'],
        primary_key,
      )
    else:
      raise ValueError(
        f"path_dict['type'] should be either 'csv' or 'parquet', found {path_dict['type']}"
      )
    return df, meta

  except Exception as e:
    logging.error(
      f"An error occurred while preparing {path_dict['write_path']}:\n {e}"
    )
    return None, None


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
  replace_old_data = cfg.get(REPLACE_OLD, True)
  if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)  

  path_dicts = get_csv_data(
    source_dir, 
    destination_dir,
    delimiter,
    categorical_threshold,
  )
  path_dicts.extend(get_parquet_data(source_dir, destination_dir))
  
  for path_dict in path_dicts:
    if os.path.exists(path_dict['write_path']):
      if replace_old_data:
        logging.info(f"Replacing already existing {path_dict['write_path']}.")
      else:
        logging.info(f"{path_dict['write_path']} already exists. Not replacing file and continuing.")
        continue
    df, meta = load_data(path_dict, primary_key)
    if df is not None:
      process_and_write(
        df=df,
        meta_dict=meta,
        write_path=path_dict['write_path'],
        primary_key=primary_key,
        immutable_cols=immutable_cols,
        restricted_substrings=restricted_substrings,
      )
      logging.info(f"{path_dict['write_path']} is written")

  for col in all_cols:
    logging.info(
        f"Column {col} is found in {len(all_cols[col])} files. "
        f"The files are: {all_cols[col]}"
    )

  logging.info(f"# of people: {len(all_primaries)}")

if __name__ == "__main__":
  main()
