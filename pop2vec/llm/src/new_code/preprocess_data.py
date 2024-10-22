import fnmatch
import json
import math
import os
import shutil
import sys
from functools import partial
import logging

import numpy as np
import pandas as pd

from pop2vec.llm.src.new_code.constants import (
    AGE,
    DAYS_SINCE_FIRST,
    FIRST_EVENT_TIME,
    INF,
    MISSING,
)
from pop2vec.llm.src.new_code.utils import get_column_names, print_now

# Set up logging configuration
logging.basicConfig(
  level=logging.INFO, 
  format='%(asctime)s - %(levelname)s - %(message)s'
)

TIME_KEY = 'TIME_KEY'
SOURCE = 'SOURCE'
DEST = 'DEST'
IMMUTABLE_COLS = 'IMMUTABLE_COLS'
PRIMARY_KEY = 'PRIMARY_KEY'
MAX_UNIQUE_PER_COLUMN = 'MAX_UNIQUE_PER_COLUMN'
RESTRICTED_SUBSTRINGS = 'RESTRICTED_SUBSTRINGS'
CALCULATE_AGE = 'CALCULATE_AGE'
DELIMITER = 'DELIMITER'
MISSING_SIGNIFIERS = 'MISSING_SIGNIFIERS'

# Global variables
missing_signifiers = []

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

def bucketize(x, current_min, current_max, new_min, new_max, missing_signifiers):
  """Bucketizes a value into a new range, handling missing values.

  Args:
    x (float): The value to bucketize.
    current_min (float): The minimum value of the current range.
    current_max (float): The maximum value of the current range.
    new_min (int): The minimum value of the new range.
    new_max (int): The maximum value of the new range.
    missing_signifiers (list): List of values signifying missing data.

  Returns:
    int or str: The bucketized value, or MISSING if x is missing.
  """
  try:
    if x in missing_signifiers:
      return MISSING
    return int(
        np.round(
            ((x - current_min) / (current_max - current_min) * (new_max - new_min))
        ) + new_min
    )
  except Exception as e:
    if not math.isnan(float(x)):
      logging.info(e)
      logging.info(f"{x}, {current_min}, {current_max}, {new_min}, {new_max}")
    return MISSING

def quantize(col, name, missing_signifiers=None):
  """Quantizes a numeric column into buckets.

  Args:
    col (pd.Series): The column to quantize.
    name (str): The name of the column.

  Returns:
    tuple: A tuple containing the quantized column and a boolean indicating
      whether there are enough non-missing values.
  """
  current_min = col.min(skipna=True)
  current_max = np.ceil(col.max(skipna=True))

  if current_min == -np.inf:
    current_min = int(-INF)
  if current_max == np.inf:
    current_max = int(INF)

  new_min = 1
  new_max = min(current_max, 100)
  assert current_min != current_max

  scaled_col = col.apply(
      bucketize,
      args=(current_min, current_max, new_min, new_max, missing_signifiers)
  )
  ok_num = len(scaled_col) - np.sum(scaled_col == 'MISSING')
  logging.info(
      f"processed column = {name}, ok count = {ok_num}, "
      f"ok % = {ok_num / len(scaled_col) * 100}"
  )
  assert ok_num > (len(scaled_col) / 100)
  return scaled_col, ok_num > (len(scaled_col) / 100)

all_cols = {}
all_primaries = set()

def process_and_write(
    file_path,
    primary_key,
    immutable_cols,
    restricted_substrings,
    max_unique_per_column,
    time_key,
    first_event_time,
    calculate_age,
    missing_signifiers,
    delimiter
):
  global all_primaries
  if 'background' in file_path:
    logging.info(f"background file {file_path} is left as it is.")
    return
  logging.info(f"processing {file_path}")
  logging.info("*" * 100)
  df = pd.read_csv(file_path, delimiter=delimiter)
  all_primaries = all_primaries | set(df[primary_key].unique())
  col_for_drop_check = DAYS_SINCE_FIRST
  if DAYS_SINCE_FIRST not in df.columns:
    col_for_drop_check = time_key
  if not calculate_age:
    if AGE not in df.columns:
      logging.info(f"Ignoring {file_path} as it does not have the column age and calculate_age is False")
      os.remove(file_path)
      return
    else:
      df[AGE] = df[AGE].to_numpy(int)
  else:
    df[AGE] = 0

  df.dropna(subset=[col_for_drop_check], inplace=True)
  for column in df.columns:
    if column in [AGE, DAYS_SINCE_FIRST, time_key]:
      df[column] = np.round(df[column].tolist()).astype(int)
    elif column == primary_key or column in immutable_cols:
      continue
    elif (
        any(item in column for item in restricted_substrings)
        or (
            df[column].dtype == 'object' and
            len(df[column].unique()) > max_unique_per_column
        )
    ):
      logging.info(f"dropping column = {column}")
      df.drop(columns=[column], inplace=True)
    elif (
        np.issubdtype(df[column].dtype, np.number) and
        len(df[column].unique()) > 100
    ):
      logging.info(f"quantizing column {column}")
      scaled_col, have_enough = quantize(df[column], column, missing_signifiers)
      if have_enough:
        df[column] = scaled_col
    else:
      logging.info(
          f"keeping categorical column {column} with "
          f"{len(df[column].unique())} items"
      )
  df.fillna(value=MISSING, inplace=True)
  for col in df.columns:
    if col not in all_cols:
      all_cols[col] = []
    all_cols[col].append(file_path)
  if DAYS_SINCE_FIRST not in df.columns:
    df[DAYS_SINCE_FIRST] = df[time_key].apply(lambda x: x - first_event_time)
    df.drop(columns=[time_key], inplace=True)
  else:
    df[DAYS_SINCE_FIRST] = df[DAYS_SINCE_FIRST].to_numpy(int)
  df.to_csv(file_path, index=False)

def main():
  global missing_signifiers
  cfg_path = sys.argv[1]
  logging.info(f"cfg_path = {cfg_path}")
  cfg = read_cfg(cfg_path)

  source_dir = cfg[SOURCE]
  destination_dir = cfg[DEST]
  immutable_cols = cfg[IMMUTABLE_COLS]
  primary_key = cfg[PRIMARY_KEY]
  restricted_substrings = cfg[RESTRICTED_SUBSTRINGS]
  max_unique_per_column = cfg[MAX_UNIQUE_PER_COLUMN]
  calculate_age = cfg[CALCULATE_AGE]

  delimiter = cfg.get(DELIMITER, ',')
  time_key = cfg.get(TIME_KEY, None)
  first_event_time = cfg.get(FIRST_EVENT_TIME) if TIME_KEY in cfg else None
  missing_signifiers = cfg.get(MISSING_SIGNIFIERS, [])

  # Copy the entire directory recursively
  if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)
  shutil.copytree(source_dir, destination_dir)

  for root, dirs, files in os.walk(destination_dir):
    for filename in fnmatch.filter(files, '*.csv'):
      current_file_path = os.path.join(root, filename)
      cols = get_column_names(current_file_path, delimiter=delimiter)

      if primary_key in cols and (DAYS_SINCE_FIRST in cols or time_key in cols):
        logging.info(f"processing {filename}")
        process_and_write(
            file_path=current_file_path,
            primary_key=primary_key,
            immutable_cols=immutable_cols,
            restricted_substrings=restricted_substrings,
            max_unique_per_column=max_unique_per_column,
            time_key=time_key,
            first_event_time=first_event_time,
            calculate_age=calculate_age,
            missing_signifiers=missing_signifiers,
            delimiter=delimiter
        )

  for col in all_cols:
    logging.info(
      f"""Column {col} is found in {len(all_cols[col])} files."
      the files are:  {all_cols[col]}"""
    )

  logging.info(f"# of people: {len(all_primaries)}")

if __name__ == "__main__":
  main()
