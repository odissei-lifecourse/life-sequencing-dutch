from functools import partial
from typing import List, Dict

import ast
import csv
import json
import time
import numpy as np
import os
import pandas as pd
import random
import logging
from pop2vec.llm.src.new_code.constants import (
    AGE,
    DAYS_SINCE_FIRST,
    SPECIAL_NUMERIC_ZERO,
    SPECIAL_STR_ZERO
)

from typing import Union

print_now = partial(print, flush=True)
logging.basicConfig(level=logging.INFO)

def replace_less_frequent(
  df_column, 
  keep_top_n = 100, 
  name_others = "Others", 
  ignore_list = ["MISSING"]
):
  """
  Replace less frequent values in a pandas Series with a placeholder.

  Args:
    df_column (pd.Series): The column to process.
    keep_top_n (int): Number of top frequent values to keep.
    name_others (str): Placeholder name for less frequent values.
    ignore_list (list): List of values to ignore during processing.

  Returns:
      pd.Series: The transformed column.
  """
  # Get the value counts of the column, excluding the values in ignore_list
  value_counts = df_column[~df_column.isin(ignore_list)].value_counts()
  
  # Identify the top 'n' most frequent values that are not in ignore_list
  top_n_values = value_counts.nlargest(keep_top_n).index
  
  # Replace values not in the top 'n' and not in ignore_list with 'name_others'
  
  # new_column = df_column.apply(
  #   lambda x: x if x in top_n_values or x in ignore_list else name_others
  # )
  # Faster option below
  mask = df_column.isin(top_n_values) | df_column.isin(ignore_list)
  new_column = df_column.where(mask, other=name_others)

  
  return new_column

def transform_to_percentiles(column_data: pd.Series, inplace: bool) -> Union[None, pd.Series]:
  """Transform numeric values in a column to integer percentiles (0-99),
  leaving non-numeric values unchanged.

  Args:
    column_data: A pandas Series to be transformed into percentiles.

  Returns:
    A pandas Series where numeric values are replaced with percentile ranks
    as integers from 0 to 99, and non-numeric values are left unchanged.
  """
  # Create a mask to identify numeric values
  numeric_mask = pd.to_numeric(column_data, errors='coerce').notna()

  # Apply the transformation only to numeric values
  percentiles = column_data[numeric_mask].rank(pct=True) * 100
  percentiles = np.floor(percentiles).astype(int).clip(0, 99)

  if inplace:
    column_data.loc[numeric_mask] = percentiles
    return
  else:
    # Create a copy of the original column_data
    transformed_data = column_data.copy()

    # Replace numeric values with their percentile ranks
    transformed_data[numeric_mask] = percentiles
    return transformed_data

def load_csv_and_create_metadata(
  file_path: str, 
  delimiter: str, 
  categorical_threshold: int,
  primary_key: str,
) -> tuple[pd.DataFrame, dict]: 
    assert file_path.endswith('.csv'), (
      '{file_path} is not a csv file.'
    )
    assert delimiter is not None, 'delimiter must not be none for csv'
    
    df = pd.read_csv(file_path, delimiter=delimiter)
    special_values = {
      '0' : SPECIAL_STR_ZERO,
      0 : SPECIAL_NUMERIC_ZERO,
    }

    meta = {}
    for col in df.columns:
      if col in [primary_key, AGE, DAYS_SINCE_FIRST]:
        continue
      if (
        df[col].dtype == 'object' or 
        df[col].nunique() <= categorical_threshold
      ):
        meta[col] = 'String'
      else:
        meta[col] = 'Numeric'
      df[col].replace(special_values, inplace=True)

    return df, meta

def _load_metadata(metadata_file: str) -> pd.DataFrame:
  """Load metadata from a parquet file.

  Args:
    metadata_file: Path to the parquet file containing metadata.

  Returns:
    A pandas DataFrame containing metadata.
  """
  metadata_df = pd.read_parquet(metadata_file)
  
  # Ensure that all types are either "Numeric" or "String"
  valid_types = ["Numeric", "String"]
  assert all(metadata_df['Type'].isin(valid_types)), (
      f"Invalid types found in metadata. "
      f"Expected: {valid_types}, Found: {metadata_df['Type'].unique()}"
  )
  
  return metadata_df

def _load_parquet_and_transform_data(
    data_file: str,
    metadata_df: pd.DataFrame,
    primary_key: str,
) -> pd.DataFrame:
  """Load data from a parquet file and apply metadata value labels.

  Args:
    data_file: Path to the parquet file containing data.
    metadata_df: DataFrame with metadata containing column names, types, 
                 and value labels.
    primary_key: primary key of df stored at data_file

  Returns:
    A pandas DataFrame with special values replaced for numeric columns.
  """
  data_df = pd.read_parquet(data_file)
  
  for _, row in metadata_df.iterrows():
    col_name = row['Name']
    if col_name in [primary_key, AGE, DAYS_SINCE_FIRST]:
      continue
    col_type = row['Type']
    value_labels = row[
      'Value Labels' if 'Value Labels' in metadata_df.columns else 'ValueLabels'
    ]
    special_values = ast.literal_eval(value_labels)  # Convert string to dictionary
    if 0 not in special_values:
      special_values[0] = SPECIAL_NUMERIC_ZERO
    if '0' not in special_values:
      special_values['0'] = SPECIAL_STR_ZERO
    # Only handle numeric columns for value replacement
    if col_type == 'Numeric' and pd.notna(value_labels):
      data_df[col_name].replace(special_values, inplace=True)
  
  return data_df

def load_parquet_with_metadata(
    data_file: str,
    metadata_file: str,
    primary_key: str,
) -> tuple[pd.DataFrame, dict]:
  """Process data and metadata to create a dataframe and column types.

  Args:
    data_file: Path to the parquet file containing the data.
    metadata_file: Path to the parquet file containing the metadata.

  Returns:
    A tuple containing:
      - A pandas DataFrame with processed data.
      - A dictionary where keys are column names, and values are either 
        "Numeric" or "String".
  """
  assert data_file.endswith('.parquet'), (
    '{data_file} is not a parquet file.'
  )
  assert metadata_file.endswith('.parquet'), (
    '{meta_path} is not a parquet file.'
  )
  # Load metadata and data
  metadata_df = _load_metadata(metadata_file)
  data_df = _load_parquet_and_transform_data(
    data_file, 
    metadata_df, 
    primary_key
  )
  
  # Create column types dictionary
  column_types = {}
  for _, row in metadata_df.iterrows():
    column_types[row['Name']] = row['Type']

  return data_df, column_types


def get_column_names(csv_file, delimiter=','):
  df = pd.read_csv(csv_file, delimiter=delimiter, nrows=2)
  return df.columns

def read_json(path):
  with open(path, 'r') as file:
    data = json.load(file)
  return data  

def shuffle_json(input_file, output_file):
  start = time.time()
  logging.info("shuffle json starting")
  if os.path.exists(output_file):
    # Generate new filename with timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_file = f"{output_file[:-5]}_{timestamp}.json"
    logging.warning(f"Output file already exists. Writing to new file: {output_file}")

  # Read lines from input file
  with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

  end = time.time()
  logging.info(f"{end-start} seconds elapsed for reading")
  start = end
  # Shuffle lines
  random.shuffle(lines)
  end = time.time()
  logging.info(f"{end-start} seconds elapsed for shuffling")
  start = end
  # Write shuffled lines to output file
  with open(output_file, 'w', encoding='utf-8') as f:
    for line in lines:
      f.write(line)
  end = time.time()
  logging.info(f"{end-start} seconds elapsed for writing")

def shuffle_json_memory_efficient(input_file_path, output_file_path):
    def index_file(file_path):
      index = []
      offset = 0
      with open(file_path, 'rb') as file:  # Use binary mode to handle bytes accurately
          while line := file.readline():
              index.append((offset, len(line)))
              offset += len(line)
      return index

    # Index the lines in the original file
    indices = index_file(input_file_path)

    # Shuffle the indices
    random.shuffle(indices)

    # Read lines in the order of shuffled indices and write to new file
    with open(input_file_path, 'rb') as input_file:  # Ensure binary mode for accurate seeking
        with open(output_file_path, 'wb') as output_file:  # Binary mode for output
            for start, length in indices:
                input_file.seek(start)
                line = input_file.read(length)
                output_file.write(line)


