from functools import partial
from typing import List, Dict

import ast
import csv
import json
import logging
import numpy as np
import os
import pandas as pd
import polars as pl
import pyarrow.parquet 
import random
import shutil
import time
import yaml

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
  ignore_list = ["MISSING"],
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

def transform_to_percentiles(
  df: pd.DataFrame, 
  column: str, 
  inplace: bool
) -> Union[None, pd.DataFrame]:
  """Transform numeric values in a column to integer percentiles (0-99),
  leaving non-numeric values unchanged.

  Args:
    column_data: A pandas Series to be transformed into percentiles.

  Returns:
    A pandas Series where numeric values are replaced with percentile ranks
    as integers from 0 to 99, and non-numeric values are left unchanged.
  """
  # Create a mask to identify numeric values
  column_data = df[column]
  numeric_mask = pd.to_numeric(column_data, errors='coerce').notna()

  # Apply the transformation only to numeric values
  percentiles = column_data[numeric_mask].rank(pct=True) * 100
  percentiles = np.floor(percentiles).astype(int).clip(0, 99)

  if inplace:
    df.loc[numeric_mask, column] = percentiles
    return
  else:
    # Create a copy of the original column_data
    transformed_data = column_data.copy()

    # Replace numeric values with their percentile ranks
    transformed_data[numeric_mask] = percentiles
    df[column] = transformed_data
    return df

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

def _replace_less_frequent_special_values(
  column_data: pd.Series,
  special_values: dict,
  keep_top_n = 100, 
  name_others = "Others", 
):
  # Step 1: Count occurrences of each key in `replacements` within `series`
  freq_counts = column_data[
    column_data.isin(special_values.keys())
  ].value_counts()

  # Step 2: Identify top_n most frequent keys
  top_frequent_keys = set(freq_counts.nlargest(keep_top_n).index)

  # Step 3: Create masks for efficient vectorized replacement
  # Mask for top_n frequent keys
  mask_top = column_data.isin(top_frequent_keys)
  # Mask for other keys in `replacements` but not in top_n
  mask_other_replacements = column_data.isin(replacements.keys()) & ~mask_top

  # Step 4: Perform vectorized replacements
  result = column_data.where(~mask_top, column_data.map(special_values))         # Replace top_n keys
  result = result.where(~mask_other_replacements, name_others)       # Replace other keys

  return result

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
    if col_name in [primary_key, AGE, DAYS_SINCE_FIRST] or col_name not in data_df.columns:
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
    keep_top_n = 100
    if col_type == 'Numeric' and pd.notna(value_labels):
      if len(special_values) <= keep_top_n:
        data_df[col_name].replace(special_values, inplace=True)
      else:
        logging.info(
          f"""Column {col_name} has {len(special_values)}. 
          We are keeping the most frequent {keep_top_n} values.
          """
        )
        data_df[col_name] = _replace_less_frequent_special_values(
          data_df[col_name],
          special_values,
          keep_top_n
        )

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
    '{metadata_file} is not a parquet file.'
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


def get_column_names(file, delimiter=','):
  if file.endswith('.csv'):
    return pd.read_csv(csv_file, delimiter=delimiter, nrows=2).columns.tolist()
  elif file.endswith('.parquet'):
    return pyarrow.parquet.ParquetFile(file).schema.names
  else:
    raise ValueError('{file} is not a csv or parquet file.')
  
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



def create_subsampled_parquets(source_dir, dest_dir, n=10000, id_column="RINPERSOON"):
    """
    1) Find a parquet file in source_dir whose name contains 'background' (excluding _meta files).
    2) Read only its id_column, randomly sample n rows, and store those ids in keep_ids.
    3) Recursively mirror the directory structure of source_dir into dest_dir.
       - Copy any *_meta.parquet files verbatim.
       - For regular parquet files, filter out rows where `id_column` is not in keep_ids
         and write out the result. Uses lazy scanning and sink to avoid loading
         the entire file into memory at once (where Polars streaming is supported).
    """
    # 1) Locate the 'background' parquet file (excluding meta)
    background_file = None
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if "background" in file and file.endswith(".parquet") and not file.endswith("_meta.parquet"):
                background_file = os.path.join(root, file)
                break
        if background_file:
            break

    if background_file is None:
        raise FileNotFoundError("No background parquet file found in source_dir.")

    # 2) Sample n IDs from the background file
    #    We only scan the id_column to reduce memory usage.
    bg_ids = pd.read_parquet(background_file, columns=[id_column])

    n = min(n, len(bg_ids))

    df_sampled = bg_ids.sample(n=n)
    keep_ids = set(df_sampled[id_column].to_list())

    # 3) Create dest_dir (raise error if it already exists)
    if os.path.exists(dest_dir):
        raise FileExistsError(f"Destination directory '{dest_dir}' already exists.")
    os.makedirs(dest_dir)

    # 4) Walk through source_dir and process each file
    for root, dirs, files in os.walk(source_dir):
        # Create the corresponding subdirectory in dest_dir
        rel_dir = os.path.relpath(root, source_dir)
        dest_subdir = (
            os.path.join(dest_dir, rel_dir) if rel_dir != "." else dest_dir
        )
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)

        for file in files:
            source_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_subdir, file)

            # a) Copy *_meta.parquet files verbatim
            if file.endswith("_meta.parquet"):
                shutil.copy2(source_file_path, dest_file_path)

            # b) For regular .parquet files, filter rows whose id_column is in keep_ids
            elif file.endswith(".parquet"):
                df = pd.read_parquet(source_file_path)
                filtered = df[df[id_column].isin(keep_ids)]
                filtered.to_parquet(dest_file_path, index=False)

def is_float(string):
    try:
      float(string)
      return True
    except ValueError:
      return False

# Read hparams from the text file
def read_hparams_from_txt(file_path):
    with open(file_path) as file:
        lines = file.readlines()
        hparams = {}
        for line in lines:
            if len(line) < 2 or line.startswith("#"):
              continue
            
            line = line.strip().split("#")[0]

            key, value = line.strip().split(": ")
            value = value.replace('"',"")
            if value in ["True", "False"]:
              if value == "True":
                value = True
              else:
                value = False
            elif value.isdigit():
              value = int(value)
            elif is_float(value):
              value = float(value)
            hparams[key] = value # float(value) if value.isdigit() else value
            
        return hparams

def read_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def read_hparams(file_path):
    if file_path.endswith('.yaml'):
        return read_yaml(file_path)
    else:
        return read_hparams_from_txt(file_path)
