import pandas as pd
import os
import sys
import numpy as np
import json
import logging 
import pyreadstat
from utils import check_column_names, sample_from_file

PII_COLS = ['RINPERSOON', 'RINADRES', 'BEID', 'BRIN', 'HUISHOUDNR', 
            'REFPERSOONHH', 'IKVID', 'SCHEIDINGNUMMER', '_crypt']
MIN_NOBS = 10
NA_ID_INPA = 9999999999.0


def process_numeric_column(name, data):
  ret_dict = {}
  tenth, q1, q2, q3, ninetieth = np.nanpercentile(data, [10, 25, 50, 75, 90])
  ret_dict[name] = {
    'median': float(np.nanmedian(data)),
    'mean': float(np.nanmean(data)),
    'std_dev': float(np.nanstd(data)),
    '10th_percentile': float(tenth),
    '90th_percentile': float(ninetieth),
    'q1': float(q1),
    'q3': float(q3),
    'null_fraction': float(data.isna().sum() / len(data))
  }
  return ret_dict

def process_categorical_column(name, data):
  ret_dict = {
    name: {}
  }
  most_frequent = data.value_counts().head(5).to_dict()
  for _ in range(len(most_frequent), 5):
    most_frequent['EMPTY'] = 0
  total = 0
  for i, category in enumerate(most_frequent):
    if most_frequent[category] >= MIN_NOBS:
      ret_dict[name][f"category_top_{i}"] = f"{category}--{most_frequent[category]/len(data)}"
      total += most_frequent[category]
  ret_dict[name]['_others'] = (len(data) - total)/len(data) 
  ret_dict[name]['null_fraction'] = float(data.isna().sum()/len(data))
  return ret_dict

def get_col_metadata(df, source_file_path):
  metadata = {
    'path': source_file_path,
    'shape': df.shape,
  }
  columns_with_dtypes = {}
  numeric_columns = []
  for i, column in enumerate(df.columns):
    columns_with_dtypes[column] = str(df.dtypes[i])
    if np.issubdtype(df[column].dtype, np.number):
      numeric_columns.append(column)
  metadata['columns_with_dtypes'] = columns_with_dtypes
  metadata["numeric_columns"] = numeric_columns
  return metadata

def get_cov_matrix(df, numeric_columns):
    numeric_data = df.loc[:, numeric_columns]
    if len(numeric_columns) == 1:
      cov_matrix = np.reshape(np.cov(numeric_data, rowvar=False), (1,1)).tolist()
    else:
      cov_matrix = np.cov(numeric_data, rowvar=False).tolist()

    assert len(cov_matrix) == len(numeric_columns)
    return cov_matrix


def process(source_file_path, target_file_path, n_rows=None):
  """Process a file from source to target

  Args:
   source_file_path (str): path to the source file.
   target_file_path (str): path to target file.
   n_rows (int or None): Read only as many rows from the file. If None, read whole file.
  """
  logging.debug("Starting with file %s.", source_file_path)
  df, nobs = sample_from_file(source_file_path, n_rows)
  
  summary_dict = {
    'metadata': get_col_metadata(df, source_file_path)
  }
  summary_dict['metadata']["total_nobs"] = nobs
  summary_dict['metadata']["nobs_sumstat"] = n_rows

  has_pii_cols = [c for c in df.columns if check_column_names([c], PII_COLS)]
  keep_cols = [c for c in df.columns if c not in has_pii_cols]
  df = df.loc[:, keep_cols]
  summary_dict['has_pii_columns'] = has_pii_cols

  numeric_columns = summary_dict["metadata"]["numeric_columns"]
  numeric_columns = [i for i in numeric_columns if i not in has_pii_cols]

  if "INPATAB" in source_file_path and len(numeric_columns) > 0:
    for col in numeric_columns: 
      df.loc[df[col] == NA_ID_INPA, col] = np.nan 


  if df.shape[0] > 1 and len(numeric_columns) > 0:
    summary_dict["metadata"].update(cov_matrix = get_cov_matrix(df, numeric_columns))

  logging.debug("Processing columns")
  for column in df.columns:
    logging.debug("current column: %s", column)

    if np.issubdtype(df[column].dtype, np.number):
      summary_dict.update(process_numeric_column(column, df[column]))
    elif df[column].dtype=='object':
      summary_dict.update(process_categorical_column(column, df[column]))
    else:
      logging.warning(
        f"Ignoring {column} from {source_file_path}; "
        f"dtype of {column} is {df[column].dtype}"
      )
  
  logging.debug("Writing output")
  with open(target_file_path, 'w') as f:
    try:
      json.dump(summary_dict, f)
    except Exception as e:
      logging.error(
        f"{target_file_path} could not be jsonified", 
        exc_info=True,
      )


def create_dir(path):
  if os.path.exists(path):
    return
  os.mkdir(path)


if __name__ == '__main__':
  logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
  )
  
  root_dir = sys.argv[1]
  target_dir = sys.argv[2]
  if sys.argv[3] != 'None':
    sample_size = int(sys.argv[3])
  else:
    sample_size = None
  if len(sys.argv) > 4:
    source_extension = '.' + sys.argv[4]
  else:
    source_extension = '.csv'

  target_extension = '.txt'

  for source_root, dirs, files in os.walk(root_dir):
    target_root = source_root.replace(root_dir, target_dir)
    create_dir(target_root)
    
    for f in files:
      if f.endswith(source_extension):
        source_path = os.path.join(source_root, f)
        target_path = (
          os.path.join(target_root, f.split(source_extension)[0]) + 
          target_extension
        )
        process(source_path, target_path, sample_size)
        
