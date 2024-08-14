import pandas as pd
import os
import sys
import numpy as np
import json
import logging 
import pyreadstat
from utils import check_column_names, subsample_from_ids, sample_from_file
import copy

REG_COLUMNS = [
  'median',
  'mean',
  'std_dev',
  '10th_percentile',
  '90th_percentile',
  'q1',
  'q3',
  'null_fraction',
  'category_top_0',
  'category_top_1',
  'category_top_2',
  'category_top_3',
  'category_top_4',
  '_others'
]

def save_meta(metadata, save_path, has_pii_columns):
  tmp_metadata = copy.deepcopy(metadata)
  tmp_metadata.pop('cov_matrix', None)
  tmp_metadata.pop('numeric_columns', None)
  tmp_metadata['has_pii_columns'] = has_pii_columns
  with open(save_path, 'w') as file:
    json.dump(tmp_metadata, file, indent=4, separators=(',', ': '))

def save_covariance(cov_matrix, cols, cov_path):
  df = pd.DataFrame(cov_matrix, columns=cols, index=cols)
  df.to_csv(cov_path)

def save_regular_columns(data, col_path):
  unrelated_for_col = ['metadata', 'has_pii_columns']
  final_dict = {}
  for attribute in REG_COLUMNS:
    final_dict[attribute] = []
  final_dict['variable_name'] = []
  for key, val in data.items():
    if key in unrelated_for_col:
      continue
    final_dict['variable_name'].append(key)
    for attribute in REG_COLUMNS:
      if attribute in val:
        final_dict[attribute].append(val[attribute])
      else:
        final_dict[attribute].append(None)

    df = pd.DataFrame(final_dict)
    df.set_index('variable_name', inplace=True)
    df.to_csv(col_path)

def process_json(data, meta_path, col_path, cov_path):
  has_pii_columns = data.get('has_pii_columns', [])
  if 'cov_matrix' in data['metadata']:
    save_covariance(
      data['metadata']['cov_matrix'],
      [col for col in data['metadata']['numeric_columns'] if col not in has_pii_columns],
      cov_path
    )
  save_meta(data['metadata'], meta_path, has_pii_columns)
  save_regular_columns(data, col_path)


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
  source_extension = '.txt'
  
  for source_root, dirs, files in os.walk(root_dir):
    target_root = source_root.replace(root_dir, target_dir)
    create_dir(target_root)
    
    for f in files:
      if f.endswith(source_extension):
        source_path = os.path.join(source_root, f)
        with open(source_path, 'r') as file:
          data = json.load(file)
        target_root = (source_path.split('.')[0]).replace(root_dir, target_dir)
        process_json(
          data, 
          target_root+'_meta.txt',
          target_root+'_columns.csv',
          target_root+'_covariance.csv',
        )
  