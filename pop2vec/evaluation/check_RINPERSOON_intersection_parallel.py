import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Manager

# Define initializer function to set global id_sets in each worker
def init_worker(id_sets_shared):
    global id_sets
    id_sets = id_sets_shared

# Function to compute similarities
def compute_similarity(pair):
    i, j = pair
    set_i = id_sets[file_names[i]]
    set_j = id_sets[file_names[j]]

    # For CSV1: percentage of ids in file i that are also in file j
    if set_i:
        sim_csv1 = len(set_i.intersection(set_j)) / len(set_i) * 100
    else:
        sim_csv1 = 0  # Avoid division by zero if set_i is empty

    # For CSV2: Jaccard similarity
    union_size = len(set_i.union(set_j))
    sim_csv2 = (len(set_i.intersection(set_j)) / union_size * 100) if union_size else 0

    return i, j, sim_csv1, sim_csv2


def read_file(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    if file_path.endswith('.parquet'):
        table = pq.read_table(file_path, columns=['RINPERSOON'])
        id_set = set(table['RINPERSOON'].to_pylist())
    else:
        table = pd.read_csv(file_path, usecols=['RINPERSOON'])
        id_set = set(table['RINPERSOON'].unique())
    return file_name, id_set


if __name__ == "__main__":
  # Define the directory path
  directory_path = '/gpfs/ostr/ossc9424/homedir/data/llm/raw/'
  manager = Manager()
  id_sets = manager.dict()
  # Step 1: Find all Parquet and CSV files
  all_files = []
  for root, dirs, files in os.walk(directory_path):
      for file in files:
          if (
              (file.endswith('.parquet') and not file.endswith('_meta.parquet'))
              or file.endswith('.csv')
          ):
              all_files.append(os.path.join(root, file))

  # Step 2: Read files in parallel and store ID sets
  print(f"# of files = {len(all_files)}")


  # Read files using multiprocessing
  with multiprocessing.Pool(processes=65) as pool:
      results = list(tqdm(pool.imap_unordered(read_file, all_files), total=len(all_files)))

  # Build id_sets dictionary
  for f_name, id_set in results:
    id_sets[f_name] = id_set
  # id_sets = dict(results)
  total = 0
  for k, v in id_sets.items():
    total += len(v)
  print(f"size of id_sets: {total}")
  # Get the list of file names
  file_names = list(id_sets.keys())
  n = len(file_names)

  # Initialize matrices for CSV1 and CSV2
  csv1_matrix = np.zeros((n + 1, n + 1), dtype=object)
  csv2_matrix = np.zeros((n + 1, n + 1), dtype=object)

  # Set the first row and column of each matrix to file names
  csv1_matrix[0, 1:] = csv1_matrix[1:, 0] = file_names
  csv2_matrix[0, 1:] = csv2_matrix[1:, 0] = file_names

  # Prepare list of index pairs (i, j)
  index_pairs = []
  for i in range(n):
      # Set diagonal elements
      csv1_matrix[i + 1, i + 1] = 100
      csv2_matrix[i + 1, i + 1] = 100
      for j in range(0, n):
          if i == j:
            continue
          index_pairs.append((i, j))

  print(f"n = {n}, len of index_pairs = {len(index_pairs  )}")

  # Compute similarities in parallel
  with multiprocessing.Pool(processes=65) as pool:
      results = list(tqdm(pool.imap_unordered(compute_similarity, index_pairs), total=len(index_pairs)))

  # Step 4: Save matrices as CSVs
  csv1_df = pd.DataFrame(csv1_matrix)
  csv2_df = pd.DataFrame(csv2_matrix)

  csv1_df.to_csv('file_similarity_percentage.csv', index=False, header=False)
  csv2_df.to_csv('file_similarity_jaccard.csv', index=False, header=False)
  