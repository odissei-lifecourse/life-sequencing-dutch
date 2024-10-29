import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from tqdm import tqdm

# Define the directory path
directory_path = 'your_directory_path_here'

# Step 1: Find all Parquet files that do not end with "_meta.parquet"
all_files = []
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if (
          (file.endswith('.parquet') and not file.endswith('_meta.parquet'))
          or file.endswith('.csv')
        ):
          all_files.append(os.path.join(root, file))
        
# Step 2: Create a dictionary to store sets of IDs for each file
id_sets = {}
print(f"# of files = {len(all_files)}")

for file_path in tqdm(all_files):
    if file_path.endswith('.parquet'):
      table = pq.read_table(file_path, columns=['RINPERSOON'])
      id_set = set(table['RINPERSOON'].to_pylist())
    else:
      table = pd.read_csv(file_path, usecols=['RINPERSOON'])
      id_set = set(table['RINPERSOON'].unique())
    
    id_sets[os.path.splitext(os.path.basename(file_path))[0]] = id_set

# Get the list of file names
file_names = list(id_sets.keys())
n = len(file_names)

# Initialize matrices for CSV1 and CSV2
csv1_matrix = np.zeros((n + 1, n + 1), dtype=object)
csv2_matrix = np.zeros((n + 1, n + 1), dtype=object)

# Set the first row and column of each matrix to file names
csv1_matrix[0, 1:] = csv1_matrix[1:, 0] = file_names
csv2_matrix[0, 1:] = csv2_matrix[1:, 0] = file_names

# Step 3: Calculate the values for CSV1 and CSV2
for i in tqdm(range(n)):
    csv1_matrix[i+1, i+1] = 100
    csv2_matrix[i+1, i+1] = 100
    for j in range(i+1, n):
        set_i = id_sets[file_names[i]]
        set_j = id_sets[file_names[j]]
        
        # For CSV1: percentage of ids in file i that are also in file j
        if set_i:
            csv1_matrix[i + 1, j + 1] = len(set_i.intersection(set_j)) / len(set_i) * 100
        else:
            csv1_matrix[i + 1, j + 1] = 0  # If set_i is empty, avoid division by zero
        csv1_matrix[j+1, i+1] = csv1_matrix[i + 1, j + 1]

        # For CSV2: Jaccard similarity = (set_i intersection set_j) / (set_i union set_j)
        union_size = len(set_i.union(set_j))
        csv2_matrix[i + 1, j + 1] = (
          len(set_i.intersection(set_j)) / union_size if union_size != 0 else 0
        ) * 100
        csv2_matrix[j+1, i+1] = csv2_matrix[i+1, j+1]

# Step 4: Save matrices as CSVs
csv1_df = pd.DataFrame(csv1_matrix)
csv2_df = pd.DataFrame(csv2_matrix)

csv1_df.to_csv('file_similarity_percentage.csv', index=False, header=False)
csv2_df.to_csv('file_similarity_jaccard.csv', index=False, header=False)
