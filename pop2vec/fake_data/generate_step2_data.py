
"""
Create completely fake data for parquet files. This is for the new data structure and code as of 01/2025.
It supersedes most of the fake data generated in create_fake_data.py.
"""


# TODO: make paths as arguments

import pandas as pd
import random
import string
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from tqdm import tqdm
from pathlib import Path

# Constants
MIN_ROWS = 100000#1_000_000
MAX_ROWS = 500000#10_000_000
GENESIS_DATE = pd.Timestamp('1971-12-30')
DATA_ROOT = "/projects/0/prjs1019/data/fake_data_v0"
NROW_BACKGROUND = 10_000_000


def generate_random_string(length=5):
    """Generate a random string of fixed length."""
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def generate_random_description(val):
    """Generate a random natural language description for a value."""
    descriptions = [
        f"Description for value {val}",
        f"This value represents {val}",
        f"Value {val} indicates a special case",
        f"An example of {val}",
        f"{val} is significant because...",
        f"The value {val} corresponds to an important category",
        f"Explanation for {val}",
        f"The code {val} stands for...",
        f"{val} is associated with certain characteristics",
        f"Note about value {val}"
    ]
    return random.choice(descriptions)

def generate_random_data(columns, num_rows, nrows_per_id=8):
    """Generate random data for given columns and number of rows.

    Creates a dataframe of categorical and continuous data. Identifiers are
    drawn at random from a global pool of identifiers.

    Arguments:
        columns: names of columns to generate.
        num_rows: number of rows to generate.
        nrows_per_id: number of rows per id.

    """
    global RANDOM_ID_POOL

    print(f'generating {num_rows} rows with {len(columns)} columns')
   
    data = {} 
    current_ids = np.random.choice(RANDOM_ID_POOL, size=num_rows//nrows_per_id, replace=False) 
    
    for col in columns:
        if col == 'RINPERSOON':
            draw_with_replacement = nrows_per_id > 1
            data[col] = np.random.choice(current_ids, size=num_rows, replace=draw_with_replacement)

        elif col == 'daysSinceFirstEvent':
            data[col] = np.random.randint(0, 18_000, size=num_rows)  # Random days since genesis (roughly 50 years)
        elif col == 'age':
            data[col] = np.random.randint(16, 100, size=num_rows)  # Age between 16 and 100
        elif col == 'month':
            data[col] = np.random.randint(1, 13, size=num_rows)
        elif col == 'year':
            data[col] = np.random.randint(1940, 2020, size=num_rows)
        elif col == 'gender':
            data[col] = np.random.randint(1, 3, size=num_rows)
        elif col == 'municipality':
            data[col] = np.random.randint(1, 500, size=num_rows)
        else:
            # Randomly decide if the column is numeric or categorical
            if random.choice([True, False]):  # 50% probability for each
                data[col] = np.random.randint(0, 1000, size=num_rows)  # Example numeric range
            else:
                data[col] = [generate_random_string(2) for _ in range(num_rows)]
    return pd.DataFrame(data)

def create_parquet_and_metadata(df, file_name):
    """Save DataFrame as Parquet and create corresponding metadata Parquet."""
    # Save data to Parquet
    parquet_path = Path(DATA_ROOT) / "step2" / f"{file_name}.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)
    
    # Generate metadata
    metadata = {
        'Name': [],
        'Type': [],
        'ValueLabels': []
    }
    for col in df.columns:
        if col in ['RINPERSOON', 'daysSinceFirstEvent', 'age']:
          continue
        metadata['Name'].append(col)
        col_type = 'Numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'String'
        metadata['Type'].append(col_type)
        x = random.randint(1, 10)
        unique_values = df[col].dropna().unique()
        if len(unique_values) > x:
            sampled_values = np.random.choice(unique_values, x, replace=False)
        else:
            sampled_values = unique_values
        value_labels = {}
        if 'background' not in file_name:
          for val in sampled_values:
              # Generate a random natural language description
              description = generate_random_description(val)
              value_labels[val] = description
        metadata['ValueLabels'].append(str(value_labels))
    
    metadata_df = pd.DataFrame(metadata)
    meta_parquet_path = Path(DATA_ROOT) / "step2" / f"{file_name}_meta.parquet"
    meta_table = pa.Table.from_pandas(metadata_df)
    pq.write_table(meta_table, meta_parquet_path)
    
    return parquet_path, meta_parquet_path

def process_files(columns_dict):
    for name, columns in tqdm(columns_dict.items()):
        if "background" in name:
            df = generate_random_data(columns, NROW_BACKGROUND, 1)
        else:
            num_rows = random.randint(MIN_ROWS, MAX_ROWS)
            df = generate_random_data(columns, num_rows, 8)
        
        create_parquet_and_metadata(df, name)

if __name__ == '__main__':
    dir_path = Path(DATA_ROOT) / Path("empty_files")
    columns_dict = {}
    RINPERSOONS_PATH = Path(DATA_ROOT) / 'fake_rinpersoons.csv'

    for file_name in os.listdir(dir_path):
        if file_name.endswith('.csv'):
            file_path = dir_path / file_name 
            df = pd.read_csv(file_path, nrows=0)  # Read only the header
            columns = df.columns.tolist()
            columns_dict[os.path.splitext(file_name)[0]] = columns
            

    RANDOM_ID_POOL = pd.read_csv(RINPERSOONS_PATH)['RINPERSOON'].unique()
    
    process_files(columns_dict)  # Execute the data generation and saving process

