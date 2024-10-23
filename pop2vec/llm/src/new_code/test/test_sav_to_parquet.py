import json
import numpy as np
import os 
import pandas as pd
import pyreadstat

from pop2vec.llm.src.new_code.utils import load_parquet_with_metadata

def create_sav(path):

  # Create the DataFrame
  data = {
      "Income": [10000, 0, 999999999, -5000, np.nan, 999999999, 500, -9999999, np.nan, 3000000],
      "Partnership": [888888888, 1, 0, np.nan, 1, 1, 888888888, 0, np.nan, 1],
      "Category": ["A", "B", "C", "D", "A", "C", "E", np.nan, "B", "D"]
  }

  df = pd.DataFrame(data)

  # Define value labels for the numeric and string columns
  variable_value_labels = {
      'Income': {
          0: 'could not find this person during survey',
          999999999: 'this person did not have any income'
      },
      'Partnership': {
          888888888: 'Did not end partnership'
      },
      'Category': {
          'A': 'Category A description',
          'B': 'Category B description',
          'C': 'Category C description',
          'D': 'Category D description',
          'E': 'Category E description'
      }
  }

  # Define metadata (optional if you want to add variable labels or column labels)
  meta_data = {
      'Income': 'Annual income of the person',
      'Partnership': 'Partnership status',
      'Category': 'Categorical description of status'
  }

  # Write the DataFrame to a .sav file with value labels
  pyreadstat.write_sav(
    df, 
    path, 
    variable_value_labels=variable_value_labels, 
    column_labels=meta_data
  )

  print("Saved .sav file with missing values, numeric and string columns.")


def create_parquet_from_sav(sav_path, data_path, meta_path):
  # Step 1: Read the .sav file and its metadata
  data, meta = pyreadstat.read_sav(sav_path)

  # Step 2: Write the data to a Parquet file
  data.to_parquet(data_path, index=False)
  print(f"Data written to {data_path}")

  # Step 3: Prepare the metadata for the second Parquet file
  metadata_info = []

  for col_name in meta.column_names:
      # Get the type (string or numeric) from metadata
      col_type = meta.readstat_variable_types[col_name]
      col_type = 'Numeric' if col_type == 'double' else 'String'
      # Get the value labels for the column (if any)
      value_labels = meta.variable_value_labels.get(col_name, {})
      
      # Convert the value labels to a JSON-like string representation
      value_labels_str = str(value_labels)
    
      # Append the row information to the list
      metadata_info.append({
          'Name': col_name,
          'Type': col_type,
          'Value Labels': value_labels_str
      })

  # Convert the metadata information to a DataFrame
  metadata_df = pd.DataFrame(metadata_info)

  # Step 4: Write the metadata to a Parquet file
  metadata_df.to_parquet(meta_path, index=False)
  print(f"Metadata written to {meta_path}")

directory = 'pop2vec/llm/src/new_code/test'
sav_path = os.path.join(directory, 'dummy.sav')
data_path = os.path.join(directory, 'data.parquet')
meta_path = os.path.join(directory, 'meta.parquet')
create_sav(sav_path)
create_parquet_from_sav(sav_path, data_path, meta_path)
df, meta = load_parquet_with_metadata(data_path, meta_path)

print(df)
print("-"*100)
print(meta)  