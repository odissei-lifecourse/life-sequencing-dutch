import logging
import os
import pandas as pd
import shutil
import sys

from collections import Counter

logging.basicConfig(
      format="%(asctime)s %(name)s %(levelname)s: %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
      level=logging.INFO
)

def subset_by_columns(source_dir, dest_dir, columns_path):
    # Read the columns file and build kept_columns set
    with open(columns_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    total_lines = len(lines)
    kept_columns = set(lines)
    print(f"Total lines read: {total_lines}")
    print(f"Unique columns count: {len(kept_columns)}")
    
    # Detect duplicates
    counter = Counter(lines)
    duplicates = {col for col, count in counter.items() if count > 1}
    if duplicates:
        print("Duplicates found:", duplicates)
    
    # Create destination directory, raise an error if it exists
    if os.path.exists(dest_dir):
        raise FileExistsError(f"Destination directory '{dest_dir}' already exists.")
    os.makedirs(dest_dir)
    
    # Walk through the source directory structure
    for root, dirs, files in os.walk(source_dir):
        # Create the corresponding directory in dest_dir
        rel_dir = os.path.relpath(root, source_dir)
        dest_subdir = os.path.join(dest_dir, rel_dir) if rel_dir != "." else dest_dir
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)
        
        for file in files:
            source_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_subdir, file)
            
            # Copy meta parquet files verbatim
            if file.endswith("_meta.parquet"):
                shutil.copy2(source_file_path, dest_file_path)
            # For regular parquet files, filter columns and then write the subset dataframe
            elif file.endswith(".parquet"):
                logging.info(f'subsetting {source_file_path}')
                df = pd.read_parquet(source_file_path)  
                # Retain only columns that are in kept_columns and exist in the DataFrame
                cols_to_keep = [col for col in df.columns if col in kept_columns]
                df = df[cols_to_keep]
                df.to_parquet(dest_file_path, index=False)
            # Skip any non-parquet files 

if __name__ == '__main__':
    subset_by_columns(
        source_dir=sys.argv[1], 
        dest_dir=sys.argv[2], 
        columns_path=sys.argv[3]
    )
