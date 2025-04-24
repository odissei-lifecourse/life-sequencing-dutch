import os
import pandas as pd

# assuming the current directory is data/graph/embeddings/
# Create the output folder if it doesn't exist
output_folder = "embeddings-32bits"
os.makedirs(output_folder, exist_ok=True)

# Iterate over the desired years
for year in range(2009, 2021):
    # Define the base directory for the current year
    input_dir = f"year={year}"
    
    # Recursively search for 'embedding.parquet' under the input_dir
    embedding_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == "embedding.parquet" and 'record_edge_type=1' in root:
                embedding_files.append(os.path.join(root, file))
    
    # Check if we found exactly one file; otherwise, raise an error
    if len(embedding_files) > 1:
        raise ValueError(f"Multiple embedding.parquet files found in directory {input_dir}: {embedding_files}")
    elif len(embedding_files) == 0:
        raise ValueError(f"No embedding.parquet file found in directory {input_dir}.")
    
    input_file = embedding_files[0]
    
    # Read the parquet file
    df = pd.read_parquet(input_file)
    
    # Convert all float64 columns to float32
    float64_cols = df.select_dtypes(include=["float64"]).columns
    for col in float64_cols:
        df[col] = df[col].astype("float32")
    
    # Define the output file name
    output_file = f"graph-{year}-epoch_50-with_edge-walklen_40-prob_0.8.parquet"
    output_path = os.path.join(output_folder, output_file)
    
    # Save the modified DataFrame to Parquet
    df.to_parquet(output_path)
    
    print(f"Processed {input_file} and saved to {output_path}")