import os
from pathlib import Path
import pandas as pd 


def split(
    src_dir,
    dest_dir,
    train_frac=0.7,
    val_frac=0.1,
    test_frac=0.2,
    random_state=42,
):
    for split in ("train", "val", "test"):
        Path(dest_dir, split).mkdir(parents=True, exist_ok=True)


    for root, _, files in os.walk(src_dir):
        for name in files:
            ext = Path(name).suffix.lower()
            print(f"checking {name}")
            if ext not in {'.csv', '.parquet'}:
                continue
            print(f"processing {name}")
            src_path = Path(root, name)

            if ext == ".csv":
                df = pd.read_csv(src_path)
            else:   
                df = pd.read_parquet(src_path)

            print(f"len(df) = {len(df)}")
            df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        
            n_total = len(df)
            n_train = int(n_total * train_frac)
            n_val = int(n_total * val_frac)
            print(f"total = {n_total}, train = {n_train}, val = {n_val}")
            train_df = df.iloc[:n_train]
            val_df = df.iloc[n_train:n_train + n_val]
            test_df = df.iloc[n_train + n_val:]
            print(len(train_df), n_train)
            outfile = Path(src_path.stem + ".parquet")
            train_df.to_parquet(Path(dest_dir, "train", outfile), index=False)
            val_df.to_parquet(Path(dest_dir, "val", outfile), index=False)
            test_df.to_parquet(Path(dest_dir, "test", outfile), index=False)