import numpy as np
import pandas as pd
import os 
from pathlib import Path
import random

from pop2vec.evaluation.split_dataset    import split

def work2(path, good_ids, write_path, sample_n=200000):
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if 'RINPERSOON1' not in df.columns:
        ids = set(df['RINPERSOON'].tolist())
    else:
        ids = set(df['RINPERSOON1'].tolist()) | set(df['RINPERSOON2'].tolist())

    print(path)
    print(f"good_ids = {len(good_ids)}, ids = {len(ids)}, intersection = {len(good_ids & ids)}")
    inter_ids = list(good_ids & ids)
    sampled = set(random.sample(inter_ids, min(len(inter_ids), sample_n)))
    if 'RINPERSOON1' not in df.columns:
        df = df[df['RINPERSOON'].isin(sampled)]
    else:
        df = df[df['RINPERSOON1'].isin(sampled) & df['RINPERSOON2'].isin(sampled)]
    print(f"final df size = {len(df)}")
    df.to_parquet(write_path)

if __name__ == '__main__':
    big_root = "/?"

    df1 = pd.read_parquet(f'{big_root}/good-ids_D3.parquet')
    good_ids = set(df1['RINPERSOON'].tolist())
    df2 = pd.read_parquet(f'{big_root}/subset-ids.parquet')
    good_ids = good_ids & set(df2['RINPERSOON'].tolist())
    root_dir = f'{big_root}/all/'
    dest_dir = f'{big_root}/subset/'
    root = Path(root_dir)

    for file_path in root.rglob("*"):
        if file_path.suffix.lower() in {'.parquet', '.csv'} and file_path.is_file():
            path = str(file_path)
            wp = Path(dest_dir, f"{file_path.stem}.parquet")
            work2(path, good_ids, wp)

    split(dest_dir, f'{big_root}/subset-splits')
        