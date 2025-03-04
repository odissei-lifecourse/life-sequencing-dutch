from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl


def sample_ids_from_parquet(
    parquet_path: str | Path, id_column: str = "rinpersoon_id", n: int = 1000, seed: int | None = None
) -> pl.DataFrame:
    """Sample from a parquet file.

    Sample n random IDs from a Parquet file and return a DataFrame with all columns
    for those sampled IDs.

    Parameters:
    -----------
    parquet_path : str or Path
        Path to the Parquet file
    id_column : str, default="rinpersoon_id"
        Name of the ID column to sample from
    n : int, default=1000
        Number of IDs to sample
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pl.DataFrame
        DataFrame containing all columns for the sampled IDs
    """
    # Convert to Path object if string is provided
    if isinstance(parquet_path, str):
        parquet_path = Path(parquet_path)

    rng = np.random.default_rng(seed)

    ids_df = pl.scan_parquet(parquet_path).select(id_column).collect()
    unique_ids = ids_df[id_column].unique()
    total_ids = len(unique_ids)

    if total_ids < n:
        raise ValueError

    sampled_ids = rng.choice(unique_ids, n, replace=False)

    return pl.scan_parquet(parquet_path).filter(pl.col(id_column).is_in(sampled_ids)).collect()
