"""Taken from code for layered walk. Can be made more general."""

from collections import OrderedDict
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def save_to_parquet(
    data: np.ndarray, id_col: str, prefix_data_cols: str, filename: str, parquet_nests: OrderedDict, parquet_root: str
) -> None:
    """Save an array to parquet with partitioning.

    Args:
        `data`: the data to save.
        `id_col`: the name for the column with record identifiers.
        `prefix_data_cols`: the prefix for the columns created from the remaining columns.
        `filename`: the name of the parquet file.
        `parquet_nests`: the nesting for parquet partitions. A key-value pair in this dictionary
        gets converted into 'key=value'.
        `parquet_root`: The root path to the parquet file.

    Notes:
        - The schema of the parquet file is determined by the `id_col` and the `prefix_data_cols`.
    """
    n_cols = data.shape[1]

    data_cols = [f"{prefix_data_cols}_{i}" for i in range(n_cols - 1)]
    col_names = [id_col, *data_cols]

    table = pa.Table.from_arrays([data[:, i] for i in range(n_cols)], names=col_names)

    partitions = [f"{key}={value}" for key, value in parquet_nests.items()]
    save_dir = Path(parquet_root) / Path(*partitions)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / Path(filename)
    pq.write_table(table, save_path)
