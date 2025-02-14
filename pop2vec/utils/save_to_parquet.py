"""Taken from code for layered walk. Can be made more general."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections import OrderedDict
    import numpy as np

# ruff: noqa: PLR0913


def save_to_parquet(
    data: np.ndarray, id_col: str, prefix_data_cols: str, filename: str, parquet_nests: OrderedDict, parquet_root: str
) -> None:
    """Save an array to parquet with partitioning.

    Args:
        data: the data to save.
        id_col: the name for the column with record identifiers.
        prefix_data_cols: the prefix for the columns created from the remaining columns.
        filename: the name of the parquet file.
        parquet_nests: the nesting for parquet partitions. A key-value pair in this dictionary
        gets converted into 'key=value'.
        parquet_root: The root path to the parquet file.

    Notes:
        - The schema of the parquet file is determined by the `id_col` and the `prefix_data_cols`.
    """
    n_cols = data.shape[1]

    data_cols = [f"{prefix_data_cols}_{i}" for i in range(n_cols - 1)]
    col_names = [id_col, *data_cols]

    table = pa.Table.from_arrays(
        [pa.array(data[:, 0], type=pa.int64())] + [data[:, i] for i in range(1, n_cols)], names=col_names
    )

    save_path = create_nested_dir(parquet_nests, parquet_root, filename)

    pq.write_table(table, save_path)


def create_nested_dir(nests: dict, root_dir: Path | str, filename: str) -> Path:
    """Create path to a file with parquet-like nesting directories.

    Args:
        nests (dict): dictionary of key-value pairs for nesting.
        root_dir (str or Path): root directory to create the nesting structure.
        filename (str): the filename to be saved.

    Returns:
        A Path to a file in the nested directory with the given filename.
    """
    partitions = [f"{key}={value}" for key, value in nests.items()]
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    save_dir = root_dir / Path(*partitions)

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / Path(filename)
