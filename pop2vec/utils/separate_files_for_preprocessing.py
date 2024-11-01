"""Script to separate data into "good" and "huge" data files for sequence modelling.

Takes a set of files from a root location, checks their metadata and columns,
loads intersection of columns present and a specified list of columns,
then saves the data in a new location, in the same nesting structure as in the origin.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pop2vec.utils.constants import DATA_ROOT

N_COL_THRESHOLD = 20


def read_selected_columns(file_path: Path, column_names: list) -> pd.DataFrame:
    """Read `column_names` from a file."""
    file_type = file_path.suffix.lower()

    if file_type == ".csv":
        file_columns = pd.read_csv(file_path, nrows=0).columns.tolist()
    elif file_type == ".parquet":
        file_columns = pd.read_parquet(file_path, columns=None).columns.tolist()
    else:
        msg = "Unsupported file type."
        raise ValueError(msg)

    columns_to_read = list(set(column_names) & set(file_columns))
    if not columns_to_read:
        msg = "No overlapping columns found."
        raise RuntimeError(msg)

    if file_type == ".csv":
        data = pd.read_csv(file_path, usecols=columns_to_read)
    else:
        data = pd.read_parquet(file_path, columns=columns_to_read)

    return data


def determine_save_type(file_path: Path, meta_file_stem_map: dict, data_source: Path) -> SavingDataContainer:
    """Determine whether a file should be saved as csv or parquet."""
    save_dir = Path(file_path.stem)
    file_path_iter = file_path.parent
    while file_path_iter.name != data_source.name:
        save_dir = file_path_iter.name / save_dir
        file_path_iter = file_path_iter.parent

    if file_path.suffix == ".csv":
        saving_data_container = SavingDataContainer(save_as="csv", save_dir=save_dir)
    else:
        lookup_stem = file_path.stem + "_meta"
        if lookup_stem in meta_file_stem_map:
            meta_data_file = meta_file_stem_map[lookup_stem]
            meta_data = pd.read_parquet(meta_data_file)
            saving_data_container = SavingDataContainer(
                save_as="parquet", save_dir=save_dir, meta_data=meta_data, meta_data_file=meta_data_file
            )
        else:
            saving_data_container = SavingDataContainer(save_as="csv", save_dir=save_dir)

    return saving_data_container


@dataclass
class SavingDataContainer:
    """Container for metadata about where and how to save a file."""

    save_as: str
    save_dir: Path
    meta_data: pd.DataFrame | None = None
    meta_data_file: Path | None = None

    def update_save_dir(self, n_cols: int, threshold: int = N_COL_THRESHOLD) -> None:
        """Update the saving directory by number of columns."""
        data_dest = {"good": "llm/raw/data-driven-good/", "huge": "llm/raw/data-driven-huge/"}

        if n_cols <= threshold:
            self.save_dir = Path(data_dest["good"]) / self.save_dir
        else:
            self.save_dir = Path(data_dest["huge"]) / self.save_dir

    def save_data(self, data_root: Path, data: pd.DataFrame) -> None:
        """Savea data frame to the location defined in the class."""
        self.save_dir = data_root / self.save_dir
        self.save_dir.parent.mkdir(parents=True, exist_ok=True)

        if self.save_as == "parquet":
            if self.meta_data_file is None or self.meta_data is None:
                msg = "Missing meta_data or meta_data_file."
                raise ValueError(msg)

            data.to_parquet(self.save_dir.with_suffix(".parquet"), index=False)
            meta_file = self.save_dir.parent / self.meta_data_file.name
            self.meta_data.to_parquet(meta_file, index=False)
        else:
            data.to_csv(self.save_dir.with_suffix(".csv"), index=False)


def main():
    """Process all files, load subset of their columns to memory, save as csv or parquet."""
    files_to_ignore = ["columns.txt"]
    data_source = "llm/raw/ana_data"
    column_file = "llm/raw/ana_data/columns.txt"

    with Path(*[DATA_ROOT, column_file]).open("r") as f:
        relevant_columns = [line.rstrip() for line in f]

    data_source = Path(DATA_ROOT, data_source)
    files = [Path(*[dirpath, f]) for (dirpath, _, filenames) in os.walk(data_source) for f in filenames]

    data_file_paths = [f for f in files if "_meta" not in f.stem and f.name not in files_to_ignore]
    meta_file_stem_map = {f.stem: f for f in files if "_meta" in f.stem}

    for data_file_path in tqdm(data_file_paths):
        data = read_selected_columns(data_file_path, relevant_columns)

        saving_data_container = determine_save_type(data_file_path, meta_file_stem_map, data_source)

        saving_data_container.update_save_dir(data.shape[1])
        saving_data_container.save_data(Path(DATA_ROOT), data)


if __name__ == "__main__":
    main()
