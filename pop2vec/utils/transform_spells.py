"""Transform spell data into start and end events. Save metadata, data as csv and parquet."""

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pop2vec.utils.constants import DATA_ROOT

SOURCE_DIR = DATA_ROOT + "llm/raw/household/"
DESTINATION_DIR = DATA_ROOT + "llm/processed/household/"
DRY_RUN = False


def read_and_convert(filename: Path, sep: str = ";", add_age: bool = True) -> pd.DataFrame:
    """Read sequence data with begin and end date. Transform to separate events for beginning and end."""
    n_rows = None
    if DRY_RUN:
        n_rows = 5_000
    d = pd.read_csv(filename, sep=sep, nrows=n_rows)

    d_beg = d.copy()
    d_beg["eventType"] = "beg"
    d_beg = d_beg.drop(columns=["beg", "end"])

    d_end = d.copy()
    d_end["eventType"] = "end"

    for colname in ["end", "beg"]:
        d_end[colname] = pd.to_datetime(d_end[colname], format="%Y-%m-%d")

    d_end["time_diff"] = (d_end["end"] - d_end["beg"]).dt.total_seconds() / (24 * 60 * 60)
    d_end["daysSinceFirstEvent"] = d_end["daysSinceFirstEvent"] + d_end["time_diff"]
    d_end = d_end.drop(columns=["end", "beg", "time_diff"])

    d_out = pd.concat([d_beg, d_end])
    d_out = d_out.rename(columns={"rinpersoon": "RINPERSOON"})

    if add_age:
        d_out["age"] = 0

    return d_out


def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract metadata from dataframe."""
    meta = []
    for col in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        dtype = "Numeric" if is_numeric else "String"
        value = str({})
        info = (col, dtype, value)
        meta.append(info)

    return pd.DataFrame.from_records(meta, columns=["Name", "Type", "Value Labels"])


def save_data(df: pd.DataFrame, meta: pd.DataFrame, filename: Path, save_path: Path) -> None:
    """Save a dataframe and its metadata in various formats.

    Creates the `save_dir` if it does not exist, but not its parents.
    """
    stem = filename.stem
    save_path.mkdir(parents=False, exist_ok=True)

    file_suffix_mapping = {"parquet": (pd.DataFrame.to_parquet, ()), "csv": (pd.DataFrame.to_csv, {"index": False})}

    for suffix, settings in file_suffix_mapping.items():
        save_func, options = settings
        save_filename = Path(stem).with_suffix("." + suffix)
        save_func(df, Path(*[save_path, save_filename]), **options)

    save_filename = Path(stem + "_meta").with_suffix(".parquet")
    meta.to_parquet(Path(*[save_path, save_filename]))


def main() -> None:
    """Read, transform and save file."""
    source_dir = SOURCE_DIR
    destination_dir = DESTINATION_DIR

    files = os.listdir(source_dir)
    if DRY_RUN:
        files = files[:3]

    for file in tqdm(files):
        filename = Path(*[source_dir, file])
        d = read_and_convert(filename, sep=";")
        meta = get_metadata(d)
        save_data(d, meta, filename, Path(destination_dir))


if __name__ == "__main__":
    main()
