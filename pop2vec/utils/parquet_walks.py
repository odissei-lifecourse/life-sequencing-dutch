from __future__ import annotations
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import duckdb
import numpy as np
from pop2vec.utils.save_to_parquet import save_to_parquet

if TYPE_CHECKING:
    import pandas as pd


def create_column_placeholders(cols) -> str:  # suggested by Claude
    """Return comma-separated columns to query."""
    return ", ".join(cols)


@dataclass
class ParquetWalks:
    """Container for loading dataframes of walks for deepwalk."""

    parquet_root: str
    parquet_nests: str # TODO: could this not be inferred from all the partitions that are arguments below?
    year: int
    iter_name: str
    record_edge_type: bool
    chunk_id: int | None = None
    n_edge_types: int = 5 # hard-coded number of edge types. should find way to determine automatically

    def remap_ids(self, mapped_indices): # TODO: don't hard-code path here
        """Remap an array of mapped indices.

        From a list of mapped indices from 0 to len(mapped_indices)-1,
        return the ids corresponding to the indices.
        """
        mapping_url = "/gpfs/ostor/ossc9424/homedir/data/graph/mappings/family_" + str(self.year) + ".pkl"
        with Path(mapping_url).open("rb") as pkl_file:
            person_mappings = dict(pickle.load(pkl_file)) # noqa: S301


        inverted_mappings = {value: key for key, value in person_mappings.items()}
        unmapped_ids = [inverted_mappings[idx] for idx in mapped_indices]
        return np.array(unmapped_ids)


    def save_embedding(self,
                       embeddings: np.ndarray,
                       parquet_root_out: str,
                       filename: str,
                       record_chunk: bool=False) -> None:
        """Save an array of embeddings to parquet.

        Args:
            embeddings (np.ndarray): array to save.
            parquet_root_out (str): parquet root from where the partitioning starts.
            filename (str): name of the file, without any paths.
            record_chunk (bool, optional): If true, the current `chunk_id` is appended to the filename.

        Notes:
            Replicates the parquet nesting from the source data, but omits the lowest nest (`dry`).
        """
        nests = OrderedDict([
            ("year", self.year),
            ("iter_name", self.iter_name),
            ("record_edge_type", int(self.record_edge_type))
            ])

        if record_chunk:
            file_path = Path(filename)
            filename = file_path.stem + "_" + str(self.chunk_id) + file_path.suffix

        save_to_parquet(
                data=embeddings,
                id_col="rinpersoon_id",
                prefix_data_cols="emb",
                filename=filename,
                parquet_nests=nests,
                parquet_root=parquet_root_out
                )



    def load_walks(self) -> pd.DataFrame:
        """Load random walks from parquet partitions."""
        if self.chunk_id is None:
            msg = "self.chunk_id is `None`, but `int` is required"
            raise ValueError(msg)

        con = duckdb.connect(":memory:")

        parquet_path = str(Path(self.parquet_root) / Path(self.parquet_nests))
        column_query = f"""
            SELECT column_name
            FROM ( DESCRIBE TABLE '{parquet_path}' )
        """
        columns = con.execute(column_query).fetchall()
        source_col = ["SOURCE"]
        step_cols = [col[0] for col in columns if "STEP" in col[0]]
        cols_to_query = source_col + step_cols

        # ruff: noqa: S608
        main_query = f"""
            SELECT {create_column_placeholders(cols_to_query)}
            FROM parquet_scan(?, filename = true)
            WHERE filename LIKE '%chunk-{self.chunk_id}.parquet'
            AND dry = 0
            AND year = ?
            AND record_edge_type = ?
            AND iter_name = ?
        """
        # ruff: enable: S608

        query_args = (
                parquet_path,
                self.year,
                int(self.record_edge_type),
                self.iter_name)

        result = con.execute(main_query, query_args)
        result_df = result.df()

        n_unique = result_df["SOURCE"].nunique()
        n_rows = result_df.shape[0]
        if n_rows == 0:
            msg = f"The query f{main_query} yielded no results with parameters {query_args}"
            raise ValueError(msg)

        if n_unique != n_rows:
            msg = "Found duplicated SOURCE nodes"
            raise RuntimeError(msg)

        con.close()
        return result_df


# this is only for development
if __name__ == "__main__":
    parquet_dir = "/gpfs/ostor/ossc9424/homedir/data/graph/walks/"

    data_file = ParquetWalks(
            parquet_root=parquet_dir,
            parquet_nests= "*/*/*/*/*.parquet",
            iter_name="walklen40_prob0.8",
            record_edge_type=False,
            year=2016)

    data_file.chunk_id = 0

    dataframe = data_file.load_walks()
