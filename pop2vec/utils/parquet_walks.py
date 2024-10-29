from dataclasses import dataclass
import duckdb
import pandas as pd


def create_column_placeholders(cols) -> str:  # suggested by Claude
    """Return comma-separated columns to query."""
    return ", ".join(cols)


@dataclass
class ParquetWalks:
    """Container for loading dataframes of walks for deepwalk."""

    parquet_path: str
    year: str
    iter_name: str
    chunk_id = int | None

    def load_walks(self) -> pd.DataFrame:
        """Load random walks from parquet partitions."""
        if self.chunk_id is None:
            msg = "self.chunk_id is `None`, but `int` is required"
            raise ValueError(msg)

        con = duckdb.connect(":memory:")

        column_query = """
            SELECT column_name
            FROM ( DESCRIBE TABLE '?' )
        """
        columns = con.execute(column_query, (self.parquet_path,)).fetchall()
        source_col = ["SOURCE"]
        step_cols = [col[0] for col in columns if "STEP" in col[0]]
        cols_to_query = source_col + step_cols

        # ruff: noqa: S608
        main_query = f"""
            SELECT {create_column_placeholders(cols_to_query)}
            FROM parquet_scan(?, filename = true)
            WHERE filename LIKE '%chunk-?.parquet'
            AND dry = 0
            AND year = ?
            AND iter_name = ?
        """
        # ruff: enable: S608

        query_args = (*cols_to_query, self.parquet_path, self.year, self.iter_name, f"%chunk-{self.chunk_id}.parquet")
        result = con.execute(main_query, query_args)
        result_df = result.df()

        n_unique = result_df["SOURCE"].nunique()
        n_rows = result_df.shape[0]
        if n_unique != n_rows:
            msg = "Found duplicated SOURCE nodes"
            raise RuntimeError(msg)

        con.close()
        return result_df
