import polars as pl
import json
import logging
import os
import pandas as pd
from pop2vec.llm.src.new_code.constants import AGE
from pop2vec.llm.src.new_code.constants import BIRTH_MONTH
from pop2vec.llm.src.new_code.constants import BIRTH_YEAR
from pop2vec.llm.src.new_code.constants import DAYS_SINCE_FIRST
from pop2vec.llm.src.new_code.constants import GENDER
from pop2vec.llm.src.new_code.constants import MISSING
from pop2vec.llm.src.new_code.constants import ORIGIN
from pop2vec.llm.src.new_code.utils import print_now

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class CreatePersonDict:
    """Optimized class to create person data dictionary from Parquet files."""

    def __init__(
        self,
        file_paths,
        primary_key,
        vocab=None,
        vocab_path=None,
    ):
        """Initializes the CreatePersonDict class.

        Args:
            file_paths (List[str]): List of Parquet file paths.
            primary_key (str): Primary key column name.
            vocab (dict, optional): Custom vocabulary. Defaults to None.
            vocab_path (str, optional): Path to the vocabulary file.
                Defaults to None.
        """
        self.source_paths = file_paths.copy()
        self.background_file_path = self._get_background_file(file_paths)
        self.source_paths.remove(self.background_file_path)
        self.vocab = self._get_vocab(vocab, vocab_path)
        self.primary_key = primary_key

    def _get_vocab(self, vocab, vocab_path):
        """Gets the vocabulary.

        Args:
            vocab (dict): Custom vocabulary.
            vocab_path (str): Path to the vocabulary file.

        Returns:
            dict: Vocabulary dictionary.
        """
        if vocab is not None:
            return vocab
        if vocab_path is not None:
            return self._load_vocab(vocab_path)
        return None

    def _load_vocab(self, vocab_path):
        """Loads vocabulary from a file.

        Args:
            vocab_path (str): Path to the vocabulary file.

        Returns:
            dict: Vocabulary dictionary.
        """
        with open(vocab_path) as f:
            return json.load(f)

    def _get_background_file(self, file_paths):
        """Gets the background file from the list of file paths.

        Args:
            file_paths (List[str]): List of file paths.

        Returns:
            str: Background file path.
        """
        background_file_path = [fp for fp in file_paths if "background" in os.path.basename(fp)]
        if len(background_file_path) != 1:
            msg = f"Unique background file not found. Instead, found this list: {background_file_path}"
            raise RuntimeError(msg)
        return background_file_path[0]

    def _process_background_data(self):
        """Processes the background data.

        Returns:
            pd.DataFrame: Processed background DataFrame.
        """
        background_df = pd.read_parquet(self.background_file_path)
        background_df = background_df.fillna(MISSING)
        background_df[self.primary_key] = background_df[self.primary_key]
        background_df.set_index(self.primary_key, inplace=True)

        logging.info(f"{len(background_df)} people in background file")
        logging.info(f"Columns in background file: {list(background_df.columns)}")

        # Create 'background' column
        background_df["background"] = background_df.apply(
            lambda row: {
                "birth_year": f"{BIRTH_YEAR}_{row[BIRTH_YEAR]}",
                "birth_month": f"{BIRTH_MONTH}_{row[BIRTH_MONTH]}",
                "gender": f"{GENDER}_{row[GENDER]}",
                "origin": f"{ORIGIN}_{row[ORIGIN]}",
            },
            axis=1,
        )

        background_df = background_df[["background"]]
        return background_df

    def process_single_file(self, source_path, valid_ids):
        """
        Reads one parquet file, filters by valid_ids, does minimal transformations,
        and returns a grouped DataFrame with columns: [self.primary_key, sentence, abspos, age].
        """
        logging.info(f"Processing file: {source_path}")
        df = pd.read_parquet(source_path)

        # Filter out IDs not in background
        initial_size = len(df)
        df = df[df[self.primary_key].isin(valid_ids)]
        logging.info(
            f"Initial size of {source_path}: {initial_size}, " 
            f"after background filtering: {len(df)}"
        )

        if df.empty:
            logging.info(f"No valid records in {source_path} after filtering.")
            return pd.DataFrame()  # return empty if no data

        df = df.fillna(MISSING)

        # Identify event columns
        event_columns = [
            col for col in df.columns 
            if col not in [self.primary_key, DAYS_SINCE_FIRST, AGE, "Index"]
        ]

        # Convert each event column entry to "colname_value"
        for col in event_columns:
            df[col] = col + "_" + df[col].astype(str)

        # Combine all event columns into a single "sentence" list
        df["sentence"] = df[event_columns].values.tolist()

        # Keep only needed columns
        df = df[[self.primary_key, "sentence", DAYS_SINCE_FIRST, AGE]]

        # Sort by primary_key, DAYS_SINCE_FIRST for consistent grouping
        df = df.sort_values(by=[self.primary_key, DAYS_SINCE_FIRST])

        # Group by primary_key to store lists of sentences, abspos, age
        grouped = df.groupby(self.primary_key).agg({
            "sentence": lambda x: list(x),  # list of lists
            DAYS_SINCE_FIRST: lambda x: list(x),
            AGE: lambda x: list(x)
        })

        grouped.rename(
            columns={
                DAYS_SINCE_FIRST: "abspos",
                AGE: "age"
            },
            inplace=True
        )

        return grouped

    def generate_people_data(self, write_path):
        """
        Main function:
          1) Read and process background data.
          2) For each source file, group the data by person, store to temp parquet.
          3) Concatenate and group all partial results to get final event aggregator.
          4) Re‐sort each person's event lists by DAYS_SINCE_FIRST across partials.
          5) Merge with background data and write the final result.
        """
        # --- 1) Process background file
        background_df = self._process_background_data()
        valid_ids = set(background_df.index)

        # # --- 2) Process each event file separately and write partial results
        temp_dir = os.path.join(os.path.dirname(write_path), "temp_event_files")
        os.makedirs(temp_dir, exist_ok=True)

        for i, source_path in enumerate(self.source_paths):
            grouped = self.process_single_file(source_path, valid_ids)

            if not grouped.empty:
                # Write partial results to disk
                temp_parquet_path = os.path.join(temp_dir, f"temp_{i}.parquet")
                # Save index (which is the person ID) so we can group again
                grouped.reset_index().to_parquet(temp_parquet_path, index=False)
            
            # Force release memory before next iteration
            del grouped

            logging.info(
                f"Finished processing file {i+1}/{len(self.source_paths)}: {source_path}"
            )

        # --- 3) Read all partial parquet files, combine them with one more groupby
        temp_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                      if f.endswith(".parquet")]
        if temp_files:
            chunks = []
            for tf in tqdm(temp_files):
                chunks.append(pl.read_parquet(tf).to_pandas())

            logging.info("read all temp files")
            all_events_df = pd.concat(chunks, ignore_index=True)
            del chunks
            logging.info("concatenation done")
            # Now group again to combine the partial lists for each ID
            all_events_df = all_events_df.groupby(self.primary_key).agg({
                "sentence": lambda x: [item for sublist in x for item in sublist],  # sum of lists concatenates them
                "abspos": lambda x: [item for sublist in x for item in sublist],
                "age": lambda x: [item for sublist in x for item in sublist]
            })

            logging.info("grouping for concatenation done")
        else:
            # If no event data at all, create an empty aggregator
            all_events_df = pd.DataFrame(
                columns=["sentence", "abspos", "age"]
            )
            all_events_df.index.name = self.primary_key
        # --- 4) Re-sort the final lists by DAYS_SINCE_FIRST
        def reorder_lists(abspos_list, age_list, sentence_list):
            # zip them together
            combined = list(zip(abspos_list, age_list, sentence_list))
            # sort by abspos (the first element in each tuple)
            combined_sorted = sorted(combined, key=lambda x: x[0])
            # unzip them back to three parallel lists
            sorted_abspos, sorted_age, sorted_sentence = zip(*combined_sorted)
            return list(sorted_abspos), list(sorted_age), list(sorted_sentence)

        # Make sure columns exist
        for col in ["sentence", "abspos", "age"]:
            if col not in all_events_df.columns:
                all_events_df[col] = [[] for _ in range(len(all_events_df))]

        # Apply reorder_lists per row
        all_events_df["abspos"], all_events_df["age"], all_events_df["sentence"] = zip(
            *all_events_df.apply(
                lambda row: reorder_lists(
                    row["abspos"],
                    row["age"],
                    row["sentence"]
                ),
                axis=1
            )
        )

        logging.info("reordering by DAYS_SINCE_FIRST done")
        # Compute segment after final sorting
        def compute_segment(abspos):
            seg = []
            for i in range(len(abspos)):
                if i > 0 and abspos[i-1] == abspos[i]:
                    seg.append(1)
                else:
                    seg.append(2)
            return seg

        all_events_df["segment"] = all_events_df["abspos"].apply(compute_segment)

        logging.info("segment computation done")
        # --- 5) Merge event aggregator with background
        people_df = background_df.merge(all_events_df, 
                                        left_index=True, 
                                        right_index=True, 
                                        how="left")

        # Replace any NaN with empty lists if necessary
        for col in ["sentence", "abspos", "age", "segment"]:
            people_df[col] = people_df[col].apply(lambda x: x if isinstance(x, list) else [])

        # Flatten to final schema
        people_df = people_df.reset_index().rename(
            columns={"index": self.primary_key}
        )
        people_df = people_df[
            [self.primary_key, "background", "sentence", "abspos", "age", "segment"]
        ]

        # Shuffle if desired
        people_df = people_df.sample(frac=1)

        logging.info("shuffling done")
        # Calculate a row-group size to avoid huge memory usage on write
        size_in_bytes = people_df.memory_usage(deep=True).sum()
        bytes_per_row = max(1, size_in_bytes / max(1, len(people_df)))
        # ensuring each row_group is at most ~64 MB
        row_group_size = int((64 * 10**6) // bytes_per_row)

        # Final write
        people_df.to_parquet(write_path, index=False, row_group_size=row_group_size)
        logging.info(f"Data written to {write_path}")
        logging.info(f"Temporary files in {temp_dir} can be cleaned up if desired.")