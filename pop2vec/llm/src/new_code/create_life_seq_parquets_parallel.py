import os
import json
import logging
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

from pop2vec.llm.src.new_code.constants import (
    GENDER,
    BIRTH_MONTH,
    BIRTH_YEAR,
    ORIGIN,
    DAYS_SINCE_FIRST,
    AGE,
    IGNORE_COLUMNS,
    MISSING,
)
from pop2vec.llm.src.new_code.utils import print_now

logging.basicConfig(level=logging.INFO)


class ParallelCreatePersonDict:
    """Parallel class to create person data dictionary using Dask."""

    def __init__(
        self,
        file_paths,
        primary_key,
        vocab=None,
        vocab_path=None,
    ):
        """Initializes the ParallelCreatePersonDict class.

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
        elif vocab_path is not None:
            return self._load_vocab(vocab_path)
        else:
            return None

    def _load_vocab(self, vocab_path):
        """Loads vocabulary from a file.

        Args:
            vocab_path (str): Path to the vocabulary file.

        Returns:
            dict: Vocabulary dictionary.
        """
        with open(vocab_path, 'r') as f:
            return json.load(f)

    def _get_background_file(self, file_paths):
        """Gets the background file from the list of file paths.

        Args:
            file_paths (List[str]): List of file paths.

        Returns:
            str: Background file path.
        """
        background_file_path = [
            fp for fp in file_paths if 'background' in os.path.basename(fp)
        ]
        assert len(background_file_path) == 1, (
            f"Unique Background file not found.\n"
            f"found background files list: {background_file_path}"
        )
        return background_file_path[0]

    def _process_background_data(self):
        """Processes the background data.

        Returns:
            dd.DataFrame: Processed background Dask DataFrame.
        """
        background_df = dd.read_parquet(self.background_file_path)
        background_df = background_df.fillna(MISSING)
        background_df[self.primary_key] = background_df[self.primary_key]
        background_df = background_df.set_index(self.primary_key)

        logging.info(f"{len(background_df)} people in background file")
        logging.info(
            f"Columns in background file: {list(background_df.columns.compute())}"
        )

        # Create 'background' column
        background_df['background'] = background_df.apply(
            lambda row: {
                'birth_year': f"{BIRTH_YEAR}_{row[BIRTH_YEAR]}",
                'birth_month': f"{BIRTH_MONTH}_{row[BIRTH_MONTH]}",
                'gender': f"{GENDER}_{row[GENDER]}",
                'origin': f"{ORIGIN}_{row[ORIGIN]}",
            },
            axis=1,
            meta=('background', 'object'),
        )

        background_df = background_df[['background']]
        return background_df

    def _process_event_files(self, valid_ids):
        """Processes event files individually and creates 'sentence' columns.

        Args:
            valid_ids (set): Set of valid primary keys present in background data.

        Returns:
            dd.DataFrame: Concatenated events Dask DataFrame with 'sentence' column.
        """
        event_dfs = []
        for file_count, source_path in enumerate(self.source_paths):
            logging.info(f"Processing file: {source_path}")

            df = dd.read_parquet(
                source_path,
                columns=lambda column: column not in IGNORE_COLUMNS,
            )

            # Filter out IDs not present in background data
            df = df[df[self.primary_key].isin(valid_ids)]

            df_size = df.shape[0].compute()
            if df_size == 0:
                logging.info(f"No valid records in {source_path} after filtering.")
                continue
            else:
                logging.info(
                    f"Final size after filtering using background = {df_size}"
                )

            # Fill missing values
            df = df.fillna(MISSING)

            # Get event-specific columns
            event_columns = [
                col
                for col in df.columns
                if col
                not in [self.primary_key, DAYS_SINCE_FIRST, AGE, 'Index']
            ]

            # Create 'sentence' column
            for col in event_columns:
                df[col] = col + '_' + df[col].astype(str)

            df['sentence'] = df[event_columns].apply(
                lambda row: list(row), axis=1, meta=('sentence', 'object')
            )

            # Keep only necessary columns
            df = df[[self.primary_key, 'sentence', DAYS_SINCE_FIRST, AGE]]

            event_dfs.append(df)

            logging.info(
                f"Files processed: {file_count + 1}, "
                f"Remaining: {len(self.source_paths) - file_count - 1}"
            )

        if event_dfs:
            merged_df = dd.concat(event_dfs, axis=0)
            logging.info(f"Total number of event records: {merged_df.shape[0].compute()}")
        else:
            merged_df = dd.from_pandas(pd.DataFrame(), npartitions=1)
            logging.info("No event data files found.")

        return merged_df

    def _process_events(self, events_df):
        """Processes the events DataFrame to group events by person and compute segments.

        Args:
            events_df (dd.DataFrame): DataFrame containing event data.

        Returns:
            dd.DataFrame: Grouped events DataFrame with aggregated event information.
        """
        # Sort and group events by primary_key and DAYS_SINCE_FIRST
        events_df = events_df.map_partitions(
            lambda df: df.sort_values(by=[self.primary_key, DAYS_SINCE_FIRST])
        )

        grouped = events_df.groupby(self.primary_key).agg(
            {'sentence': 'list', DAYS_SINCE_FIRST: 'list', AGE: 'list'}
        )

        grouped = grouped.rename(
            columns={DAYS_SINCE_FIRST: 'abspos', AGE: 'age'}
        )

        # Compute 'segment' per person
        grouped['segment'] = grouped['abspos'].apply(
            self._compute_segment, meta=('segment', 'object')
        )

        return grouped

    def _compute_segment(self, abspos):
        """Computes the 'segment' list based on 'abspos'.

        Args:
            abspos (List[int]): List of absolute positions.

        Returns:
            List[int]: List of segment identifiers.
        """
        segment = []
        for i in range(len(abspos)):
            if i > 0 and abspos[i - 1] == abspos[i]:
                segment.append(1)
            else:
                segment.append(2)
        return segment

    def generate_people_data(self, write_path):
        """Generates people data and writes to a Parquet file.

        Args:
            write_path (str): Path to write the Parquet file.
        """
        # Initialize Dask client
        cluster = LocalCluster(
            n_workers=72,
            threads_per_worker=1,
            memory_limit='4GB',
        )
        client = Client(cluster)
        logging.info("Dask client initialized.")

        # Process background data
        background_df = self._process_background_data()
        valid_ids = client.compute(background_df.index).result()
        valid_ids = set(valid_ids)

        # Process event data
        events_df = self._process_event_files(valid_ids)

        if events_df.shape[0].compute() > 0:
            # Process events and group by primary_key
            grouped_events = self._process_events(events_df)

            # Merge background and events data
            people_df = background_df.merge(
                grouped_events,
                how='left',
                left_index=True,
                right_index=True,
            )
        else:
            # If no events, create empty columns
            people_df = background_df.copy()
            people_df['sentence'] = [[] for _ in range(len(people_df))]
            people_df['abspos'] = [[] for _ in range(len(people_df))]
            people_df['age'] = [[] for _ in range(len(people_df))]
            people_df['segment'] = [[] for _ in range(len(people_df))]

        # Replace NaN lists with empty lists
        for col in ['sentence', 'abspos', 'age', 'segment']:
            people_df[col] = people_df[col].apply(
                lambda x: x if isinstance(x, list) else [],
                meta=(col, 'object'),
            )

        # Reset index and reorder columns
        people_df = people_df.reset_index().rename(
            columns={'index': self.primary_key}
        )
        people_df = people_df[
            [
                self.primary_key,
                'background',
                'sentence',
                'abspos',
                'age',
                'segment',
            ]
        ]

        # Write to Parquet file
        people_df.to_parquet(write_path, index=False, compute=True)
        logging.info(f"Data written to {write_path}")

        # Close the Dask client
        client.close()
        cluster.close()
