import os
import json
import pandas as pd
import logging

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
            pd.DataFrame: Processed background DataFrame.
        """
        background_df = pd.read_parquet(self.background_file_path)
        background_df = background_df.fillna(MISSING)
        background_df[self.primary_key] = background_df[self.primary_key]
        background_df.set_index(self.primary_key, inplace=True)

        logging.info(f"{len(background_df)} people in background file")
        logging.info(
          f"Columns in background file: {list(background_df.columns)}"
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
        )

        background_df = background_df[['background']]
        return background_df

    def _process_event_files(self, valid_ids):
        """Processes event files individually and creates 'sentence' columns.

        Returns:
            pd.DataFrame: Concatenated events DataFrame with 'sentence' column.
        """
        event_dfs = []
        for file_count, source_path in enumerate(self.source_paths):
            logging.info(f"Processing file: {source_path}")

            df = pd.read_parquet(
                source_path,
                # columns=lambda column: column not in IGNORE_COLUMNS,
            )
            # Filter out IDs not present in background data
            initial_size = len(df)
            df = df[df[self.primary_key].isin(valid_ids)]

            if df.empty:
                logging.info(f"No valid records in {source_path} after filtering.")
                continue
            else:
              logging.info(
                f"""Initial size of {source_path} df = {initial_size}
                Final size after filtering using background = {len(df)}"""
              )
            # Fill missing values
            df = df.fillna(MISSING)

            # Get event-specific columns (excluding primary_key, DAYS_SINCE_FIRST, AGE, 'Index')
            event_columns = [
                col
                for col in df.columns
                if col
                not in [self.primary_key, DAYS_SINCE_FIRST, AGE, 'Index']
            ]

            # Create 'sentence' column
            for col in event_columns:
                df[col] = col + '_' + df[col].astype(str)

            df['sentence'] = df[event_columns].values.tolist()

            # Keep only necessary columns
            df = df[[self.primary_key, 'sentence', DAYS_SINCE_FIRST, AGE]]

            event_dfs.append(df)

            logging.info(
                f"Files processed: {file_count + 1}, "
                f"Remaining: {len(self.source_paths) - file_count - 1}"
            )

        if event_dfs:
            merged_df = pd.concat(event_dfs, ignore_index=True)
            logging.info(f"Total number of event records: {len(merged_df)}")
        else:
            merged_df = pd.DataFrame()
            logging.info("No event data files found.")

        return merged_df

    def _process_events(self, events_df):
        """Processes the events DataFrame to group events by person and compute segments.

        Args:
            events_df (pd.DataFrame): DataFrame containing event data.

        Returns:
            pd.DataFrame: Grouped events DataFrame with aggregated event information.
        """


        # events_df = events_df.copy()
        
        # Sort and group events by id and DAYS_SINCE_FIRST
        events_df = events_df.sort_values(
            by=[self.primary_key, DAYS_SINCE_FIRST]
        )
        grouped = events_df.groupby(self.primary_key).agg(
            {'sentence': list, DAYS_SINCE_FIRST: list, AGE: list}
        )
        grouped = grouped.rename(
            columns={DAYS_SINCE_FIRST: 'abspos', AGE: 'age'}
        )

        # Compute 'segment' per person
        grouped['segment'] = grouped['abspos'].apply(self._compute_segment)
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
        # Process background data
        background_df = self._process_background_data()
        valid_ids = set(background_df.index)

        # Process event data
        events_df = self._process_event_files(valid_ids)
        
        if not events_df.empty:
            # Process events and group by id
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
                lambda x: x if isinstance(x, list) else []
            )

        # Reset index and reorder columns
        people_df = people_df.reset_index().rename(
            columns={'index': self.primary_key}
        )
        people_df = people_df[
            [self.primary_key, 'background', 'sentence', 'abspos', 'age', 'segment']
        ]

        # Write to Parquet file
        people_df.to_parquet(write_path, index=False, row_group_size=len(people_df)//65)
        logging.info(f"Data written to {write_path}")
