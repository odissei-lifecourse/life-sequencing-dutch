import pandas as pd

from dataclasses import dataclass, field
from pop2vec.llm.src.data_new.vocabulary import Vocabulary
from typing import TYPE_CHECKING, Generic, List, NewType, Optional, TypeVar

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, cast

import dask
import pandas as pd

from pop2vec.llm.src.data_new.decorators import save_pickle, save_tsv
from pop2vec.llm.src.data_new.serialize import DATA_ROOT
from pop2vec.llm.src.data_new.sources.base import TokenSource

from pop2vec.llm.src.new_code.constants import BIRTH_YEAR, BIRTH_MONTH, ORIGIN, GENDER, TIME_COLUMNS, MISSING, DELIMITER

from tqdm import tqdm
from multiprocessing import Pool
import logging

logging.basicConfig(level=logging.DEBUG)

class DataFile():
  
  def __init__(self, path, primary_key, name=None):
    self.path = path
    self.name = name
    self.df = None
    self.primary_key = primary_key

  def _get_df(self):
    if self.df is None:
      if self.path.endswith('.csv'):
        self.df = pd.read_csv(self.path)
      elif self.path.endswith('.parquet'):
        self.df = pd.read_parquet(self.path)
      else:
        raise ValueError(f'{path} is not a csv or parquet file!')
    return self.df

  def _get_unique_tokens_for_column(self, data, column):
    tokens_with_column_name = [f"{column}_{str(d)}" for d in data.unique()]

    return pd.DataFrame({
        "TOKEN": tokens_with_column_name,
        "CATEGORY": f"{self.name}_{column}",
      }
    )

  def get_all_unique_tokens_with_category(self):
    df = self._get_df()
    unique_tokens_by_category = []
    for column in df.columns:
      if (
        column not in TIME_COLUMNS and 
        column != self.primary_key
      ):
        unique_tokens_by_category.append(
          self._get_unique_tokens_for_column(df[column], column,)
        )
    return unique_tokens_by_category

def get_all_unique_tokens_with_category(source_file):
  return source_file.get_all_unique_tokens_with_category()
          

@dataclass
class CustomVocabulary(Vocabulary):
    """
    Generate a vocabulary from the tokenized training data of a corpus.

    :param corpus: The :class:`src.data_new.Corpus` to generate the vocabulary from.
    :param name: Name of the vocabulary.
    :param general_tokens: General tokens.
    :param background_tokens: Background tokens.
    :param year_range: Range of years (inclusive) to generate tokens for.
    :param min_token_count: The minimum number of occurances of a token to be included
        in the vocabulary.
    :param min_token_count_field: Field-specific minimum token counts.

    """
    
    name: str
    data_files: List[DataFile] = field(
        default_factory=lambda: []
    )
    general_tokens: List[str] = field(
        default_factory=lambda: [
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[UNK]",
        ]
    )
    background_tokens: List[str] = field(
        default_factory=lambda: [f"{GENDER}_1", f"{GENDER}_2", f"{GENDER}_{MISSING}"]
    )
    # year_range: Tuple[int, int] = field(
    #     default_factory=lambda: (1820, 2023)  # inclusive
    # )
    # origin_range: Tuple[int, int] = field(
    #     default_factory=lambda: (1, 2) # inclusive
    # )
    # min_token_count: int = 1000 
    # min_token_count_field: Dict[str, int] = field(default_factory=dict)

    vocab_df = None

    def load_vocab(self, path: str) -> None:
        """Load the vocabulary DataFrame from a CSV file."""
        if path.endswith('.csv'):
          self.vocab_df = pd.read_csv(path, index_col='ID')
        elif path.endswith('.parquet'):
          self.vocab_df = pd.read_parquet(path)
          self.vocab_df.set_index('ID', inplace=True)
        else:
          raise ValueError(f"{path} is not a csv or parquet!")
        self.vocab_df = self.vocab_df.rename_axis(index="ID")
        # the following looks unnecessary but the code breaks without this
        # TODO: Figure out why this is necessary
        self.vocab_df['ID'] = self.vocab_df.index

    # @save_tsv(DATA_ROOT / "processed/vocab/{self.name}/", on_validation_error="error")
    def vocab(self, num_processes=1) -> pd.DataFrame:
        """Filters the tokens by count, sorts them lexicographically for each source,
        and computes the voculary with the field labels as categories.
        """
        if self.vocab_df is not None:
          return self.vocab_df

        general = pd.DataFrame({"TOKEN": self.general_tokens, "CATEGORY": "GENERAL"})
        background = pd.DataFrame(
            {"TOKEN": self.background_tokens, "CATEGORY": "BACKGROUND"}
        )
        month = pd.DataFrame(
            {"TOKEN": [f"{BIRTH_MONTH}_{i}" for i in range(1, 13)] + [f"{BIRTH_MONTH}_{MISSING}"], "CATEGORY": BIRTH_MONTH}
        )
        # year = pd.DataFrame(
        #     {
        #         "TOKEN": [
        #             f"{BIRTH_YEAR}_{i}"
        #             for i in range(self.year_range[0], self.year_range[1] + 1)
        #         ] + [f"{BIRTH_YEAR}_{MISSING}"],
        #         "CATEGORY": BIRTH_YEAR,
        #     }
        # )
        # origin = pd.DataFrame(
        #     {
        #         "TOKEN": [
        #             f"{ORIGIN}_{i}"
        #             for i in range(self.origin_range[0], self.origin_range[1] + 1)
        #         ] + [f"{ORIGIN}_{MISSING}"],
        #         "CATEGORY": ORIGIN,
        #     }
        # )
        
        vocab_parts = [general, background]#, month, year, origin]
        if num_processes == 1:
          for source_file in tqdm(self.data_files):
            vocab_parts.extend(
              source_file.get_all_unique_tokens_with_category()
            )
        else:
          logging.info("Starting multiprocessing")
          with Pool(processes=num_processes) as pool:
            results = list(tqdm(
              pool.imap_unordered(
                get_all_unique_tokens_with_category, 
                self.data_files
              ), 
              total=len(self.data_files)
            ))
          for tokens in results:
            vocab_parts.extend(tokens)  
        # debug
        # total = 0
        # for part in vocab_parts:
        #   print(f"{part.iloc[0]}")
        #   print(f"length = {len(part)}")
        #   total += len(part)
        #   print(f"current total = {total}")

        # concatenates the vocab_parts and drops duplicate tokens
        # then resets the index to be contiguous from 0 to n-1
        self.vocab_df = pd.concat(
          vocab_parts, ignore_index=True
        ).drop_duplicates(subset=['TOKEN'], keep='first').reset_index(drop=True)
        self.vocab_df = self.vocab_df.rename_axis(index="ID")
        # the following looks unnecessary but the code breaks without this
        # TODO: Figure out why this is necessary
        self.vocab_df['ID'] = self.vocab_df.index

        return self.vocab_df  

    def save_vocab(self, path, num_processes=1):
      vocab_df = self.vocab(num_processes)
      if path.endswith('.csv'):
        vocab_df.to_csv(path, index=False)
      elif path.endswith('.parquet'):
        vocab_df.to_parquet(path, index=False)
      else:
        raise ValueError(f'{path} is neither a csv nor a parquet file!')
