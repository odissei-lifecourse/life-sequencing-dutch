from pop2vec.llm.src.new_code.create_person_dict import CreatePersonDict
from pop2vec.llm.src.new_code.utils import get_column_names, print_now, read_json
from pop2vec.llm.src.new_code.constants import DAYS_SINCE_FIRST, AGE

import os
import fnmatch
import sys
import logging



PRIMARY_KEY = "PRIMARY_KEY"
DATA_PATH = 'DATA_PATH'
SEQUENCE_WRITE_PATH = "SEQUENCE_WRITE_PATH"

logging.basicConfig(level=logging.INFO)



def create_person_sequence(file_paths, custom_vocab, write_path, primary_key):
    """Creates person sequence data and writes it to a Parquet file.

    Args:
        file_paths (List[str]): List of Parquet file paths.
        custom_vocab (dict): Custom vocabulary.
        write_path (str): Path to write the output Parquet file.
        primary_key (str): Primary key column name.
    """
    creator = CreatePersonDict(
        file_paths=file_paths,
        primary_key=primary_key,
        vocab=custom_vocab,
    )
    creator.generate_people_data(write_path)

def data_integrity_check(fp, primary_key):
  columns = get_column_names(fp)
  if (
      primary_key in columns and
      ("background" in fp or {DAYS_SINCE_FIRST, AGE}.issubset(columns))
  ):
    logging.info(f"{fp} passes data_integrity_check")
    return True
  else:
    logging.info(
      f"""{fp} cannot be processed
      columns = {columns}
      primary_key ({primary_key}) in columns = {primary_key in columns}
      is this the background file = {'background' in fp}
      {DAYS_SINCE_FIRST} in columns = {DAYS_SINCE_FIRST in columns}
      {AGE} in columns = {AGE in columns}"""
    )
    return False

def get_data_files_from_directory(directory, primary_key):
    """Gets data file paths from a directory.

    Args:
        directory (str): Directory path to search for Parquet files.
        primary_key (str): Primary key column name.

    Returns:
        List[str]: List of Parquet file paths.
    """
    data_files = []
    for root, _, files in os.walk(directory):
        for filename in fnmatch.filter(files, '*.parquet'):
            current_file_path = os.path.join(root, filename)
            if data_integrity_check(current_file_path, primary_key):
              data_files.append(current_file_path)
    return data_files


if __name__ == "__main__":
    CFG_PATH = sys.argv[1]
    cfg = read_json(CFG_PATH)

    data_file_paths = get_data_files_from_directory(
        cfg[DATA_PATH],
        cfg[PRIMARY_KEY]
    )
    print_now(f"# of data_files_paths = {len(data_file_paths)}")

    create_person_sequence(
        file_paths=data_file_paths,
        custom_vocab=None,  # Replace with custom_vocab if available
        write_path=cfg[SEQUENCE_WRITE_PATH],
        primary_key=cfg[PRIMARY_KEY],
    )
