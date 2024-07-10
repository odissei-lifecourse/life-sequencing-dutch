import numpy as np 
import pyreadstat
import logging
import pandas as pd 

def check_column_names(column_names, names_to_check):
    "Check if a colum name matches exactly, or on a substring, the names to check."
    for column_name in column_names:
        for name in names_to_check:
            if name in column_name:
                return True
    return False


def subsample_from_ids(df, id_col="RINPERSOON", frac=0.1):
  """Draw all rows from a random sample of record ids.

  Args:
    df (pd.DataFrame): dataframe to sample from. 
    id_col (str): column with the identifier. 
    frac (float): Sampling fraction of RINPERSOON records.  
  """
  assert frac > 0 and frac < 1, "frac needs to be between 0 and 1"
  ids = df[id_col].unique()
  n_ids = len(list(ids))
  rng = np.random.default_rng(1234)
  sampled_ids = rng.choice(a=ids, size=int(n_ids*frac), replace=False)
  mask = df[id_col].isin(sampled_ids)
  return df.loc[mask, :]


def sample_from_file(source_file_path, n_rows):
  """Sample n_rows from a file. 
  
  Returns subsampled df and the total number of rows in the file.
  If n_rows is None, the whole file is read. For csvs related to SPOLISBUS and GBAHUISHOUDENS,
  this is using the python engine from pandas, which is slow on large files.
  """
  if source_file_path.endswith('.sav'):
    df, _ = pyreadstat.read_sav(source_file_path, metadataonly=True)
    columns = df.columns

    df, _ = pyreadstat.read_sav(source_file_path, usecols=[columns[0]])
    nobs = df.shape[0]

    if n_rows is not None:
      df, _ = pyreadstat.read_sav(source_file_path, row_limit=n_rows)
    else:
      df, _ = pyreadstat.read_sav(source_file_path)
       
  elif source_file_path.endswith(".csv"):
    sep = None
    if "SPOLISBUS" in source_file_path:
      sep = ";"
    elif "GBAHUISHOUDENS" in source_file_path:
      sep = ","
    engine = "python" if sep is None else "c"
    
    columns = pd.read_csv(source_file_path, 
                          index_col=0, 
                          nrows=0, 
                          sep=sep,
                          engine=engine).columns.tolist()

    df = pd.read_csv(source_file_path, usecols=[columns[0]], engine=engine, sep=sep)
    nobs = df.shape[0]

    if n_rows is not None:
      df = pd.read_csv(source_file_path, nrows=n_rows, engine=engine, sep=sep)
    else:
      df = pd.read_csv(source_file_path, engine=engine, sep=sep)
  else:
    raise ValueError(f"wrong file extension found for {source_file_path}")
  
  if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

  return df, nobs

