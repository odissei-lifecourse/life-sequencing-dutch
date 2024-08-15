import numpy as np 
import pyreadstat
import pandas as pd 
import os 
import re 
from pathlib import Path 

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


def yield_filepaths(root):
  "Yield full paths to all files in `root` and its subdirs"
  for root, _, files in os.walk(root):
      for filename in files:
          filename = os.path.join(root, filename)
          if os.path.isfile(filename): 
              yield filename   


def get_unique_source_files(root):
  """Return full paths to unique data files.

  Deals with the files of summary statistics generated from data. 

  Args:
    root (str): root directory to search


  Example: from ["data1_meta.json", "data1_covariance.csv", "data1_columns.csv"] it returns "data1".
  """
  files = list(yield_filepaths(root))
  src_files = set()

  for f in files:
    src_file, _ = split_at_last_match(f, "_")
    src_files.add(src_file)

  return list(src_files)



def split_at_last_match(s, p):
  "Split a string `s` at the last occurence of `p`"
  last_match = list(re.finditer(p, s))[-1]
  first = s[:last_match.end()-1]
  second = s[last_match.end():]
  return first, second


def replace_numeric_in_path(input_path, level, replace_value):
  """Replace a 4-digit numeric a path
  
  Args:
    input_path (str or pathlib.Path): path to operate on.
    level (int): level at which to replace on. Level is relative to the filename in a directory.
      Thus, a change in the filename requires level=0. A change in the parent directory of the file
      requires level=1, etc. See example.
    replace_value (str): value for the replacement. 

  Example
    replace_numeric_in_path('path/to/my2011file.csv', 0, 2015) -> 'path/to/my2015file.csv' 
  """

  path = Path(input_path)
  parts = list(path.parts)
  
  index = -1 - level
  part = parts[index]
  match = re.search(r'\d{4}', part)
  assert match, f"found none or multiple 4-digit numeric elements in {path}"
  
  new_part = part[:match.start()] + str(replace_value) + part[match.end():]
  parts[index] = new_part
  
  new_path = Path(*parts)
  return str(new_path)


def split_classes_and_probs(cats):
  """Split a string of categories and probabilities
  
  Summary statistics of categorical variables are stored as "class--prob". 
  Sometimes, the class is "--", preventing a simple splitting. This function
  deals with this.  
  """
  cats_splitted = []

  for cat in cats:
    if "---" in cat:
        splitted = split_single_category(cat) 
    else:
        splitted = cat.split("--")
    cats_splitted.append(splitted)

  return cats_splitted


def split_single_category(x):
  """Split a pair of a class value and a probability
  
  Example: "abc--0.44" -> ["abc", 0.44]
  """
  prob = re.findall(r'\d+\.\d+', x)[0]
  string_part = x.split(prob)[0]

  x_prob = float(prob)
  x_class = string_part[:-2]

  result = [x_class, x_prob]
  return result
