import pandas as pd

data_dir = "/gpfs/ostor/ossc9424/homedir/data/"
background_path = data_dir + "llm/raw/background.csv"


df_str = pd.read_csv(background_path, dtype={"RINPERSOON": "object"}, usecols="RINPERSOON")
df_int = pd.read_csv(background_path, dtype={"RINPERSOON": "int"}, usecols="RINPERSOON")

list1 = list(df_str["col1"].unique())
set1 = set([int(v) for v in list1])
set2 = set(df_int["col2"].unique())

assert set1 == set2, "the two unique sets are not equal"
assert len(set1) == len(df_str), f"all RINPERSOON IDs are not unique. df size = {len(df)}, set size = {len(set1)}"