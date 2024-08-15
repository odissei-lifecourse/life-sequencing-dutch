import os
import pandas as pd


directory_paths = ['.', 'income_yearly', 'job_yearly']
for filename in os.listdir(directory_paths):
  path = os.path.join(directory_path, filename)
  if path.endswith('.csv'):
    print(filename)
    #input()
    df = pd.read_csv(filename)
    print("# of rows =", len(df))
    print("# of columns =", len(df.columns))
    print("columns, dtypes =", df.dtypes)
    for column in columns:
      print("working for column", column)
      print(f"dtype = {df.dtype[column]}")
      unique_items = df[column].unique()
      print(f"# of unique items in column = {len(unique_items)}")
      print(f"The unique items are",)
      for item in unique_items:
        print(item)
      print("-"*30)
    print(f"work completed for {filename}")
    print("*"*30)