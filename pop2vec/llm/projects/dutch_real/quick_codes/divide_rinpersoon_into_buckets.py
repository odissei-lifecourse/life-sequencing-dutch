import pandas as pd
import json

background_path = 'projects/dutch_real/data/background.csv'

df = pd.read_csv(background_path, usecols=['RINPERSOON'], dtype=int)

ids = df['RINPERSOON'].tolist()

buckets = []
n = len(ids)
last = 0
for i in range(4):
  next_last = int(last + (n/4)+1)
  buckets.append(ids[last:next_last])
  with open(f'projects/dutch_real/gen_data/rinpersoon{i}.json', 'w') as f:
    json.dump(buckets[-1], f)
    print(f"step {i}: dumped {len(buckets[-1])} ids from idx {last} to {next_last}")
  last = next_last