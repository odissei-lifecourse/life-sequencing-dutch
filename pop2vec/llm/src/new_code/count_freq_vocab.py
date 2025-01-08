import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


vocab_df = pd.read_csv('?.csv')
freq = {}
for token in vocab_df['TOKEN']:
  freq[token] = 0


with h5py.File('?.h5', 'r') as hdf5:
  original_sequence = torch.from_numpy(hdf5['original_sequence'][indices]).share_memory_()
  for seq in tqdm(original_sequence):
    for sent in seq:
      for token in sent:
        if token in freq:
          freq[token] += 1

vocab_df['count'] = vocab_df['TOKEN'].map(freq)

vocab_df = vocab_df.sort_values(by='count')
print(vocab_df[:10])
print("-"*10)
print(vocab_df[-10:])
vocab_df.to_csv('good_vocab_with_count.csv')

