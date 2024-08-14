import os
import sys
import random, string 
import numpy as np
import resource
import h5py
# Get the current working directory
# current_directory = os.getcwd()
# sys.path.append('/Users/tanzir5/Documents/GitHub/life-sequencing-dutch/code/llm')
# Print the current directory
# print("Current Directory:", current_directory)
# print(sys.path)

from src.new_code.pipeline import init_hdf5_datasets, write_to_hdf5, convert_to_numpy

def get_data_dict():
  SIZE = 5000
  VOCAB_SIZE = 1
  CONTEXT_LEN = 512
  random_positions = np.random.randint(1, CONTEXT_LEN, size=SIZE)
  padding_mask = np.where(np.arange(CONTEXT_LEN) < random_positions[:, None], 0, 1).tolist()
  random_positions = np.full(SIZE, CONTEXT_LEN)#np.random.randint(1, CONTEXT_LEN, size=SIZE)
  target_tokens = [np.random.randint(0, VOCAB_SIZE, size=random_positions[i]) for i in range(SIZE)]
  target_pos = [np.random.randint(0, 1, size=random_positions[i]) for i in range(SIZE)]
  target_cls = [np.random.randint(0, 3) for _ in range(SIZE)]
  data_dict = {
    'sequence_id': [
      ''.join(
        random.choices(
          string.ascii_letters + string.digits, k=random.randint(1, 5)
        )
      ) for _ in range(SIZE)
    ],
    'original_sequence': np.random.randint(0, VOCAB_SIZE, (SIZE, CONTEXT_LEN)).tolist(),
    'input_ids': np.random.randint(0, VOCAB_SIZE, (SIZE, 4, CONTEXT_LEN)).tolist(),
    'padding_mask': padding_mask,
    'target_tokens': target_tokens,
    'target_pos': target_pos,
    'target_cls': target_cls,
  }
  return data_dict

def test_write_to_hdf5():
  write_path = 'test.h5'
  #for _ in range(3):
  data_dict = get_data_dict()
  convert_to_numpy(data_dict)
  write_to_hdf5(write_path, data_dict)

def get_peak_memory():
    # Get peak memory usage (in kilobytes) on Unix systems
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024*1024)
    return usage

if __name__ == '__main__':
  file_path = 'test.h5'
  with h5py.File(file_path, 'r') as hdf5:
    for key in hdf5:
      print(key, hdf5[key].shape)
  exit(0)
  print(f"Peak Memory Usage: {get_peak_memory()} GB")

  test_write_to_hdf5()
  print(f"Peak Memory Usage: {get_peak_memory()} GB")

  # test_write_to_hdf5()
  # print(f"Peak Memory Usage: {get_peak_memory()} GB")
