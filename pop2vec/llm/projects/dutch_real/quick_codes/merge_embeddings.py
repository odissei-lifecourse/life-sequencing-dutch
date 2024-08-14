import json

path = 'projects/dutch_real/gen_data/64'

def merge(x, path):
  merged_embedding_dict = {}
  for i in range(4):
    with open(f'{path}/{x}_embedding_2017_{i}.json', 'r') as f:
      cur_emb_dict = json.load(f)
      old_len = len(merged_embedding_dict)
      merged_embedding_dict.update(cur_emb_dict)
      new_len = len(merged_embedding_dict)
      assert (new_len - old_len)==len(cur_emb_dict)

  with open(f'{path[:-3]}/{x}_embedding_2017_64.json', 'w') as f:
    json.dump(merged_embedding_dict, f)

merge('mean', path)
merge('cls', path)