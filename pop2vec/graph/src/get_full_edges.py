import random
import csv
import sys
import argparse
import pickle

with open("connected_user_set_2020.pkl", "rb") as pkl_file:
    resident_set = set(pickle.load(pkl_file))
    
with open("mappings/family_2020.pkl", 'rb') as pkl_file:
    mappings = dict(pickle.load(pkl_file))

def get_edges(path_to_original, save_path):
    start_row = 1

    original_row_count = 0
    # First compute edgelist in adjacency list
    adjacency_dict = {}
    with open(path_to_original, newline="\n") as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        i = 0
        for j, row in enumerate(reader):
            if i < start_row:
                i += 1
                continue
            if (j+1) % 10000000 == 0:
                print(j+1, flush=True)
            rin_source = int(row[1])
            rin_target = int(row[3])
            
            if rin_source in mappings:
                source = mappings[rin_source]
            else:
                continue
                
            if rin_target in mappings:
                target = mappings[rin_target]
            else:
                continue
            
            if source not in resident_set or target not in resident_set:
                continue
            
            original_row_count += 1
            if source not in adjacency_dict:
                adjacency_dict[source] = [target]
            else:
                if target not in adjacency_dict[source]:
                    adjacency_dict[source].append(target)

            if target not in adjacency_dict:
                adjacency_dict[target] = [source]
            else:
                if source not in adjacency_dict[target]:
                    adjacency_dict[target].append(source)

    print("Converted", original_row_count, "rows from", path_to_original, "into adjacency dict", flush=True)

    with open(save_path, "wb") as pkl_file:
        pickle.dump(adjacency_dict, pkl_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Edges")
    parser.add_argument(
        "--network_path",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    args = parser.parse_args()
    network_path = args.network_path
    save_path = args.save_path

    get_edges(network_path, save_path)
