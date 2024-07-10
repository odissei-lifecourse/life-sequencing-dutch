import random
import csv
import sys
import argparse
import pickle

def get_edges(path_to_original, save_path):

    global running_id
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
            source = int(row[1])
            target = int(row[3])
            
            if source not in resident_set or target not in resident_set:
                continue
            
            original_row_count += 1
            
            # Get the mapping
            if source not in mappings:
                mappings[source] = running_id
                running_id += 1
            if target not in mappings:
                mappings[target] = running_id
                running_id += 1
                
            mapped_source = mappings[source]
            mapped_target = mappings[target]
            
            if mapped_source not in adjacency_dict:
                adjacency_dict[mapped_source] = [mapped_target]
            else:
                if mapped_target not in adjacency_dict[mapped_source]:
                    adjacency_dict[mapped_source].append(mapped_target)

            if mapped_target not in adjacency_dict:
                adjacency_dict[mapped_target] = [mapped_source]
            else:
                if mapped_source not in adjacency_dict[mapped_target]:
                    adjacency_dict[mapped_target].append(mapped_source)

    print("Converted", original_row_count, "rows from", path_to_original, "into adjacency dict", flush=True)

    with open(save_path, "wb") as pkl_file:
        pickle.dump(adjacency_dict, pkl_file)

year = "2009"

with open("gron_" + year + "_resident_list.pkl", "rb") as pkl_file:
    resident_set = set(pickle.load(pkl_file))

mappings = {}
running_id = 0

get_edges("KLASGENOTENNETWERK" + year + "TABV1.csv", "gron_" + year + "_classmate_edges.pkl")
get_edges("COLLEGANETWERK" + year + "TABV1.csv", "gron_" + year + "_colleague_edges.pkl")
get_edges("FAMILIENETWERK" + year + "TABV1.csv", "gron_" + year + "_family_edges.pkl")
get_edges("HUISGENOTENNETWERK" + year + "TABV1.csv", "gron_" + year + "_household_edges.pkl")
get_edges("BURENNETWERK" + year + "TABV1.csv", "gron_" + year + "_neighbor_edges.pkl")

with open("mappings/gron_" + year + "_mappings.pkl", "wb") as pkl_file:
    pickle.dump(mappings, pkl_file)
    
mapped_residents = set()
for resident in resident_set:
    if resident in mappings:
        mapped_resident = mappings[resident]
        mapped_residents.add(mapped_resident)
print(len(mapped_residents), flush=True)
        
with open("gron_mapped_" + year + "_resident_list.pkl", "wb") as pkl_file:
    pickle.dump(mapped_residents, pkl_file)