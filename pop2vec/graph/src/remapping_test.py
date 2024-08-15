import pickle
import csv

path_to_original = "../FAMILIENETWERK2018TABV1.csv"

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
        original_row_count += 1
        source = int(row[1])
        target = int(row[3])
        if source not in adjacency_dict:
            adjacency_dict[source] = set()
            adjacency_dict[source].add(target)
        else:
            if target not in adjacency_dict[source]:
                adjacency_dict[source].add(target)

        if target not in adjacency_dict:
            adjacency_dict[target] = set()
            adjacency_dict[target].add(source)
        else:
            if source not in adjacency_dict[target]:
                adjacency_dict[target].add(source)

print("Converted", original_row_count, "rows from", path_to_original, "into adjacency dict", flush=True)

# Load previous mapping file
with open("mappings/family_2018.pkl", 'rb') as pkl_file:
    original_map = dict(pickle.load(pkl_file))
    
# Invert the map
inverted_map = {}
for key, value in original_map.items():
    inverted_map[int(value)] = int(key)
    
# Load a clean edgelist
with open('clean_family_edges_2018.csv', newline="\n") as csv_file:

    reader = csv.reader(csv_file, delimiter=' ')
    for row in reader:
        mapped_source = int(row[0])
        mapped_target = int(row[1])
        
        original_source = inverted_map[mapped_source]
        original_target = inverted_map[mapped_target]
        
        connections = adjacency_dict[original_source]
        assert original_target in connections, print("Unable to find connection between", original_source, "and", original_target, flush=True)

print("All mappings restored! Great success.", flush=True)            