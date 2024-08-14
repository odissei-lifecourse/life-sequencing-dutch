import random
import csv
import sys
import argparse
import pickle
import time


def find_and_reduce_component(path_to_original, save_path):

    root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/intermediates/"

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
            #if (j+1) % 10000000 == 0:
            #    print(j+1, flush=True)
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

    #print("Converted", original_row_count, "rows from", path_to_original, "into adjacency dict", flush=True)

    user_list = list(adjacency_dict.keys())

    visited_nodes = set()
    search_stack = []
    seed_node = random.sample(user_list, 1)[0]
    search_stack.append(seed_node)
    visited_nodes.add(seed_node)

    # def depth_first_search(seed_node):
    #     # Base case
    #     if seed_node in visited_nodes:
    #         return
    #
    #     visited_nodes.add(seed_node)
    #     print(len(visited_nodes))
    #
    #     connected_nodes = adjacency_dict[seed_node]
    #     for node in connected_nodes:
    #         if node not in visited_nodes:
    #             depth_first_search(node)

    def depth_first_search(search_stack):

        while len(search_stack) > 0:
            node_to_explore = search_stack.pop()
            connected_nodes = adjacency_dict[node_to_explore]
            for node in connected_nodes:
                if node not in visited_nodes:
                    visited_nodes.add(node)
                    search_stack.append(node)

    depth_first_search(search_stack)

    while len(visited_nodes) < 300000:
        print("Only found", len(visited_nodes), "connected nodes this run", flush=True)
        print("Performing another search to find the largest component...", flush=True)
        visited_nodes = set()
        seed_node = random.sample(user_list, 1)[0]
        search_stack = [seed_node]
        visited_nodes.add(seed_node)

        depth_first_search(search_stack)

    # Now that we (presumably) have the largest connected component, reduce the edge list to only these people and resave it
    trimmed_row_count = 0

    # A sufficient component has been found
    print('A sufficient component has been found, containing', len(visited_nodes), 'nodes', flush=True)
    with open(path_to_original, newline="\n") as in_csvfile, open(save_path, 'w', newline="\n") as out_csvfile:
        reader = csv.reader(in_csvfile, delimiter=';')
        writer = csv.writer(out_csvfile, delimiter=';')

        i = 0
        for j, row in enumerate(reader):
            if i < start_row:
                i += 1
                continue
            #if (j+1) % 10000000 == 0:
            #    print(j+1, flush=True)
                
            source = int(row[1])
            target = int(row[3])

            if source in visited_nodes:
                writer.writerow([source, target])
                trimmed_row_count += 1

    #print("Wrote", trimmed_row_count, "rows to a new edgefile named", save_path, flush=True)

###################################################################################################################################################################################
# Takes a messy edgelist in csv format and converts it to something similar to the ones saved by network X
def write_edgelist(path_to_original, path_to_edgelist, path_to_mapping, start_row=0):

    running_id = 0
    original_reindex_map = {}

    # Open the original csvfile and the new edgelist file
    # Write as a stream to avoid high memory loads
    with open(path_to_original, newline="\n") as in_csvfile, open(path_to_edgelist, 'w', newline="\n") as out_csvfile:
        reader = csv.reader(in_csvfile, delimiter=';')
        writer = csv.writer(out_csvfile, delimiter=' ')

        row_counter = 0
        for row in reader:
            if row_counter < start_row:
                row_counter += 1
                continue

            assert len(row) == 2, print(row)
            source = int(row[0])
            target = int(row[1])

            if source not in original_reindex_map:
                original_reindex_map[source] = running_id
                source_id = running_id
                running_id += 1
            else:
                source_id = original_reindex_map[source]

            if target not in original_reindex_map:
                original_reindex_map[target] = running_id
                target_id = running_id
                running_id += 1
            else:
                target_id = original_reindex_map[target]

            # Write this row worth of data
            writer.writerow([source_id, target_id])

    # Done writing edgelist, now write index mapping
    with open(path_to_mapping, 'wb') as pkl_file:
        pickle.dump(original_reindex_map, pkl_file)

#####################################################################################################################################################################

def get_family_set(year):
    root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/intermediates/"

    adjacency_dict = {}
    user_set = set()

    with open(root + "clean_family_edges_" + year + ".csv", newline="\n") as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')

        start_row = 0
        row_counter = 0
        num_connections = 0
        for i, row in enumerate(reader):
            if row_counter < start_row:
                row_counter += 1
                continue

            #if (i+1) % 10000000 == 0:
            #    print(i+1, flush=True)

            assert len(row) == 2, print(row)
            source = int(row[0])
            target = int(row[1])
            
            user_set.add(source)
            user_set.add(target)

            num_connections += 1

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

    #print(num_connections, flush=True)
    # Done writing edgelist, now write index mapping
    with open(root + "family_" + year + "_adjacency_dict.pkl", 'wb') as pkl_file:
        pickle.dump(adjacency_dict, pkl_file)
        
    with open(root + "connected_user_set_" + year + ".pkl", 'wb') as pkl_file:
        pickle.dump(user_set, pkl_file)

######################################################################################################################################################################################

def get_edges(path_to_original, save_path, year):
    
    root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/"
    
    with open(root + "intermediates/connected_user_set_" + year + ".pkl", "rb") as pkl_file:
        resident_set = set(pickle.load(pkl_file))
    
    with open(root + "mappings/family_" + year + ".pkl", 'rb') as pkl_file:
        mappings = dict(pickle.load(pkl_file))
    
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
            #if (j+1) % 10000000 == 0:
            #    print(j+1, flush=True)
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

    #print("Converted", original_row_count, "rows from", path_to_original, "into adjacency dict", flush=True)

    with open(save_path, "wb") as pkl_file:
        pickle.dump(adjacency_dict, pkl_file)

#############################################################################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PreDeep")
    parser.add_argument(
        "--year",
        type=int,
        default=2010
    )
    args = parser.parse_args()
    year = str(args.year)
    
    full_start = time.time()
    section_start = time.time()
    
    root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/"
    
    # First find the largest component in the family network and reduce the edgelist to only those people
    input_path = root + "FAMILIENETWERK" + year + "TABV1.csv"
    output_path = root + "intermediates/reduced_family_edges_" + year + ".csv"
    find_and_reduce_component(input_path, output_path)
    
    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Found family component in", str(delta), "minutes", flush=True)
    
    #----------------------------------------------------------------------------------------------------------#
    
    section_start = time.time()
    # Now clean the family edges and get the mappings
    path_to_reduced = root + "intermediates/reduced_family_edges_" + year + ".csv"
    path_to_clean = root + "intermediates/clean_family_edges_" + year + ".csv"
    path_to_mapping = root + "mappings/family_" + year + ".pkl"
    write_edgelist(path_to_reduced, path_to_clean, path_to_mapping, start_row=0)
    
    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Got family mappings and cleaned edgelist in", str(delta), "minutes", flush=True)
    
    #----------------------------------------------------------------------------------------------------------#
    
    section_start = time.time()
    # Now get the user set and family adjacency dict
    get_family_set(year)
    
    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Produced family adjacency dict in", str(delta), "minutes", flush=True)
    
    #----------------------------------------------------------------------------------------------------------#
    
    # Now get adjacency dicts for all the other network layers
    section_start = time.time()
    # Classmates
    save_path = root + "intermediates/classmate_" + year + "_adjacency_dict.pkl"
    network_path = "KLASGENOTENNETWERK" + year + "TABV1.csv"
    get_edges(network_path, save_path, year)
    
    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Produced classmate adjacency dict in", str(delta), "minutes", flush=True)
    
    #----------------------------------------------------------------------------------------------------------#
    
    section_start = time.time()
    # Neighbors
    save_path = root + "intermediates/neighbor_" + year + "_adjacency_dict.pkl"
    network_path = "BURENNETWERK" + year + "TABV1.csv"
    get_edges(network_path, save_path, year)
    
    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Produced neighbor adjacency dict in", str(delta), "minutes", flush=True)
    
    #----------------------------------------------------------------------------------------------------------#
    
    section_start = time.time()
    # Householders
    save_path = root + "intermediates/household_" + year + "_adjacency_dict.pkl"
    network_path = "HUISGENOTENNETWERK" + year + "TABV1.csv"
    get_edges(network_path, save_path, year)
        
    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Produced household adjacency dict in", str(delta), "minutes", flush=True)
    
    #----------------------------------------------------------------------------------------------------------#
    
    section_start = time.time()
    # Colleagues
    save_path = root + "intermediates/colleague_" + year + "_adjacency_dict.pkl"
    network_path = "COLLEGANETWERK" + year + "TABV1.csv"
    get_edges(network_path, save_path, year)

    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Produced colleague adjacency dict in", str(delta), "minutes", flush=True)
    
    #----------------------------------------------------------------------------------------------------------#
    
    full_end = time.time()
    delta = (full_end - full_start) / 60.
    print("Finished preprocessing for year", year, flush=True)
    print("The process took", str(delta/60.), "hours", flush=True)
    print("Don't forget to delete any unneeded files (Keep mappings, adjacency dicts, connected_user_sets)", flush=True)
