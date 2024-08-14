import random
import csv
import sys
import argparse
# 1. Reads an edgelist and computes the largest connected component
# 2. Saves a new edgelist which contains only members of that connected component

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

    visited_nodes = set()
    search_stack = []
    search_stack.append(source)
    visited_nodes.add(source)

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
        seed_node = random.sample(adjacency_dict.keys(), 1)[0]
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
            if (j+1) % 10000000 == 0:
                print(j+1, flush=True)
                
            source = int(row[1])
            target = int(row[3])

            if source in visited_nodes:
                writer.writerow([source, target])
                trimmed_row_count += 1

    print("Wrote", trimmed_row_count, "rows to a new edgefile named", save_path, flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="largest_cc")
    parser.add_argument(
        "--year",
        type=int,
        default=2010
    )

    args = parser.parse_args()
    year = str(args.year)
    
    root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/"
    
    input_path = root + "FAMILIENETWERK" + year + "TABV1.csv"
    output_path = root + "reduced_family_edges_" + year + ".csv"
    
    find_and_reduce_component(input_path, output_path)