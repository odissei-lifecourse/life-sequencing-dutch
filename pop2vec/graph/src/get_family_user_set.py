import pandas as pd
import csv
import pickle
import argparse

if __name__ == '__main__':

    root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/intermediates/"

    parser = argparse.ArgumentParser(description="get full user set for a year")
    parser.add_argument(
        "--year",
        type=int,
        default=2010
    )
    args = parser.parse_args()
    year = str(args.year)

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

            if (i+1) % 10000000 == 0:
                print(i+1, flush=True)

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

    print(num_connections, flush=True)
    # Done writing edgelist, now write index mapping
    with open(root + "family_" + year + "_adjacency_dict.pkl", 'wb') as pkl_file:
        pickle.dump(adjacency_dict, pkl_file)
        
    with open(root + "connected_user_set_" + year + ".pkl", 'wb') as pkl_file:
        pickle.dump(user_set, pkl_file)
