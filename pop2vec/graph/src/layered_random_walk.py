#advanced_random_walk
import random
import pickle
import csv
import time
import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="random_walk_generator")
    parser.add_argument(
        "--year",
        type=int,
        default=2016
    )
    
    parser.add_argument(
        "--start_int",
        type=int,
        default=0
    )
    args = parser.parse_args()
    start_int = args.start_int
    year = str(args.year)
    
    # Probability of maintaining the same layer
    p = 0.8

    layers = []
    # load the 5 adjacency dicts

    root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/intermediates/"

    print("Loading family edges...", flush=True)
    with open(root + "family_" + year + "_adjacency_dict.pkl", "rb") as pkl_file:
        family_edges = dict(pickle.load(pkl_file))
        layers.append(family_edges)
        
    print("Loading household edges...", flush=True)
    with open(root + "household_" + year + "_adjacency_dict.pkl", "rb") as pkl_file:
        household_edges = dict(pickle.load(pkl_file))
        layers.append(household_edges)
        
    print("Loading neighbor edges...", flush=True)
    with open(root + "neighbor_" + year + "_adjacency_dict.pkl", "rb") as pkl_file:
        neighbor_edges = dict(pickle.load(pkl_file))
        layers.append(neighbor_edges)
        
    print("Loading classmate edges...", flush=True)
    with open(root + "classmate_" + year + "_adjacency_dict.pkl", "rb") as pkl_file:
        classmate_edges = dict(pickle.load(pkl_file))
        layers.append(classmate_edges)
        
    print("Loading colleague edges...", flush=True)
    with open(root + "colleague_" + year + "_adjacency_dict.pkl", "rb") as pkl_file:
        colleague_edges = dict(pickle.load(pkl_file))
        layers.append(colleague_edges)
        
    # [Family, Household, Neighbor, Classmate, Colleague]

    print("Loading user set...", flush=True)
    with open(root + "connected_user_set_" + year + ".pkl", "rb") as pkl_file:
        unique_users = set(pickle.load(pkl_file))

    # Increment the user set by 5 so we can use the first 5 integers as layer tokens
    for user in unique_users:
        user += 5
        
    walk_len = 40
    num_walks = 5
    
    print("Preparing node layer dict...", flush=True)
    node_layer_dict = {}
    
    for user in unique_users:
        node_layer_dict[user] = []
        
        for i, layer in enumerate(layers):
            if user in layer:
                if len(layer[user]) > 0:
                    node_layer_dict[user].append(i)
    print("Generating walks...", flush=True)
    
    for k in range(num_walks):
        start_time = time.time()
        
        num_users = len(unique_users)
        # Pregen a bunch of random numbers
        random_nums = np.random.rand(num_users, walk_len)
        
        with open("walks/layered_full_network_" + year + "_" + str(k + start_int) + ".csv", 'w', newline="\n") as out_csvfile:
            writer = csv.writer(out_csvfile, delimiter=',')
            header_row = ["SOURCE"] + ["STEP_" + str(i) for i in range(walk_len-1)]
            writer.writerow(header_row)
            
            rows = []
            
            for user_idx, user in enumerate(unique_users):
            
                if (user + 1) % 1000000 == 0:
                    print(user + 1, flush=True)
            
                walk = [user]
                current_node = user
                
                layer_indices = node_layer_dict[current_node]
                layer_index = np.random.choice(layer_indices)
                current_layer = layers[layer_index]
                
                while len(walk) < walk_len:
                
                    layer_indices = node_layer_dict[current_node]
                
                    roll = random_nums[user_idx][len(walk)]
                    
                    if roll > p:
                        # Random chance for each layer
                        layer_index = np.random.choice(layer_indices)
                        current_layer = layers[layer_index]
                        
                    adjacent_nodes = current_layer[current_node]

                    # Layer index should encode the layer type in an integer 0-4
                    walk.append(layer_index)
                    
                    next_node = np.random.choice(adjacent_nodes)
                    walk.append(next_node)
                    current_node = next_node
                
                #assert len(walk) == walk_len, print(len(walk))
                rows.append(walk)
                
            writer.writerows(rows)
         
        end_time = time.time()
        delta = end_time - start_time
        print(len(unique_users), "walks generated over", delta, flush=True)