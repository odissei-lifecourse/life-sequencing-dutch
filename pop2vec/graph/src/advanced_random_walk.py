#advanced_random_walk
import random
import pickle
import csv
import time
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="random_walk_generator")
    parser.add_argument(
        "--start_int",
        type=int,
        default=0
    )
    args = parser.parse_args()
    start_int = args.start_int

    layers = []
    # load the 5 adjacency dicts

    print("Loading family edges...", flush=True)
    with open("family_2010_adjacency_dict.pkl", "rb") as pkl_file:
        family_edges = dict(pickle.load(pkl_file))
        layers.append(family_edges)
        
    print("Loading household edges...", flush=True)
    with open("household_2010_adjacency_dict.pkl", "rb") as pkl_file:
        household_edges = dict(pickle.load(pkl_file))
        layers.append(household_edges)
        
    print("Loading neighbor edges...", flush=True)
    with open("neighbor_2010_adjacency_dict.pkl", "rb") as pkl_file:
        neighbor_edges = dict(pickle.load(pkl_file))
        layers.append(neighbor_edges)
        
    print("Loading classmate edges...", flush=True)
    with open("classmate_2010_adjacency_dict.pkl", "rb") as pkl_file:
        classmate_edges = dict(pickle.load(pkl_file))
        layers.append(classmate_edges)
        
    print("Loading colleague edges...", flush=True)
    with open("colleague_2010_adjacency_dict.pkl", "rb") as pkl_file:
        colleague_edges = dict(pickle.load(pkl_file))
        layers.append(colleague_edges)
        
    # [Family, Household, Neighbor, Classmate, Colleague]

    print("Loading user set...", flush=True)
    with open("connected_user_set_2010.pkl", "rb") as pkl_file:
        unique_users = set(pickle.load(pkl_file))
        
    walk_len = 40
    num_walks = 5
    
    for k in range(num_walks):
        start_time = time.time()
        with open("walks/full_network_2010_" + str(k + start_int) + ".csv", 'w', newline="\n") as out_csvfile:
            writer = csv.writer(out_csvfile, delimiter=',')
            header_row = ["SOURCE"] + ["STEP_" + str(i) for i in range(walk_len-1)]
            writer.writerow(header_row)
            
            for user in unique_users:
            
                if (user + 1) % 1000000 == 0:
                    print(user + 1, flush=True)
            
                #print(user, flush=True)
                walk = [user]
                current_node = user
                
                while len(walk) < walk_len:
                
                    layer_indices = []
                    for i, layer in enumerate(layers):
                        if current_node in layer:
                            if len(layer[current_node]) > 0:
                                layer_indices.append(i)
                
                    # Random chance for each layer
                    layer_index = random.choice(layer_indices)
                    current_layer = layers[layer_index]
                        
                    adjacent_nodes = list(current_layer[current_node])
                        
                    next_node = random.choice(adjacent_nodes)
                    walk.append(next_node)
                    current_node = next_node
                
                assert len(walk) == walk_len, print(len(walk))
                writer.writerow(walk)
         
        end_time = time.time()
        delta = end_time - start_time
        print(len(unique_users), "walks generated over", delta, flush=True)