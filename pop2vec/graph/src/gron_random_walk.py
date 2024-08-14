#advanced_random_walk, capable of switching layers at predefined probabilities
# Can be used to give priority to one layer to distinguish where predictive power is coming from.
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
    
    parser.add_argument(
        "--priority",
        type=str,
        default=None
    )
    
    args = parser.parse_args()
    
    start_int = args.start_int
    year = str(args.year)
    priority = args.priority
    
    # Probability of maintaining the same layer
    # 0 = family    
    # 1 = household
    # 2 = neighbor
    # 3 = classmate
    # 4 = colleague

    if priority is None:
        layer_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    elif priority == 'family':
        layer_probs = [0.6, 0.1, 0.1, 0.1, 0.1]
    elif priority == 'household':
        layer_probs = [0.1, 0.6, 0.1, 0.1, 0.1]
    elif priority == 'neighbor':
        layer_probs = [0.1, 0.1, 0.6, 0.1, 0.1]
    elif priority == 'classmate':
        layer_probs = [0.1, 0.1, 0.1, 0.6, 0.1]
    elif priority == 'colleague':
        layer_probs = [0.1, 0.1, 0.1, 0.1, 0.6]

    layers = []
    # load the 5 adjacency dicts

    gron_root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/intermediates/groningen/"

    print("Loading family edges...", flush=True)
    with open(gron_root + "gron_" + year + "_family_edges.pkl", "rb") as pkl_file:
        family_edges = dict(pickle.load(pkl_file))
        layers.append(family_edges)
        
    print("Loading household edges...", flush=True)
    with open(gron_root + "gron_" + year + "_household_edges.pkl", "rb") as pkl_file:
        household_edges = dict(pickle.load(pkl_file))
        layers.append(household_edges)
        
    print("Loading neighbor edges...", flush=True)
    with open(gron_root + "gron_" + year + "_neighbor_edges.pkl", "rb") as pkl_file:
        neighbor_edges = dict(pickle.load(pkl_file))
        layers.append(neighbor_edges)
        
    print("Loading classmate edges...", flush=True)
    with open(gron_root + "gron_" + year + "_classmate_edges.pkl", "rb") as pkl_file:
        classmate_edges = dict(pickle.load(pkl_file))
        layers.append(classmate_edges)
        
    print("Loading colleague edges...", flush=True)
    with open(gron_root + "gron_" + year + "_colleague_edges.pkl", "rb") as pkl_file:
        colleague_edges = dict(pickle.load(pkl_file))
        layers.append(colleague_edges)
        
    # [Family, Household, Neighbor, Classmate, Colleague]

    print("Loading user set...", flush=True)
    with open(gron_root + "gron_mapped_" + year + "_resident_list.pkl", "rb") as pkl_file:
        unique_users = set(pickle.load(pkl_file))
        
    walk_len = 40
    num_walks = 100
    
    for k in range(num_walks):
        start_time = time.time()
        if priority is not None:
            url = "walks/gron_" + priority + "_priority_" + year + "_" + str(k + start_int) + ".csv"
        else:
            url = "walks/gron_" + year + "_" + str(k + start_int) + ".csv"
        with open(url, 'w', newline="\n") as out_csvfile:
            writer = csv.writer(out_csvfile, delimiter=',')
            header_row = ["SOURCE"] + ["STEP_" + str(i) for i in range(walk_len-1)]
            writer.writerow(header_row)
            
            for user in unique_users:
            
                if (user + 1) % 1000000 == 0:
                    print(user + 1, flush=True)
            
                #print(user, flush=True)
                walk = [user]
                current_node = user
                
                layer_indices = []
                for i, layer in enumerate(layers):
                    if current_node in layer:
                        if len(layer[current_node]) > 0:
                            layer_indices.append(i)
                
                # Starting layer is our priority
                layer_index = np.argmax(layer_probs)
                current_layer = layers[layer_index]
                
                while len(walk) < walk_len:
                
                    layer_indices = []
                    temp_layer_probs = []
                    for i, layer in enumerate(layers):
                        if current_node in layer:
                            if len(layer[current_node]) > 0:
                                layer_indices.append(i)
                                temp_layer_probs.append(layer_probs[i])
                
                    # Random chance for each layer
                    layer_index = random.choices(layer_indices, weights=temp_layer_probs, k=1)[0]
                    current_layer = layers[layer_index]
                        
                    adjacent_nodes = current_layer[current_node]
                        
                    next_node = random.choice(adjacent_nodes)
                    walk.append(next_node)
                    current_node = next_node
                
                assert len(walk) == walk_len, print(len(walk))
                writer.writerow(walk)
         
        end_time = time.time()
        delta = end_time - start_time
        print(len(unique_users), "walks generated over", delta, flush=True)
        
    #for user in bad_users:
    #    unique_users.discard(user)
    #with open("updated_gron_2010_resident_list.pkl", "wb") as pkl_file:
    #    unique_users = pickle.dump(unique_users, pkl_file)