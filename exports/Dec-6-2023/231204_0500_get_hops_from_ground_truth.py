import pickle
import pandas as pd
import numpy as np
import random
import time


def bfs(person, num_steps):

    seen = set()
    
    ret_list = []
    
    connections = adjacency_dict[person]
    seen.add(person)
    for i in range(num_steps):
    
        new_additions = []
        if len(connections) > 2:
            connections = random.sample(connections, 2)
        for connection in connections:
        
            seen.add(connection)
                
            ret_list.append((connection, i+1))
            
            further = adjacency_dict[connection]
            for other_connection in further:
                if other_connection not in seen:
                    new_additions.append(other_connection)
                    
        connections = new_additions
        
    return ret_list
    
########################################################################################

truth_type = 'full'

#years = [str(x) for x in range(2010, 2021)] 
years = ['2016']
#years = ['2010', '2020']

for year in years:


    print(year, flush=True)
    start_time = time.time()

    url = '/gpfs/ostor/ossc9424/homedir/Dakota_network/ground_truth_adjacency/' + truth_type + '_' + year + '_adjacency.pkl'

    with open(url, 'rb') as pkl_file:
        adjacency_dict = dict(pickle.load(pkl_file))

    person_hops = {}
    num_hops = 3
    people = list(adjacency_dict.keys())

    for person in people:
        person_hops[person] = {}
        for i in range(num_hops):
            person_hops[person][i+1] = []
            
        hop_list = bfs(person, num_hops)
        for hop in hop_list:
            other = hop[0]
            dist = hop[1]
            
            person_hops[person][dist].append(other)
            
        # Subsample each number of hops so this doesn't get huge/take forever
        #for i in range(num_hops):
        #    length = len(person_hops[person][i+1])
        #    if length > 10:
        #        person_hops[person][i+1] = random.sample(person_hops[person][i+1], 10)
                
       
    save_url = '/gpfs/ostor/ossc9424/homedir/Dakota_network/ground_truth_hops/' + truth_type + '_' + year + '_hops.pkl'
    with open(save_url, 'wb') as pkl_file:
        pickle.dump(person_hops, pkl_file)
    
    end_time = time.time()
    delta = end_time - start_time
    print("Hops computed for", year, "in", str(np.round(delta/60., 2)), 'minutes', flush=True)