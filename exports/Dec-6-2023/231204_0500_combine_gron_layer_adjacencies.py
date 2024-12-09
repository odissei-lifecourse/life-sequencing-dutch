import argparse
import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combining Adjacencies')

    parser.add_argument(
        "--year",
        default='2010',
        type=str,
        help='year for which to combine adjacencies'
    )

    args = parser.parse_args()

    year = args.year
    
    root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/intermediates/groningen/"
    
    prefixes = ['classmate', 'colleague', 'family', 'household', 'neighbor']
    
    full_adjacency = {}
    
    mapping_url = "/gpfs/ostor/ossc9424/homedir/Dakota_network/mappings/gron_" + year + "_mappings.pkl"
    
    with open(mapping_url, 'rb') as pkl_file:
        mappings = dict(pickle.load(pkl_file))
        
    inverse_mappings = {}
    for key, value in mappings.items():
        inverse_mappings[value] = key
    
    for prefix in prefixes:
        url = root + 'gron_' + year + '_' + prefix + '_edges.pkl'
        
        with open(url, 'rb') as pkl_file:
            tiny_adjacency = dict(pickle.load(pkl_file))
        
        for idx in tiny_adjacency:
            person_id = inverse_mappings[idx]
            
            if person_id not in full_adjacency:
                full_adjacency[person_id] = []
                
            for other_idx in tiny_adjacency[idx]:
                other_id = inverse_mappings[other_idx]
                
                if other_id not in full_adjacency[person_id]:
                    full_adjacency[person_id].append(other_id)
    
    print(len(full_adjacency), flush=True)
    
    save_url = "/gpfs/ostor/ossc9424/homedir/Dakota_network/ground_truth_adjacency/gron_" + year + '_adjacency.pkl'  
    with open(save_url, 'wb') as pkl_file:
        pickle.dump(full_adjacency, pkl_file)