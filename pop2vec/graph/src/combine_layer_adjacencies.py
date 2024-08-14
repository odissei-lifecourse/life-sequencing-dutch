import argparse
import pickle

if __name__ == '__main__':

    years = ["2010", '2020']

    for year in years:

        year = args.year
        
        root = "/gpfs/ostor/ossc9424/homedir/Dakota_network/intermediates/"
        
        prefixes = ['classmate', 'colleague', 'family', 'household', 'neighbor']
        
        full_adjacency = {}
        
        mapping_url = "/gpfs/ostor/ossc9424/homedir/Dakota_network/mappings/family_" + year + ".pkl"
        
        with open(mapping_url, 'rb') as pkl_file:
            mappings = dict(pickle.load(pkl_file))
            
        inverse_mappings = {}
        for key, value in mappings.items():
            inverse_mappings[value] = key
        
        for prefix in prefixes:
            url = root + prefix + '_' + year + '_adjacency_dict.pkl'
            
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
        
        for person in full_adjacency:
            full_adjacency[person] = random.sample(full_adjacency[person], 5)
        
        save_url = "/gpfs/ostor/ossc9424/homedir/Dakota_network/ground_truth_adjacency/full_" + year + '_adjacency.pkl'  
        with open(save_url, 'wb') as pkl_file:
            pickle.dump(full_adjacency, pkl_file)