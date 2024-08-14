import pickle

filenames = ["gron_2010_classmate_edges.pkl",
			"gron_2010_colleague_edges.pkl",
			"gron_2010_family_edges.pkl",
			"gron_2010_household_edges.pkl",
			"gron_2010_neighbor_edges.pkl"]
		
gron_mappings = {}
running_id = 0
        
for filename in filenames:
    print(filename, flush=True)
    with open(filename, "rb") as pkl_file:
        old_edges = dict(pickle.load(pkl_file))
    updated_edges = {}
    
    for key, value in old_edges.items():
        if key not in gron_mappings:
            gron_mappings[key] = running_id
            running_id += 1
        
        mapped_values = []
        for elem in value:
            if elem not in gron_mappings:
                gron_mappings[elem] = running_id
                running_id += 1
                
            mapped_values.append(gron_mappings[elem])
            
        mapped_key = gron_mappings[key]
        updated_edges[mapped_key] = mapped_values
    
    with open(filename, "wb") as pkl_file:
        pickle.dump(updated_edges, pkl_file)

with open("mappings/gron_2010_mappings.pkl", "wb") as pkl_file:
    pickle.dump(gron_mappings, pkl_file)
    
with open("gron_2010_resident_list.pkl", "rb") as pkl_file:
	user_set = set(pickle.load(pkl_file))
updated_user_set = set()
for user in user_set:
    if user in gron_mappings:
        updated_user = gron_mappings[user]
        updated_user_set.add(updated_user)
        
with open("gron_mapped_2010_resident_list.pkl", "wb") as pkl_file:
	pickle.dump(updated_user_set, pkl_file)   