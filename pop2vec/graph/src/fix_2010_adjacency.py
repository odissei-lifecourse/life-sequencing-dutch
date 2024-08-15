import pickle

filenames = ["family_2010_adjacency_dict.pkl",
			"household_2010_adjacency_dict.pkl",
			"neighbor_2010_adjacency_dict.pkl",
			"classmate_2010_adjacency_dict.pkl",
			"colleague_2010_adjacency_dict.pkl"]
			
for filename in filenames:
	print(filename, flush=True)
	with open(filename, "rb") as pkl_file:
		old_edges = dict(pickle.load(pkl_file))
	updated_edges = {}
	for key, value in old_edges.items():
		updated_edges[key] = list(value)
		
	with open(filename, "wb") as pkl_file:
		pickle.dump(updated_edges, pkl_file)