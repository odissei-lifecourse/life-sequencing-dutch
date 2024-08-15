import numpy as np
import csv
import pickle
import random
import matplotlib.pyplot as plt

# Load embeddings
with open("embeddings/lr_steve_full_network_2010_34.emb", 'rb') as pkl_file:
    embeddings = pickle.load(pkl_file)

unique_ids = [i for i in range(len(embeddings))]

layers = []

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
            
num_samples = 1
real_distances = []
fake_distances = []

print("Computing distances...", flush=True)
for i in range(len(embeddings)):
    if (i+1) % 1000000 == 0:
        print(i+1, flush=True)
        
    source_embedding = embeddings[i]
    
    for k in range(len(layers)):
        adjacency_dict = layers[k]
        if i not in adjacency_dict:
            continue
        if len(adjacency_dict[i]) == 0:
            continue
        real_target_list = adjacency_dict[i]
    
        # Positive samples
        for j in range(num_samples):
            real_target_index = random.sample(real_target_list, 1)[0]
            real_target_embedding = embeddings[real_target_index]
            
            real_distance = np.linalg.norm(source_embedding - real_target_embedding)
            real_distances.append(real_distance)
            
        # Negative samples
        for j in range(num_samples):
            fake_target_index = random.sample(unique_ids, 1)[0]
            if fake_target_index in real_target_list:
                j -= 1
                continue
            fake_target_embedding = embeddings[fake_target_index]
            
            fake_distance = np.linalg.norm(source_embedding - fake_target_embedding)
            fake_distances.append(fake_distance)
        
fig, _ = plt.subplots(figsize=(5, 3), dpi=150)

plt.hist(real_distances, color='blue', bins=100, label="Distance of real connections")
plt.hist(fake_distances, color='red', bins=100, label="Distance of fake connections")
plt.title("Distributions of Distances in Embedding Space")
plt.xlabel("Distance in Embedding Space")
plt.ylabel("# of samples")
plt.legend()
plt.savefig("results/lr_steve_full_2010_34_distances.png", bbox_inches='tight')