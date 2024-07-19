import h5py
import pickle
import numpy as np

def convert_file(filename, mapping_url):

    embeddings = []
    ids = []

    # First get mappings
    with open(mapping_url, 'rb') as pkl_file:
        mappings = dict(pickle.load(pkl_file))

    inverse_mappings = {}
    for key, value in mappings.items():
        inverse_mappings[value] = key

    with open(filename, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    for i in range(len(data)):
        embedding = data[i]
        RINPERSOON = int(inverse_mappings[i])

        embeddings.append(embedding)
        ids.append(RINPERSOON)

    ids = np.array(ids)
    embeddings = np.array(embeddings)

    filename = filename.split(".emb")[0]
    filename += '.h5'

    # Now open the new h5 file and create datasets
    with h5py.File(filename, 'w') as f:
        f.create_dataset('sequence_id', data=ids)
        f.create_dataset('embeddings', data=embeddings)

if __name__ == '__main__':
    convert_file("/gpfs/ostor/ossc9424/homedir/Dakota_network/embeddings/lr_steve_full_network_2020.emb",
                 "/gpfs/ostor/ossc9424/homedir/Dakota_network/mappings/family_2020.pkl")

    convert_file("/gpfs/ostor/ossc9424/homedir/Dakota_network/embeddings/lr_steve_full_network_2010_30.emb",
                 "/gpfs/ostor/ossc9424/homedir/Dakota_network/mappings/family_2010.pkl")