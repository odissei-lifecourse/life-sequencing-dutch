import h5py
import pickle
import numpy as np

def convert_file(filename):

    # First load the embeddings from the old pickle file
    with open(filename, 'rb') as pkl_file:
        # Assumes we have a dictionary where keys are RINPERSOON ids and values are embedding arrays
        data_dict = dict(pickle.load(pkl_file))

    ids = np.array(data_dict.keys())
    embeddings = np.array(data_dict.values())

    filename = filename.split(".emb")[0]
    filename += '.h5'

    # Now open the new h5 file and create datasets
    with h5py.File(filename, 'w') as f:
        f.create_dataset('sequence_id', data=ids)
        f.create_dataset('embeddings', data=embeddings)

if __name__ == '__main__':
    convert_file("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/Dakota_network/embeddings/lr_steve_full_network_2020.emb")
    convert_file("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/Dakota_network/embeddings/lr_steve_full_network_2010.emb")