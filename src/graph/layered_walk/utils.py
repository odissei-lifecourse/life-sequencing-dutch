
import pickle 
import numpy as np 
import timeit 
import numba 

root = "/home/flavio/datasets/synthetic_layered_graph_1mil/"


def load_data():
    layers = []
    layer_types = ["family", "colleague", "education", "neighbor", "household"]
    layer_types = ["neighbor", "colleague"]
    for ltype in layer_types:
        with open(root + "fake_" + ltype + "_adjacency_dict.pkl", "rb") as pkl_file:
            edges = dict(pickle.load(pkl_file))
            # edges_keep = dict((u, edges[u]) for u in users)
            layers.append(edges)

    users = list(layers[0].keys())


    node_layer_dict = {}
    for user in users:
        node_layer_dict[user] = []
        
        for i, layer in enumerate(layers):
            if user in layer:
                if len(layer[user]) > 0:
                    node_layer_dict[user].append(i)

    return users, layers, node_layer_dict
