"""
The signature of the C function is:
void random_walk(int const* ptr, int const* neighs, int n, int num_walks,
                 int num_steps, int seed, int nthread, int* walks);
"""
import numpy as np
import numpy.ctypeslib as npct
import csv
import pickle
from ctypes import c_int
from os.path import dirname
from time import time
import os
import multiprocessing
import random
import argparse

array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')

librwalk = npct.load_library("librwalk", '/gpfs/ostor/ossc9424/homedir/Network_Embeddings/')

print("rwalk: Loading library from: {}".format('/gpfs/ostor/ossc9424/homedir/Network_Embeddings/'))
librwalk.random_walk.restype = None
librwalk.random_walk.argtypes = [array_1d_int, array_1d_int, c_int, c_int, c_int, c_int, c_int, array_1d_int]

########################################################################################################################

def random_walk(ptr, neighs, num_walks=10, num_steps=3, nthread=-1, seed=111413):
    assert (ptr.flags['C_CONTIGUOUS'])
    assert (neighs.flags['C_CONTIGUOUS'])
    assert (ptr.dtype == np.int32)
    assert (neighs.dtype == np.int32)

    n = ptr.size - 1
    walks = -np.ones((n * num_walks, (num_steps + 1)), dtype=np.int32, order='C')
    assert (walks.flags['C_CONTIGUOUS'])

    librwalk.random_walk(
        ptr,
        neighs,
        n,
        num_walks,
        num_steps,
        seed,
        nthread,
        np.reshape(walks, (walks.size,), order='C'))

    return walks

########################################################################################################################

# Assumes a perfectly 0-indexed edge list such as the type produced by NetworkX
def read_edgelist(fname, comments='#', delimiter=None):
    edges = np.genfromtxt(fname, comments=comments, delimiter=delimiter,
                          defaultfmt='%d', dtype=np.int32)
    assert (len(edges.shape) == 2)
    assert (edges.shape[1] == 2)

    # Sort so smaller index comes first
    edges.sort(axis=1)
    n = np.amax(edges) + 1

    # Duplicate
    duplicated_edges = np.vstack((edges, edges[:, ::-1]))

    # Sort duplicated edges by first index
    _tmp = np.zeros((duplicated_edges.shape[0],), dtype=np.int64)
    _tmp += duplicated_edges[:, 0]
    _tmp *= np.iinfo(np.int32).max
    _tmp += duplicated_edges[:, 1]

    ind_sort = np.argsort(_tmp)
    sorted_edges = duplicated_edges[ind_sort, :]

    # Calculate degree and create ptr, neighs
    vals, counts = np.unique(sorted_edges[:, 0], return_counts=True)
    degs = np.zeros((n,), dtype=np.int32)
    degs[vals] = counts
    ptr = np.zeros((n + 1,), dtype=np.int32)
    ptr[1:] = np.cumsum(degs)
    neighs = np.copy(sorted_edges[:, 1])

    # Check ptr, neighs
    ptr.flags.writeable = False
    assert (ptr.flags.owndata == True)
    assert (ptr.flags.aligned == True)
    assert (ptr.flags.c_contiguous == True)

    neighs.flags.writeable = False
    assert (neighs.flags.owndata == True)
    assert (neighs.flags.aligned == True)
    assert (neighs.flags.c_contiguous == True)

    return ptr, neighs

########################################################################################################################

# Takes a messy edgelist in csv format and converts it to something similar to the ones saved by network X
def write_edgelist(path_to_original, path_to_edgelist, path_to_mapping, start_row=0):

    running_id = 0
    original_reindex_map = {}

    # Open the original csvfile and the new edgelist file
    # Write as a stream to avoid high memory loads
    with open(path_to_original, newline="\n") as in_csvfile, open(path_to_edgelist, 'w', newline="\n") as out_csvfile:
        reader = csv.reader(in_csvfile, delimiter=';')
        writer = csv.writer(out_csvfile, delimiter=' ')

        row_counter = 0
        for row in reader:
            if row_counter < start_row:
                row_counter += 1
                continue

            assert len(row) == 2, print(row)
            source = int(row[0])
            target = int(row[1])

            if source not in original_reindex_map:
                original_reindex_map[source] = running_id
                source_id = running_id
                running_id += 1
            else:
                source_id = original_reindex_map[source]

            if target not in original_reindex_map:
                original_reindex_map[target] = running_id
                target_id = running_id
                running_id += 1
            else:
                target_id = original_reindex_map[target]

            # Write this row worth of data
            writer.writerow([source_id, target_id])

    # Done writing edgelist, now write index mapping
    with open(path_to_mapping, 'wb') as pkl_file:
        pickle.dump(original_reindex_map, pkl_file)

########################################################################################################################

def save_walks(headers, walks, save_path):
    with open(save_path, 'w', newline="\n") as out_csvfile:
        writer = csv.writer(out_csvfile, delimiter=',')

        writer.writerow(headers)
        writer.writerows(walks)

########################################################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="random_walk_generator")
    parser.add_argument(
        "--dirty_edge_path",
        type=str,
    )
    parser.add_argument(
        "--clean_edge_path",
        type=str,
    )
    parser.add_argument(
        "--walk_path",
        type=str,
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
    )
    parser.add_argument(
        "--clean_edges",
        default=True,
        action='store_false'
    )
    parser.add_argument(
        "--num_walks", default=1, type=int
    )
    parser.add_argument(
        "--walk_length", default=30, type=int
    )
    parser.add_argument(
        "--start_index", default=0, type=int
    )
    
    args = parser.parse_args()
    dirty_edge_path = args.dirty_edge_path
    clean_edge_path = args.clean_edge_path
    walk_path = args.walk_path
    mapping_path = args.mapping_path
    clean_edges = args.clean_edges
    num_walks = args.num_walks
    walk_length = args.walk_length
    start_index = args.start_index

    # Converts sloppy edgelists to NetworkX format
    if clean_edges:
        print("Cleaning edgelist...", flush=True)
        write_edgelist(dirty_edge_path, clean_edge_path, mapping_path, start_row=0)

    # Converts NetworkX edgelist into C compatible datatypes
    #print("Converting edgelist into C compatible datatypes...", flush=True)
    #ptr, neighs = read_edgelist(clean_edge_path)

    # Use as many threads as there are cpus
    #n_cpus = np.arange(1, multiprocessing.cpu_count() + 1)
    #nthread = len(n_cpus)

    #print("Generating walks using", nthread, "cpus", flush=True)
    # Generate the walks
    #for i in range(num_walks):
    #    seed = random.randint(0, 1000000)
    #    walks = random_walk(ptr, neighs, num_walks=1, num_steps=walk_length-1, nthread=nthread, seed=seed)
    #    print(len(walks), "walks have been generated", flush=True)

    #    header_row = ["SOURCE"] + ["STEP_" + str(i) for i in range(walk_length-1)]
    #    # Save the walks
    #    walk_id = str(i + start_index)
    #    save_walks(header_row, walks, walk_path + walk_id + ".csv")
