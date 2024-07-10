"Read in the file output by pipeline, and get summary stats from random sample"

import h5py 
import numpy as np 


file_path = "/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/gen_data/"
filename = file_path + "mlm_encoded_upto_2017.h5"
sample_size = 1_000

rng = np.random.default_rng()


with h5py.File(filename, "r") as file:
    for key in file: 
        shapes = file[key].shape 
        print(key, shapes, flush=True)
        start = rng.integers(low=0, high=shapes[0])
        try:
            end = start + sample_size
            sample = file[key][start:end]
            smean, smin, smax = sample.mean(), sample.min(), sample.max()
            print(f"mean, min, max:", smean, smin, smax)
        except: 
            print(f"No summary for {key}")


