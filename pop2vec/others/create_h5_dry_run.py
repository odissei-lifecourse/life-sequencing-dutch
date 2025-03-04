import h5py
import numpy as np
import sys

def copy_attributes(source_obj, target_obj):
    """
    Copy attributes from source_obj to target_obj
    """
    for attr_name, attr_value in source_obj.attrs.items():
        target_obj.attrs[attr_name] = attr_value

def reduce_hdf5_file(input_file, output_file, keep_count):
    """
    Create a reduced HDF5 file from the input_file by
    keeping only keep_count # of elements of each dataset.
    """
    with h5py.File(input_file, 'r') as f_in, \
         h5py.File(output_file, 'w') as f_out:

        # Copy global file attributes (if any)
        copy_attributes(f_in, f_out)
        print("attribute copying done")
        def copy_group(group_in, group_out):
            """
            Recursively copy groups and datasets from group_in to group_out,
            downsampling each dataset to keep_count rows.
            """
            print(f"keys = {group_in.keys()}")
            for key, item in group_in.items():
                print(f"working for key = {key}")
                if isinstance(item, h5py.Dataset):

                    # Create the downsampled dataset
                    dset_out = group_out.create_dataset(
                        key,
                        data=item[:keep_count],
                        compression='gzip'
                    )

                    # Copy dataset attributes
                    copy_attributes(item, dset_out)

                elif isinstance(item, h5py.Group):
                    subgroup_out = group_out.create_group(key)
                    # Copy group attributes
                    copy_attributes(item, subgroup_out)
                    # Recurse
                    copy_group(item, subgroup_out)

        # Start the recursive copying from the root group
        copy_group(f_in, f_out)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    keep_count = int(sys.argv[3])
    reduce_hdf5_file(input_file, output_file, keep_count)

