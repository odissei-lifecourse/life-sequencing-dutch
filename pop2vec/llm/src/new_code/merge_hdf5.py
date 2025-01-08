def merge_hdf5_files(input_files, output_file, chunk_size=1000):
    if os.path.exists(output_file):
        os.remove(output_file)

    # Open all input files
    h5_files = [h5py.File(f, 'r') for f in input_files]

    # Open the output file
    with h5py.File(output_file, 'w') as h5f_out:
        # Initialize datasets in the output file
        for key in h5_files[0].keys():
            shape = list(h5_files[0][key].shape)
            shape[0] = None  # Unlimited size along the first axis
            maxshape = tuple(shape)
            dtype = h5_files[0][key].dtype
            if key == 'sequence_id':
                dtype = h5py.special_dtype(vlen=str)
            h5f_out.create_dataset(
                key,
                shape=(0,) + h5_files[0][key].shape[1:],
                maxshape=maxshape,
                dtype=dtype,
                chunks=True,
                compression="gzip"
            )

        # Iterate over datasets and copy data in chunks
        for key in h5f_out.keys():
            total_size = sum(h5f[key].shape[0] for h5f in h5_files)
            h5f_out[key].resize(total_size, axis=0)
            current_index = 0
            for h5f in h5_files:
                dataset = h5f[key]
                num_rows = dataset.shape[0]
                for start in range(0, num_rows, chunk_size):
                    end = min(start + chunk_size, num_rows)
                    data_chunk = dataset[start:end]
                    h5f_out[key][current_index:current_index + (end - start)] = data_chunk
                    current_index += (end - start)

    # Close all input files
    for h5f in h5_files:
        h5f.close()

    print(f"Merged {len(input_files)} files into {output_file}")
