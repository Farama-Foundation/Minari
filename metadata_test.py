import h5py


with h5py.File(
    "/home/rodrigo/.minari/datasets/hammer-human-v0/data/main_data.hdf5", "r"
) as f:
    print(dict(f.attrs.items()))
