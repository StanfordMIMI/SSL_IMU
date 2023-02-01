import matplotlib.pyplot as plt
import h5py
import json
import random


def show_each_dset(dset):
    rows = random.sample(range(ssl_data[dset].shape[0]), min(windows_to_plot, ssl_data[dset].shape[0]))
    plt.figure()
    plt.plot(ssl_data[dset][rows, 24].T, 'C0')
    plt.show()


windows_to_plot = 3
with h5py.File('D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/Combined_walking_knee_moment.h5', 'r') as hf:
    ssl_data = {dset: dset_data[:windows_to_plot, :, :] for dset, dset_data in hf.items()}
    ssl_columns = json.loads(hf.attrs['columns'])
show_each_dset('dset5')



