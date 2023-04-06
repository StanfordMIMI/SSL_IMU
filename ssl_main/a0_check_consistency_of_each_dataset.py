import h5py
import numpy as np
import pandas as pd
import json
from a0_process_combined_dataset import CombinedDatasetLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')


def plot_trial_data(trial_data, title, i_imu=0):
    plt.figure()
    plt.plot(trial_data[:, i_imu*3:(i_imu+1)*3])
    plt.title(title)
    # plt.show()


if __name__ == '__main__':
    data_loc = 'D:/Local/Data/DataAntoine/imu_in_h5/'
    data_reader = CombinedDatasetLoader(data_loc, 100)
    x=1

    with h5py.File('D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/walking_knee_moment.h5', 'r') as hf:
        data_columns = json.loads(hf.attrs['columns'])
        data_list = [data_[:, :128, :] for sub_, data_ in hf.items()]     # only keep 128 time steps
        kam_data = np.concatenate(data_list, axis=0).transpose([0, 2, 1])
        """ [step, feature, time] """
        plot_trial_data(data_list[0], 'kam_data')

    for dset in data_reader.data_contin.keys():
        for sub in data_reader.data_contin[dset].keys():

            plot_trial_data(data_reader.data_contin[dset][sub][0], dset)

    plt.show()



