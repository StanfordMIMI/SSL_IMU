import h5py
import numpy as np
import pandas as pd
import json
from const import STANDARD_IMU_SEQUENCE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')


def plot_trial_data(trial_data, title, i_imu=0):
    plt.figure()
    plt.plot(trial_data[0, i_imu*3:(i_imu+1)*3])
    plt.title(title)


if __name__ == '__main__':

    # with h5py.File('D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/walking_knee_moment.h5', 'r') as hf:
    #     data_columns = json.loads(hf.attrs['columns'])
    #     data_list = [data_[:, :128, :] for sub_, data_ in hf.items()]     # only keep 128 time steps
    #     kam_data = np.concatenate(data_list, axis=0).transpose([0, 2, 1])
    #     """ [step, feature, time] """
    #     plot_trial_data(data_list[0], 'kam_data')

    IMU_LIST = [segment + sensor + axis for sensor in ['_Accel_', '_Gyro_'] for segment in STANDARD_IMU_SEQUENCE for axis in ['X', 'Y', 'Z']]
    with h5py.File('D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/MoVi.h5', 'r') as hf:
        data_columns = json.loads(hf.attrs['columns'])
        data_list = [data_[:] for sub_, data_ in hf.items()]
        movi_data = np.concatenate(data_list, axis=0)
        for i_sensor in range(8):
            col_acc = IMU_LIST[i_sensor*3:(i_sensor+1)*3]
            col_loc = [data_columns.index(col_) for col_ in col_acc]
            plt.figure()
            for i_axis in range(3):
                plt.plot(movi_data[:100, col_loc[i_axis]].ravel())
            plt.title(STANDARD_IMU_SEQUENCE[i_sensor])


    # data_loc = 'D:/Local/Data/DataAntoine/imu_in_h5/'
    # data_reader = CombinedDatasetLoader(data_loc, 100)
    # for dset in data_reader.data_contin.keys():
    #     for sub in data_reader.data_contin[dset].keys():
    #         for i in range(len(data_reader.data_contin[dset][sub])):
    #             plot_trial_data(data_reader.data_contin[dset][sub][i], dset)

    # with h5py.File('D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/combined_walking_knee_moment.h5', 'r') as hf:
    #     data_columns = json.loads(hf.attrs['columns'])
    #     data_dict = {dset_: data_[:] for dset_, data_ in hf.items()}
    # for dset_, data_ in data_dict.items():
    #     print('dset: ', dset_, 'max: ', np.max(np.abs(data_)), 'argmax: ', np.argmax(np.abs(data_)))
    #     for i in range(8):
    #         plt.figure()
    #         for j in range(3):
    #             plt.plot(data_[:5, 6*i+j, :].T, f'C{j}')
    #         plt.title(dset_ + ' ' + data_columns[6*i])
    #         plt.figure()
    #         for j in range(3):
    #             plt.plot(data_[:5, 6*i+j+3, :].T, f'C{j}')
    #         plt.title(dset_ + ' ' + data_columns[6*i+3])
    #     plt.show()

    # with h5py.File('D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/Camargo_100.h5', 'r') as hf:
    #     data_columns = json.loads(hf.attrs['columns'])
    #     data_dict = {dset_: data_[:] for dset_, data_ in hf.items()}
    #
    # grf_col_loc = [data_columns.index(col_) for col_ in ['fx', 'fy', 'fz']]
    # plt.figure()
    # plt.plot(data_dict['AB06'][0, grf_col_loc, :].T)
    plt.show()

