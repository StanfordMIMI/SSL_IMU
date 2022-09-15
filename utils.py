
import numpy as np
from scipy.signal import filtfilt, butter
import random
import torch
from scipy.signal import find_peaks


def fix_seed():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def find_peak_max(data_clip, height, width=None, prominence=None):
    """
    find the maximum peak
    :return:
    """
    peaks, properties = find_peaks(data_clip, width=width, height=height, prominence=prominence)
    if len(peaks) == 0:
        return None, None
    peak_heights = properties['peak_heights']
    max_index = np.nanargmax(peak_heights)
    return peaks[max_index], np.nanmax(peak_heights)


def data_filter(data, cut_off_fre, sampling_fre, filter_order=4):
    fre = cut_off_fre / (sampling_fre / 2)
    b, a = butter(filter_order, fre, 'lowpass')
    if len(data.shape) == 1:
        data_filtered = filtfilt(b, a, data)
    else:
        data_filtered = filtfilt(b, a, data, axis=0)
    return data_filtered


def get_data_by_merging_data_struct(data_struct_list):
    x_imu, x_emg, y = [], [], []
    for data_struct in data_struct_list:
        x_imu_trial, x_emg_trial, y_trial = data_struct.get_all_data()
        x_imu.append(x_imu_trial)
        x_emg.append(x_emg_trial)
        y.append(y_trial)
    return {'IMU': np.concatenate(x_imu, axis=0), 'EMG': np.concatenate(x_emg, axis=0), 'y': np.concatenate(y, axis=0)}


class DataStruct:
    def __init__(self, feature_num_200, feature_num_1000, output_num_200, step_len_max):

        self.step_len_max = step_len_max
        self.num_of_step_allocate_one_time = 2000
        self.x_200 = np.zeros([self.num_of_step_allocate_one_time, step_len_max, feature_num_200])
        self.x_1000 = np.zeros([self.num_of_step_allocate_one_time, step_len_max * 5, feature_num_1000])
        self.y_200 = np.zeros([self.num_of_step_allocate_one_time, step_len_max, output_num_200])
        self.i_step = 0

    def add_new_step(self, x_step_200, x_step_1000, y_step_200):
        if self.i_step == self.x_200.shape[0] - 1:
            self.x_200 = np.concatenate([self.x_200, np.zeros(self.x_200.shape)], axis=0)
            self.x_1000 = np.concatenate([self.x_1000, np.zeros(self.x_1000.shape)], axis=0)
            self.y_200 = np.concatenate([self.y_200, np.zeros(self.y_200.shape)], axis=0)
        self.x_200[self.i_step, :x_step_200.shape[0], :] = x_step_200
        self.x_1000[self.i_step, :x_step_1000.shape[0], :] = x_step_1000
        self.y_200[self.i_step, :y_step_200.shape[0], :] = y_step_200
        self.i_step += 1

    def get_all_data(self):
        return self.x_200[:self.i_step], self.x_1000[:self.i_step], self.y_200[:self.i_step]


