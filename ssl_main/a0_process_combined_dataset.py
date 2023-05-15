from utils import resample_to_target_fre
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from a0_process_camargo import BaseSegmentation, DataStruct


class CombinedDatasetSegmentation(BaseSegmentation):
    def __init__(self, win_len, name, imu_num):
        self.imu_num = imu_num
        self.win_len = win_len
        self.name = name
        self.win_step = int(0.5 * win_len)

    @staticmethod
    def is_window_healthy(data_):
        for i_axis in range(data_.shape[1]):
            diff_std = np.std(data_[1:, i_axis] - data_[:-1, i_axis])
            if np.max(np.abs(data_[1:, i_axis] - data_[:-1, i_axis])) > 6 * diff_std:
                return False
            if np.max(np.abs(data_[1:-1, i_axis] - 0.5 * data_[:-2, i_axis] - 0.5 * data_[2:, i_axis])) > 6 * diff_std:
                return False
            if np.max(np.abs(data_)) > 1000:
                return False
        return True

    def start_segment(self, trial_data):
        trial_len = trial_data.shape[0]
        i_current = 0
        while i_current+self.win_len < trial_len:
            data_ = trial_data[i_current:i_current+self.win_len]
            i_current += self.win_step
            if self.is_window_healthy(data_):
                self.data_struct.add_new_step(data_)
            else:
                plt.figure()
                plt.plot(data_)
                plt.show()


class CombinedDatasetLoader:
    @staticmethod
    def resample_to_target_fre(trial_data, target_fre, ori_fre):
        if target_fre == ori_fre:
            return trial_data
        else:
            return resample_to_target_fre(trial_data, target_fre, ori_fre)

    def add_additional_info(self, trial_data, i_dataset, i_subject, i_trial):
        return trial_data

    def __init__(self, data_loc, target_fre):
        self.target_fre = target_fre
        dataset_info = pd.read_excel(data_loc + '../database.xlsx', index_col=0)
        self.columns = []
        self.data_contin = {}

        dataset_frequencies = {i_dataset: dataset_info['Sampling frequency'][i_dataset] for i_dataset in range(30)}

        for (dirpath, dirnames, trial_names) in os.walk(data_loc):
            for trial in trial_names:
                dataset_name, sub_name = dirpath.replace('\\', '/').split('/')[-2:]
                if 'static' in trial:
                    continue
                # if '0' not in sub_name:
                #     continue
                i_dataset = int(dataset_name.split('dataset')[-1])
                try:
                    with h5py.File(dirpath + '/' + trial, 'r') as hf:
                        trial_data = hf['data/block0_values'][:]
                        trial_data = self.resample_to_target_fre(trial_data=trial_data, target_fre=target_fre, ori_fre=dataset_frequencies[i_dataset])
                        if len(self.columns) == 0:
                            self.columns = [item.decode("utf-8") for item in hf['data/axis0']]
                        if i_dataset not in self.data_contin.keys():
                            self.data_contin[i_dataset] = {}
                        if sub_name not in self.data_contin[i_dataset].keys():
                            self.data_contin[i_dataset][sub_name] = []
                        self.data_contin[i_dataset][sub_name].append(trial_data)
                except:
                    print('Error in reading: ', dirpath + '/' + trial)
                    continue

    def loop_all_the_trials(self, segment_methods):
        for i_dataset, dataset_data in self.data_contin.items():
            print('Looping dataset: ', i_dataset)
            [method.set_data_struct(DataStruct(len(self.columns), method.win_len)) for method in segment_methods]
            for subject, data_trials in dataset_data.items():
                for data_current_trial in data_trials:
                    for method in segment_methods:
                        method.start_segment(data_current_trial)
            [method.export(self.columns, f'dset{i_dataset}') for method in segment_methods]


if __name__ == '__main__':
    data_loc = 'D:/Local/Data/DataAntoine/imu_in_h5_gait/'
    data_reader = CombinedDatasetLoader(data_loc, 100)
    data_reader.loop_all_the_trials([CombinedDatasetSegmentation(80, 'Combined_sun_drop_jump', 8)])



















