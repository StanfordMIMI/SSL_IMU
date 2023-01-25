from a0_process_MoVi import ContinuousDatasetLoader, WindowSegmentation, DataStruct, BaseSegmentation
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class CombinedDatasetSegmentation(BaseSegmentation):
    def __init__(self, win_len, name, imu_num):
        self.imu_num = imu_num
        self.win_len = win_len
        self.name = name
        self.win_step = int(0.5 * win_len)

    @staticmethod
    def is_window_healthy(data_):
        if np.max(np.abs(data_[1:] - data_[:-1])) > 10:
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


class CombinedDatasetLoader:
    @staticmethod
    def resample_to_target_fre(trial_data, target_fre, ori_fre):
        if target_fre == ori_fre:
            return trial_data
        else:
            return ContinuousDatasetLoader.resample_to_target_fre(trial_data, target_fre, ori_fre)

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
                i_dataset = int(dataset_name.split('dataset')[-1])
                with h5py.File(dirpath + '/' + trial, 'r') as hf:
                    trial_data = hf['data/block0_values'][:]
                    trial_data = self.resample_to_target_fre(trial_data=trial_data, target_fre=target_fre, ori_fre=dataset_frequencies[i_dataset])
                    if len(self.columns) == 0:
                        self.columns = [item.decode("utf-8") for item in hf['data/axis0']]
                    # trial_data = self.add_additional_info(trial_data, i_dataset, i_subject, i_trial)
                    if i_dataset not in self.data_contin.keys():
                        self.data_contin[i_dataset] = {}
                    if sub_name not in self.data_contin[i_dataset].keys():
                        self.data_contin[i_dataset][sub_name] = []
                    self.data_contin[i_dataset][sub_name].append(trial_data)

    def loop_all_the_trials(self, segment_methods):
        for i_dataset, dataset_data in self.data_contin.items():
            [method.set_data_struct(DataStruct(len(self.columns), method.win_len)) for method in segment_methods]
            for subject, data_trials in dataset_data.items():
                for data_current_trial in data_trials:
                    for method in segment_methods:
                        method.start_segment(data_current_trial)
            [method.export(self.columns, f'dset{i_dataset}') for method in segment_methods]



if __name__ == '__main__':
    data_loc = 'D:/Data/DataAntoine/imu_in_h5/'
    dataset_num = 30
    data_reader = CombinedDatasetLoader(data_loc, 100)
    data_reader.loop_all_the_trials([CombinedDatasetSegmentation(128, 'Combined_walking_knee_moment', 8)])



















