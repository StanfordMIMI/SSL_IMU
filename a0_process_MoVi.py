
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import ast
from scipy.signal import medfilt
from utils import get_data_by_merging_data_struct, find_peak_max, data_filter
from const import TRIAL_TYPES, GRAVITY, STANCE_V_GRF_THD, DICT_TRIAL_MOVI
from const import DICT_SUBJECT_ID, DICT_TRIAL_TYPE_ID
from scipy.interpolate import interp1d
from a0_generate_windows import BaseSegment, DataStruct

DATA_PATH_MOVI = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/MoVi/data_processed/'


class ContinuousDatasetLoader:
    def __init__(self, subject_list):
        self.columns = self.load_columns()
        self.subject_list = subject_list
        self.data_contin = {}
        for i_subject, subject in enumerate(subject_list):
            self.data_contin[subject] = []
            print("Loading subject " + subject)
            with h5py.File(DATA_PATH_MOVI + subject + '.h5', 'r') as hf:
                for trial, trial_data in hf.items():
                    i_trial = DICT_TRIAL_MOVI[trial]
                    trial_data = trial_data[:].T
                    trial_data = self.resample_120_to_100hz(trial_data)
                    trial_data = self.add_additional_info(trial_data, i_subject, i_trial)
                    self.data_contin[subject].append(trial_data)
                # if np.mean(trial_data[:10, 32]) > 0.8:
                #     print(subject)
                # plt.plot(trial_data[:10, 26])
        # plt.show()


    @staticmethod
    def resample_120_to_100hz(trial_data):
        x, step = np.linspace(0., 1., trial_data.shape[0], retstep=True)
        new_x = np.arange(0., 1., step*120/100)
        f = interp1d(x, trial_data, axis=0)
        trial_data_resampled = f(new_x)

        return trial_data_resampled

    def add_additional_info(self, trial_data, i_subject, i_trial):
        data_len = trial_data.shape[0]
        subject_id_array = np.full([data_len], i_subject)
        trial_type_id_array = np.full([data_len], i_trial)

        trial_data = np.column_stack([trial_data, subject_id_array, trial_type_id_array])
        if len(self.columns) != trial_data.shape[1]:
            self.columns.extend(['sub_id', 'trial_type_id'])
        return trial_data

    @staticmethod
    def load_columns():
        columns = list(np.array(ast.literal_eval(open(DATA_PATH_MOVI + '/columns.txt').read()), dtype=object))
        return columns

    def loop_all_the_trials(self, segment_methods):
        for subject, data_trials in self.data_contin.items():
            [method.set_data_struct(DataStruct(len(self.columns), method.data_len)) for method in segment_methods]
            for trial_data in data_trials:
                # trial_data = self.clean_imu_data(trial_data, self.columns)
                for method in segment_methods:
                    method.start_segment(trial_data)
            [method.export(self.columns, subject) for method in segment_methods]


class WindowSegment(BaseSegment):
    def __init__(self, name='UnivariantWinTest'):
        self.data_len = 100
        self.name = name
        self.win_len, self.win_step = self.data_len, int(self.data_len/5)

    def start_segment(self, trial_data):
        trial_len = trial_data.shape[0]
        i_current = 0
        acc_cols = [6*i for i in range(17)] + [6*i + 1 for i in range(17)] + [6*i + 2 for i in range(17)]
        acc_cols.sort()
        while i_current+self.data_len < trial_len:
            data_ = trial_data[i_current:i_current+self.data_len]
            acc_std = np.std(data_[:, acc_cols], axis=0)
            if acc_std.mean() >= 0.2:
                self.data_struct.add_new_step(data_)
            i_current += self.win_step


if __name__ == '__main__':
    sub_list = ['sub_' + str(i+1) for i in range(88)]
    data_reader = ContinuousDatasetLoader(sub_list)
    data_reader.loop_all_the_trials([WindowSegment('MoVi')])