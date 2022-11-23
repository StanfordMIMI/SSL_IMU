import h5py
import numpy as np
import ast
from const import TRIAL_TYPES, GRAVITY, STANCE_V_GRF_THD, DICT_TRIAL_MOVI
from scipy.interpolate import interp1d
from a0_generate_windows import BaseSegment, DataStruct
import matplotlib.pyplot as plt

DATA_PATH_MOVI = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/MoVi/data_processed/'


class ContinuousDatasetLoader:
    def __init__(self, subject_list, target_fre):
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
                    if target_fre is not None:
                        trial_data = self.resample_120hz_to_target_fre(trial_data, target_fre)
                    trial_data = self.add_additional_info(trial_data, i_subject, i_trial)
                    self.data_contin[subject].append(trial_data)

    @staticmethod
    def resample_120hz_to_target_fre(trial_data, target_fre):
        x, step = np.linspace(0., 1., trial_data.shape[0], retstep=True)
        new_x = np.arange(0., 1., step*120/target_fre)
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
            [method.set_data_struct(DataStruct(len(self.columns), method.win_len)) for method in segment_methods]
            for trial_data in data_trials:
                # trial_data = self.clean_imu_data(trial_data, self.columns)
                for method in segment_methods:
                    method.start_segment(trial_data)
            [method.export(self.columns, subject) for method in segment_methods]


class WindowSegment(BaseSegment):
    def __init__(self, win_len, name='UnivariantWinTest'):
        self.win_len = win_len
        self.name = name
        # self.win_step = int(self.win_len/4)
        self.win_step = win_len

    def start_segment(self, trial_data):
        trial_len = trial_data.shape[0]
        i_current = 0
        acc_cols = [6*i for i in range(17)] + [6*i + 1 for i in range(17)] + [6*i + 2 for i in range(17)]
        acc_cols.sort()
        while i_current+self.win_len < trial_len:
            data_ = trial_data[i_current:i_current+self.win_len]
            acc_std = np.std(data_[:, acc_cols], axis=0)
            if acc_std.mean() >= 0.2:
                self.data_struct.add_new_step(data_)
            i_current += self.win_step


if __name__ == '__main__':
    sub_list = ['sub_' + str(i+1) for i in range(90)]

    data_reader = ContinuousDatasetLoader(sub_list, 200)
    data_reader.loop_all_the_trials([WindowSegment(64, 'MoVi_hw_running')])

    data_reader = ContinuousDatasetLoader(sub_list, 200)
    data_reader.loop_all_the_trials([WindowSegment(128, 'MoVi_Carmargo')])

    data_reader = ContinuousDatasetLoader(sub_list, 100)
    data_reader.loop_all_the_trials([WindowSegment(128, 'MoVi_walking_knee_moment')])

    # data_reader = ContinuousDatasetLoader(sub_list, 100)
    # data_reader.loop_all_the_trials([WindowSegment(80, 'MoVi_sun_drop_jump')])




