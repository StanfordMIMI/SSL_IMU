import h5py
import numpy as np
import ast
from const import TRIAL_TYPES, GRAVITY, STANDARD_IMU_SEQUENCE, DICT_TRIAL_MOVI
from utils import resample_to_target_fre
from a0_process_camargo import DataStruct, BaseSegmentation
DATA_PATH_MOVI = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/MoVi/data_processed/'


""" 
IMU orientations
Thigh: X down
Shank: X down
Foot: X forward
"""


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
                        trial_data = resample_to_target_fre(trial_data, target_fre)
                    trial_data = self.add_additional_info(trial_data, i_subject, i_trial)
                    self.data_contin[subject].append(trial_data)

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

    @staticmethod
    def calibrate_imu_orientation(real_imu, columns):
        for imu in STANDARD_IMU_SEQUENCE:
            col_loc_acc = [columns.index(item) for item in [imu + '_Accel_' + axis for axis in ['X', 'Y', 'Z']]]
            col_loc_gyr = [columns.index(item) for item in [imu + '_Gyro_' + axis for axis in ['X', 'Y', 'Z']]]
            for i_sample in range(real_imu.shape[0]):
                real_imu[i_sample, col_loc_acc] = np.matmul(movi_orientation_transform_mat[imu], real_imu[i_sample, col_loc_acc])
                real_imu[i_sample, col_loc_gyr] = np.matmul(movi_orientation_transform_mat[imu], real_imu[i_sample, col_loc_gyr])
        return real_imu

    @staticmethod
    def update_imu_name(columns):
        name_map = {'Spine1': 'CHEST', 'Hip': 'WAIST', 'RightUpLeg': 'R_THIGH', 'LeftUpLeg': 'L_THIGH',
                    'RightLeg': 'R_SHANK', 'LeftLeg': 'L_SHANK', 'RightFoot': 'R_FOOT', 'LeftFoot': 'L_FOOT'}
        old_name_list = list(name_map.keys())
        for i_col, col in enumerate(columns):
            for old_name in old_name_list:
                if old_name in col:
                    columns[i_col] = col.replace(old_name, name_map[old_name])
        return columns

    def loop_all_the_trials(self, segment_methods):
        for subject, data_trials in self.data_contin.items():
            [method.set_data_struct(DataStruct(len(self.columns), method.win_len)) for method in segment_methods]
            for trial_data in data_trials:
                self.columns = self.update_imu_name(self.columns)
                trial_data = self.calibrate_imu_orientation(trial_data, self.columns)
                for method in segment_methods:
                    method.start_segment(trial_data)
            [method.export(self.columns, subject) for method in segment_methods]


class WindowSegmentation(BaseSegmentation):
    def __init__(self, win_len, name='UnivariantWinTest', imu_num=17):
        self.imu_num = imu_num
        self.win_len = win_len
        self.name = name
        # self.win_step = int(self.win_len/4)
        self.win_step = win_len

    def start_segment(self, trial_data):
        trial_len = trial_data.shape[0]
        i_current = 0
        acc_cols = [6*i for i in range(self.imu_num)] + [6*i + 1 for i in range(self.imu_num)] + [6*i + 2 for i in range(self.imu_num)]
        acc_cols.sort()
        while i_current+self.win_len < trial_len:
            data_ = trial_data[i_current:i_current+self.win_len]
            acc_std = np.std(data_[:, acc_cols], axis=0)
            if acc_std.mean() >= 0.2:
                self.data_struct.add_new_step(data_)
            i_current += self.win_step


movi_orientation_transform_mat = {
    'CHEST': np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]), 'WAIST': np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
    'R_THIGH': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]), 'L_THIGH': np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
    'R_SHANK': np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]), 'L_SHANK': np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
    'R_FOOT': np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]), 'L_FOOT': np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])}

if __name__ == '__main__':
    sub_list = ['sub_' + str(i+1) for i in range(88)]

    # data_reader = ContinuousDatasetLoader(sub_list, 200)
    # data_reader.loop_all_the_trials([WindowSegmentation(64, 'MoVi_hw_running')])
    #
    # data_reader = ContinuousDatasetLoader(sub_list, 200)
    # data_reader.loop_all_the_trials([WindowSegmentation(128, 'MoVi_Camargo')])

    data_reader = ContinuousDatasetLoader(sub_list, 100)
    data_reader.loop_all_the_trials([WindowSegmentation(128, 'MoVi')])

    # data_reader = ContinuousDatasetLoader(sub_list, 100)
    # data_reader.loop_all_the_trials([WindowSegment(80, 'MoVi_sun_drop_jump')])




