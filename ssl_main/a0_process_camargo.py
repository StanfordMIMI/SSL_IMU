import json
import h5py
import matplotlib.pyplot as plt
import numpy as np
import ast
from scipy.signal import medfilt
from utils import get_data_by_merging_data_struct, find_peak_max, data_filter
from const import DATA_PATH_CAMARGO_WIN, TRIAL_TYPES, GRAVITY, IMU_CAMARGO_SAMPLE_RATE, EMG_CAMARGO_SAMPLE_RATE,\
    GRF_CAMARGO_SAMPLE_RATE, IMU_CAMARGO_SEGMENT_LIST, DICT_LABEL, STANCE_V_GRF_THD
from config import DATA_PATH
from const import DICT_SUBJECT_ID, DICT_TRIAL_TYPE_ID


class ContinuousDatasetLoader:
    def __init__(self, subject_list):
        self.sample_rate = IMU_CAMARGO_SAMPLE_RATE
        self.columns_raw = self.load_columns()
        for key_ in self.columns_raw.keys():
            self.columns_raw[key_]['200'] = self.update_column_names(self.columns_raw[key_]['200'])
        self.col_200 = INFO_LIST + IMU_LIST
        self.col_1000 = FORCE_LIST
        self.subject_list = subject_list
        self.data_contin_merged = {}
        for subject in subject_list:
            print("Loading subject " + subject)
            with h5py.File(DATA_PATH_CAMARGO_WIN + subject + '.h5', 'r') as hf:
                data_trials = {}
                [data_trials.update({trial: [trial_data['data_200'][:], trial_data['data_1000'][:]]})
                 for trial, trial_data in hf.items()]

                data_trials = self.add_additional_info(data_trials, subject)
                data_trials = self.process_grf(data_trials)
                data_trial_merged, columns = self.merge_1000_to_200(data_trials)
                data_trial_merged = self.resample_to_100(data_trial_merged)
                self.data_contin_merged[subject] = data_trial_merged
        columns = self.update_column_names(columns)
        self.columns = columns

        # grf_col_loc = [self.columns.index(col_) for col_ in ['fx', 'fy', 'fz']]
        # plt.figure()
        # plt.plot(data_trial_merged['Treadmill_01_01'][:, grf_col_loc])
        # plt.show()

        self.trials = list(data_trials.keys())

    def resample_to_100(self, data_trial_merged):
        for trial, trial_data in data_trial_merged.items():
            data_trial_merged[trial] = trial_data[::2, :]
        return data_trial_merged

    @staticmethod
    def update_column_names(columns):
        column_name_map = {'trunk': 'CHEST', 'thigh': 'R_THIGH', 'shank': 'R_SHANK', 'foot': 'R_FOOT'}
        for key in column_name_map.keys():
            for column in columns:
                if key in column:
                    columns[columns.index(column)] = column.replace(key, column_name_map[key])
        return columns

    def add_additional_info(self, data_trials, subject):
        for trial, trial_data in data_trials.items():
            trial_info = trial.split('_')
            condition = trial_info[0]
            data_len = trial_data[0].shape[0]
            subject_id_array = np.full([data_len], DICT_SUBJECT_ID[subject])
            trial_type_id_array = np.full([data_len], DICT_TRIAL_TYPE_ID[condition])

            trial_data[0] = np.column_stack([trial_data[0], subject_id_array, trial_type_id_array])
            data_trials[trial] = trial_data
            if len(self.columns_raw[condition]['200']) != data_trials[trial][0].shape[1]:
                self.columns_raw[condition]['200'].extend(['sub_id', 'trial_type_id'])
        return data_trials

    def merge_1000_to_200(self, data_trials):
        for trial, trial_data in data_trials.items():
            condition = trial.split('_')[0]
            col_loc_200 = [self.columns_raw[condition]['200'].index(x) for x in self.col_200]
            data_200_to_keep = trial_data[0][:, col_loc_200]

            col_loc_1000 = [self.columns_raw[condition]['1000'].index(x) for x in self.col_1000]
            data_1000_to_keep = trial_data[1][::5, col_loc_1000]
            if data_200_to_keep.shape[0] != data_1000_to_keep.shape[0]:
                shorter_len = min(data_200_to_keep.shape[0], data_1000_to_keep.shape[0])
                data_200_to_keep, data_1000_to_keep = data_200_to_keep[:shorter_len], data_1000_to_keep[:shorter_len]
            trial_data_merged = np.column_stack([data_200_to_keep, data_1000_to_keep])
            data_trials[trial] = trial_data_merged
        columns = self.col_200 + self.col_1000
        return data_trials, columns

    def process_grf(self, data_trials):
        for trial, trial_data in data_trials.items():
            trial_info = trial.split('_')
            condition = trial_info[0]
            if condition == 'LevelGround':
                condition_force = condition + '_' + trial_info[1]
            elif condition in ['Ramp', 'Stair']:
                condition_force = condition + '_' + trial_info[2]
            else:
                condition_force = condition
            grf_cols = RIGHT_FOOT_FORCE_COL[condition_force]
            grf_col_loc = [self.columns_raw[condition]['1000'].index(x) for x in grf_cols]

            cop_cols = RIGHT_FOOT_COP_COL[condition_force]
            cop_col_loc = [self.columns_raw[condition]['1000'].index(x) for x in cop_cols]
            cop_data = trial_data[1][:, cop_col_loc]
            grf_data = trial_data[1][:, grf_col_loc]

            for i_plate, v_grf_plate_col in enumerate(grf_col_loc[1::3]):
                v_grf_plate = trial_data[1][:, v_grf_plate_col]
                for i_axis in range(3):
                    cop_data[:, 3*i_plate + i_axis][v_grf_plate < STANCE_V_GRF_THD] = 0

            for data_, data_name in zip([cop_data, grf_data], [['cx', 'cy', 'cz'], ['fx', 'fy', 'fz']]):

                data_combined = np.zeros([data_.shape[0], 3])
                for i in range(0, data_.shape[1], 3):
                    data_combined += data_[:, i:i+3]
                data_combined = data_filter(data_combined, 15, GRF_CAMARGO_SAMPLE_RATE)

                trial_data[1] = np.column_stack([trial_data[1], data_combined])
                data_trials[trial] = trial_data
                if len(self.columns_raw[condition]['1000']) != data_trials[trial][1].shape[1]:
                    self.columns_raw[condition]['1000'].extend(data_name)
        return data_trials

    def process_emg(self, data_trials):
        for trial, trial_data in data_trials.items():
            condition = trial.split('_')[0]
            emg_col_loc = [self.columns_raw[condition]['1000'].index(x) for x in EMG_LIST]
            emg_data = trial_data[1][:, emg_col_loc]
            emg_data = data_filter(np.abs(emg_data), 20, EMG_CAMARGO_SAMPLE_RATE)
            trial_data[1][:, emg_col_loc] = emg_data
        return data_trials

    @staticmethod
    def clean_imu_data(data, columns):
        imu_locs = [columns.index(col) for col in IMU_LIST]
        for loc in imu_locs:
            data[:, loc] = medfilt(data[:, loc], 3)

        sensor_and_thd = {'trunk_Accel_': 7.8, 'trunk_Gyro_': 15.,
                          'thigh_Accel_': 2, 'thigh_Gyro_': 15.,
                          'shank_Accel_': 7.8, 'shank_Gyro_': 15.,
                          'foot_Accel_': 7.8, 'foot_Gyro_': 15.}
        # for sensor, thd in sensor_and_thd.items():
        #     col_locs = [columns.index(sensor + axis) for axis in ['X', 'Y', 'Z']]
        #     for loc in col_locs:
        #         data_axis = data[:, loc]
        #         data_axis[data_axis > thd] = thd
        #         data_axis[data_axis < -thd] = -thd
        #         data[:, loc] = data_axis

        return data

    @staticmethod
    def are_kinematics_correct(data_200, columns, kinematic_range):
        kinematic_col_loc = [columns.index(col) for col in
                             ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r']]
        if np.min(data_200[:, kinematic_col_loc]) < kinematic_range[0] or np.max(data_200[:, kinematic_col_loc]) > kinematic_range[1]:
            return False
        return True

    @staticmethod
    def load_columns():
        columns = {ambulation: {} for ambulation in TRIAL_TYPES}
        for ambulation in TRIAL_TYPES:
            for frequency in ['200', '1000']:
                columns[ambulation][frequency] = list(np.array(ast.literal_eval(open(
                    DATA_PATH_CAMARGO_WIN + ambulation + '_' + frequency + '_columns.txt').read()), dtype=object))
        return columns

    def transform_imu_orientation(self, data, columns):
        """ To follow the convention of z pointing skin surface norm, y pointing upwards  """
        rot_mat = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
        imu_locs = [columns.index(col) for col in IMU_LIST]
        for i_sensor in range(0, len(imu_locs), 3):
            data[:, imu_locs[i_sensor:i_sensor+3]] = np.matmul(rot_mat, data[:, imu_locs[i_sensor:i_sensor+3]].T).T
        return data

    def loop_all_the_trials(self, segment_methods):
        for subject in self.subject_list:
            [method.set_data_struct(DataStruct(len(self.columns), method.data_len)) for method in segment_methods]
            print('processing ' + subject)
            for trial in self.trials:
                if trial not in self.data_contin_merged[subject].keys():
                    continue
                # trial_type = trial.split('_')[0]
                trial_data = self.data_contin_merged[subject][trial]
                if not self.are_kinematics_correct(trial_data, self.columns, (-180, 180)):
                    continue
                # trial_data = self.clean_imu_data(trial_data, self.columns)
                self.transform_imu_orientation(trial_data, self.columns)
                for method in segment_methods:
                    method.start_segment(trial_data, self.columns)
            [method.export(self.columns, subject) for method in segment_methods]


class DataStruct:
    """ 3 dimensions of self.data: [window, columns, time_step]"""
    def __init__(self, col_num, step_len_max):
        self.step_len_max = step_len_max
        self.num_of_step_allocate_one_time = 2000
        self.data = np.zeros([self.num_of_step_allocate_one_time, col_num, step_len_max])
        self.i_step = 0

    def add_new_step(self, data_step):
        if self.i_step == self.data.shape[0] - 1:
            self.data = np.concatenate([self.data, np.zeros(self.data.shape)], axis=0)
        data_feature_first = np.transpose(data_step)
        self.data[self.i_step, :, :data_step.shape[0]] = data_feature_first
        self.i_step += 1

    def get_all_data(self):
        return self.data[:self.i_step]


class BaseSegmentation:
    def set_data_struct(self, data_struct):
        self.data_struct = data_struct

    def export(self, columns, dataset_name):
        all_data = self.data_struct.get_all_data()
        if all_data.shape[0] == 0:
            return
        with h5py.File(DATA_PATH + self.name + '.h5', 'a') as hf:
            try: del hf[dataset_name]
            except KeyError: pass
            dset = hf.create_dataset(dataset_name, all_data.shape, data=all_data)
            hf.attrs['columns'] = json.dumps(columns)


class WindowSegmentation(BaseSegmentation):
    def __init__(self, name='Camargo'):
        self.win_len = self.data_len = 128
        self.name = name
        self.win_step = int(self.win_len/2)

    def start_segment(self, trial_data, columns):
        trial_len = trial_data.shape[0]
        label,  = [trial_data[:, columns.index(col)] for col in ['label']]
        walking_turning_stair_ramp_treadmill = np.sum(np.array([label == i for i in [-1, 0, 3, 4, 5, 6, 7, 8, 9]]), axis=0).astype(dtype=bool)
        if walking_turning_stair_ramp_treadmill.all():
            start_loc = [0]
        else:
            start_loc = np.where(walking_turning_stair_ramp_treadmill[1:] & ~walking_turning_stair_ramp_treadmill[:-1])[0] + 1
        for i_start in start_loc:
            i_current = i_start
            while(i_current+self.win_len < trial_len and walking_turning_stair_ramp_treadmill[i_current:i_current+self.win_len].all()):
                data_ = trial_data[i_current:i_current+self.win_len]
                self.data_struct.add_new_step(data_)
                i_current += self.win_step


IMU_LIST = [segment + sensor + axis for sensor in ['_Accel_', '_Gyro_'] for segment in IMU_CAMARGO_SEGMENT_LIST for axis in ['X', 'Y', 'Z']]
INFO_LIST = ['sub_id', 'trial_type_id', 'label', 'treadmill_speed', 'ramp', 'heel_strike', 'toe_off',
             'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'knee_angle_r_moment']

EMG_LIST = ['gastrocmed', 'tibialisanterior', 'soleus', 'vastusmedialis', 'vastuslateralis', 'rectusfemoris',
            'bicepsfemoris', 'semitendinosus', 'gracilis', 'gluteusmedius']
FORCE_LIST = ['fx', 'fy', 'fz']     # 'cx', 'cy', 'cz'

FORCE_AXIS, COP_AXIS = ['vx', 'vy', 'vz'], ['px', 'py', 'pz']
RIGHT_FOOT_PLATES = {'Treadmill': ['Treadmill_R'], 'LevelGround_cw': ['Combined', 'FP2', 'FP5'],
                     'LevelGround_ccw': ['FP1'], 'Stair_R': ['FP2', 'FP3', 'FP4'], 'Stair_L': ['FP1', 'FP5'],
                     'Ramp_R': ['FP2', 'FP4'], 'Ramp_L': ['FP1', 'FP3', 'FP5']}
RIGHT_FOOT_FORCE_COL = {type: [plate + '_' + axis for plate in plates for axis in FORCE_AXIS] for type, plates in RIGHT_FOOT_PLATES.items()}
RIGHT_FOOT_COP_COL = {type: [plate + '_' + axis for plate in plates for axis in COP_AXIS] for type, plates in RIGHT_FOOT_PLATES.items()}

PARAM_TO_STORE = []

if __name__ == '__main__':
    sub_list = [
        'AB06',
        'AB07',
        # 'AB08', 'AB09', 'AB10',
        # 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB17',
        # 'AB18', 'AB19', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB30'
    ]
    data_reader = ContinuousDatasetLoader(sub_list)
    data_reader.loop_all_the_trials([WindowSegmentation('Camargo_100')])

