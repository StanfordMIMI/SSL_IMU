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


def add_clinical_metrics(data, trial, columns):
    # TODO: Add peak knee adduction moment impulse

    # add peak vertical GRF during walking
    gc_strike, gc_off, label = [data[:, columns.index(col)] for col in ['heel_strike', 'toe_off', 'label']]
    strike_end_loc = np.where(gc_strike == 100.)[0]
    strike_loc = np.where((gc_strike == 0)[:-1] & (gc_strike > 0)[1:])[0]
    if len(strike_loc) > 0:
        if len(strike_end_loc) == 0 or strike_loc[-1] > strike_end_loc[-1]:
            strike_loc = strike_loc[:-1]

    # add kinetics
    v_grf = data[:, columns.index('fy')]
    kinetic_to_process = {'fy': 1, 'knee_angle_r_moment': -1}
    peak_kinetic = np.zeros([v_grf.shape[0], len(kinetic_to_process)])
    strike_and_walking_and_has_grf_loc = [loc for loc in strike_loc if label[loc] in [-1, 0, 4, 5, 7, 8] and v_grf[loc+10] > STANCE_V_GRF_THD]  #
    for i_kin, kinetic_metric_name in enumerate(kinetic_to_process.keys()):
        kin_data = data[:, columns.index(kinetic_metric_name)]
        for step_start in strike_and_walking_and_has_grf_loc:
            step_end = strike_end_loc[strike_end_loc > step_start][0]
            clip_start = step_start + 10
            try:
                clip_end = np.where(v_grf[clip_start:step_end] < STANCE_V_GRF_THD)[0][0] + clip_start - 10
            except IndexError:
                continue
            kin_clip = kin_data[clip_start:clip_end]
            peak_loc, peak_val = find_peak_max(kinetic_to_process[kinetic_metric_name] * kin_clip, -1e5)
            if peak_loc:
                peak_kinetic[peak_loc+clip_start, i_kin] = kinetic_to_process[kinetic_metric_name] * peak_val

    # add kinematics
    kinematic_to_process = {'hip_flexion_r': -1}
    peak_kinematic = np.zeros([v_grf.shape[0], len(kinematic_to_process)])
    strike_and_walking = [loc for loc in strike_loc if label[loc] in [-1, 0]]
    for i_kin, kinematic_metric_name in enumerate(kinematic_to_process.keys()):
        kin_data = data[:, columns.index(kinematic_metric_name)]
        for step_start in strike_and_walking:
            step_end = strike_end_loc[strike_end_loc > step_start][0]
            clip_start = step_start + int(0.1 * (step_end - step_start))
            clip_end = step_start + int(0.7 * (step_end - step_start))
            kin_clip = kin_data[clip_start:clip_end]
            peak_loc, peak_val = find_peak_max(kinematic_to_process[kinematic_metric_name] * kin_clip, -1e5)
            if peak_loc:
                peak_kinematic[peak_loc+clip_start, i_kin] = kinematic_to_process[kinematic_metric_name] * peak_val
    #
    # if len(strike_and_walking):
    #     plt.figure()
    #     plt.title(trial)
    #     plt.plot(gc_strike)
    #     plt.plot(data[:, columns.index('hip_flexion_r')])       # knee_angle_r
    #     plt.plot(strike_and_walking, [0 for i in strike_and_walking], '*')
    #     plt.plot(peak_kinematic[:, 0])
    #     plt.show()

    data = np.column_stack([data, peak_kinetic, peak_kinematic])
    peak_kin_names = ['peak_' + item for item in list(kinetic_to_process.keys()) + list(kinematic_to_process.keys())]
    return data, peak_kin_names


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
                self.data_contin_merged[subject] = data_trial_merged
        columns = self.update_column_names(columns)
        self.columns = columns
        self.trials = list(data_trials.keys())

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

    def add_additional_columns(self):
        for subject in self.subject_list:
            for trial in self.trials:
                if trial not in self.data_contin_merged[subject].keys():
                    continue
                self.data_contin_merged[subject][trial], new_cols = add_clinical_metrics(
                    self.data_contin_merged[subject][trial], trial, self.columns)
        if new_cols[0] not in self.columns: self.columns.extend(new_cols)

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


class CenterSegmentation(BaseSegmentation):
    def __init__(self, name='Camargo'):
        self.win_len = self.data_len = 400
        self.half_win_len = int(self.win_len/2)
        self.name = name
        self.peak_outcome = 'peak_fy'

    def start_segment(self, trial_data, columns):
        trial_len = trial_data.shape[0]

        outcome_loc = np.where(trial_data[:, columns.index(self.peak_outcome)] != 0)[0]
        for loc in outcome_loc:
            if loc - self.half_win_len > 0 and loc + self.half_win_len < trial_len:
                data_ = trial_data[loc-self.half_win_len:loc + self.half_win_len]
                self.data_struct.add_new_step(data_)


class StepSegmentation(BaseSegmentation):
    def __init__(self, name):
        self.data_len = 256
        self.name = name
        self.win_len, self.win_step = self.data_len, int(self.data_len/2)
        self.step_len_max, self.step_len_min = int(256 * IMU_CAMARGO_SAMPLE_RATE / 200), int(40 * IMU_CAMARGO_SAMPLE_RATE / 200)

    @staticmethod
    def strike_off_to_step_and_remove_incorrect_step(gyr_y, strike_list, off_list, step_len_max, step_len_min, from_strike_to_off=True):
        SAMPLES_BEFORE_STEP, SAMPLES_AFTER_STEP = 0, 0
        steps_ = []
        if from_strike_to_off:
            event_1, event_2 = np.array(strike_list), np.array(off_list)
            event_1_height_thd, event_2_height_thd = 0, 4
        else:
            event_2, event_1 = np.array(strike_list), np.array(off_list)
            event_2_height_thd, event_1_height_thd = 0, 4
        for i_event_1 in range(len(event_1)):
            potential_event_2 = event_2[event_1[i_event_1] + step_len_min < event_2]
            potential_event_2 = potential_event_2[potential_event_2 < event_1[i_event_1] + step_len_max - SAMPLES_BEFORE_STEP - SAMPLES_AFTER_STEP]
            if len(potential_event_2) == 1:
                if gyr_y[event_1[i_event_1]] > event_1_height_thd and gyr_y[potential_event_2] > event_2_height_thd:
                    steps_.append([event_1[i_event_1] - SAMPLES_BEFORE_STEP, potential_event_2[0] + SAMPLES_AFTER_STEP])
        return steps_

    @staticmethod
    def initalize_steps_and_stance_phase(data_df, strike_list, off_list, sample_after_thd=10):
        """The name "stance phase" is not accurate. It starts from gyr < thd + sample_after_thd sample,
         ends in the middle of the stance"""

        gyr_all = np.deg2rad(data_df[['gyr_x', 'gyr_y', 'gyr_z']])
        gyr_magnitude = np.linalg.norm(gyr_all, axis=1)

        imu_sample_rate = IMU_CAMARGO_SAMPLE_RATE
        stance_phase_sample_thd_lower = 0.3 * imu_sample_rate
        stance_phase_sample_thd_higher = 1 * imu_sample_rate
        data_len = data_df.shape[0]
        strike_array, off_array = np.array(strike_list), np.array(off_list)
        strike_num = len(strike_array)
        steps = []
        stance_phase_flag = np.zeros([data_len], dtype=bool)
        abandoned_step_num = 0
        last_off = 0
        for i_strike in range(strike_num):
            strike = strike_array[i_strike]
            offs_near_strike = off_array[max(0, i_strike - 70): i_strike + 70]
            off = offs_near_strike[offs_near_strike > strike + stance_phase_sample_thd_lower]
            off = off[off < strike + stance_phase_sample_thd_higher]
            if len(off) == 1:  # stance phase detected
                if strike < last_off:
                    continue
                off = off[0]
                steps.append([int(strike), int(off)])
                flag_start = strike + 20
                flag_end = int(round((strike + off) / 2))
                for i_sample in range(strike, off):
                    if all(gyr_magnitude[i_sample:i_sample + 5] < 1.7):
                        flag_start = i_sample + sample_after_thd
                        break
                stance_phase_flag[flag_start:flag_end] = True
                last_off = off
            else:
                abandoned_step_num += 1
        return steps, stance_phase_flag

    def get_walking_strike_off(self, acc_data, gyr_data, cut_off_fre_strike_off=None, verbose=False):
        """ Reliable algorithm used in TNSRE first submission"""
        gyr_thd = 2.6
        acc_thd = 1.2 / GRAVITY
        max_distance = IMU_CAMARGO_SAMPLE_RATE * 2  # distance from stationary phase should be smaller than 2 seconds
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        gyr_magnitude = np.linalg.norm(gyr_data, axis=1)
        gyr_y = gyr_data[:, 1]
        data_len = gyr_data.shape[0]

        if cut_off_fre_strike_off is not None:
            acc_magnitude = data_filter(acc_magnitude, cut_off_fre_strike_off, IMU_CAMARGO_SAMPLE_RATE, filter_order=2)
            gyr_magnitude = data_filter(gyr_magnitude, cut_off_fre_strike_off, IMU_CAMARGO_SAMPLE_RATE, filter_order=2)
            gyr_y = data_filter(gyr_y, cut_off_fre_strike_off, IMU_CAMARGO_SAMPLE_RATE, filter_order=2)

        acc_magnitude = acc_magnitude - 1       # remove the gravity

        stationary_flag = self.__find_stationary_phase(
            gyr_magnitude, acc_magnitude, acc_thd, gyr_thd)

        strike_list, off_list = [], []
        i_sample = 0

        while i_sample < data_len:
            # step 0, go to the next stationary phase
            if not stationary_flag[i_sample]:
                i_sample += 1
            else:
                front_crossing, back_crossing = self.__find_zero_crossing(gyr_y, gyr_thd, i_sample)

                if not back_crossing:  # if back zero crossing not found
                    break
                if not front_crossing:  # if front zero crossing not found
                    i_sample = back_crossing
                    continue

                the_strike, _ = find_peak_max(gyr_y[front_crossing:i_sample], height=0)
                the_off, _ = find_peak_max(gyr_y[i_sample:back_crossing], height=0)

                if the_strike is not None and i_sample - (the_strike + front_crossing) < max_distance:
                    strike_list.append(the_strike + front_crossing)
                if the_off is not None and the_off < max_distance:
                    off_list.append(the_off + i_sample)
                i_sample = back_crossing
        if verbose:
            plt.figure()
            plt.plot(stationary_flag)
            plt.plot(gyr_y)
            plt.plot(strike_list, gyr_y[strike_list], 'g*')
            plt.plot(off_list, gyr_y[off_list], 'r*')

        return strike_list, off_list, gyr_y

    @staticmethod
    def __find_stationary_phase(gyr_magnitude, acc_magnitude, foot_stationary_acc_thd, foot_stationary_gyr_thd):
        data_len = gyr_magnitude.shape[0]
        stationary_flag, stationary_flag_temp = np.zeros(gyr_magnitude.shape), np.zeros(gyr_magnitude.shape)
        stationary_flag_temp[
            (acc_magnitude < foot_stationary_acc_thd) & (abs(gyr_magnitude) < foot_stationary_gyr_thd)] = 1
        for i_sample in range(data_len):
            if stationary_flag_temp[i_sample - 12:i_sample + 12].all():
                stationary_flag[i_sample] = 1
        return stationary_flag

    @staticmethod
    def __find_zero_crossing(gyr_x, foot_stationary_gyr_thd, i_sample):
        """
        Detected as a zero crossing if the value is lower than negative threshold.
        :return:
        """
        max_search_range = IMU_CAMARGO_SAMPLE_RATE * 3  # search 3 second front data at most
        front_crossing, back_crossing = False, False
        for j_sample in range(i_sample, max(0, i_sample - max_search_range), -1):
            if gyr_x[j_sample] < - foot_stationary_gyr_thd:
                front_crossing = j_sample
                break
        for j_sample in range(i_sample+1, gyr_x.shape[0]):
            if gyr_x[j_sample] < - foot_stationary_gyr_thd:
                back_crossing = j_sample
                break
        return front_crossing, back_crossing

    def start_segment(self, trial_data, columns):
        acc_loc = [columns.index('foot_Accel_' + axis) for axis in ['X', 'Y', 'Z']]
        gyr_loc = [columns.index('foot_Gyro_' + axis) for axis in ['X', 'Y', 'Z']]
        acc_data, gyr_data = trial_data[:, acc_loc], trial_data[:, gyr_loc]
        strike_list, off_list, gyr_y = self.get_walking_strike_off(acc_data, gyr_data, 10)
        steps_ = self.strike_off_to_step_and_remove_incorrect_step(
            gyr_y, strike_list, off_list, self.step_len_max, self.step_len_min)
        for step_ in steps_:
            data_ = trial_data[step_[0]:step_[1]]
            self.data_struct.add_new_step(data_)


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
        'AB08', 'AB09', 'AB10',
        'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB17',
        'AB18', 'AB19', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB30'
    ]
    data_reader = ContinuousDatasetLoader(sub_list)
    data_reader.add_additional_columns()
    data_reader.loop_all_the_trials([WindowSegmentation()])

