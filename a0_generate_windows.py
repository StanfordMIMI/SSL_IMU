import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import ast

from utils import get_data_by_merging_data_struct, fix_seed, data_filter
from const import SUB_LIST, DATA_PATH, AMBULATIONS, GRAVITY, IMU_SAMPLE_RATE, EMG_SAMPLE_RATE, GRF_SAMPLE_RATE, \
    IMU_SEGMENT_LIST
from const import DICT_SUBJECT_ID, DICT_TRIAL_TYPE_ID
from scipy.signal import find_peaks


class ContinuousDatasetLoader:
    def __init__(self, subject_list):
        self.sample_rate = IMU_SAMPLE_RATE
        self.columns_raw = self.load_columns()
        self.col_200 = KIN_LIST + IMU_LIST
        self.col_1000 = EMG_LIST + FORCE_LIST
        self.subject_list = subject_list
        self.data_contin_merged = {}
        for subject in subject_list:
            with h5py.File(DATA_PATH + subject + '.h5', 'r') as hf:
                data_trials = {}
                [data_trials.update({trial: [trial_data['data_200'][:], trial_data['data_1000'][:]]})
                 for trial, trial_data in hf.items()]

                data_trials = self.add_additional_info(data_trials, subject)
                data_trials = self.process_grf(data_trials)
                data_trials = self.process_emg(data_trials)
                data_trial_merged, columns = self.merge_1000_to_200(data_trials)
                self.data_contin_merged[subject] = data_trial_merged
        self.columns = columns
        self.trials = list(data_trials.keys())

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
            grf_data = trial_data[1][:, grf_col_loc]
            grf_combined = np.zeros([grf_data.shape[0], 3])
            for i in range(0, grf_data.shape[1], 3):
                grf_combined += grf_data[:, i:i+3]

            # # FOR DEBUG
            # if condition_force == 'Ramp_R':
            #     plt.figure()
            #     imu_index = [self.columns[condition]['200'].index('foot_Gyro_Y')]
            #     plt.plot(trial_data[0][:, imu_index]*100)
            #     plt.plot(grf_combined[::5])
            #     plt.show()

            grf_combined = data_filter(grf_combined, 15, GRF_SAMPLE_RATE)
            trial_data[1] = np.column_stack([trial_data[1], grf_combined])
            data_trials[trial] = trial_data
            if len(self.columns_raw[condition]['1000']) != data_trials[trial][1].shape[1]:
                self.columns_raw[condition]['1000'].extend(['fx', 'fy', 'fz'])
        return data_trials

    def process_emg(self, data_trials):
        for trial, trial_data in data_trials.items():
            condition = trial.split('_')[0]
            emg_col_loc = [self.columns_raw[condition]['1000'].index(x) for x in EMG_LIST]
            emg_data = trial_data[1][:, emg_col_loc]
            emg_data = data_filter(np.abs(emg_data), 20, EMG_SAMPLE_RATE)
            trial_data[1][:, emg_col_loc] = emg_data
        return data_trials

    def generate_additional_col(self, win_len, win_step):
        pass

    @staticmethod
    def interpo_extreme_large_data(data, columns, thd):
        # col_and_thd = {'shank_Accel_': 8}
        # for col, thd in col_and_thd.items():
        data_loc = [columns.index(col) for col in IMU_LIST]
        for axis in data_loc:
            data_axis = data[:, axis]
            ok = np.abs(data_axis) < thd
            if (~ok).any():
                xp = ok.ravel().nonzero()[0]
                fp = data_axis[ok]
                x = (~ok).ravel().nonzero()[0]
                data_axis[~ok] = np.interp(x, xp, fp)
                data[:, axis] = data_axis

    @staticmethod
    def are_kinematics_correct(data_200, columns, kinematic_range):
        kinematic_col_loc = [columns.index(col) for col in
                             ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r']]
        if np.min(data_200[:, kinematic_col_loc]) < kinematic_range[0] or np.max(data_200[:, kinematic_col_loc]) > kinematic_range[1]:
            return False
        return True

    @staticmethod
    def load_columns():
        columns = {ambulation: {} for ambulation in AMBULATIONS}
        for ambulation in AMBULATIONS:
            for frequency in ['200', '1000']:
                columns[ambulation][frequency] = list(np.array(ast.literal_eval(open(
                    DATA_PATH + ambulation + '_' + frequency + '_columns.txt').read()), dtype=object))
        return columns

    def loop_all_the_trials(self, segment_methods):
        # data_structs = {method.name: DataStruct(len(self.columns), method.step_len_max)
        #                 for method in segment_methods}
        [method.set_data_struct(DataStruct(len(self.columns), method.step_len_max)) for method in segment_methods]
        for subject in self.subject_list:
            for trial in self.trials:
                if trial not in self.data_contin_merged[subject].keys():
                    continue
                # trial_type = trial.split('_')[0]
                trial_data = self.data_contin_merged[subject][trial]
                if not self.are_kinematics_correct(trial_data, self.columns, (-180, 180)):
                    continue
                self.interpo_extreme_large_data(trial_data, self.columns, 15)
                for method in segment_methods:
                    method.start_segment(trial_data, self.columns)

        [method.export(self.columns) for method in segment_methods]


class DataStruct:
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


class BaseSegment:
    def set_data_struct(self, data_struct):
        self.data_struct = data_struct

    def export(self, columns):
        all_data = self.data_struct.get_all_data()
        with h5py.File(DATA_PATH + self.name + '.h5', 'w') as hf:
            dset = hf.create_dataset('data', all_data.shape, data=all_data)
            hf.attrs['columns'] = json.dumps(columns)


class WindowSegment(BaseSegment):
    def __init__(self):
        self.data_len = 256
        self.name = 'UnivariantWin'
        self.win_len, self.win_step = self.data_len, int(self.data_len/2)

    def start_segment(self):

        acc_data, gyr_data = trial_data_200[:, acc_loc], trial_data_200[:, gyr_loc]
        strike_list, off_list, gyr_y = self.get_walking_strike_off(acc_data, gyr_data, 10)
        steps_200, steps_1000 = self.strike_off_to_step_and_remove_incorrect_step(
            gyr_y, strike_list, off_list, self.step_len_max, self.step_len_min)
        for step_200, step_1000 in zip(steps_200, steps_1000):
            data_200 = trial_data_200[step_200[0]:step_200[1], input_col_loc_imu]       # 遇到 out of xxx，写exception 跳过该步
            data_1000 = trial_data_1000[step_1000[0]:step_1000[1], input_col_loc_emg]
            data_struct.add_new_step(data_x_200, data_x_1000, data_y_200)


class StepSegment(BaseSegment):
    def __init__(self):
        self.data_len = 256
        self.name = 'StepWin'
        self.win_len, self.win_step = self.data_len, int(self.data_len/2)
        self.step_len_max, self.step_len_min = int(256*IMU_SAMPLE_RATE/200), int(40*IMU_SAMPLE_RATE/200)

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

        imu_sample_rate = IMU_SAMPLE_RATE
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
        max_distance = IMU_SAMPLE_RATE * 2  # distance from stationary phase should be smaller than 2 seconds
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        gyr_magnitude = np.linalg.norm(gyr_data, axis=1)
        gyr_y = gyr_data[:, 1]
        data_len = gyr_data.shape[0]

        if cut_off_fre_strike_off is not None:
            acc_magnitude = data_filter(acc_magnitude, cut_off_fre_strike_off, IMU_SAMPLE_RATE, filter_order=2)
            gyr_magnitude = data_filter(gyr_magnitude, cut_off_fre_strike_off, IMU_SAMPLE_RATE, filter_order=2)
            gyr_y = data_filter(gyr_y, cut_off_fre_strike_off, IMU_SAMPLE_RATE, filter_order=2)

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

                the_strike = self.find_peak_max(gyr_y[front_crossing:i_sample], height=0)
                the_off = self.find_peak_max(gyr_y[i_sample:back_crossing], height=0)

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
        max_search_range = IMU_SAMPLE_RATE * 3  # search 3 second front data at most
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

    @staticmethod
    def find_peak_max(data_clip, height, width=None, prominence=None):
        """
        find the maximum peak
        :return:
        """
        peaks, properties = find_peaks(data_clip, width=width, height=height, prominence=prominence)
        if len(peaks) == 0:
            return None
        peak_heights = properties['peak_heights']
        max_index = np.argmax(peak_heights)
        return peaks[max_index]

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


IMU_LIST = [segment + sensor + axis for sensor in ['_Accel_', '_Gyro_'] for segment in IMU_SEGMENT_LIST for axis in ['X', 'Y', 'Z']]
KIN_LIST = ['sub_id', 'trial_type_id', 'label', 'speed', 'ramp',
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
            'hip_flexion_r_moment', 'hip_adduction_r_moment', 'hip_rotation_r_moment',
            'knee_angle_r_moment', 'ankle_angle_r_moment']
EMG_LIST = ['gastrocmed', 'tibialisanterior', 'soleus', 'vastusmedialis', 'vastuslateralis', 'rectusfemoris',
            'bicepsfemoris', 'semitendinosus', 'gracilis', 'gluteusmedius']
FORCE_LIST = ['fx', 'fy', 'fz']
INFO = ['subject_id', 'trial_type', 'trial_id', 'kinetics_completeness', 'plate_id']
RIGHT_FOOT_FORCE_COL = {
    'Treadmill': ['Treadmill_R_vx', 'Treadmill_R_vy', 'Treadmill_R_vz'],

    'LevelGround_cw': ['Combined_vx', 'Combined_vy', 'Combined_vz',
                       'FP2_vx', 'FP2_vy', 'FP2_vz',
                       'FP5_vx', 'FP5_vy', 'FP5_vz'],
    'LevelGround_ccw': ['FP1_vx', 'FP1_vy', 'FP1_vz'],

    'Stair_R': ['FP2_vx', 'FP2_vy', 'FP2_vz',
                'FP3_vx', 'FP3_vy', 'FP3_vz',
                'FP4_vx', 'FP4_vy', 'FP4_vz'],
    'Stair_L': ['FP1_vx', 'FP1_vy', 'FP1_vz',
                'FP5_vx', 'FP5_vy', 'FP5_vz'],

    'Ramp_R': ['FP2_vx', 'FP2_vy', 'FP2_vz',
               'FP4_vx', 'FP4_vy', 'FP4_vz'],
    'Ramp_L': ['FP1_vx', 'FP1_vy', 'FP1_vz',
               'FP3_vx', 'FP3_vy', 'FP3_vz',
               'FP5_vx', 'FP5_vy', 'FP5_vz']}


if __name__ == '__main__':
    sub_list = [
        'AB06', 'AB07', 'AB08', 'AB09',
        'AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB17',
        'AB18', 'AB19', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28',
        'AB30'
    ]
    data_reader = ContinuousDatasetLoader(sub_list)
    data_reader.loop_all_the_trials([StepSegment()])

