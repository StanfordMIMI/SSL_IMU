import copy
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import scipy.interpolate as interpo
import matplotlib.pyplot as plt
from utils import data_filter
from scipy import linalg
import h5py, json


def rigid_transform_3d(a, b):
    """
    Get the Rotation Matrix and Translation array between A and B.
    return:
        R: Rotation Matrix, 3*3
        T: Translation Array, 1*3
    """
    assert len(a) == len(b)

    N = a.shape[0]  # total points
    centroid_A = np.mean(a, axis=0)
    centroid_B = np.mean(b, axis=0)
    # centre the points
    AA = a - np.tile(centroid_A, (N, 1))
    BB = b - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.dot(AA.T, BB)
    U, _, V_t = linalg.svd(np.nan_to_num(H))
    R = np.dot(V_t.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        # print
        # "Reflection detected"
        V_t[2, :] *= -1
        R = np.dot(V_t.T, U.T)
    T = -np.dot(R, centroid_A.T) + centroid_B.T
    return R, T


def sync_via_correlation(data1, data2, verbose=False):
    data1_processed = data_filter(copy.deepcopy(data1), 5, 100)
    data2_processed = data_filter(copy.deepcopy(data2), 5, 100)
    data1_processed = data1_processed - np.mean(data1_processed)
    data2_processed = data2_processed - np.mean(data2_processed)

    # correlations = []
    # data2_padded = np.concatenate([np.zeros([len(data1_processed)]), data2_processed, np.zeros([len(data1_processed)])])
    # for i_ in range(5, len(data1_processed) + len(data2_processed) - 5):
    #     correlations.append(np.corrcoef(data1_processed, data2_padded[i_:i_+len(data1_processed)])[0, 1])
    # delay = np.argmax(correlations) - len(data1_processed) + 5
    correlations = np.correlate(data1_processed, data2_processed, 'full')
    delay = len(data2) - np.argmax(correlations) - 1
    if verbose:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # fig, axes = plt.subplots(2, 2)
        plt.subplot(2, 1, 1)
        plt.plot(data1)
        plt.plot(data2)
        plt.subplot(2, 2, 3)
        plt.plot(correlations)
        plt.title(np.max(correlations))
        plt.subplot(2, 2, 4)
        if delay > 0:
            plt.plot(data1)
            plt.plot(data2[delay:])
        else:
            plt.plot(data1[-delay:])
            plt.plot(data2)
    return delay


def get_angular_velocity_theta(marker_data, sampling_rate, check_len):
    check_len = min(marker_data.shape[0], check_len)
    marker_number = int(marker_data.shape[1] / 3)
    angular_velocity_theta = np.zeros([check_len])

    next_marker_matrix = marker_data[0, :].reshape([marker_number, 3])
    for i_frame in range(check_len):
        if i_frame == 0:
            continue
        current_marker_matrix = next_marker_matrix
        next_marker_matrix = marker_data[i_frame, :].reshape([marker_number, 3])
        R_one_sample, _ = rigid_transform_3d(current_marker_matrix, next_marker_matrix)
        theta = np.math.acos((np.matrix.trace(R_one_sample) - 1) / 2)

        angular_velocity_theta[i_frame] = theta * sampling_rate
    return angular_velocity_theta


def linear_resample_data(trial_data, original_fre, target_fre):
    x, step = np.linspace(0., 1., trial_data.shape[0], retstep=True)
    new_x = np.arange(0., 1., step*original_fre/target_fre)
    f = interp1d(x, trial_data, axis=0)
    trial_data_resampled = f(new_x)
    return trial_data_resampled


def resample_via_spline_fitting(data_, step_to_resample):
    tck, step = interpo.splprep(data_[:, :].T, u=data_[:, 0], s=0)
    data_resampled = interpo.splev(step_to_resample, tck, der=0)
    data_resampled = np.column_stack(data_resampled)
    return data_resampled


def load_opencap_imu(sub_folder, trial_name):
    real_imu = {}
    acc_thd, gyr_thd = 40, 8.73        # Range of measurements of IMUs
    common_start, common_end = 0, np.inf
    for opencap_name, opensim_name in opencap_to_opensim_body_name_map.items():
        file = f'{sub_folder}/{trial_name}-000_{imu_file_map[sub_][opencap_name]}.txt'
        data_ = pd.read_csv(file, skiprows=12, sep='\t', usecols=[
            'PacketCounter', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']).values

        if data_[0, 0] > data_[0, -1]:
            data_[:, 0] = np.arange(data_[0, 0], data_[0, 0] + len(data_))

        # Clip extreme values
        data_[:, 1:4][data_[:, 1:4] > acc_thd] = acc_thd
        data_[:, 1:4][data_[:, 1:4] < -acc_thd] = -acc_thd
        data_[:, 4:7][data_[:, 4:7] > gyr_thd] = gyr_thd
        data_[:, 4:7][data_[:, 4:7] < -gyr_thd] = -gyr_thd

        # Lowpass filtering
        data_[:, 1:7] = data_filter(data_[:, 1:7], 10, 40)

        # Sync start end
        if data_[0, 0] > common_start:
            common_start = data_[0, 0]
        if data_[-1, 0] < common_end:
            common_end = data_[-1, 0]
        real_imu[opensim_name] = data_

    step_to_resample = np.arange(common_start, common_end, 0.4)
    for _, opensim_name in opencap_to_opensim_body_name_map.items():
        real_imu[opensim_name] = resample_via_spline_fitting(real_imu[opensim_name], step_to_resample)

    # for _, opensim_name in opencap_to_opensim_body_name_map.items():
    #     plt.figure()
    #     plt.plot(real_imu[opensim_name][:, 1:4])
    #     plt.title(opensim_name)
    # plt.show()

    return real_imu


def load_body_marker_data(file, body_name):
    with open(file, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 3:
                marker_name_all = line
                break
    marker_name_all = [marker for marker in marker_name_all.split('\t') if marker not in ['Frame#', 'Time', '', '\n']]
    columns = ['Frame', 'Time'] + [marker+axis for marker in marker_name_all for axis in ['_x', '_y', '_z']]
    df = pd.read_csv(file, skiprows=5, header=None, sep='\t').iloc[:, :155]
    df.columns = columns
    marker_list = body_marker_map[body_name]
    target_columns = ['Time'] + [marker + axis for marker in marker_list for axis in ['_x', '_y', '_z']]
    return df[target_columns]


def combine_osim_id_with_imu(id_file, shank_marker, real_imu, delay):
    for body_, value_ in real_imu.items():
        for i_sample in range(value_.shape[0]):
            value_[i_sample, 1:4] = np.matmul(xsens_orientation_transform_mat[body_], value_[i_sample, 1:4])
            value_[i_sample, 4:7] = np.matmul(xsens_orientation_transform_mat[body_], value_[i_sample, 4:7])
        real_imu[body_] = value_
    imu_data_list = [pd.DataFrame(value_[:, 1:], columns=[key_ + sensor + axis for sensor in ['_Accel_', '_Gyro_'] for axis in ['X', 'Y', 'Z']]) for key_, value_ in real_imu.items()]
    real_imu_df = pd.concat(imu_data_list, axis=1)
    id_data = pd.read_csv(id_file, skiprows=6, sep='\t')
    id_marker_diff = np.where(shank_marker['Time'] == id_data['time'][0])[0][0]
    delay -= id_marker_diff
    if delay > 0:
        print('bad trial: IMU data is incomplete')
    else:
        data_len = np.min([real_imu_df.shape[0]+delay, id_data.shape[0]])
        real_imu_df = real_imu_df.iloc[-delay:data_len-delay, :]
        id_data = id_data.iloc[:data_len, :]
    id_data.reset_index(drop=True, inplace=True)
    real_imu_df.reset_index(drop=True, inplace=True)
    data_df = pd.concat([id_data, real_imu_df], axis=1)
    return data_df


def segment_df_into_windows_and_store_as_h5(df_list, file_name, win_len=128, step_len=64):
    with h5py.File(export_dir + file_name, 'w') as hf:
        hf.attrs['columns'] = json.dumps(df_list[subject_list[0]][0].columns.tolist())
        for sub_ in subject_list:
            win_list = []
            for df_ in df_list[sub_]:
                values = df_.values
                if values.shape[0] < win_len:
                    win_ = np.zeros((win_len, values.shape[1]))
                    win_[:values.shape[0], :] = values
                    win_list.append(win_)     # If the data is too short, just pad zeros
                else:
                    for i_win in range(0, values.shape[0]-win_len, step_len):
                        win_list.append(values[i_win:i_win+win_len, :])
            if len(win_list) == 0:
                print(sub_ + file_name + ' has no data')
                continue
            data_ = np.stack(win_list, axis=0).transpose((0, 2, 1))
            hf.create_dataset(sub_, data_.shape, data=data_)


incorrect_trials = {
    'subject2': [],
    'subject4': ['STS1', 'STSweakLegs1'],
    'subject5': [],
    'subject6': ['DJAsym3'],
    'subject7': ['DJ4', 'STS1'],
    'subject8': ['DJ3'],
    'subject9': ['squats1'],
    'subject10': ['STS1'],
    'subject11': [],
}
subject_list = list(incorrect_trials.keys())
opencap_imu_map_df = pd.read_excel('D:/Local/Data/DataOpenCap/XSENS Title Numbers.xlsx')
imu_file_map = {row['Location']: row['XSENS Number'] for index, row in opencap_imu_map_df.iterrows()}
imu_file_map = {sub_: copy.deepcopy(imu_file_map) for sub_ in subject_list}
# File correction
# imu_file_map['subject3']['LLEG R'] = '00B4226D'

opencap_data_path = 'D:/Local/Data/DataOpenCap/LabValidation_withoutVideos/'
export_dir = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/'

opencap_to_opensim_body_name_map = {
    'STERN': 'CHEST', 'PELV': 'WAIST', 'ULEG R': 'R_THIGH', 'ULEG L': 'L_THIGH',
    'LLEG R': 'R_SHANK', 'LLEG L': 'L_SHANK', 'FOOT R': 'R_FOOT', 'FOOT L': 'L_FOOT'}
body_marker_map = {'R_SHANK': ['r_knee', 'r_shank_antsup', 'r_sh2', 'r_sh3', 'r_sh4'],
                   'R_THIGH': ['r_thigh1', 'r_thigh2', 'r_thigh3', 'r_thigh4', 'r_thigh5']}

xsens_orientation_transform_mat = {
    'CHEST': np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]), 'WAIST': np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
    'R_THIGH': np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), 'L_THIGH': np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    'R_SHANK': np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]), 'L_SHANK': np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
    'R_FOOT': np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]), 'L_FOOT': np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])}

dj_trials, squat_trials, sts_trials = [{sub_: [] for sub_ in subject_list} for _ in range(3)]
for sub_ in subject_list[:]:
    print(sub_)
    marker_folder = opencap_data_path + sub_ + '/MarkerData/Mocap/'
    trial_list = [file[:-4] for file in os.listdir(marker_folder) if file[-4:] == '.trc']

    for trial_name in trial_list[:]:
        if 'static' in trial_name or 'walk' in trial_name or trial_name in incorrect_trials[sub_]:
            continue
        print(trial_name)
        dir_imu = opencap_data_path + sub_ + '/IMUData/' + trial_name + '/'
        shank_marker = load_body_marker_data(marker_folder + trial_name + '.trc', 'R_THIGH')
        shank_angular_vel = get_angular_velocity_theta(shank_marker.iloc[:, 1:].values, 100, 1000)
        real_imu = load_opencap_imu(dir_imu, trial_name)
        delay = sync_via_correlation(np.linalg.norm(real_imu['R_THIGH'][:, 4:], axis=1), shank_angular_vel)
        id_file = opencap_data_path + sub_ + '/OpenSimData/Mocap/ID/' + trial_name + '.sto'
        combined_df = combine_osim_id_with_imu(id_file, shank_marker, real_imu, delay)
        combined_df['sub_id'] = np.full(combined_df.shape[0], int(sub_.split('subject')[-1]))

        if 'DJ' in trial_name:
            dj_trials[sub_].append(combined_df)
        elif 'squat' in trial_name:
            squat_trials[sub_].append(combined_df)
        elif 'STS' in trial_name:
            sts_trials[sub_].append(combined_df)

        plt.show()

segment_df_into_windows_and_store_as_h5(dj_trials, 'opencap_dj.h5')
segment_df_into_windows_and_store_as_h5(squat_trials, 'opencap_squat.h5')
segment_df_into_windows_and_store_as_h5(sts_trials, 'opencap_sts.h5')









