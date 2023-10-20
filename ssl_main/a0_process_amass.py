import copy
import sys
sys.path.insert(0, '..')
import articulate as art
from utils import data_filter
import numpy as np
from tqdm import tqdm
import glob
from utils import resample_to_target_fre
import matplotlib.pyplot as plt
import os
import torch
import scipy.interpolate as interpo
import h5py, json
from config import AMASS_PATH, DATA_PATH
import time
import random


""" 
Notes: 
AMASS body frames: x points left; y points up, z points forward
[step, feature, time]
['TRUNK', 'PELVIS', 'R_THIGH', 'L_THIGH', 'R_SHANK', 'L_SHANK', 'R_FOOT', 'L_FOOT'] 
"""
IMU_CONFIGS = {
    'TRUNK': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 1329, 'ji_sel': 9},
    'PELVIS': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 3147, 'ji_sel': 0},
    'L_THIGH': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 4434, 'ji_sel': 2},
    'R_THIGH': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 945, 'ji_sel': 1},
    'L_SHANK': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 4563, 'ji_sel': 5},
    'R_SHANK': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 1075, 'ji_sel': 4},
    'L_FOOT': {'R_sw': np.array([[1, 0, 0], [0, 0.5, - np.sqrt(3)/2], [0, np.sqrt(3)/2, 0.5]]), 'vi_sel': 6740, 'ji_sel': 8},
    'R_FOOT': {'R_sw': np.array([[1, 0, 0], [0, 0.5, - np.sqrt(3)/2], [0, np.sqrt(3)/2, 0.5]]), 'vi_sel': 3343, 'ji_sel': 7}
}


class SyntheticAccGyr:
    def __init__(self, imu_name, R_wb, traj, R_sb):
        self.R_wb = R_wb
        self.traj = traj
        self.R_sb = R_sb
        self.imu_name = imu_name
        self.acc = self._compute_acc()
        self.gyr = self._compute_gyr()

    def get_acc_gyr(self):
        return self.acc, self.gyr

    def _compute_gyr(self):
        data_len = self.traj.shape[0]
        gyr_middle = np.zeros([data_len, 3])
        for i_frame in range(data_len - 1):
            R_one_sample = np.matmul(self.R_wb[i_frame+1].T, self.R_wb[i_frame])
            val = (torch.trace(R_one_sample) - 1) / 2
            val = np.max(np.min((val, 1)), -1)
            theta = np.math.acos(val)
            a, b = np.linalg.eig(R_one_sample)
            for i_eig in range(a.__len__()):
                if abs(a[i_eig].imag) < 1e-12:
                    vector = b[:, i_eig].real
                    break
                if i_eig == a.__len__():
                    raise RuntimeError('no eig')

            if (R_one_sample[2, 1] - R_one_sample[1, 2]) * vector[0] > 0:  # check the direction of the rotation axis
                vector = -vector
            gyr_middle[i_frame, :] = theta * vector * target_frame_rate

        step_middle = np.arange(0.5 / target_frame_rate, data_len / target_frame_rate + 0.5 / target_frame_rate - 1e-10,
                                1 / target_frame_rate)
        step_gyr = np.arange(0, data_len / target_frame_rate - 1e-10, 1 / target_frame_rate)
        # in splprep, s the amount of smoothness.
        tck, step = interpo.splprep(gyr_middle.T, u=step_middle, s=0)
        gyr_b = interpo.splev(step_gyr, tck, der=0)
        gyr_b = np.column_stack([gyr_b[0], gyr_b[1], gyr_b[2]])

        gyr_s = np.zeros([data_len, 3])
        for i_sample in range(data_len):
            gyr_s[i_sample, :] = np.matmul(self.R_sb, gyr_b[i_sample, :])

        return gyr_s

    def _compute_acc(self):
        data_len = self.traj.shape[0]
        step_marker = np.arange(0, data_len / target_frame_rate - 1e-12, 1 / target_frame_rate)
        tck, step_marker = interpo.splprep(self.traj.T, u=step_marker, s=0)
        acc_w = np.column_stack(interpo.splev(step_marker, tck, der=2))  # der=2 means take the second derivation

        acc_w += np.array([0, 0, 9.81])

        acc_s = np.zeros([data_len, 3])
        for i_sample in range(data_len):
            R_bw = self.R_wb[i_sample].T
            acc_b = np.matmul(R_bw, acc_w[i_sample, :])
            acc_s[i_sample, :] = np.matmul(self.R_sb, acc_b)

        return acc_s

    @staticmethod
    def _rigid_transform_3D(A, B):
        assert len(A) == len(B)
        N = A.shape[0]  # total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))
        # dot is matrix multiplication for array
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = - np.dot(R, centroid_A.T) + centroid_B.T
        return R, t


def plot_3d_scatter(vert):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    data_ = vert[0].numpy()
    data_ = data_ - np.mean(data_, axis=0)
    # for i_, data__ in enumerate(data_[::10]):
    #     ax.text(data__[0], data__[1], data__[2], str(i_*10), 'x', fontsize=8)
    ax.scatter(data_[:, 0], data_[:, 1], data_[:, 2], edgecolors='none', c='gray', s=5, zorder=10)
    for segment, config in IMU_CONFIGS.items():
        ax.scatter(data_[config['vi_sel'], 0], data_[config['vi_sel'], 1], data_[config['vi_sel'], 2], c='r', s=25, zorder=20)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.view_init(elev=0., azim=90)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.set_axis_off()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)

    plt.show()


def process_one_trial(npz_fname, body_model_path, imu_names, columns, i_trial, trial_num, dset_name):
    body_model = art.ParametricModel(body_model_path)
    if i_trial % (trial_num // 10) == 0:
        print(f'{round(i_trial/trial_num*100, 2)}%')
    try: cdata = np.load(npz_fname)
    except: return []
    try:
        if 'mocap_framerate' in cdata.keys():
            ori_frame_rate = int(cdata['mocap_framerate'])
        else:
            ori_frame_rate = int(cdata['mocap_frame_rate'])
    except KeyError:
        return
    if ori_frame_rate < 50: print('abandoned low sampling rate trial', npz_fname); return []

    data_pose_current = data_filter(cdata['poses'].astype(np.float32), amass_cut_off_fre, ori_frame_rate)
    data_trans_current = data_filter(cdata['trans'].astype(np.float32), amass_cut_off_fre, ori_frame_rate)

    data_pose_current = resample_to_target_fre(data_pose_current, target_frame_rate, ori_frame_rate)
    data_trans_current = resample_to_target_fre(data_trans_current, target_frame_rate, ori_frame_rate)

    data_pose = torch.tensor(np.asarray(data_pose_current, np.float32)).view(data_pose_current.shape[0], -1, 3)
    data_trans = torch.tensor(np.asarray(data_trans_current, np.float32))
    data_shape = torch.tensor(np.asarray(cdata['betas'][:10], np.float32))       # TODO: Augment the data by changing betas
    length = torch.tensor(data_pose_current.shape[0])
    data_pose[:, 23] = data_pose[:, 37]     # right hand, [UNCLEAR]
    data_pose = data_pose[:, :24].clone()   # only use body, [UNCLEAR]

    if length < win_len: return []
    p = art.math.axis_angle_to_rotation_matrix(data_pose).view(-1, 24, 3, 3)
    grot, joint, traj = body_model.forward_kinematics(p, data_shape, data_trans, calc_mesh=True)

    # plot_3d_scatter(traj)
    # exit()

    trial_data = []
    for imu_name in imu_names:
        imu_config = IMU_CONFIGS[imu_name]
        synthetic = SyntheticAccGyr(imu_name, R_wb=grot[:, imu_config['ji_sel']], traj=traj[:, imu_config['vi_sel']], R_sb=imu_config['R_sw'])
        acc_, gyr_ = synthetic.get_acc_gyr()
        trial_data.append(acc_)
        trial_data.append(gyr_)
    trial_data_continuous = np.concatenate(trial_data, axis=1).T.astype(np.float32)

    temp_save_file = npy_data_path + f'{dset_name}/{i_trial}.npy'
    temp_dir = os.path.dirname(temp_save_file)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with open(temp_save_file, 'wb') as f:
        np.save(f, trial_data_continuous)


def load_npy_and_to_h5():
    with h5py.File(DATA_PATH + 'amass_with_extreme_values.h5', 'w') as hf:
        hf.attrs['columns'] = json.dumps(columns)
        num_of_removed_static, num_of_removed_extreme, num_of_windows_all = 0, 0, 0
        for dset_name in next(os.walk(npy_data_path))[1]:

            dset_wins = []
            temp_files = glob.glob(os.path.join(npy_data_path, dset_name, '*.npy'))
            print(f'Saving dset: {dset_name}, {len(temp_files)} trials')
            for temp_file in temp_files:
                with open(temp_file, 'rb') as f:
                    trial_data_continuous = np.load(f)
                    if imu_cut_off_fre is not None:
                        trial_data_continuous = data_filter(trial_data_continuous.T, imu_cut_off_fre, target_frame_rate).T
                    """ segment trial data from [feature, time] into windows [step, feature, time] """
                    acc_cols = [i for i, col in enumerate(columns) if 'Accel' in col]
                    gyr_cols = [i for i, col in enumerate(columns) if 'Gyro' in col]
                    trial_len = trial_data_continuous.shape[1]
                    for i_ in range(0, trial_len - win_len, win_stride):
                        win_data = trial_data_continuous[:, i_:i_+win_len]
                        acc_std = np.std(win_data[acc_cols], axis=1)

                        num_of_windows_all += 1
                        if acc_std.mean() < 2:
                            num_of_removed_static += 1
                            continue
                        if np.max(np.abs(win_data[acc_cols])) > 160 or np.max(np.abs(win_data[gyr_cols])) > 35:
                            num_of_removed_extreme += 1
                            continue
                        dset_wins.append(win_data)

            print(f'{num_of_removed_static} / {num_of_removed_extreme} / {len(dset_wins)}; static_win/ extreme_win / kept_win')
            if len(dset_wins) > 0:
                dset_data = np.stack(dset_wins)
                dset = hf.create_dataset(dset_name, dset_data.shape, data=dset_data)
        print(f'{num_of_removed_static} / {num_of_removed_extreme} / {num_of_windows_all},'
              f' {num_of_removed_static / num_of_windows_all * 100:.1f}% static, '
              f' {num_of_removed_extreme / num_of_windows_all * 100:.1f}% extreme'
              f' {num_of_windows_all - num_of_removed_extreme - num_of_removed_static}% kept')


def npy_to_h5_small_size():
    with h5py.File(DATA_PATH + 'amass_small.h5', 'w') as hf:
        hf.attrs['columns'] = json.dumps(columns)
        for dset_name in next(os.walk(os.path.join(DATA_PATH, 'npy_temp')))[1]:
            print(f'Saving dset: {dset_name}')
            dset_wins = []
            temp_files = glob.glob(os.path.join(npy_data_path, dset_name, '*.npy'))
            for temp_file in temp_files:
                with open(temp_file, 'rb') as f:
                    dset_wins.append(np.load(f))
            if len(dset_wins) > 10:
                dset_data = np.concatenate(dset_wins, axis=0)
                sampled_rows = np.sort(random.sample(range(dset_data.shape[0]), int(0.1 * dset_data.shape[0])))
                dset_data = dset_data[sampled_rows]
                dset = hf.create_dataset(dset_name, dset_data.shape, data=dset_data)


def check_extreme_values():
    with h5py.File(DATA_PATH + 'amass.h5', 'r') as hf:
        ssl_data = {sub_: sub_data[:int(1 * hf[sub_].shape[0]), :, :] for sub_, sub_data in hf.items()}
        ssl_columns = json.loads(hf.attrs['columns'])
    acc_col_loc = [i for i, col in enumerate(ssl_columns) if 'Accel' in col]
    gyr_col_loc = [i for i, col in enumerate(ssl_columns) if 'Gyro' in col]

    total_num, total_bad_num = 0, 0
    for dset_name, data_ in ssl_data.items():
        total_win = data_.shape[0]
        bad_win_num = 0
        badbad_win_num = 0
        for i_win in range(total_win):
            if np.max(np.abs(data_[i_win, acc_col_loc])) > 320 or np.max(np.abs(data_[i_win, gyr_col_loc])) > 70:
                badbad_win_num += 1
            if np.max(np.abs(data_[i_win, acc_col_loc])) > 160 or np.max(np.abs(data_[i_win, gyr_col_loc])) > 35:
                bad_win_num += 1
            plt.subplot(2, 2, 1)
            plt.plot(data_[i_win, acc_col_loc].T)
            plt.subplot(2, 2, 2)
            plt.plot(data_[i_win, gyr_col_loc].T)
            plt.title(dset_name)
            plt.show()
        print(dset_name, ': {} / {} / {}'.format(badbad_win_num, bad_win_num, total_win))
        total_num += total_win
        total_bad_num += bad_win_num
    print('Total: {} / {}'.format(total_bad_num, total_num))


def check_num_of_npy_vs_num_of_npz():
    print('Checking')
    for dset_name in amass_dset_names:
        trials = glob.glob(os.path.join(AMASS_PATH, dset_name, '*/*.npz'))
        num_of_trials = len(trials)
        num_of_npy = len(glob.glob(os.path.join(npy_data_path, dset_name, '*.npy')))
        print(dset_name, f' {num_of_npy}/{num_of_trials} trials processed.')


def process_amass_single_thread():
    body_model_path = AMASS_PATH + 'ModelFiles/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    for dset_name in amass_dset_names:
        t0 = time.time()
        trials = glob.glob(os.path.join(AMASS_PATH, dset_name, '*/*.npz'))
        print('Processing ', dset_name, '\t Num of trials ', len(trials))
        trial_num = len(trials)
        for i_trial, npz_fname in enumerate(trials):
            try:
                process_one_trial(npz_fname, body_model_path, imu_names, columns, i_trial, trial_num, dset_name)
            except Exception as e:
                print(e)
                print('Failed to process ', npz_fname)
        print('Time cost:', time.time() - t0)


imu_names = tuple([imu_name for imu_name, imu_config in IMU_CONFIGS.items()])
columns = tuple([segment + sensor + axis for segment in imu_names for sensor in ['_Accel_', '_Gyro_'] for axis in ['X', 'Y', 'Z']])
amass_dset_names = [
    'ACCAD', 'BMLhandball',
    'BioMotionLab_NTroje',
    'BMLmovi',
    'CNRS',
    'DFaust_67',
    'EKUT', 'Eyes_Japan_Dataset', 'GRAB', 'HUMAN4D', 'HumanEva',
    'MPI_HDM05',
    'MPI_Limits',
    'MPI_mosh',
    'SFU',
    'SOMA', 'SSM_synced', 'TotalCapture', 'Transitions_mocap',
    'WEIZMANN',
    'CMU',      # 40 GB memory
    'DanceDB',      # 40 GB memory
    'KIT',      # 40 GB memory
]

target_frame_rate = 100
win_len, win_stride = 128, 64
amass_cut_off_fre = 15
imu_cut_off_fre = None
npy_data_path = f'{DATA_PATH}../AMASS_data_cutoff_{amass_cut_off_fre}/'

if __name__ == '__main__':
    # process_amass_single_thread()
    load_npy_and_to_h5()

    # check_num_of_npy_vs_num_of_npz()
    # check_extreme_values()




