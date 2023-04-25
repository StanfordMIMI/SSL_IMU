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
from a0_process_MoVi import ContinuousDatasetLoader, WindowSegmentation
import h5py, json
from config import AMASS_PATH, DATA_PATH
from multiprocessing import Pool, cpu_count
import time


""" 
Notes: 
[step, feature, time]
['CHEST', 'WAIST', 'R_THIGH', 'L_THIGH', 'R_SHANK', 'L_SHANK', 'R_FOOT', 'L_FOOT'] 
"""
IMU_CONFIGS = {
    'CHEST': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 1329, 'ji_sel': 9},
    'WAIST': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 3147, 'ji_sel': 0},
    'L_THIGH': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 4434, 'ji_sel': 2},
    'R_THIGH': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 945, 'ji_sel': 1},
    'L_SHANK': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 4563, 'ji_sel': 5},
    'R_SHANK': {'R_sw': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'vi_sel': 1075, 'ji_sel': 4},
    'L_FOOT': {'R_sw': np.array([[1, 0, 0], [0, 0.5, - np.sqrt(3)/2], [0, np.sqrt(3)/2, 0.5]]), 'vi_sel': 6740, 'ji_sel': 8},
    'R_FOOT': {'R_sw': np.array([[1, 0, 0], [0, 0.5, - np.sqrt(3)/2], [0, np.sqrt(3)/2, 0.5]]), 'vi_sel': 3343, 'ji_sel': 7}
}


class SyntheticAccGyr:
    def __init__(self, imu_name, R_wb, traj, R_sb):
        traj = data_filter(traj, 10, 100)
        self.R_wb = R_wb
        self.traj = traj
        self.R_sb = R_sb
        self.imu_name = imu_name
        self.acc = self._compute_acc()
        self.gyr = self._compute_gyr()
        self.acc = data_filter(self.acc, 15, 100)
        self.gyr = data_filter(self.gyr, 15, 100)

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

        # plt.figure()
        # plt.plot(gyr_s)
        # plt.title(self.imu_name + ' gyr')
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

        # plt.figure()
        # plt.plot(acc_s)
        # plt.title(self.imu_name + ' acc')
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
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    data_ = vert[0].numpy()
    # for i_, data__ in enumerate(data_[::10]):
    #     ax.text(data__[0], data__[1], data__[2], str(i_*10), 'x', fontsize=8)
    ax.scatter(data_[:, 0], data_[:, 1], data_[:, 2])
    # ax.scatter(data_[vi_sel, 0], data_[vi_sel, 1], data_[vi_sel, 2], c='r', s=50)
    ax.scatter([0, 2], [0, 2], [0, 2], s=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=0., azim=90)
    plt.show()


def process_one_trial(npz_fname, body_model_path, imu_names, columns, progress_percent):
    # sys.stdout.write(f'{round(progress_percent*100, 2)}%')
    # sys.stdout.flush()
    print(f'{round(progress_percent*100, 2)}%', end='\t')
    body_model = art.ParametricModel(body_model_path)
    trial_wins = []
    try: cdata = np.load(npz_fname)
    except: return []
    ori_frame_rate = int(cdata['mocap_framerate'])
    if ori_frame_rate < 50: print('abandoned low sampling rate trial', npz_fname); return []
    data_pose_current = resample_to_target_fre(cdata['poses'].astype(np.float32), target_frame_rate, ori_frame_rate)
    data_trans_current = resample_to_target_fre(cdata['trans'].astype(np.float32), target_frame_rate, ori_frame_rate)

    data_pose = torch.tensor(np.asarray(data_pose_current, np.float32)).view(-1, 52, 3)
    data_trans = torch.tensor(np.asarray(data_trans_current, np.float32))
    data_shape = torch.tensor(np.asarray(cdata['betas'][:10], np.float32))       # TODO: Augment the data by changing betas
    length = torch.tensor(data_pose_current.shape[0])
    data_pose[:, 23] = data_pose[:, 37]     # right hand, [UNCLEAR]
    data_pose = data_pose[:, :24].clone()   # only use body, [UNCLEAR]

    if length < win_len: return []
    p = art.math.axis_angle_to_rotation_matrix(data_pose).view(-1, 24, 3, 3)
    grot, joint, traj = body_model.forward_kinematics(p, data_shape, data_trans, calc_mesh=True)

    # plot_3d_scatter(traj)

    trial_data = []
    for imu_name in imu_names:
        imu_config = IMU_CONFIGS[imu_name]
        synthetic = SyntheticAccGyr(imu_name, R_wb=grot[:, imu_config['ji_sel']], traj=traj[:, imu_config['vi_sel']], R_sb=imu_config['R_sw'])
        acc_, gyr_ = synthetic.get_acc_gyr()
        trial_data.append(acc_)
        trial_data.append(gyr_)
    trial_data_continuous = np.concatenate(trial_data, axis=1).T.astype(np.float32)

    # segment trial data from [feature, time] into windows [step, feature, time]
    gyr_cols = [i for i, col in enumerate(columns) if 'Accel' in col]
    trial_len = trial_data_continuous.shape[1]
    for i_ in range(0, trial_len - win_len, win_stride):
        win_data = trial_data_continuous[:, i_:i_+win_len]
        acc_std = np.std(win_data[gyr_cols], axis=1)
        # plt.figure()
        # plt.plot(win_data[gyr_cols].T)
        # plt.title(npz_fname.split('\\')[-1] + '\t' + str(acc_std.mean()))
        # plt.show()
        if acc_std.mean() >= 2:
            trial_wins.append(win_data)
    return trial_wins


def process_amass_multi_thread():
    def record_wins(trial_wins):
        dset_wins.extend(trial_wins)

    imu_names = tuple([imu_name for imu_name, imu_config in IMU_CONFIGS.items()])
    body_model_path = AMASS_PATH + 'ModelFiles/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    columns = tuple([segment + sensor + axis for segment in imu_names for sensor in ['_Accel_', '_Gyro_'] for axis in ['X', 'Y', 'Z']])
    for dset_name in amass_data:
        print(dset_name)
        dset_wins = []
        t0 = time.time()
        pool = Pool(processes=5)
        trials = glob.glob(os.path.join(AMASS_PATH, dset_name, '*/*_poses.npz'))[:10]
        trial_num = len(trials)
        for i_trial, npz_fname in enumerate(trials):
            pool.apply_async(process_one_trial, args=(npz_fname, body_model_path, imu_names, columns, (i_trial+1)/trial_num), callback=record_wins)
        pool.close()
        pool.join()
        print('Time cost:', time.time() - t0)
        if len(dset_wins) == 0: continue
        dset_data = np.stack(dset_wins, axis=0)
        with h5py.File(DATA_PATH + 'amass.h5', 'w') as hf:
            try:
                del hf[dset_name]
            except KeyError:
                pass
            dset = hf.create_dataset(dset_name, dset_data.shape, data=dset_data)        # , compression='gzip', compression_opts=9
            hf.attrs['columns'] = json.dumps(columns)


amass_data = ['ACCAD']      # , 'BioMotionLab_NTroje', 'BMLmovi', 'BMLhandball', 'CMU', 'CNRS', 'DFaust_67']      #      Transitions_mocap
target_frame_rate = 100
win_len, win_stride = 128, 64
if __name__ == '__main__':
    process_amass_multi_thread()
