import copy

from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt, butter
import random
import torch
from scipy.signal import find_peaks
import datetime
import prettytable as pt
from customized_logger import logger as logging, add_file_handler
from sklearn.metrics import r2_score, mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import kendalltau


def fix_seed():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


def get_step_len(data, feature_col=[0, 1, 2]):
    """
    :param data: Numpy array, 3d (step, sample, feature)
    :param feature_col: int, feature column id for step length detection. Different id would probably return
           the same results
    :return:
    """
    data_the_feature = data[:, feature_col, :]
    zero_loc = data_the_feature == 0.
    data_len = np.sum(~zero_loc, axis=2)
    data_len = np.max(data_len, axis=1)
    return data_len


def preprocess_modality(data_columns, data_scalar, data_, channel_names, norm_method):
    processed_data = {}
    for group_name, cols in channel_names.items():
        col_loc = [data_columns.index(col) for col in cols]
        group_data = data_[:, col_loc, :]
        group_data = normalize_data(data_scalar, group_data, group_name, norm_method, 'by_all_columns')
        processed_data[group_name] = group_data
    return processed_data


def get_profile_scores(y_true, y_pred, field, test_step_lens):
    def get_column_score(arr_true, arr_pred):
        r2, rmse, cor_value = [np.zeros(arr_true.shape[0]) for _ in range(3)]
        for i in range(arr_true.shape[0]):
            arr_true_i = arr_true[i, :test_step_lens[i]]
            arr_pred_i = arr_pred[i, :test_step_lens[i]]
            r2[i] = r2_score(arr_true_i, arr_pred_i)
            rmse[i] = np.sqrt(mse(arr_true_i, arr_pred_i))
            cor_value[i] = kendalltau(arr_true_i, arr_pred_i)[0]
        return {'r2': np.mean(r2), 'rmse': np.mean(rmse), 'cor_value': np.mean(cor_value)}

    scores = []
    score_one_field = {'field': field}
    score_one_field.update(get_column_score(y_true, y_pred))
    scores.append(score_one_field)
    return scores


def get_scores(y_true, y_pred, y_fields, lens):
    scores = []
    for col, field in enumerate(y_fields):
        if len(y_true.shape) == 2:
            r2 = r2_score(y_true[:, col], y_pred[:, col])
            rmse = np.sqrt(mse(y_true[:, col], y_pred[:, col]))
            cor_value = kendalltau(y_true[:, col], y_pred[:, col])[0]
        else:
            r2, rmse, cor_value = [np.zeros(y_true.shape[0]) for _ in range(3)]
            for i_step in range(y_true.shape[0]):
                y_true_one_step = y_true[i_step, col, :lens[i_step]]
                y_pred_one_step = y_pred[i_step, col, :lens[i_step]]
                r2[i_step] = r2_score(y_true_one_step, y_pred_one_step)
                rmse[i_step] = np.sqrt(mse(y_true_one_step, y_pred_one_step))
                cor_value[i_step] = pearsonr(y_true_one_step, y_pred_one_step)[0]
        score_one_field = {'field': field, 'r2': r2, 'rmse': rmse, 'cor_value': cor_value}
        scores.append(score_one_field)
    return scores


def prepare_dl(data_list, batch_size, shuffle, drop_last=False):
    data_list_torch = [torch.from_numpy(data).float() for data in data_list]
    ds = TensorDataset(*data_list_torch)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dl


def set_dtype_and_model(device, model):
    if device == 'cuda':
        dtype = torch.cuda.FloatTensor
        model.cuda()
    else:
        dtype = torch.FloatTensor
    return dtype, model


def get_non_zero_max(data_):
    zero_loc = (data_ == 0.).all(axis=1).reshape([data_.shape[0], 1, -1])
    zeros_to_exclude = np.concatenate([zero_loc for i in range(data_.shape[1])], axis=1)
    data_[zeros_to_exclude] = np.nan
    max_vals = np.nanmax(data_[:, :, 40:], axis=2)
    max_vals[np.isnan(max_vals)] = 0.
    return max_vals


def result_folder():
    folder_name = str(datetime.datetime.now())[:-7]
    for item in ['.', ':', '-', ' ']:
        folder_name = folder_name.replace(item, '_')
    return folder_name


def normalize_data(data_scalar, data, name, method, scalar_mode, with_mean=False):
    """
    :param data_scalar:
    :param data: []
    :param name:
    :param method:
    :param scalar_mode:
    :return:
    """
    if method == 'fit_transform':
        data_scalar[name] = copy.deepcopy(data_scalar['base_scalar'](with_mean=with_mean))
    assert (scalar_mode in ['by_each_column', 'by_all_columns'])

    input_data = data.copy()
    size, channel, length = input_data.shape
    target_shape = [-1, input_data.shape[1]] if scalar_mode == 'by_each_column' else [-1, 1]
    zero_loc = (input_data == 0.).all(axis=1)
    for i in range(input_data.shape[1]):
        input_data[:, i][zero_loc] = np.nan
    input_data = input_data.transpose(0, 2, 1).reshape(target_shape) if scalar_mode == 'by_each_column' else input_data.reshape(target_shape)
    scaled_data = getattr(data_scalar[name], method)(input_data)
    scaled_data = scaled_data.reshape(size, length, channel).transpose(0, 2, 1) if scalar_mode == 'by_each_column' else scaled_data.reshape(size, channel, length)
    scaled_data[np.isnan(scaled_data)] = 0.
    return scaled_data


def define_channel_names(the_task):
    imu_segments = the_task['imu_segments']
    channel_names = {
        'acc': [segment + '_Accel_' + axis for segment in imu_segments for axis in ['X', 'Y', 'Z']],
        'gyr': [segment + '_Gyro_' + axis for segment in imu_segments for axis in ['X', 'Y', 'Z']]}
    return channel_names


def print_table(results):
    tb = pt.PrettyTable()
    for test_result in results:
        tb.field_names = test_result.keys()
        tb.add_row([np.round(np.mean(value), 3) if isinstance(value, (np.ndarray, float)) else value
                    for value in test_result.values()])
    logging.info(tb)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def find_peak_max(data_clip, height, width=None, prominence=None):
    """
    find the maximum peak
    :return:
    """
    peaks, properties = find_peaks(data_clip, width=width, height=height, prominence=prominence)
    if len(peaks) == 0:
        return None, None
    peak_heights = properties['peak_heights']
    max_index = np.nanargmax(peak_heights)
    return peaks[max_index], np.nanmax(peak_heights)


def data_filter(data, cut_off_fre, sampling_fre, filter_order=4):
    fre = cut_off_fre / (sampling_fre / 2)
    b, a = butter(filter_order, fre, 'lowpass')
    if len(data.shape) == 1:
        data_filtered = filtfilt(b, a, data)
    else:
        data_filtered = filtfilt(b, a, data, axis=0)
    return data_filtered


def get_data_by_merging_data_struct(data_struct_list):
    x_imu, x_emg, y = [], [], []
    for data_struct in data_struct_list:
        x_imu_trial, x_emg_trial, y_trial = data_struct.get_all_data()
        x_imu.append(x_imu_trial)
        x_emg.append(x_emg_trial)
        y.append(y_trial)
    return {'IMU': np.concatenate(x_imu, axis=0), 'EMG': np.concatenate(x_emg, axis=0), 'y': np.concatenate(y, axis=0)}


def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


class DataStruct:
    def __init__(self, feature_num_200, feature_num_1000, output_num_200, step_len_max):

        self.step_len_max = step_len_max
        self.num_of_step_allocate_one_time = 2000
        self.x_200 = np.zeros([self.num_of_step_allocate_one_time, step_len_max, feature_num_200])
        self.x_1000 = np.zeros([self.num_of_step_allocate_one_time, step_len_max * 5, feature_num_1000])
        self.y_200 = np.zeros([self.num_of_step_allocate_one_time, step_len_max, output_num_200])
        self.i_step = 0

    def add_new_step(self, x_step_200, x_step_1000, y_step_200):
        if self.i_step == self.x_200.shape[0] - 1:
            self.x_200 = np.concatenate([self.x_200, np.zeros(self.x_200.shape)], axis=0)
            self.x_1000 = np.concatenate([self.x_1000, np.zeros(self.x_1000.shape)], axis=0)
            self.y_200 = np.concatenate([self.y_200, np.zeros(self.y_200.shape)], axis=0)
        self.x_200[self.i_step, :x_step_200.shape[0], :] = x_step_200
        self.x_1000[self.i_step, :x_step_1000.shape[0], :] = x_step_1000
        self.y_200[self.i_step, :y_step_200.shape[0], :] = y_step_200
        self.i_step += 1

    def get_all_data(self):
        return self.x_200[:self.i_step], self.x_1000[:self.i_step], self.y_200[:self.i_step]


