import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from const import LINE_WIDTH
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE
import h5py
import json
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.signal import find_peaks


def load_da_data(test_dir):
    with h5py.File(test_dir, 'r') as hf:
        result_all_tests = {test_name: {sub_name: sub_data[:] for sub_name, sub_data in test_results.items()}
                            for test_name, test_results in hf.items()}
        return result_all_tests


def results_dict_to_pd(result_all_tests):
    result_list = []
    for test_name, test_results in result_all_tests.items():
        only_linear = test_name.split('linear_protocol_')[1].split(',')[0] == 'True'
        use_ssl = test_name.split('use_ssl_')[1].split(',')[0] == 'True'
        ratio = float(test_name.split('ratio_')[1])
        for sub_name, subject_data in test_results.items():
            sub_id = int(sub_name[4:])
            rmse = np.sqrt(mse(subject_data[:, 0], subject_data[:, 1]))
            r_rmse = rmse / (np.max(subject_data[:, 0]) - np.min(subject_data[:, 0])) * 100
            r2 = r2_score(subject_data[:, 0], subject_data[:, 1])
            correlation, _ = kendalltau(subject_data[:, 0], subject_data[:, 1])
            result_list.append([only_linear, use_ssl, ratio, sub_id, rmse, r_rmse, r2, correlation])

    result_df = pd.DataFrame(result_list, columns=['only_linear', 'use_ssl', 'ratio', 'sub_id', 'rmse', 'r_rmse', 'r2', 'correlation'])
    return result_df


def results_dict_to_pd_ssl_sub_num(result_all_tests):
    result_list = []
    for test_name, test_results in result_all_tests.items():
        only_linear = test_name.split('linear_protocol_')[1].split(',')[0] == 'True'
        use_ssl = test_name.split('use_ssl_')[1].split(',')[0] == 'True'
        da_ratio = float(test_name.split('ratio_')[1].split(',')[0])
        ssl_sub_num = int(test_name.split('ssl_sub_num_')[1].split(',')[0])
        for sub_name, subject_data in test_results.items():
            sub_id = int(sub_name[4:])
            rmse = np.sqrt(mse(subject_data[:, 0], subject_data[:, 1]))
            r_rmse = rmse / (np.max(subject_data[:, 0]) - np.min(subject_data[:, 0])) * 100
            r2 = r2_score(subject_data[:, 0], subject_data[:, 1])
            correlation, _ = pearsonr(subject_data[:, 0], subject_data[:, 1])
            result_list.append([only_linear, use_ssl, da_ratio, ssl_sub_num, sub_id, rmse, r_rmse, r2, correlation])

    result_df = pd.DataFrame(result_list, columns=['only_linear', 'use_ssl', 'da_ratio', 'ssl_sub_num', 'sub_id', 'rmse', 'r_rmse', 'r2', 'correlation'])
    return result_df


def results_dict_to_pd_convergence_speed(result_all_tests):
    result_list = []
    for test_name, test_results in result_all_tests.items():
        only_linear = test_name.split('linear_protocol_')[1].split(',')[0] == 'True'
        use_ssl = test_name.split('use_ssl_')[1].split(',')[0] == 'True'
        da_ratio = float(test_name.split('ratio_')[1].split(',')[0])
        try:
            i_optimize = int(test_name.split('i_optimize_')[1])
        except IndexError:
            continue
        for sub_name, subject_data in test_results.items():
            sub_id = int(sub_name[4:])
            rmse = np.sqrt(mse(subject_data[:, 0], subject_data[:, 1]))
            r_rmse = rmse / (np.max(subject_data[:, 0]) - np.min(subject_data[:, 0])) * 100
            r2 = r2_score(subject_data[:, 0], subject_data[:, 1])
            correlation, _ = pearsonr(subject_data[:, 0], subject_data[:, 1])
            result_list.append([only_linear, use_ssl, da_ratio, i_optimize, sub_id, rmse, r_rmse, r2, correlation])

    result_df = pd.DataFrame(result_list, columns=['only_linear', 'use_ssl', 'da_ratio', 'i_optimize', 'sub_id', 'rmse', 'r_rmse', 'r2', 'correlation'])
    return result_df


def get_step_len(data, feature_col_num=0):
    """
    :param data: Numpy array, 3d (step, sample, feature)
    :param feature_col_num: int, feature column id for step length detection. Different id would probably return
           the same results
    :return:
    """
    data_the_feature = data[:, :, feature_col_num]
    zero_loc = data_the_feature == 0.
    step_lens = np.sum(~zero_loc, axis=1)
    return step_lens


def find_peak_max(data_clip, height, width=None, prominence=None):
    """
    find the maximum peak
    :return:
    """
    peaks, properties = find_peaks(data_clip, width=width, height=height, prominence=prominence)
    if len(peaks) == 0:
        return None
    peak_heights = properties['peak_heights']
    return np.max(peak_heights)


def save_fig(name, dpi=300):
    plt.savefig('exports/' + name + '.png', dpi=dpi)


def format_axis(line_width=LINE_WIDTH):
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(width=line_width)
    ax.yaxis.set_tick_params(width=line_width)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)


def hide_axis_add_grid():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(color='lightgray', linewidth=1.5)
    ax.tick_params(color='lightgray', width=1.5)


def get_score(arr_true, arr_pred, w):
    assert(len(arr_true.shape) == 1 and arr_true.shape == arr_pred.shape == w.shape)
    locs = np.where(w.ravel())[0]
    arr_true, arr_pred = arr_true.ravel()[locs], arr_pred.ravel()[locs]
    mae = np.mean(np.abs(arr_true - arr_pred))
    r_rmse = np.sqrt(mse(arr_true, arr_pred)) / (arr_true.max() - arr_true.min()) * 100
    cor_value = pearsonr(arr_true, arr_pred)[0]
    rmse = np.sqrt(mse(arr_true, arr_pred))
    return {'MAE': mae, 'RMSE': rmse, 'rRMSE': r_rmse, 'r':  cor_value}





