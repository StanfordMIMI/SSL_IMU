import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ssl_main.const import LINE_WIDTH
import h5py
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
            correlation, _ = pearsonr(subject_data[:, 0], subject_data[:, 1])
            result_list.append([only_linear, use_ssl, ratio, sub_id, rmse, r_rmse, r2, correlation])

    result_df = pd.DataFrame(result_list, columns=['only_linear', 'use_ssl', 'ratio', 'sub_id', 'rmse', 'r_rmse', 'r2', 'correlation'])
    return result_df


def results_dict_to_pd_profiles_masking_patchlen(result_all_tests, result_field_id, block_swing_phase=True):
    result_list = []
    for test_name, test_results in result_all_tests.items():
        only_linear = test_name.split('linear_protocol_')[1].split(',')[0] == 'True'
        use_ssl = test_name.split('use_ssl_')[1].split(',')[0] == 'True'
        ratio, rest_ = test_name.split('ratio_')[1].split(', masking_')
        mask_patch_num, patch_len = rest_.split(', patchlen_')
        ratio, mask_patch_num, patch_len = float(ratio), float(mask_patch_num), float(patch_len)
        for sub_name, subject_data in test_results.items():
            sub_id = int(sub_name[4:])
            data_true, data_pred = subject_data[:, result_field_id].ravel(), subject_data[:, result_field_id+3].ravel()

            if block_swing_phase:
                stance_phase_loc = np.where(np.abs(data_true) > 0.02)[0]
                data_true, data_pred = data_true[stance_phase_loc], data_pred[stance_phase_loc]
            rmse = np.sqrt(mse(data_true, data_pred))
            r_rmse = rmse / (np.max(data_true) - np.min(data_true)) * 100
            r2 = r2_score(data_true, data_pred)
            correlation, _ = pearsonr(data_true, data_pred)
            result_list.append([only_linear, use_ssl, ratio, mask_patch_num, patch_len, sub_id, rmse, r_rmse, r2, correlation])
    result_df = pd.DataFrame(result_list, columns=['only_linear', 'use_ssl', 'ratio', 'mask_patch_num', 'patch_len', 'sub_id',
                                                   'rmse', 'r_rmse', 'r2', 'correlation'])
    return result_df


def results_dict_to_pd_profiles(result_all_tests, result_field_id, block_swing_phase=True):
    result_list = []
    for test_name, test_results in result_all_tests.items():
        param_tuples = [param_tuple.split('_') for param_tuple in test_name.split(', ')]
        for param_tuple in param_tuples:
            if param_tuple[1] in ['True', 'False']:
                param_tuple[1] = param_tuple[1] == 'True'
            else:
                param_tuple[1] = float(param_tuple[1])
        # only_linear = test_name.split('linear_protocol_')[1].split(',')[0] == 'True'
        # use_ssl = test_name.split('use_ssl_')[1].split(',')[0] == 'True'
        # ratio, rest_ = test_name.split('ratio_')[1].split(', masking_')
        # mask_patch_num, patch_len = rest_.split(', patchlen_')
        # ratio, mask_patch_num, patch_len = float(ratio), float(mask_patch_num), float(patch_len)
        for sub_name, subject_data in test_results.items():
            sub_id = int(sub_name[4:])
            data_true, data_pred = subject_data[:, result_field_id].ravel(), subject_data[:, result_field_id+int(subject_data.shape[1]/2)].ravel()
            if block_swing_phase:
                stance_phase_loc = np.where(np.abs(data_true) > 0.02)[0]
                data_true, data_pred = data_true[stance_phase_loc], data_pred[stance_phase_loc]
            rmse = np.sqrt(mse(data_true, data_pred))
            r_rmse = rmse / (np.max(data_true) - np.min(data_true)) * 100
            r2 = r2_score(data_true, data_pred)
            correlation, _ = pearsonr(data_true, data_pred)
            result_list.append([param_tuple[1] for param_tuple in param_tuples] + [sub_id, rmse, r_rmse, r2, correlation])
    result_df = pd.DataFrame(result_list, columns=[param_tuple[0] for param_tuple in param_tuples] + ['sub_id',
                                                   'rmse', 'r_rmse', 'r2', 'correlation'])
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





