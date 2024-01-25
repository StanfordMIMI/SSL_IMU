import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ssl_main.const import LINE_WIDTH
import h5py
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import json
from ssl_main.const import vgrf_names


def load_da_data(test_dir):
    with h5py.File(test_dir, 'r') as hf:
        result_all_tests = {test_name: {sub_name: sub_data[:] for sub_name, sub_data in test_results.items()}
                            for test_name, test_results in hf.items()}
        data_columns = json.loads(hf.attrs['columns'])
        return result_all_tests, data_columns


def format_errorbar_cap(caplines, size=15):
    for i_cap in range(1):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(size)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH)


def results_to_pd_summary(result_all_tests, results_columns, block_swing_phase=1):
    """

    :param result_all_tests:
    :param results_columns:
    :param block_swing_phase: 0 for no blocking, 1 for blocking stance phase, 2 for blocking swing phase
    :return:
    """
    result_list = []
    for test_name, test_results in result_all_tests.items():
        param_tuples = [param_tuple.split('_') for param_tuple in test_name.split(', ')]
        for param_tuple in param_tuples:
            if param_tuple[-1] in ['True', 'False']:
                param_tuple[-1] = param_tuple[-1] == 'True'
            else:
                param_tuple[-1] = float(param_tuple[-1])

        for sub_name, subject_data in test_results.items():
            sub_id = int(sub_name[4:])
            current_sub_result = [param_tuple[-1] for param_tuple in param_tuples] + [sub_id]
            grf_z = abs(subject_data[:, int((len(results_columns) - 1)/2)])
            stance_phase_loc = np.where(grf_z.ravel() > 0.02)[0]
            flight_phase_loc = np.where(grf_z.ravel() <= 0.02)[0]
            biom_param_names = [col[:-4] for col in results_columns if 'true' in col]
            for i_col, col_name in enumerate(biom_param_names):
                data_true, data_pred = subject_data[:, i_col].ravel(), subject_data[:, i_col+int(subject_data.shape[1]/2)].ravel()
                if block_swing_phase == 1:
                    data_true, data_pred = data_true[stance_phase_loc], data_pred[stance_phase_loc]
                elif block_swing_phase == 2:
                    data_true, data_pred = data_true[flight_phase_loc], data_pred[flight_phase_loc]
                rmse = np.sqrt(mse(data_true, data_pred))
                r_rmse = rmse / (np.max(data_true) - np.min(data_true)) * 100
                r2 = r2_score(data_true, data_pred)
                correlation, _ = pearsonr(data_true, data_pred)
                current_sub_result += [rmse, r_rmse, r2, correlation]
            result_list.append(current_sub_result)
    metric_col_names = [param_ + metric_ for param_ in biom_param_names for metric_ in ['rmse', 'r_rmse', 'r2', 'correlation']]
    result_df = pd.DataFrame(result_list, columns=[param_tuple[0] for param_tuple in param_tuples] + ['sub_id'] + metric_col_names)
    return result_df


def results_to_pd_summary_only_peaks(result_all_tests, result_field_id, sign_of_peak=1):
    result_list = []
    for test_name, test_results in result_all_tests.items():
        param_tuples = [param_tuple.split('_') for param_tuple in test_name.split(', ')]
        for param_tuple in param_tuples:
            if param_tuple[1] in ['True', 'False']:
                param_tuple[1] = param_tuple[1] == 'True'
            else:
                param_tuple[1] = float(param_tuple[1])
        for sub_name, subject_data in test_results.items():
            sub_id = int(sub_name[4:])
            data_true, data_pred = subject_data[:, result_field_id], subject_data[:, result_field_id+int(subject_data.shape[1]/2)]
            peaks = np.array([[np.max(sign_of_peak * data_step_true), np.max(sign_of_peak * data_step_pred)]
                              for data_step_true, data_step_pred in zip(data_true, data_pred)])
            rmse = np.sqrt(mse(peaks[:, 0], peaks[:, 1]))
            r_rmse = rmse / (np.max(peaks[:, 0]) - np.min(peaks[:, 0])) * 100
            r2 = r2_score(peaks[:, 0], peaks[:, 1])
            try:
                correlation, _ = pearsonr(peaks[:, 0], peaks[:, 1])
            except:
                correlation = np.nan
            result_list.append([param_tuple[1] for param_tuple in param_tuples] + [sub_id, rmse, r_rmse, r2, correlation])
    result_df = pd.DataFrame(result_list, columns=[param_tuple[0] for param_tuple in param_tuples] + ['sub_id', 'rmse', 'r_rmse', 'r2', 'correlation'])
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


def format_axis(ax=None, line_width=LINE_WIDTH):
    if ax is None:
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







