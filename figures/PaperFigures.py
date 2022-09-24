import numpy as np
import matplotlib.pyplot as plt
from const import LINE_WIDTH
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE
import h5py
import json
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.signal import find_peaks


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


def get_mean_gait_cycle_then_find_peak(data, columns, moment_name, search_percent_from_start):
    search_sample = int(100 * search_percent_from_start)
    moment_name = translate_moment_name(moment_name)
    true_row, pred_row, weight_row = columns.index(moment_name), columns.index('pred_' + moment_name), columns.index('force_phase')
    true_resampled = BaseFramework.keep_stance_then_resample(data[:, :, true_row:true_row+1], data[:, :, weight_row:weight_row+1])[0][:, :, 0]
    pred_resampled = BaseFramework.keep_stance_then_resample(data[:, :, pred_row:pred_row+1], data[:, :, weight_row:weight_row+1])[0][:, :, 0]
    true_average_of_gait_cycles = np.mean(true_resampled, axis=0)
    pred_average_of_gait_cycles = np.mean(pred_resampled, axis=0)
    #
    # plt.figure()
    # plt.plot(true_average_of_gait_cycles)
    # plt.plot(pred_average_of_gait_cycles)
    # plt.show()

    true_peak = np.max(true_average_of_gait_cycles[:search_sample])
    pred_peak = np.max(pred_average_of_gait_cycles[:search_sample])
    return true_peak, pred_peak


def save_fig(name, dpi=300):
    plt.savefig('exports/' + name + '.png', dpi=dpi)


def get_peak_of_each_gait_cycle(data, columns, moment_name, search_percent_from_start):
    step_lens = get_step_len(data)
    search_lens = (search_percent_from_start * step_lens).astype(int)
    moment_name = translate_moment_name(moment_name)
    true_row, pred_row = columns.index(moment_name), columns.index('pred_' + moment_name)
    true_peaks, pred_peaks = [], []
    peak_not_found = 0
    for i_step in range(data.shape[0]):
        true_peak = np.max(data[i_step, :search_lens[i_step], true_row])
        pred_peak = np.max(data[i_step, :search_lens[i_step], pred_row])
        # true_peak = find_peak_max(data[i_step, :search_lens[i_step], true_row], 0.1)
        # if true_peak is None:
        #     peak_not_found += 1
        #     continue
        # pred_peak = find_peak_max(data[i_step, :search_lens[i_step], pred_row], 0.1)
        # if pred_peak is None:
        #     pred_peak = np.max(data[i_step, :search_lens[i_step], pred_row])
        true_peaks.append(true_peak)
        pred_peaks.append(pred_peak)
    # print('Peaks of {:3.1f}% steps not found.'.format(peak_not_found/data.shape[0]*100))
    return true_peaks, pred_peaks


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
















