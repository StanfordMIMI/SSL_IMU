import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from figures.PaperFigures import save_fig, load_da_data
from ssl_main.config import RESULTS_PATH
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
import pandas as pd


# def plot_map_with_number(ax, data_, x_ticks, y_ticks, title):
#     data_ = data_.T
#     mav_val = np.max(np.abs(data_))
#     cmap = plt.colormaps.get_cmap('RdBu')
#     im = ax.imshow(data_, interpolation='nearest', cmap=cmap, vmax=mav_val, vmin=-mav_val)
#     # Add text to the matrix to display the values
#     for i in range(len(data_)):
#         for j in range(len(data_[i])):
#             ax.text(j, i, round(data_[i, j], 3), ha='center', va='center', color='black')
#     # Set the x and y axis labels and tick marks
#     ax.set_xticks(np.arange(len(x_ticks)))
#     ax.set_yticks(np.arange(len(y_ticks)))
#     ax.set_xticklabels(x_ticks)
#     ax.set_yticklabels(y_ticks)
#     plt.setp(ax.get_xticklabels(), ha='center')
#     # Add a title and colorbar
#     ax.set_title(title)
#     return im
#
#
# def plot_example_windows(results_task):
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
#     for i_subplot, (ax, data_key_) in enumerate(zip(axes.flatten(), results_task.items())):
#         key_, data_ = data_key_
#         params = dict([param_tuple.split('_') for param_tuple in key_.split(', ')])
#         output_num = int(list(data_.values())[2].shape[1] / 2)
#         random_step = np.random.randint(0, list(data_.values())[2].shape[0])
#         ax.plot(list(data_.values())[2][random_step, :output_num].ravel())
#         ax.plot(list(data_.values())[2][random_step, output_num:].ravel())
#         ax.title.set_text('UseSSL: ' + params['UseSsl'] + ', ' + 'Linear Prob: ' + params['LinearProb'])
#
#     plt.suptitle('Example window', size=20)
#     plt.tight_layout(rect=[0., 0., 1., 1.], w_pad=6)
#     pdf.savefig()
#
#
# def plot_r2_distribution_of_windows(results_task):
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
#     plt.figure()
#     tick_list = []
#     r2_conditions = {}
#     for i_subplot, (ax, data_key_) in enumerate(zip(axes.flatten(), results_task.items())):
#         key_, data_ = data_key_
#         params = dict([param_tuple.split('_') for param_tuple in key_.split(', ')])
#         name_1 = 'ssl' if params['UseSsl'] == 'True' else 'no ssl'
#         name_2 = ', linear prob' if params['LinearProb'] == 'True' else ', tuning'
#         condition_name = name_1 + name_2
#         tick_list.append(condition_name)
#         output_num = int(list(data_.values())[2].shape[1] / 2)
#         data_true = list(data_.values())[2][:, :output_num]
#         data_pred = list(data_.values())[2][:, output_num:]
#         r2_list = []
#         for i_win in range(data_true.shape[0]):
#             r2_list.append(r2_score(data_true[i_win, :].ravel(), data_pred[i_win, :].ravel()))
#         r2_conditions[condition_name] = np.array(r2_list)
#         plt.violinplot(r2_list, [i_subplot + int(i_subplot / 2)], points=20, widths=0.3, showmeans=True, showextrema=True)
#
#     plt.violinplot(r2_conditions['ssl, tuning'] - r2_conditions['no ssl, tuning'],
#                    [2], points=20, widths=0.3, showmeans=True, showextrema=True)
#     tick_list.insert(2, 'Difference')
#     plt.violinplot(r2_conditions['ssl, linear prob'] - r2_conditions['no ssl, linear prob'],
#                    [5], points=20, widths=0.3, showmeans=True, showextrema=True)
#     tick_list.insert(5, 'Difference')
#
#     ax = plt.gca()
#     ax.set_xticks(np.arange(len(tick_list)))
#     ax.set_xticklabels(tick_list, size=7)
#     plt.title('R2 distribution of windows', size=20)
#     plt.grid()
#     plt.tight_layout(rect=[0., 0., 1., 1.], w_pad=6)
#     pdf.savefig()
#
#
# def plot_spectrum(results_task):
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
#     for i_subplot, (ax, data_key_) in enumerate(zip(axes.flatten(), results_task.items())):
#         key_, data_ = data_key_
#         params = dict([param_tuple.split('_') for param_tuple in key_.split(', ')])
#         output_data = list(data_.values())[2]
#         output_num = int(output_data.shape[1] / 2)
#
#         step_spectrum_true, step_spectrum_pred = [], []
#         for i_output in range(output_num):
#             for i_step in range(output_data.shape[0]):
#                 step_spectrum_true.append(np.abs(np.fft.fft(output_data[i_step, i_output])))
#                 step_spectrum_pred.append(np.abs(np.fft.fft(output_data[i_step, i_output+output_num])))
#
#                 # plt.figure()
#                 # plt.plot(step_spectrum_true[-1])
#                 # plt.plot(step_spectrum_pred[-1])
#                 # plt.show()
#
#         step_spectrum_true = np.mean(np.array(step_spectrum_true), axis=0)
#         step_spectrum_pred = np.mean(np.array(step_spectrum_pred), axis=0)
#
#         ax.plot(step_spectrum_true[:int(len(step_spectrum_true)/2)])
#         ax.plot(step_spectrum_pred[:int(len(step_spectrum_pred)/2)])
#         ax.title.set_text('UseSSL: ' + params['UseSsl'] + ', ' + 'Linear Prob: ' + params['LinearProb'])
#
#     plt.suptitle('Frequency analysis', size=20)
#     plt.tight_layout(rect=[0., 0., 1., 1.], w_pad=6)
#     pdf.savefig()
#
#
# def plot_map_with_number_all_four(data_all, i_metric, axes, x_ticks, y_ticks):
#     min_val, max_val = np.min(data_all), np.max(data_all)
#     for i_subplot, (data_, title_) in enumerate(zip(data_all, title_list)):
#         ax = axes.flatten()[i_metric * 4 + i_subplot]
#         data_ = data_.T
#         cmap = colormaps.get_cmap('RdBu')
#         if 'no SSL' in title_:
#             data_ = data_[:1]
#             y_ticks = ['']
#
#         im = ax.imshow(data_, interpolation='nearest', cmap=cmap, vmax=max_val, vmin=min_val)
#         # Add text to the matrix to display the values
#         for i in range(len(data_)):
#             for j in range(len(data_[i])):
#                 ax.text(j, i, round(data_[i, j], 2), size=8, ha='center', va='center', color='black')
#         ax.set_xticks(np.arange(len(x_ticks)))
#         ax.set_yticks(np.arange(len(y_ticks)))
#         ax.set_xticklabels(x_ticks, size=8)
#         ax.set_yticklabels(y_ticks, size=8)
#         plt.setp(ax.get_xticklabels(), ha='center')
#         ax.set_title(title_, size=8)
#     return im
#
#
# def init_fig():
#     fig, ax = plt.subplots()
#     return fig, ax
#
#
# def finalize_fig(fig, ax, im):
#     fig.colorbar(im, ax=ax)
#     pdf.savefig()
#
#
# def results_to_dict(result_all_tests, result_field_id, block_swing_phase=True):
#     result_dict = {}
#     for test_name, test_results in result_all_tests.items():
#         param_tuples = [param_tuple.split('_') for param_tuple in test_name.split(', ')]
#         for param_tuple in param_tuples:
#             if param_tuple[-1] in ['True', 'False']:
#                 param_tuple[-1] = param_tuple[-1] == 'True'
#             else:
#                 param_tuple[-1] = float(param_tuple[-1])
#         result_dict[test_name] = []
#         for sub_name, subject_data in test_results.items():
#             sub_id = int(sub_name[4:])
#             data_true, data_pred = subject_data[:, result_field_id].ravel(), subject_data[:, result_field_id+int(subject_data.shape[1]/2)].ravel()
#             if block_swing_phase:
#                 stance_phase_loc = np.where(np.abs(data_true) > 0.02)[0]
#                 data_true, data_pred = data_true[stance_phase_loc], data_pred[stance_phase_loc]
#             rmse = np.sqrt(mse(data_true, data_pred))
#             result_dict[test_name].append(rmse)
#     return result_dict


def results_to_pd_summary(result_all_tests, result_field_id, block_swing_phase=True):
    result_list = []
    for test_name, test_results in result_all_tests.items():
        param_tuples = [param_tuple.split('_') for param_tuple in test_name.split(', ')]
        for param_tuple in param_tuples:
            if param_tuple[-1] in ['True', 'False']:
                param_tuple[-1] = param_tuple[-1] == 'True'
            else:
                param_tuple[-1] = float(param_tuple[-1])
        rmse, r2 = [], []
        for sub_name, subject_data in test_results.items():
            data_true, data_pred = subject_data[:, result_field_id].ravel(), subject_data[:, result_field_id+int(subject_data.shape[1]/2)].ravel()
            if block_swing_phase:
                stance_phase_loc = np.where(np.abs(data_true) > 0.02)[0]
                data_true, data_pred = data_true[stance_phase_loc], data_pred[stance_phase_loc]
            rmse.append(np.sqrt(mse(data_true, data_pred)))
            r2.append(r2_score(data_true, data_pred))
        result_list.append([param_tuple[-1] for param_tuple in param_tuples] + [np.mean(rmse), np.mean(r2)])
    result_df = pd.DataFrame(result_list, columns=[param_tuple[0] for param_tuple in param_tuples] + ['rmse', 'r2'])
    return result_df


if __name__ == "__main__":
    da_name = 'Camargo_levelground_output'
    test_folder = '2023_07_12_22_06_50_ssl_hyper'
    data_path = RESULTS_PATH + test_folder + '/'
    metric = 'r2'

    with PdfPages(data_path + f'f9_{test_folder}.pdf') as pdf:
        results_task = load_da_data(data_path + da_name + '.h5')
        hyper_list = [param_tuple.split('_')[0] for param_tuple in list(results_task.keys())[0].split(', ')][3:]
        # hyper_value_set = {}

        # result_dict = results_to_dict(results_task, 0)
        result_summary = results_to_pd_summary(results_task, 0)
        result_summary = result_summary[(result_summary['LinearProb'] == False) & (result_summary['UseSsl'] == True)]

        for hyper in hyper_list:
            hyper_values = np.sort(result_summary[hyper].unique())
            hyper_lines = []
            for hyper_value in hyper_values:
                df_current_hyper = result_summary[result_summary[hyper] == hyper_value]
                hyper_lines.append(df_current_hyper[metric].values)

            plt.figure()
            for line_ in np.array(hyper_lines).T:
                plt.plot(hyper_values, line_)
            plt.xticks(hyper_values)
            plt.xlabel(hyper)
            plt.ylabel(metric)
            # plt.show()
            pdf.savefig()









