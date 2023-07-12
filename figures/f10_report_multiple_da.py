import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from figures.PaperFigures import save_fig, load_da_data, results_to_pd_summary
from ssl_main.config import RESULTS_PATH
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score


def plot_map_with_number(ax, data_, x_ticks, y_ticks, title):
    data_ = data_.T
    mav_val = np.max(np.abs(data_))
    cmap = plt.colormaps.get_cmap('RdBu')
    im = ax.imshow(data_, interpolation='nearest', cmap=cmap, vmax=mav_val, vmin=-mav_val)
    # Add text to the matrix to display the values
    for i in range(len(data_)):
        for j in range(len(data_[i])):
            ax.text(j, i, round(data_[i, j], 3), ha='center', va='center', color='black')
    # Set the x and y axis labels and tick marks
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    plt.setp(ax.get_xticklabels(), ha='center')
    # Add a title and colorbar
    ax.set_title(title)
    return im


def plot_example_windows(results_task):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    for i_subplot, (ax, data_key_) in enumerate(zip(axes.flatten(), results_task.items())):
        key_, data_ = data_key_
        params = dict([param_tuple.split('_') for param_tuple in key_.split(', ')])
        output_num = int(list(data_.values())[2].shape[1] / 2)
        random_step = np.random.randint(0, list(data_.values())[2].shape[0])
        ax.plot(list(data_.values())[2][random_step, :output_num].ravel())
        ax.plot(list(data_.values())[2][random_step, output_num:].ravel())
        ax.title.set_text('UseSSL: ' + params['UseSsl'] + ', ' + 'Linear Prob: ' + params['LinearProb'])

    plt.suptitle('Example window', size=20)
    plt.tight_layout(rect=[0., 0., 1., 1.], w_pad=6)
    pdf.savefig()


def plot_r2_distribution_of_windows(results_task):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    plt.figure()
    tick_list = []
    r2_conditions = {}
    for i_subplot, (ax, data_key_) in enumerate(zip(axes.flatten(), results_task.items())):
        key_, data_ = data_key_
        params = dict([param_tuple.split('_') for param_tuple in key_.split(', ')])
        name_1 = 'ssl' if params['UseSsl'] == 'True' else 'no ssl'
        name_2 = ', linear prob' if params['LinearProb'] == 'True' else ', tuning'
        condition_name = name_1 + name_2
        tick_list.append(condition_name)
        output_num = int(list(data_.values())[2].shape[1] / 2)
        data_true = list(data_.values())[2][:, :output_num]
        data_pred = list(data_.values())[2][:, output_num:]
        r2_list = []
        for i_win in range(data_true.shape[0]):
            r2_list.append(r2_score(data_true[i_win, :].ravel(), data_pred[i_win, :].ravel()))
        r2_conditions[condition_name] = np.array(r2_list)
        plt.violinplot(r2_list, [i_subplot + int(i_subplot / 2)], points=20, widths=0.3, showmeans=True, showextrema=True)

    plt.violinplot(r2_conditions['ssl, tuning'] - r2_conditions['no ssl, tuning'],
                   [2], points=20, widths=0.3, showmeans=True, showextrema=True)
    tick_list.insert(2, 'Difference')
    plt.violinplot(r2_conditions['ssl, linear prob'] - r2_conditions['no ssl, linear prob'],
                   [5], points=20, widths=0.3, showmeans=True, showextrema=True)
    tick_list.insert(5, 'Difference')

    ax = plt.gca()
    ax.set_xticks(np.arange(len(tick_list)))
    ax.set_xticklabels(tick_list, size=7)
    plt.title('R2 distribution of windows', size=20)
    plt.grid()
    plt.tight_layout(rect=[0., 0., 1., 1.], w_pad=6)
    pdf.savefig()


def plot_spectrum(results_task):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    for i_subplot, (ax, data_key_) in enumerate(zip(axes.flatten(), results_task.items())):
        key_, data_ = data_key_
        params = dict([param_tuple.split('_') for param_tuple in key_.split(', ')])
        output_data = list(data_.values())[2]
        output_num = int(output_data.shape[1] / 2)

        step_spectrum_true, step_spectrum_pred = [], []
        for i_output in range(output_num):
            for i_step in range(output_data.shape[0]):
                step_spectrum_true.append(np.abs(np.fft.fft(output_data[i_step, i_output])))
                step_spectrum_pred.append(np.abs(np.fft.fft(output_data[i_step, i_output+output_num])))

                # plt.figure()
                # plt.plot(step_spectrum_true[-1])
                # plt.plot(step_spectrum_pred[-1])
                # plt.show()

        step_spectrum_true = np.mean(np.array(step_spectrum_true), axis=0)
        step_spectrum_pred = np.mean(np.array(step_spectrum_pred), axis=0)

        ax.plot(step_spectrum_true[:int(len(step_spectrum_true)/2)])
        ax.plot(step_spectrum_pred[:int(len(step_spectrum_pred)/2)])
        ax.title.set_text('UseSSL: ' + params['UseSsl'] + ', ' + 'Linear Prob: ' + params['LinearProb'])

    plt.suptitle('Frequency analysis', size=20)
    plt.tight_layout(rect=[0., 0., 1., 1.], w_pad=6)
    pdf.savefig()


def plot_map_with_number_all_four(data_all, i_metric, axes, x_ticks, y_ticks):
    min_val, max_val = np.min(data_all), np.max(data_all)
    for i_subplot, (data_, title_) in enumerate(zip(data_all, title_list)):
        ax = axes.flatten()[i_metric * 4 + i_subplot]
        data_ = data_.T
        cmap = colormaps.get_cmap('RdBu')
        if 'no SSL' in title_:
            data_ = data_[:1]
            y_ticks = ['']

        im = ax.imshow(data_, interpolation='nearest', cmap=cmap, vmax=max_val, vmin=min_val)
        # Add text to the matrix to display the values
        for i in range(len(data_)):
            for j in range(len(data_[i])):
                ax.text(j, i, round(data_[i, j], 2), size=8, ha='center', va='center', color='black')
        ax.set_xticks(np.arange(len(x_ticks)))
        ax.set_yticks(np.arange(len(y_ticks)))
        ax.set_xticklabels(x_ticks, size=8)
        ax.set_yticklabels(y_ticks, size=8)
        plt.setp(ax.get_xticklabels(), ha='center')
        ax.set_title(title_, size=8)
    return im


def init_fig():
    fig, ax = plt.subplots()
    return fig, ax


def finalize_fig(fig, ax, im):
    fig.colorbar(im, ax=ax)
    pdf.savefig()


colors = [np.array([125, 172, 80]) / 255, np.array([130, 130, 130]) / 255]


if __name__ == "__main__":
    # 'walking_knee_moment', 'Camargo_100', 'sun_drop_jump', 'opencap_dj', 'opencap_squat'
    da_names = ['Camargo_levelground_robustness', 'Camargo_levelground_output']
    test_folder = '2023_06_05_10_10_35_SSL_COMBINED'
    data_path = RESULTS_PATH + test_folder + '/'

    title_list = ['SSL - Fine-tuning', 'SSL - Linear', 'no SSL - Fine-tuning', 'no SSL - Linear']
    with PdfPages(data_path + f'f9_{test_folder}.pdf') as pdf:
        with open(data_path + 'training_log.txt') as f:
            test_name = f.readline()

        page = plt.figure()
        title_page_text = test_name
        page.text(0.5, 0.2, title_page_text, transform=page.transFigure, size=20, ha="center")
        plt.tight_layout()
        pdf.savefig()

        for da_name in da_names:
            results_task = load_da_data(data_path + da_name + '.h5')

            result_df = results_to_pd_summary(results_task, 0)
            result_df['PercentOfMasking'] = result_df['MaskPatchNum'] / (128 / result_df['PatchLen'])
            patch_len_list = np.sort(list(set(result_df['PatchLen'])))
            percent_of_masking_list = np.sort(list(set(result_df['PercentOfMasking'])))
            percent_of_masking_list_str = [str(round(i * 100, 1)) + '%' for i in percent_of_masking_list]
            print(patch_len_list)
            result_mean_map = np.zeros([4, len(patch_len_list), len(percent_of_masking_list)])
            fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(8, 12), height_ratios=[4, 1, 4, 1, 4, 1])
            for i_metric, metric in enumerate(['r2', 'rmse', 'correlation']):
                for i_patch, patch_len in enumerate(patch_len_list):
                    for i_percent, percent_of_masking in enumerate(percent_of_masking_list):
                        data_cond = result_df[(result_df['PatchLen'] == patch_len) & (result_df['PercentOfMasking'] == percent_of_masking) & (result_df['NumGradDeSsl'] == 30000)]
                        data_cond_0 = data_cond[~data_cond['LinearProb']&data_cond['UseSsl']]
                        result_mean_map[0, i_patch, i_percent] = np.mean(data_cond_0[metric])
                        data_cond_1 = data_cond[data_cond['LinearProb']&data_cond['UseSsl']]
                        result_mean_map[1, i_patch, i_percent] = np.mean(data_cond_1[metric])
                        data_cond_2 = data_cond[~data_cond['LinearProb']&~data_cond['UseSsl']]
                        result_mean_map[2, i_patch, i_percent] = np.mean(data_cond_2[metric])
                        data_cond_3 = data_cond[data_cond['LinearProb']&~data_cond['UseSsl']]
                        result_mean_map[3, i_patch, i_percent] = np.mean(data_cond_3[metric])
                plt.gcf().text(0.1, 0.8 - 0.3*i_metric, metric, size=20, ha='center', va='center', color='black')
                im = plot_map_with_number_all_four(result_mean_map, i_metric, axes, patch_len_list, percent_of_masking_list_str)

                # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                # fig.colorbar(im, cax=cbar_ax)

            plt.tight_layout(rect=[0., 0., 1., 1.], w_pad=3, h_pad=3)
            fig.subplots_adjust(left=0.2, top=0.9)
            plt.suptitle('Dataset: ' + da_name[:-7], size=20)
            pdf.savefig()

            results_task_example = {key_: value_ for key_, value_ in results_task.items() if 'PatchLen_8' in key_ and 'MaskPatchNum_6' in key_}
            plot_r2_distribution_of_windows(results_task_example)
            plot_spectrum(results_task_example)
            plot_example_windows(results_task_example)


