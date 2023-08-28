from ssl_main.const import LINE_WIDTH, FONT_DICT, LINE_WIDTH_THICK
from figures.PaperFigures import format_axis, results_to_pd_summary_only_peaks
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from figures.PaperFigures import save_fig, load_da_data, results_to_pd_summary, format_errorbar_cap
from ssl_main.config import RESULTS_PATH


def init_fig():
    fig = plt.figure(figsize=(12, 4.5))
    return fig


def draw_line_1(line_config, amount_list):
    def format_ticks():
        ax = plt.gca()
        ax.set_xlabel('Size of Training Set', fontdict=FONT_DICT)
        ax.set_ylabel('Correlation Coefficients - vGRF Profile', fontdict=FONT_DICT)
        ax.set_xscale('log')
        x_ticks = win_number_list[::2]
        ax.set_xticks(x_ticks)
        x_ticks_label = [str(round(x_tick / 1000, 1)) + f'k\n{round(x_tick*100/(0.8*win_number_total))}%' for x_tick in x_ticks]
        ax.set_xticklabels(x_ticks_label, fontdict=FONT_DICT)
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0)
        ax.set_ylim(0.9, 0.98)
        ax.set_yticks([0.9, 0.92, 0.94, 0.96, 0.98])
        ax.set_yticklabels([0.9, 0.92, 0.94, 0.96, 0.98], fontdict=FONT_DICT)
        plt.title(test_names_print[i_da], fontdict=FONT_DICT, pad=15)
    rc('font', family='Arial')
    mean_, std_ = [], []
    win_number_list = [int(0.8 * win_number_total * amount) for amount in amount_list]      # 0.8 for 5-fold CV
    for amount in amount_list:
        data_ = line_config['data'][line_config['data'][:, 0] == amount]
        mean_.append(np.mean(data_[:, 1]))
        std_.append(np.std(data_[:, 1]))
    plt.plot(win_number_list, mean_, line_config['style'], linewidth=LINE_WIDTH_THICK, markersize=10,
             color=line_config['color'], label=line_config['label'])
    format_ticks()
    format_axis()
    print(np.round(mean_[-1], 2), np.round(std_[-1], 2))


def draw_line_0_2(line_config, amount_list):
    def format_ticks():
        ax = plt.gca()
        ax.set_xlabel('Size of Training Set', fontdict=FONT_DICT)
        ax.set_ylabel('Correlation Coefficients - vGRF Profile', fontdict=FONT_DICT)
        ax.set_xlim(30, 415)
        x_ticks = win_number_list[:1] + [130, 220, 310] + win_number_list[-1:]
        x_ticks_label = [f'{x_tick}\n{round(x_tick*100/(0.8*win_number_total))}%' for x_tick in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_label, fontdict=FONT_DICT)

        if i_da == 0:
            ax.set_ylim(0.91, 0.951)
            ax.set_yticks([0.91, 0.92, 0.93, 0.94, 0.95])
            ax.set_yticklabels([0.91, 0.92, 0.93, 0.94, 0.95], fontdict=FONT_DICT)
        elif i_da == 2:
            ax.set_ylim(0.91, 0.95)
            ax.set_yticks([0.91, 0.92, 0.93, 0.94, 0.95])
            ax.set_yticklabels([0.91, 0.92, 0.93, 0.94, 0.95], fontdict=FONT_DICT)

        plt.title(test_names_print[i_da], fontdict=FONT_DICT, pad=15)

    rc('font', family='Arial')
    mean_, std_ = [], []
    win_number_list = [int(0.8 * win_number_total * amount) for amount in amount_list]
    for amount in amount_list:
        data_ = line_config['data'][line_config['data'][:, 0] == amount]
        mean_.append(np.mean(data_[:, 1]))
        std_.append(np.std(data_[:, 1]))
    plt.plot(win_number_list, mean_, line_config['style'], linewidth=LINE_WIDTH_THICK, markersize=10,
             color=line_config['color'], label=line_config['label'])
    format_ticks()
    format_axis()
    print(np.round(np.max(mean_), 2), np.round(std_[np.argmax(mean_)], 2))


def finalize_fig():
    plt.tight_layout(rect=[0., 0., 1., .86], w_pad=2)
    plt.legend(['Real IMU', 'Synthetic IMU', 'Real and Synthetic IMU', 'Baseline'], bbox_to_anchor=(1, 1.4),
               ncol=4, fontsize=FONT_DICT['fontsize'], frameon=False, handlelength=4)


if __name__ == "__main__":
    colors = [np.array(x) / 255 for x in [[96, 201, 230], [8, 141, 175], [2, 83, 100], [180, 180, 180]]]
    metric = 'correlation'
    target_param = 'ratio'
    patch_len = 1
    mask_patch_num = 16

    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]
    test_folders = ['2023_08_25_12_05_00_data_ratio_MOVI', '2023_08_25_12_05_33_data_ratio_amass', '2023_08_25_09_11_52_data_ratio_combined', 'baseline']
    test_names_print = ('Task 1 - Overground Walking', 'Task 2 - Treadmill Walking', 'Task 3 - Drop Landing')

    rc('font', family='Arial')
    bars = []
    fig = init_fig()

    for i_da, (da_name, sign) in enumerate(zip(da_names, [1, -1, 1])):
        plt.subplot(1, 3, i_da + 1)
        for i_test, test_folder in enumerate(test_folders):
            if test_folder == 'baseline':
                test_folder = test_folders[i_test - 1]
                use_ssl = False
            else:
                use_ssl = True
            data_path = RESULTS_PATH + test_folder + '/'
            results_task = load_da_data(data_path + da_name + '.h5')
            results_task = {key_: value_ for key_, value_ in results_task.items()
                            if f'PatchLen_{patch_len}' in key_ and
                            f'MaskPatchNum_{mask_patch_num}' in key_ and
                            'LinearProb_False' in key_}
            full_size_task = [value_ for key_, value_ in results_task.items() if 'ratio_1' in key_][0]
            win_number_total = np.sum([data_.shape[0] for _, data_ in full_size_task.items()])

            result_df = results_to_pd_summary(results_task, 0)

            param_set = list(set(result_df[target_param]))
            param_set.sort()

            data_cond = result_df[result_df['UseSsl'] == use_ssl][[target_param, metric]]
            line_config = {'color': colors[i_test], 'style': '.-', 'label': 'Self-Supervised Models Models', 'data': data_cond.values}
            if i_da == 1:
                draw_line_1(line_config, param_set)
            else:
                draw_line_0_2(line_config, param_set)
            plt.grid(True, linewidth=1, alpha=0.5)
    finalize_fig()
    save_fig('f4_lines')
    plt.show()
