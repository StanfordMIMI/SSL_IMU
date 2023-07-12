import os, sys
from ssl_main.const import FONT_DICT


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from figures.PaperFigures import save_fig, load_da_data, results_to_pd_summary
from ssl_main.config import RESULTS_PATH


def plot_map(i_task, data_all, x_ticks, y_ticks):
    min_val, max_val = np.min(data_all), np.max(data_all)
    min_val, max_val = min_val - 0.5 * (max_val - min_val), max_val + 0.5 * (max_val - min_val)

    cmap = colormaps.get_cmap('RdBu')

    ax = axes[0, i_task]
    data_ssl = data_all[0].T
    im = ax.imshow(data_ssl, interpolation='nearest', cmap=cmap, vmax=max_val, vmin=min_val)
    # Add text to the matrix to display the values
    for i in range(len(data_ssl)):
        for j in range(len(data_ssl[i])):
            ax.text(j, i, round(data_ssl[i, j], 2), ha='center', va='center', color='black')
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(y_ticks)))
    if i_task == 0:
        ax.set_yticklabels(y_ticks)
        ax.tick_params(which='both', top=False,  bottom=False)
        ax.set_ylabel('Percentage of Masking', labelpad=15)
    else:
        ax.set_yticklabels([])
        ax.tick_params(which='both', left=False,  right=False, top=False,  bottom=False)
    plt.setp(ax.get_xticklabels(), ha='center')
    ax.title.set_text(test_names_print[i_task])

    ax = axes[1, i_task]
    data_bl = data_all[1, :, 0:1].T
    im = ax.imshow(data_bl, interpolation='nearest', cmap=cmap, vmax=max_val, vmin=min_val)
    # Add text to the matrix to display the values
    for i in range(np.size(data_bl)):
        ax.text(i, 0, round(data_bl[0, i], 2), ha='center', va='center', color='black')

    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels(x_ticks)
    ax.set_yticks(np.arange(1))
    ax.set_xlabel('Patch Length')
    if i_task == 0:
        ax.set_yticklabels(['Baseline'])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', left=False,  right=False)

    plt.tight_layout(rect=[0.1, 0., 1., 1.], h_pad=-0.1)


def plot_curve(i_task, data_all, x_ticks, curve_labels):
    ax = axes[i_task]
    data_combined = np.concatenate([data_all[1, :, :1], data_all[0]], axis=1).T
    data_combined = np.flip(data_combined, axis=1)
    curve_labels = reversed(curve_labels)
    ax.plot(data_combined, marker="o", markersize=4)
    ax.set_xlabel('Percentage of Masking', fontdict=FONT_DICT, labelpad=0)
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks, fontdict=FONT_DICT)
    ax.set_ylabel(r'$R^2$', fontdict=FONT_DICT)
    # ax.set_yticks(plt.yticks(), fontdict=FONT_DICT)
    # ax.tick_params(axis='y', which='major', fontdict=FONT_DICT)

    ax.title.set_text(test_names_print[i_task])
    plt.tight_layout(rect=[0., 0.15, 1., 1.], h_pad=1.6)
    if i_task == 2:
        plt.legend([f'Patch Length = {int(x)}' for x in curve_labels],
                   bbox_to_anchor=(0.8, -0.6), ncol=1, fontsize=FONT_DICT['fontsize'])


def init_fig():
    fig, ax = plt.subplots()
    return fig, ax


def finalize_fig(fig, ax, im):
    fig.colorbar(im, ax=ax)


colors = [np.array([125, 172, 80]) / 255, np.array([130, 130, 130]) / 255]
plot_type = 'curve'

if __name__ == "__main__":
    test_names = ['/Camargo_levelground', '/walking_knee_moment', '/sun_drop_jump']
    test_names_print = ('Task 1 - Overground Walking', 'Task 2 - Treadmill Walking',
                        'Task 3 - Drop Jump')
    test_folder = '2023_05_21_15_03_25_SSL_COMBINED_f2'
    data_path = RESULTS_PATH + test_folder
    if plot_type == 'curve':
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4.5, 9))
    else:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 3.5), height_ratios=[3, 1])
    metric = 'r2'
    for i_task, test_name in enumerate(test_names):
        results_task = load_da_data(data_path + test_name + '_output' + '.h5')
        result_df = results_to_pd_summary(results_task, 0)
        result_df['PercentOfMasking'] = result_df['MaskPatchNum'] / (128 / result_df['PatchLen'])

        patch_len_list = np.sort(list(set(result_df['PatchLen'])))[:-1]     #        # !!!
        percent_of_masking_list = np.sort(list(set(result_df['PercentOfMasking'])))
        percent_of_masking_list_str = [str(round(i * 100, 1)) + '%' for i in percent_of_masking_list]
        result_mean_map = np.zeros([2, len(patch_len_list), len(percent_of_masking_list)])
        for i_patch, patch_len in enumerate(patch_len_list):
            for i_percent, percent_of_masking in enumerate(percent_of_masking_list):
                data_cond = result_df[(result_df['PatchLen'] == patch_len) & (result_df['PercentOfMasking'] == percent_of_masking) & (result_df['NumGradDeSsl'] == 30000)]
                data_cond_0 = data_cond[~data_cond['LinearProb']&data_cond['UseSsl']]
                result_mean_map[0, i_patch, i_percent] = np.mean(data_cond_0[metric])
                data_cond_2 = data_cond[~data_cond['LinearProb']&~data_cond['UseSsl']]
                result_mean_map[1, i_patch, i_percent] = np.mean(data_cond_2[metric])
        if plot_type == 'curve':
            plot_curve(i_task, result_mean_map, ['0%\n(baseline)'] + percent_of_masking_list_str, patch_len_list)
        else:
            plot_map(i_task, result_mean_map, patch_len_list, percent_of_masking_list_str)
    plt.show()


