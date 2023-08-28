import os, sys
from ssl_main.const import FONT_DICT, LINE_WIDTH_THICK
import copy
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from figures.PaperFigures import save_fig, load_da_data, results_to_pd_summary, format_axis
from ssl_main.config import RESULTS_PATH


def plot_map(i_da, data_all, x_ticks, y_ticks):
    min_val, max_val = np.min(data_all), np.max(data_all)
    min_val, max_val = min_val - 0.5 * (max_val - min_val), max_val + 0.5 * (max_val - min_val)

    cmap = colormaps.get_cmap('RdBu')

    ax = axes[0, i_da]
    data_ssl = data_all[0].T
    im = ax.imshow(data_ssl, interpolation='nearest', cmap=cmap, vmax=max_val, vmin=min_val)
    # Add text to the matrix to display the values
    for i in range(len(data_ssl)):
        for j in range(len(data_ssl[i])):
            ax.text(j, i, round(data_ssl[i, j], 2), ha='center', va='center', color='black')
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(y_ticks)))
    if i_da == 0:
        ax.set_yticklabels(y_ticks)
        ax.tick_params(which='both', top=False,  bottom=False)
        ax.set_ylabel('Percentage of Masking', labelpad=15)
    else:
        ax.set_yticklabels([])
        ax.tick_params(which='both', left=False,  right=False, top=False,  bottom=False)
    plt.setp(ax.get_xticklabels(), ha='center')
    ax.title.set_text(test_names_print[i_da])

    ax = axes[1, i_da]
    data_bl = data_all[1, :, 0:1].T
    im = ax.imshow(data_bl, interpolation='nearest', cmap=cmap, vmax=max_val, vmin=min_val)
    # Add text to the matrix to display the values
    for i in range(np.size(data_bl)):
        ax.text(i, 0, round(data_bl[0, i], 2), ha='center', va='center', color='black')

    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels(x_ticks)
    ax.set_yticks(np.arange(1))
    ax.set_xlabel('Patch Length')
    if i_da == 0:
        ax.set_yticklabels(['Baseline'])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', left=False,  right=False)

    plt.tight_layout(rect=[0.1, 0., 1., 1.], w_pad=0.2, h_pad=-0.1)


def plot_curve(i_da, data_all, x_ticks, curve_labels):
    ax = plt.gca()
    data_combined = np.concatenate([data_all[1, :, :1], data_all[0]], axis=1).T
    data_combined = np.flip(data_combined, axis=1)
    curve_labels = reversed(curve_labels)
    for i_col in range(data_combined.shape[1]):
        ax.plot([float(ele.split('\n')[0]) for ele in x_ticks],
                data_combined[:, i_col], color=colors[i_col], marker="o", markersize=4, linewidth=LINE_WIDTH_THICK)
    ax.set_xlabel('Percentage of Masking (%)', fontdict=FONT_DICT, labelpad=5)
    ax.set_xticks([float(ele.split('\n')[0]) for ele in x_ticks[:1]+x_ticks[2:]])
    ax.set_xlim(-1, float(x_ticks[-1]) + 1)
    ax.set_xticklabels(x_ticks[:1]+x_ticks[2:], fontdict=FONT_DICT)
    ax.set_ylabel('Correlation Coefficients - vGRF Profile', fontdict=FONT_DICT)

    if i_da == 0:
        ax.set_yticks([0.92, 0.93, 0.94, 0.95])
        ax.set_yticklabels([0.92, 0.93, 0.94, 0.95], fontdict=FONT_DICT)
        ax.set_ylim(0.92, 0.95)
    elif i_da == 1:
        ax.set_yticks([0.94, 0.95, 0.96, 0.97, 0.98])
        ax.set_yticklabels([0.94, 0.95, 0.96, 0.97, 0.98], fontdict=FONT_DICT)
        ax.set_ylim(0.938, 0.98)
    elif i_da == 2:
        ax.set_yticks([0.88, 0.9, 0.92, 0.94])
        ax.set_yticklabels([0.88, 0.9, 0.92, 0.94], fontdict=FONT_DICT)
        ax.set_ylim(0.88, 0.94)

    plt.title(test_names_print[i_da], fontdict=FONT_DICT, pad=15)
    plt.tight_layout(rect=[0., 0., 1., 0.87], h_pad=2)
    if i_da == 2:
        plt.legend([f'Patch Length = {int(x)}' for x in curve_labels], frameon=False,
                   bbox_to_anchor=(0.8, 1.4), ncol=4, fontsize=FONT_DICT['fontsize'])
    format_axis()


def init_fig():
    fig, ax = plt.subplots()
    return fig, ax


def finalize_fig(fig, ax, im):
    fig.colorbar(im, ax=ax)


colors = [np.array(list_) / 255 for list_ in [(84, 39, 143), (117, 107, 177), (158, 154, 200), (203, 201, 226)]]
plot_type = 'curve'

if __name__ == "__main__":
    test_names = ['/Camargo_levelground', '/walking_knee_moment', '/sun_drop_jump']
    test_names_print = ('Task 1 - Overground Walking', 'Task 2 - Treadmill Walking',
                        'Task 3 - Drop Jump')
    test_folder = '2023_08_24_16_57_19_mask_ratio'
    data_path = RESULTS_PATH + test_folder
    if plot_type == 'curve':
        fig = plt.figure(figsize=(12, 4.5))
    else:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 3.5), height_ratios=[3, 1])
    metric = 'correlation'
    for i_da, test_name in enumerate(test_names):
        plt.subplot(1, 3, i_da + 1)
        results_task = load_da_data(data_path + test_name + '_output' + '.h5')
        result_df = results_to_pd_summary(results_task, 0)
        result_df['PercentOfMasking'] = result_df['MaskPatchNum'] / (128 / result_df['PatchLen'])

        patch_len_list = np.sort(list(set(result_df['PatchLen'])))     #        # !!!
        percent_of_masking_list = np.sort(list(set(result_df['PercentOfMasking'])))
        percent_of_masking_list_str = [str(round(i * 100, 1)) + '' for i in percent_of_masking_list]
        result_mean_map = np.zeros([2, len(patch_len_list), len(percent_of_masking_list)])
        for i_patch, patch_len in enumerate(patch_len_list):
            for i_percent, percent_of_masking in enumerate(percent_of_masking_list):
                data_cond = result_df[(result_df['PatchLen'] == patch_len) & (result_df['PercentOfMasking'] == percent_of_masking)]
                data_cond_0 = data_cond[~data_cond['LinearProb']&data_cond['UseSsl']]
                result_mean_map[0, i_patch, i_percent] = np.mean(data_cond_0[metric])
                data_cond_2 = data_cond[~data_cond['LinearProb']&~data_cond['UseSsl']]
                result_mean_map[1, i_patch, i_percent] = np.mean(data_cond_2[metric])
        if plot_type == 'curve':
            plot_curve(i_da, result_mean_map, ['0\n(baseline)'] + percent_of_masking_list_str, patch_len_list)
        else:
            plot_map(i_da, result_mean_map, patch_len_list, percent_of_masking_list_str)
        plt.grid(True, linewidth=1, alpha=0.5)
    save_fig('f3')
    plt.show()


