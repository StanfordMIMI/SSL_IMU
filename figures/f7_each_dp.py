import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from ssl_main.const import LINE_WIDTH, FONT_DICT, LINE_WIDTH_THICK
from figures.PaperFigures import save_fig, load_da_data, format_axis
from ssl_main.config import RESULTS_PATH
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error as mse


def my_pearson(data_):
    return pearsonr(data_[:128], data_[128:])[0]


def my_r2(data_):
    return r2_score(data_[:128], data_[128:])


def plot_spectrum(results_task):
    # def format_ticks():
    #     ax.set_xlim(10, 50)
    #     ax.set_xticks(range(10, 51, 10))
    #     ax.set_xticklabels(range(10, 51, 10), fontdict=FONT_DICT)
    #     ax.set_xlabel('Frequency (Hz)', fontdict=FONT_DICT)
    #
    #     ax.set_ylabel('vGRF Error (BW)', fontdict=FONT_DICT)
    #     if i_da == 0 or i_da == 1:
    #         ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1])
    #         ax.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1.], fontdict=FONT_DICT)
    #     elif i_da == 2:
    #         ax.set_yticks([0., 0.4, 0.8, 1.2, 1.6, 2.])
    #         ax.set_yticklabels([0., 0.4, 0.8, 1.2, 1.6, 2.], fontdict=FONT_DICT)

    data_0, data_1 = [np.concatenate(list(value_.values()), axis=0) for value_ in results_task.values()]
    assert data_0.shape == data_1.shape
    assert np.max(np.abs(data_0[:, 0] - data_1[:, 0])) < 1e-10
    result_0 = np.apply_along_axis(my_pearson, -1, data_0.reshape([data_0.shape[0], -1]))
    result_1 = np.apply_along_axis(my_pearson, -1, data_1.reshape([data_0.shape[0], -1]))
    # max_val = np.max([np.max(result_0), np.max(result_1)])
    # min_val = np.min([np.min(result_0), np.min(result_1)])
    max_val = np.partition(np.concatenate([result_0, result_1]).flatten(), -1)[-1]
    min_val = np.partition(np.concatenate([result_0, result_1]).flatten(), 3)[3]

    ax = plt.gca()
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.plot(result_0, result_1, '.',)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=LINE_WIDTH_THICK)
    ax.set_xlabel('Correlation - Supervised', fontdict=FONT_DICT)
    ax.set_ylabel('Correlation - SSL', fontdict=FONT_DICT)
    plt.title(test_names_print[i_da], fontdict=FONT_DICT)

    # format_ticks()
    format_axis(ax)
    # return lines_handle, fill_handle


def finalize_fig(lines_handle, fill_handle):
    plt.tight_layout(rect=[0., 0., 1., .92], w_pad=2)
    plt.legend([(lines_handle[0], fill_handle[0]), (lines_handle[1], fill_handle[1])],
               ['Baseline', 'Self-Supervised Learning'], fontsize=FONT_DICT['fontsize'], ncol=2,
               frameon=False, bbox_to_anchor=(0.7, 1.28))


colors = [np.array(x) / 255 for x in [[110, 110, 110], [3, 136, 170]]]


if __name__ == "__main__":
    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]     # walking_knee_moment
    test_folder = '2023_07_17_11_12_25_SSL_AMASS'
    test_names_print = ('Task 1 - Overground Walking', 'Task 2 - Treadmill Walking', 'Task 3 - Drop Landing')
    data_path = RESULTS_PATH + test_folder + '/'
    plt.figure(figsize=(12, 4))

    for i_da, da_name in enumerate(da_names):
        plt.subplot(1, 3, i_da + 1)
        results_task = load_da_data(data_path + da_name + '.h5')
        results_task_ = {key_: value_ for key_, value_ in results_task.items()
                         if 'PatchLen_8' in key_ and 'MaskPatchNum_6' in key_ and 'LinearProb_False' in key_ and
                         'ratio_1' in key_}
        plot_spectrum(results_task_)
        # plot_example_windows(axes[1, i_da], results_task_)

    # finalize_fig(lines_handle, fill_handle)
    # save_fig('f6_fft')
    plt.tight_layout()
    plt.show()















