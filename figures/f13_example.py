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


def plot_example(results_task):
    def format_ticks():
        pass
        ax.set_xlim(0, 1.3)
        ax.set_xticks([0, 0.3, 0.6, 0.9, 1.2])
        ax.set_xticklabels([0, 0.3, 0.6, 0.9, 1.2], fontdict=FONT_DICT)
        ax.set_xlabel('Time (s)', fontdict=FONT_DICT)

        ax.set_ylabel('vGRF (N/kg)', fontdict=FONT_DICT)
        # if i_da == 0 or i_da == 1:
        #     ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1])
        #     ax.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1.], fontdict=FONT_DICT)
        # elif i_da == 2:
        #     ax.set_yticks([0., 0.4, 0.8, 1.2, 1.6, 2.])
        #     ax.set_yticklabels([0., 0.4, 0.8, 1.2, 1.6, 2.], fontdict=FONT_DICT)

    data_key_bl, data_key_ssl = list(results_task.values())
    ax = plt.gca()
    data_combind_bl = np.concatenate(list(data_key_bl.values()), axis=0) * 9.81
    data_combind_ssl = np.concatenate(list(data_key_ssl.values()), axis=0) * 9.81
    if - np.min(data_combind_bl) > np.max(data_combind_bl):
        data_combind_bl = - data_combind_bl
        data_combind_ssl = - data_combind_ssl
    data_true = data_combind_bl[:, 0, :]
    data_pred_bl = data_combind_bl[:, 1, :]
    data_pred_ssl = data_combind_ssl[:, 1, :]
    rmse = np.sqrt(np.mean((data_true - data_pred_ssl) ** 2, axis=1))
    median_index = np.argsort(rmse)[len(rmse)//2]
    ax.plot(np.arange(0, 1.279, 0.01), data_true[median_index, :], color=colors[0], linewidth=LINE_WIDTH*2, label='Force Plate Measurements')
    ax.plot(np.arange(0, 1.279, 0.01), data_pred_bl[median_index, :], color=colors[1], linewidth=LINE_WIDTH, label='Baseline Model Predictions')
    ax.plot(np.arange(0, 1.279, 0.01), data_pred_ssl[median_index, :], color=colors[2], linewidth=LINE_WIDTH, label='SSL Model Predictions')
    plt.title(test_names_print[i_da], fontdict=FONT_DICT)
    format_ticks()


def finalize_fig():
    plt.tight_layout(rect=[0., 0., 1., .9], w_pad=2)
    plt.legend(fontsize=FONT_DICT['fontsize'], ncol=3, frameon=False, bbox_to_anchor=(0.2, 1.3))


colors = [np.array(x) / 255 for x in [[200, 200, 200], [110, 110, 110], [3, 136, 170]]]
test_names_print = ('Task 1 - Overground Walking', 'Task 2 - Treadmill Walking', 'Task 3 - Drop Landing')


if __name__ == "__main__":
    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]
    test_folder = '2023_09_20_20_19_44_AMASS'
    test_names_print = ('Task 1 - Overground Walking', 'Task 2 - Treadmill Walking', 'Task 3 - Drop Landing')
    data_path = RESULTS_PATH + test_folder + '/'
    plt.figure(figsize=(12, 4))

    for i_da, da_name in enumerate(da_names):
        plt.subplot(1, 3, i_da + 1)
        results_task, results_columns = load_da_data(data_path + da_name + '.h5')
        results_task_ = {key_: value_ for key_, value_ in results_task.items()
                         if 'PatchLen_1' in key_ and 'MaskPatchNum_16' in key_ and 'LinearProb_False' in key_ and
                         'ratio_1' in key_}
        plot_example(results_task_)

    finalize_fig()
    plt.show()
    save_fig('f13_example')















