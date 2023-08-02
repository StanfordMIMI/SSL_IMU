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
    fig = plt.figure()
    return fig


def draw_line(line_config, amount_list):
    def format_ticks():
        ax = plt.gca()
        ax.set_xlabel('Size of Training Set', fontdict=FONT_DICT)
        ax.set_ylabel('Correlation Coefficients - vGRF Peak', fontdict=FONT_DICT)
        ax.tick_params(bottom=False)
        ax.set_xscale('log')
        # ax.set_ylim(100, 10000)
        x_ticks = [10**x for x in range(2, 5)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontdict=FONT_DICT)

        ax.set_ylim(0., 1.)
        ax.set_yticks([.0, 0.2, 0.4, 0.6, 0.8, 1.])
        ax.set_yticklabels([.0, 0.2, 0.4, 0.6, 0.8, 1.], fontdict=FONT_DICT)
        # ylim = 0.98
        # ax.set_ylim(0.88, ylim)
        # ax.set_yticks([.88, 0.90, 0.92, 0.94, 0.96, 0.98])
        # ax.set_yticklabels([.88, 0.90, 0.92, 0.94, 0.96, 0.98], fontdict=FONT_DICT)
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


def finalize_fig():
    format_axis()
    plt.tight_layout(rect=[0., 0., 1., .84])
    plt.legend(['MoVi', 'AMASS', 'Combined', 'Baseline'], bbox_to_anchor=(0.7, 1.3), ncol=2, fontsize=FONT_DICT['fontsize'], frameon=False, handlelength=4)


colors = [np.array(x) / 255 for x in [[3, 166, 200], [2, 83, 100], [153, 181, 210], [180, 180, 180]]]

if __name__ == "__main__":
    metric = 'correlation'
    # metric = 'r2'
    target_param = 'ratio'
    patch_len = 8
    mask_patch_num = 6

    da_names = [element + '_output' for element in ['walking_knee_moment']]
    test_folders = ['2023_07_17_11_07_10_SSL_MOVI', '2023_07_17_11_12_25_SSL_AMASS', '2023_07_17_11_12_25_SSL_AMASS', 'baseline']

    rc('font', family='Arial')
    bars = []
    fig = init_fig()

    for i_da, da_name in enumerate(da_names):
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

            results_task = {test_name: {'sub_000': np.concatenate(list(test_results.values()), axis=0)} for test_name, test_results in results_task.items()}
            result_df = results_to_pd_summary_only_peaks(results_task, 0, sign_of_peak=-1)
            # result_df = results_to_pd_summary(results_task, 0)

            param_set = list(set(result_df[target_param]))
            param_set.sort()

            data_cond = result_df[result_df['UseSsl'] == use_ssl][[target_param, metric]]
            line_config = {'color': colors[i_test], 'style': '.-', 'label': 'Self-Supervised Models Models', 'data': data_cond.values}
            draw_line(line_config, param_set)
    finalize_fig()
    plt.show()
