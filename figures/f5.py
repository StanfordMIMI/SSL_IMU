from scipy.stats import ttest_rel
from ssl_main.const import LINE_WIDTH, FONT_DICT
from figures.PaperFigures import save_fig, load_da_data, results_dict_to_pd, format_axis
from matplotlib.patches import Patch
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
    fig = plt.figure(figsize=(6, 4))
    return fig


def draw_box(result_df, i_da, i_test):
    with_ssl_ = result_df[result_df['UseSsl'] == True][metric].values
    no_ssl_ = result_df[result_df['UseSsl'] == False][metric].values

    data_ = with_ssl_ - no_ssl_
    x_loc = i_da * 4 + i_test
    bar = plt.bar(x_loc, np.mean(data_), width=0.7, color=colors[i_test], edgecolor='none', linewidth=LINE_WIDTH)
    lolims = np.mean(data_) > 0
    uplims = not lolims
    ebar, caplines, barlinecols = plt.errorbar([x_loc], np.mean(data_), np.std(data_),
                                               capsize=0, ecolor='black', fmt='none',
                                               uplims=uplims, lolims=lolims, elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines, 10)
    return bar


def finalize_fig(bars):
    def format_ticks():
        ax.set_ylabel(r'$\Delta R^2$', fontdict=FONT_DICT)
        x_range = (-1, 11)
        ax.set_xlim(x_range[0], x_range[1])
        ax.tick_params(bottom=False)
        ax.set_xticks(np.arange(1, x_range[1], 4))
        ax.set_xticklabels(['Task 1 -\nOverground Walking', 'Task 2 -\nTreadmill Walking',
                            'Task 3 -\nDrop Jump'], fontdict=FONT_DICT)
        # ax.set_ylim(ylim_down, ylim_up)
        # ax.set_yticks(ticks)     # [.4, .5, .6, .7, .8, .9, 1., 1.1]
        # ax.set_yticklabels(ticks, fontdict=FONT_DICT)

    ax = plt.gca()
    ylim_ori = ax.get_ylim()
    format_ticks()
    format_axis()
    # legend_elements = [Patch(facecolor=colors[0], label='Color Patch'),
    #                    Patch(facecolor=colors[1], label='Color Patch')]
    plt.legend(bars[:3], ['MoVi', 'AMASS', 'Combined'], bbox_to_anchor=(0.65, 1.22 - 0.2 * (ylim_ori[0])),
               ncol=1, fontsize=FONT_DICT['fontsize'], frameon=False)
    plt.tight_layout(rect=[0., 0., 1., 1.01])
    plt.show()


# colors = [np.array(x) / 255 for x in [[255, 166, 0], [0, 98, 204], [0, 138, 113]]]
colors = [np.array(x) / 255 for x in [[3, 166, 200], [2, 83, 100], [173, 201, 230]]]

if __name__ == "__main__":
    metric = 'r2'
    patch_len = 8
    mask_patch_num = 8

    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]
    test_folders = ['2023_05_17_08_23_49_SSL_MOVI', '2023_05_17_08_23_03_SSL_AMASS', '2023_05_17_00_08_22_SSL_KAM']

    rc('font', family='Arial')
    bars = []

    for i_da, da_name in enumerate(da_names):
        for i_test, test_folder in enumerate(test_folders):
            data_path = RESULTS_PATH + test_folder + '/'
            results_task = load_da_data(data_path + da_name + '.h5')
            results_task = {key_: value_ for key_, value_ in results_task.items()
                            if f'PatchLen_{patch_len}' in key_ and
                            f'MaskPatchNum_{mask_patch_num}' in key_ and
                            'LinearProb_False' in key_}
            result_df = results_to_pd_summary(results_task, 0)
            bars.append(draw_box(result_df, i_da, i_test))

    finalize_fig(bars)



