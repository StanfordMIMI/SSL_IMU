from ssl_main.const import LINE_WIDTH, FONT_DICT
from figures.PaperFigures import save_fig, load_da_data, results_dict_to_pd, format_axis
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from figures.PaperFigures import save_fig, load_da_data, results_to_pd_summary_only_peaks, results_to_pd_summary,\
    format_errorbar_cap
from ssl_main.config import RESULTS_PATH
from scipy.stats import tukey_hsd, f_oneway
import matplotlib.lines as lines


def draw_box(result_df, i_da, i_test):
    # with_ssl_ = result_df[result_df['UseSsl'] == True][metric].values
    # no_ssl_ = result_df[result_df['UseSsl'] == False][metric].values

    data_ = result_df[metric].values
    x_loc = i_da * 5 + i_test
    bar = plt.bar(x_loc, np.mean(data_), width=0.7, color=colors[i_test], edgecolor='none', linewidth=LINE_WIDTH)
    lolims = np.mean(data_) > 0
    uplims = not lolims
    ebar, caplines, barlinecols = plt.errorbar([x_loc], np.mean(data_), np.std(data_),
                                               capsize=0, ecolor='black', fmt='none',
                                               uplims=uplims, lolims=lolims, elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines, 10)
    return bar


def draw_error_bars(values_for_anova):
    mean_ = [np.mean(data_) for data_ in values_for_anova]
    std_ = [np.std(data_) for data_ in values_for_anova]
    y_max = np.max([a + b for a, b in zip(mean_, std_)])
    for i_ in range(3):
        top_line = y_max + (3 - i_) * 0.075
        diff_line_x = [i_ + 5, i_ + 5, 8, 8]
        diff_line_y = [mean_[i_] + std_[i_] + 0.05, top_line, top_line, mean_[3] + std_[3] + 0.05]
        plt.plot(diff_line_x, diff_line_y, 'black', linewidth=LINE_WIDTH)
        plt.text((i_ + 13) / 2 - 0.15, top_line - 0.01, '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=20)



def finalize_fig(bars):
    def format_ticks():
        # ax.set_ylabel(r'$\Delta R^2$', fontdict=FONT_DICT)
        ax.set_ylabel('Correlation Coefficients - vGRF Peak', fontdict=FONT_DICT)
        # ax.set_ylabel('Correlation Coefficients - vGRF profile', fontdict=FONT_DICT)
        x_range = (-1, 14)
        ax.set_xlim(x_range[0], x_range[1])
        ax.tick_params(bottom=False)
        ax.set_xticks(np.arange(1.5, x_range[1], 5))
        ax.set_xticklabels(['Task 1 -\nOverground Walking', 'Task 2 -\nTreadmill Walking',
                            'Task 3 -\nDrop Landing'], fontdict=FONT_DICT)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2])
        ax.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2], fontdict=FONT_DICT)

    ax = plt.gca()
    ylim_ori = ax.get_ylim()
    format_ticks()
    format_axis()
    plt.tight_layout(rect=[0., 0., 1., 1.])
    plt.legend(bars[:4], ['MoVi', 'AMASS', 'Combined', 'Baseline'], bbox_to_anchor=(0.7, 1. - 0.2 * (ylim_ori[0])),
               ncol=1, fontsize=FONT_DICT['fontsize'], frameon=False)
    save_fig('f5_bars_peak')
    plt.show()


colors = [np.array(x) / 255 for x in [[76, 181, 210], [3, 136, 170], [2, 83, 100], [180, 180, 180]]]


if __name__ == "__main__":
    metric = 'correlation'
    patch_len = 1
    mask_patch_num = 16

    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]
    da_sign_of_peak = [1, -1, 1]
    test_folders = ['2023_08_25_12_05_00_data_ratio_MOVI', '2023_08_25_12_05_33_data_ratio_amass', '2023_08_25_09_11_52_data_ratio_combined', 'baseline']

    rc('font', family='Arial')
    bars = []
    fig = plt.figure()
    rmse_value = []
    for i_da, (da_name, sign_of_peak) in enumerate(zip(da_names, da_sign_of_peak)):
        values_for_anova = []
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
                            'LinearProb_False' in key_ and
                            f'UseSsl_{use_ssl}' in key_ and
                            'ratio_1' in key_}
            print(da_name, sum([data_.shape[0] for sub_, data_ in list(results_task.values())[0].items()]))
            result_df = results_to_pd_summary_only_peaks(results_task, 0, sign_of_peak)
            bars.append(draw_box(result_df, i_da, i_test))
            values_for_anova.append(result_df[metric].values)

            # record rmse of models pretrained on AMASS
            if i_test == 1:
                rmse_value.append(result_df['rmse'].values * 9.81)
        if i_da == 1:
            draw_error_bars(values_for_anova)
        values_for_anova_np = np.array(values_for_anova)
        print(f'{da_name}: {f_oneway(*values_for_anova)}')
        print(f'{tukey_hsd(*values_for_anova)}')

    l1 = lines.Line2D([0.405, 0.405], [0.01, 0.96], linestyle='-', transform=fig.transFigure, color=[0.75]*3)
    l2 = lines.Line2D([0.695, 0.695], [0.01, 0.96], linestyle='-', transform=fig.transFigure, color=[0.75]*3)
    fig.lines.extend([l1, l2])
    finalize_fig(bars)

    [print(str(np.round(np.mean(data_), 2)) + ' \pm ' + str(np.round(np.std(data_), 2))) for data_ in rmse_value]


