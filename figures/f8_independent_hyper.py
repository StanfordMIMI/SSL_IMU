import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from figures.PaperFigures import save_fig, load_da_data, results_dict_to_pd_profiles, format_axis
from ssl_main.config import RESULTS_PATH
from ssl_main.const import FONT_DICT, LINE_WIDTH_THICK


def init_fig():
    fig = plt.figure()
    return fig


def draw_line(line_config, amount_list):
    def format_ticks():
        ax = plt.gca()
        ax.set_xlabel(target_param, fontdict=FONT_DICT)
        ax.set_ylabel('Correlation Coefficient of KFM Estimation', fontdict=FONT_DICT)
        # ax.set_xlim(line_config['data'][0, 0], 1)
        ax.tick_params(bottom=False)
        ax.set_xscale('log')
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        ylim = 1.4
        ax.set_ylim(0., ylim)
        ax.set_yticks([0., .2, .4, .6, .8, 1.])     # [.4, .5, .6, .7, .8, .9, 1., 1.1]
        ax.set_yticklabels([.0, .2, .4, .6, .8, 1.], fontdict=FONT_DICT)
    rc('font', family='Arial')
    mean_, std_ = [], []
    for amount in amount_list:
        data_ = line_config['data'][line_config['data'][:, 0] == amount]
        mean_.append(np.mean(data_[:, 1]))
        std_.append(np.std(data_[:, 1]))
    plt.plot(amount_list, mean_, line_config['style'], linewidth=LINE_WIDTH_THICK, markersize=10,
             color=line_config['color'], label=line_config['label'])
    format_ticks()


def finalize_fig():
    format_axis()
    plt.legend(bbox_to_anchor=(0.88, 1.1), ncol=1, fontsize=FONT_DICT['fontsize'], frameon=False, handlelength=4)
    plt.tight_layout(rect=[0., 0., 1., 1.])


test_name = '/walking_knee_moment_output'     # hw_running_VALR   walking_knee_moment_output  Camargo_output
colors = [np.array([125, 172, 80]) / 255, np.array([130, 130, 130]) / 255]


if __name__ == "__main__":
    data_path = RESULTS_PATH + 't03_test_camargo'
    target_param = 'NumGradDeSsl'
    metric = 'r2'
    results_task = load_da_data(data_path + test_name + '.h5')
    result_df = results_dict_to_pd_profiles(results_task, 1)

    param_set = np.sort(list(set(result_df[target_param])))

    data_cond = result_df[(result_df['PatchLen'] == 8) & (result_df['MaskPatchNum'] == 6)]
    data_cond_0 = data_cond[~data_cond['LinearProb']&data_cond['UseSsl']][[target_param, metric]]
    line_config_0 = {'color': colors[0], 'style': '.-', 'label': 'Self-Supervised Encoders - Fine-tuning', 'data': data_cond_0.values}
    data_cond_1 = data_cond[data_cond['LinearProb']&data_cond['UseSsl']][[target_param, metric]]
    line_config_1 = {'color': colors[0], 'style': '--.', 'label': 'Self-Supervised Encoders - Linear', 'data': data_cond_1.values}
    data_cond_2 = data_cond[~data_cond['LinearProb']&~data_cond['UseSsl']][[target_param, metric]]
    line_config_2 = {'color': colors[1], 'style': '.-', 'label': 'Randomly Initialized Encoders - Fine-tuning', 'data': data_cond_2.values}
    data_cond_3 = data_cond[data_cond['LinearProb']&~data_cond['UseSsl']][[target_param, metric]]
    line_config_3 = {'color': colors[1], 'style': '--.', 'label': 'Randomly Initialized Encoders - Linear', 'data': data_cond_3.values}

    fig = init_fig()
    draw_line(line_config_0, param_set)
    draw_line(line_config_1, param_set)
    draw_line(line_config_2, param_set)
    draw_line(line_config_3, param_set)
    finalize_fig()
    plt.show()


