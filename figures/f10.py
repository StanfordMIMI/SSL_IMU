
import h5py
import json
import pandas as pd
import numpy as np
from const import LINE_WIDTH, FONT_DICT
from figures.PaperFigures import save_fig
from figures.PaperFigures import get_peak_of_each_gait_cycle, format_axis, get_mean_gait_cycle_then_find_peak
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.lines as lines
from scipy.stats import ttest_rel
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


def init_f10():
    rc('font', family='Arial')
    fig = plt.figure(figsize=(6, 4))
    return fig


def draw_line(line_config):
    def format_ticks():
        ax = plt.gca()
        ax.set_xlabel('Percentage of data used for training', fontdict=FONT_DICT)
        ax.set_ylabel(r'$\bf{Correlation}$ between KAM_true and KAM_predicted', fontdict=FONT_DICT)
        ax.set_xscale('log')
        ax.set_xlim(line_config['data'][0, 0], 1)
        ax.set_xticks([.01, 0.033, .1, 0.33, 1.])
        ax.set_xticklabels(['1.0%', '3.3%', '10%', '33.3%', '100%'], fontdict=FONT_DICT)

        ax.set_ylim(0, 1)
        ax.set_yticks([0., .2, .4, .6, .8, 1.])
        ax.set_yticklabels([0., .2, .4, .6, .8, 1.], fontdict=FONT_DICT)

    rc('font', family='Arial')
    plt.plot(line_config['data'][:, 0], line_config['data'][:, 1], line_config['style'],
             color=line_config['color'], label=line_config['label'])
    format_ticks()


def finalize_f10():
    plt.tight_layout(rect=[0., 0., 1., 1.])
    plt.legend(frameon=False, fontsize=FONT_DICT['fontsize'])
    # plt.gca().add_artist(legend_0)
    # save_fig('f10', 600)


if __name__ == "__main__":
    data_path = 'D:\ssl_training_results\\2022-10-01 08_57_39'      # 2022-10-02 22_43_13
    with h5py.File(data_path + '/results.h5', 'r') as hf:
        data_all_tests = {test_name: test_results[:] for test_name, test_results in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

    result_list = []
    for test_name, test_results in data_all_tests.items():
        only_linear = test_name.split('linear_protocol_')[1].split(',')[0] == 'True'
        use_ssl = test_name.split('use_ssl_')[1].split(',')[0] == 'True'
        ratio = float(test_name.split('ratio_')[1])
        # metric = np.sqrt(mse(test_results[:, 0], test_results[:, 1]))
        # metric = r2_score(test_results[:, 0], test_results[:, 1])
        metric = spearmanr(test_results[:, 0], test_results[:, 1])[0]
        result_list.append([only_linear, use_ssl, ratio, metric])

    result_df = pd.DataFrame(result_list, columns=['only_linear', 'use_ssl', 'ratio', 'metric'])

    line_0_data = result_df[~result_df['only_linear']&result_df['use_ssl']].sort_values(by=['ratio'])[['ratio', 'metric']]
    line_config_0 = {'color': 'C2', 'style': '-', 'label': 'SSL + all_param', 'data': line_0_data.values}
    line_1_data = result_df[result_df['only_linear']&result_df['use_ssl']].sort_values(by=['ratio'])[['ratio', 'metric']]
    line_config_1 = {'color': 'C1', 'style': '-', 'label': 'SSL + one_linear_layer', 'data': line_1_data.values}
    line_2_data = result_df[~result_df['only_linear']&~result_df['use_ssl']].sort_values(by=['ratio'])[['ratio', 'metric']]
    line_config_2 = {'color': 'C2', 'style': '--', 'label': 'no_SSL + all_param', 'data': line_2_data.values}
    line_3_data = result_df[result_df['only_linear']&~result_df['use_ssl']].sort_values(by=['ratio'])[['ratio', 'metric']]
    line_config_3 = {'color': 'C1', 'style': '--', 'label': 'no_SSL + one_linear_layer', 'data': line_3_data.values}

    fig = init_f10()
    draw_line(line_config_0)
    draw_line(line_config_1)
    draw_line(line_config_2)
    draw_line(line_config_3)
    finalize_f10()
    plt.show()

