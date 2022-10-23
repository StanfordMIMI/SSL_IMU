import h5py
import json
import pandas as pd
import numpy as np
from const import LINE_WIDTH, FONT_DICT
from figures.PaperFigures import save_fig, load_da_data, results_dict_to_pd
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import pearsonr, spearmanr


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

        ax.set_ylim(0, 4)
        ax.set_yticks([0., .2, .4, .6, .8, 1.])
        ax.set_yticklabels([0., .2, .4, .6, .8, 1.], fontdict=FONT_DICT)

    rc('font', family='Arial')
    amount_list = list(set(line_config['data'][:, 0]))
    amount_list.sort()
    mean_, std_ = [], []
    for amount in amount_list:
        data_ = line_config['data'][line_config['data'][:, 0] == amount]
        mean_.append(np.mean(data_[:, 1]))
        std_.append(np.std(data_[:, 1]))
    plt.plot(amount_list, mean_, line_config['style'],
             color=line_config['color'], label=line_config['label'])
    format_ticks()


def finalize_f10():
    plt.tight_layout(rect=[0., 0., 1., 1.])
    plt.legend(frameon=False, fontsize=FONT_DICT['fontsize'])


data_path = 'D:\ssl_training_results\\2022-10-23 23_17_08'
test_name = '\\walking_knee_moment_KFM'
if __name__ == "__main__":
    # !!! 加一组线，KAM dataset for SSL
    metric = 'rmse'
    results_task = load_da_data(data_path + test_name + '.h5')
    # for i_sub in range(11, 16):
    #     plt.figure()
    #     data_ = results_task['linear_protocol_False, use_ssl_True, ratio_0.01']['sub_'+ str(i_sub)]
    #     plt.plot(data_[:, 0], data_[:, 1], '.')
    #     plt.plot([0, 8], [0, 8], 'black')
    # plt.show()

    result_df = results_dict_to_pd(results_task)

    line_0_data = result_df[~result_df['only_linear']&result_df['use_ssl']].sort_values(by=['ratio'])[['ratio', metric]]
    line_config_0 = {'color': 'C2', 'style': '-', 'label': 'SSL + all_param', 'data': line_0_data.values}
    line_1_data = result_df[result_df['only_linear']&result_df['use_ssl']].sort_values(by=['ratio'])[['ratio', metric]]
    line_config_1 = {'color': 'C1', 'style': '-', 'label': 'SSL + one_linear_layer', 'data': line_1_data.values}
    line_2_data = result_df[~result_df['only_linear']&~result_df['use_ssl']].sort_values(by=['ratio'])[['ratio', metric]]
    line_config_2 = {'color': 'C2', 'style': '--', 'label': 'no_SSL + all_param', 'data': line_2_data.values}
    line_3_data = result_df[result_df['only_linear']&~result_df['use_ssl']].sort_values(by=['ratio'])[['ratio', metric]]
    line_config_3 = {'color': 'C1', 'style': '--', 'label': 'no_SSL + one_linear_layer', 'data': line_3_data.values}

    fig = init_f10()
    draw_line(line_config_0)
    draw_line(line_config_1)
    draw_line(line_config_2)
    draw_line(line_config_3)
    finalize_f10()
    plt.show()

