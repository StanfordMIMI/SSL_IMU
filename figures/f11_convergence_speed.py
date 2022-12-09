import numpy as np
from ssl_main.const import FONT_DICT, LINE_WIDTH_THICK
import matplotlib.pyplot as plt
from matplotlib import rc
from figures.PaperFigures import save_fig, load_da_data, format_axis, \
    results_dict_to_pd_convergence_speed


def init_fig():
    fig = plt.figure(figsize=(6, 4))
    return fig


def draw_line(line_config):
    def format_ticks():
        ax = plt.gca()
        ax.set_xlabel('Number of Gradient Descents', fontdict=FONT_DICT)
        ax.set_ylabel('Correlation Coefficient of vGRF Estimation', fontdict=FONT_DICT)
        # ax.set_xlim(line_config['data'][0, 0], 1)
        # ax.tick_params(bottom=False)
        ax.set_xscale('log')
        # ax.set_xticks([.01, .033, .1, .333, 1])
        # ax.set_xticklabels(['1.0%', '3.3%', '10%', '33.3%', '100%'], fontdict=FONT_DICT)
        #
        ylim = 1.5
        ax.set_ylim(-0.4, ylim)
        # ax.set_yticks([.4, .6, .8, 1., 1.2])     # [.4, .5, .6, .7, .8, .9, 1., 1.1]
        # ax.set_yticklabels([.4, .6, .8, 1., 1.2], fontdict=FONT_DICT)

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
    plt.tight_layout(rect=[0., 0., 1., 1.])
    plt.legend(bbox_to_anchor=(0.88, 1.1), ncol=1, fontsize=FONT_DICT['fontsize'], frameon=False, handlelength=4)
    save_fig('f9')


data_path = 'D:\ssl_training_results\\2022-11-10 22_22_19'      # 2022-11-11 00_23_14   2022-11-10 22_22_19
test_name = '\\Carmargo_peak_fy'     # hw_running_VALR
rc('font', family='Arial')
colors = [np.array([125, 172, 80]) / 255, np.array([130, 130, 130]) / 255]

if __name__ == "__main__":
    metric = 'correlation'
    results_task = load_da_data(data_path + test_name + '.h5')

    result_df = results_dict_to_pd_convergence_speed(results_task)

    amount_list = list(set(result_df['i_optimize']))
    amount_list.sort()
    # amount_list = amount_list[7:]

    line_0_data = result_df[~result_df['only_linear']&result_df['use_ssl']].sort_values(by=['i_optimize'])[['i_optimize', metric]]
    line_config_0 = {'color': colors[0], 'style': '-', 'label': 'Self-Supervised Encoders - Fine-tuning', 'data': line_0_data.values}
    line_1_data = result_df[result_df['only_linear']&result_df['use_ssl']].sort_values(by=['i_optimize'])[['i_optimize', metric]]
    line_config_1 = {'color': colors[0], 'style': '--', 'label': 'Self-Supervised Encoders - Linear', 'data': line_1_data.values}
    line_2_data = result_df[~result_df['only_linear']&~result_df['use_ssl']].sort_values(by=['i_optimize'])[['i_optimize', metric]]
    line_config_2 = {'color': colors[1], 'style': '-', 'label': 'Initial Encoders - Fine-tuning', 'data': line_2_data.values}
    line_3_data = result_df[result_df['only_linear']&~result_df['use_ssl']].sort_values(by=['i_optimize'])[['i_optimize', metric]]
    line_config_3 = {'color': colors[1], 'style': '--', 'label': 'Initial Encoders - Linear', 'data': line_3_data.values}

    fig = init_fig()
    draw_line(line_config_0)
    draw_line(line_config_1)
    draw_line(line_config_2)
    draw_line(line_config_3)
    finalize_fig()
    plt.show()

