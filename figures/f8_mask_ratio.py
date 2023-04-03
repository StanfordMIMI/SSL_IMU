import numpy as np
from ssl_main.const import FONT_DICT, LINE_WIDTH_THICK
import matplotlib.pyplot as plt
from matplotlib import rc
from figures.PaperFigures import save_fig, load_da_data, results_dict_to_pd_profiles, format_axis


def init_fig():
    fig = plt.figure()
    return fig


def draw_line(line_config):
    def format_ticks():
        ax = plt.gca()
        ax.set_xlabel('Percentage of Masking', fontdict=FONT_DICT)
        ax.set_ylabel('Correlation Coefficient of KFM Estimation', fontdict=FONT_DICT)
        # ax.set_xlim(line_config['data'][0, 0], 1)
        ax.tick_params(bottom=False)
        ax.set_xticks([.1, .3, .5, .7])
        ax.set_xticklabels(['10%', '30%', '50%', '70%'], fontdict=FONT_DICT)

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
    save_fig('f9')


# data_path = './results/2023-03-29 23_47_13'
data_path = './results/2023-03-30 10_57_38'
test_name = '/walking_knee_moment_output'     # hw_running_VALR   walking_knee_moment_output  Carmargo_output
rc('font', family='Arial')
colors = [np.array([125, 172, 80]) / 255, np.array([130, 130, 130]) / 255]

if __name__ == "__main__":
    metric = 'correlation'
    results_task = load_da_data(data_path + test_name + '.h5')

    result_df = results_dict_to_pd_profiles(results_task, 1)
    result_df['percent_of_masking'] = result_df['mask_patch_num'] / (128 / result_df['patch_len'])

    result_df = result_df[result_df['patch_len'] == 16]      # !!! remove for all the data

    amount_list = list(set(result_df['percent_of_masking']))
    amount_list.sort()

    line_0_data = result_df[~result_df['only_linear']&result_df['use_ssl']].sort_values(by=['percent_of_masking'])[['percent_of_masking', metric]]
    line_config_0 = {'color': colors[0], 'style': '.-', 'label': 'Self-Supervised Encoders - Fine-tuning', 'data': line_0_data.values}
    line_1_data = result_df[result_df['only_linear']&result_df['use_ssl']].sort_values(by=['percent_of_masking'])[['percent_of_masking', metric]]
    line_config_1 = {'color': colors[0], 'style': '--.', 'label': 'Self-Supervised Encoders - Linear', 'data': line_1_data.values}
    line_2_data = result_df[~result_df['only_linear']&~result_df['use_ssl']].sort_values(by=['percent_of_masking'])[['percent_of_masking', metric]]
    line_config_2 = {'color': colors[1], 'style': '.-', 'label': 'Initial Encoders - Fine-tuning', 'data': line_2_data.values}
    line_3_data = result_df[result_df['only_linear']&~result_df['use_ssl']].sort_values(by=['percent_of_masking'])[['percent_of_masking', metric]]
    line_config_3 = {'color': colors[1], 'style': '--.', 'label': 'Initial Encoders - Linear', 'data': line_3_data.values}

    fig = init_fig()
    draw_line(line_config_0)
    draw_line(line_config_1)
    draw_line(line_config_2)
    draw_line(line_config_3)
    finalize_fig()
    plt.show()

