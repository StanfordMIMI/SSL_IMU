from scipy.stats import ttest_rel
import numpy as np
from const import LINE_WIDTH, FONT_DICT
from figures.PaperFigures import save_fig, load_da_data, results_dict_to_pd, format_axis
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Patch


def init_f10():
    fig = plt.figure(figsize=(6, 4))
    return fig


def draw_box_pair(amount_df, i_box_pair, sigifi_sign_fun):
    def format_ticks():
        ax = plt.gca()
        ax.set_xlabel('Percentage of Data Used for Training Encoders', fontdict=FONT_DICT)
        ax.set_ylabel('Correlation Coefficient of KFM Estimation', fontdict=FONT_DICT)
        x_range = (-1, 4 * len(amount_list) - 2)
        ax.set_xlim(x_range[0], x_range[1])
        ax.tick_params(bottom=False)

        ax.set_xticks(np.arange(0.5, x_range[1], 4))
        ax.set_xticklabels(['1.0%', '3.3%', '10%', '33.3%', '100%'], fontdict=FONT_DICT)

        ylim = 1.2
        ax.set_ylim(0.4, ylim)
        ax.set_yticks([.4, .6, .8, 1., 1.2])     # [.4, .5, .6, .7, .8, .9, 1., 1.1]
        ax.set_yticklabels([.4, .6, .8, 1., 1.2], fontdict=FONT_DICT)

    with_ssl_ = amount_df[amount_df['use_ssl'] == True][metric].values
    no_ssl_ = amount_df[amount_df['use_ssl'] == False][metric].values
    colors = [np.array([125, 172, 80]) / 255, np.array([130, 130, 130]) / 255]
    boxes_ = []
    for i_cond, data_ in enumerate([with_ssl_, no_ssl_]):
        meanpointprops = dict(marker='D', markeredgecolor='none', markerfacecolor='white', markersize=7)
        box_ = plt.boxplot(data_, positions=[i_cond + 4 * i_box_pair], widths=[0.8], patch_artist=True,
                           meanprops=meanpointprops, showmeans=True)
        for field in ['medians', 'whiskers', 'caps', 'boxes']:
            [box_[field][i].set(linewidth=LINE_WIDTH, color=colors[i_cond]) for i in range(len(box_[field]))]
        [box_['fliers'][i].set(marker='o', markeredgecolor=colors[i_cond], markerfacecolor=colors[i_cond],
                               markersize=2.5) for i in range(len(box_['fliers']))]
        box_['medians'][0].set(linewidth=LINE_WIDTH, color=[1, 1, 1])
        boxes_.append(box_['whiskers'])

    legend_elements = [Patch(facecolor=colors[0], label='Color Patch'),
                       Patch(facecolor=colors[1], label='Color Patch')]
    plt.legend(legend_elements, ['Self-Supervised Encoders', 'Initial Encoders'], bbox_to_anchor=(0.6, 1.1),
               ncol=1, fontsize=FONT_DICT['fontsize'], frameon=False)
    format_ticks()
    p_val = ttest_rel(with_ssl_, no_ssl_).pvalue
    top_line_y = 1.04
    if p_val < 0.05:
        plt.plot([4 * i_box_pair, 4 * i_box_pair, 4 * i_box_pair + 1, 4 * i_box_pair + 1],
                 [max(with_ssl_) + 0.02, top_line_y, top_line_y, max(no_ssl_) + 0.02], linewidth=LINE_WIDTH, color='black')
        plt.text(4 * i_box_pair + 0.3, top_line_y + 0.01, '*', fontdict={'fontname': 'Times New Roman'}, size=20, zorder=20)


def finalize_f10():
    format_axis()
    plt.tight_layout(rect=[0., 0., 1., 1.01])
    save_fig('f4')


data_path = 'D:\ssl_training_results\\2022-10-24 16_30_35_all_test'
test_name = '\\walking_knee_moment_KFM'     # hw_running_VALR
rc('font', family='Arial')
if __name__ == "__main__":
    metric = 'correlation'
    only_linear = False
    results_task = load_da_data(data_path + test_name + '.h5')
    result_df = results_dict_to_pd(results_task)
    result_df = result_df[result_df['only_linear']==only_linear]

    amount_list = list(set(result_df['ratio']))
    amount_list.sort()

    for i_amount, amount in enumerate(amount_list):
        amount_df = result_df[result_df['ratio']==amount]
        draw_box_pair(amount_df, i_amount, None)

    finalize_f10()
    plt.show()

