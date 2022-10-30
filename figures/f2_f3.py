from scipy.stats import ttest_rel
import numpy as np
from const import LINE_WIDTH, FONT_DICT
from figures.PaperFigures import save_fig, load_da_data, results_dict_to_pd, format_axis
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Patch
from tab_2 import task_names


def init_fig():
    fig = plt.figure(figsize=(6, 4))
    return fig


def draw_box_pair(result_df, i_box_pair):
    with_ssl_ = result_df[result_df['use_ssl'] == True][metric].values
    no_ssl_ = result_df[result_df['use_ssl'] == False][metric].values
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

    p_val = ttest_rel(with_ssl_, no_ssl_).pvalue
    top_line_y = 1.04
    if p_val < 0.05:
        plt.plot([4 * i_box_pair, 4 * i_box_pair, 4 * i_box_pair + 1, 4 * i_box_pair + 1],
                 [max(with_ssl_) + 0.04, top_line_y, top_line_y, max(no_ssl_) + 0.04], linewidth=LINE_WIDTH, color='black')
        plt.text(4 * i_box_pair + 0.3, top_line_y + 0.01, '*', fontdict={'fontname': 'Times New Roman'}, size=20, zorder=20)


def finalize_fig():
    def format_ticks():
        ax.set_ylabel('Correlation Coefficient - {}'.format(fig_config[i_fig]['ylabel']), fontdict=FONT_DICT)
        x_range = (-1, 10)
        ax.set_xlim(x_range[0], x_range[1])
        ax.tick_params(bottom=False)

        ax.set_xticks(np.arange(0.5, x_range[1], 4))
        ax.set_xticklabels(['Task 1 -\nRunning VALR', 'Task 2 -\nWalking vGRF',
                            'Task 3 -\nWalking KFM'], fontdict=FONT_DICT)

        ylim_up = 1.2
        if np.min(ylim_ori[0]) < 0.4:
            ylim_down = 0.2
            ticks = [.2, .4, .6, .8, 1., 1.2]
        else:
            ylim_down = 0.4
            ticks = [.4, .6, .8, 1., 1.2]
        ax.set_ylim(ylim_down, ylim_up)
        ax.set_yticks(ticks)     # [.4, .5, .6, .7, .8, .9, 1., 1.1]
        ax.set_yticklabels(ticks, fontdict=FONT_DICT)

    ax = plt.gca()
    ylim_ori = ax.get_ylim()
    format_ticks()
    format_axis()
    legend_elements = [Patch(facecolor=colors[0], label='Color Patch'),
                       Patch(facecolor=colors[1], label='Color Patch')]
    plt.legend(legend_elements, ['Self-Supervised Encoders', 'Initial Encoders'], bbox_to_anchor=(0.65, 1.22 - 0.2 * (ylim_ori[0])),
               ncol=1, fontsize=FONT_DICT['fontsize'], frameon=False)
    plt.tight_layout(rect=[0., 0., 1., 1.01])
    save_fig(fig_name)


data_path = 'D:\ssl_training_results\\2022-10-30 11_00_59_all_test'
colors = [np.array([125, 172, 80]) / 255, np.array([130, 130, 130]) / 255]
rc('font', family='Arial')
fig_config = [
    {'ylabel': 'Fine-Tuning'},
    {'ylabel': 'Linear'},
]

if __name__ == "__main__":
    metric = 'correlation'
    for i_fig, (only_linear, fig_name) in enumerate(zip([False, True], ['f2', 'f3'])):
        init_fig()
        for i_task, task_name in enumerate(task_names):
            results_task = load_da_data(data_path + task_name + '.h5')
            result_df = results_dict_to_pd(results_task)
            result_df = result_df[(result_df['ratio'] == 1.) & (result_df['only_linear'] == only_linear)]
            draw_box_pair(result_df, i_task)
        finalize_fig()
    plt.show()








