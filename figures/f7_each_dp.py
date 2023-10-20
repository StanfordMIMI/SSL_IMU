import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from ssl_main.const import LINE_WIDTH, FONT_DICT, LINE_WIDTH_THICK
from figures.PaperFigures import save_fig, load_da_data
from ssl_main.config import RESULTS_PATH
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error as mse


def pearson_cor_fun(data_):
    return pearsonr(data_[:128], data_[128:])[0]


def my_r2(data_):
    return r2_score(data_[:128], data_[128:])


def plot_benefits_of_each_dp(results_task):
    data_0, data_1 = [np.concatenate(list(value_.values()), axis=0) for value_ in results_task.values()]
    assert data_0.shape == data_1.shape
    assert np.max(np.abs(data_0[:, 0] - data_1[:, 0])) < 1e-10
    result_0 = np.apply_along_axis(pearson_cor_fun, -1, data_0.reshape([data_0.shape[0], -1]))
    result_1 = np.apply_along_axis(pearson_cor_fun, -1, data_1.reshape([data_0.shape[0], -1]))
    max_val = np.partition(np.concatenate([result_0, result_1]).flatten(), -1)[-1]
    min_val = np.partition(np.concatenate([result_0, result_1]).flatten(), 3)[3]

    ax = plt.gca()
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.plot(result_0, result_1, '.',)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=LINE_WIDTH_THICK)
    ax.set_xlabel('Correlation - Supervised', fontdict=FONT_DICT)
    ax.set_ylabel('Correlation - SSL', fontdict=FONT_DICT)
    plt.title(test_names_print[i_da], fontdict=FONT_DICT, pad=15)
    format_axis(ax)


def scatter_plot_of_peak(results_task):
    data_0, data_1 = [np.concatenate(list(value_.values())[config['start_sub']:config['end_sub']], axis=0) for value_ in results_task.values()]
    assert data_0.shape == data_1.shape
    assert np.max(np.abs(data_0[:, 0] - data_1[:, 0])) < 1e-10
    data_0, data_1 = data_0 * 9.81, data_1 * 9.81

    if i_da == 1:
        index = np.random.choice(data_0.shape[0], config['num_to_plot'], replace=False)
        data_0, data_1 = data_0[index], data_1[index]

    data_0, data_1 = data_0 * da_sign_of_peak[i_da], data_1 * da_sign_of_peak[i_da]
    max_true = np.max(data_0[:, 0], axis=-1)
    max_pred_bl = np.max(data_0[:, 1], axis=-1)
    max_pred_ssl = np.max(data_1[:, 1], axis=-1)
    ax = plt.gca()
    min_val, max_val = format_axis(ax)
    for i_scatter, (max_pred, label_) in enumerate(zip([max_pred_bl, max_pred_ssl], ['Baseline', 'SSL'])):
        correlation_ = pearsonr(max_true, max_pred)[0]
        coef = np.polyfit(max_true, max_pred, 1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(np.array([min_val, max_val]), poly1d_fn([min_val, max_val]), color=colors[i_scatter], linewidth=LINE_WIDTH, alpha=0.8)
        ax.scatter(max_true, max_pred, s=config['dot_size'], c=colors[i_scatter], alpha=0.55, edgecolors='none',
                   label=fr'{label_} ($\rho$={round(correlation_, 2)}, ' + r'$y$={:4.2f}$x$+{:4.2f})'.format(coef[0], coef[1]))

    # max_val = np.max(np.concatenate([max_true, max_pred_bl, max_pred_ssl]))
    # min_val = np.min(np.concatenate([max_true, max_pred_bl, max_pred_ssl]))
    # ax.set_ylim(min_val, max_val + 0.3 * (max_val - min_val))

    ax.set_xlabel(r'vGRF Peak: Ground Truth ($N/kg$)', fontdict=FONT_DICT)
    ax.set_ylabel(r'vGRF Peak: Model Prediction ($N/kg$)', fontdict=FONT_DICT)
    plt.title(test_names_print[i_da], fontdict=FONT_DICT, pad=15)
    l = plt.legend(fontsize=FONT_DICT['fontsize']-1, ncol=1, frameon=False, bbox_to_anchor=(1.04, 1.03),
                   handlelength=0, handletextpad=0)
    [text.set_color(colors[i_l]) for i_l, text in enumerate(l.get_texts())]
    [item.set_visible(False) for item in l.legendHandles]


def format_axis(ax=None, line_width=LINE_WIDTH):
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(width=line_width)
    ax.yaxis.set_tick_params(width=line_width)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)
    if i_da == 0:
        min_val, max_val = 6, 18
        ticks_label = [round(x, 1) for x in np.arange(min_val, max_val + 1, 3)]
        ax.set_ylim(min_val, max_val)
    elif i_da == 1:
        min_val, max_val = 9, 19
        ticks_label = [round(x, 1) for x in np.arange(min_val, max_val + 1, 2)]
        ax.set_ylim(min_val, max_val+0.9)
    else:
        min_val, max_val = 7, 35
        ticks_label = [round(x, 1) for x in np.arange(min_val, max_val + 1, 7)]
        ax.set_ylim(min_val, max_val)
    ax.set_xlim(min_val, max_val)
    # ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=LINE_WIDTH)
    ax.set_xticks(ticks_label)
    ax.set_yticks(ticks_label)
    ax.set_xticklabels(ticks_label, fontdict=FONT_DICT)
    ax.set_yticklabels(ticks_label, fontdict=FONT_DICT)
    return min_val, max_val


def finalize_fig():
    plt.tight_layout(rect=[0., 0., 1., 1.], w_pad=3)


colors = [np.array(x) / 255 for x in [[110, 110, 110], [3, 136, 170]]]
da_sign_of_peak = [1, -1, 1]
plot_configs = [
    {'start_sub': 0, 'end_sub': -1, 'dot_size': 20, 'num_to_plot': 500},
    # {'start_sub': 2, 'end_sub': 3, 'dot_size': 15, 'num_to_plot': 30},
]


if __name__ == "__main__":
    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]     # walking_knee_moment
    test_folder = '2023_09_20_20_19_44_AMASS'
    test_names_print = ('Task 1 - Overground Walking', 'Task 2 - Treadmill Walking', 'Task 3 - Drop Landing')
    data_path = RESULTS_PATH + test_folder + '/'

    for i_config, config in enumerate(plot_configs):
        plt.figure(figsize=(12, 4.5))
        for i_da, da_name in enumerate(da_names):
            plt.subplot(1, 3, i_da + 1)
            results_task = load_da_data(data_path + da_name + '.h5')
            results_task_ = {key_: value_ for key_, value_ in results_task.items()
                             if 'PatchLen_1' in key_ and 'MaskPatchNum_16' in key_ and 'LinearProb_False' in key_ and
                             'ratio_1' in key_}
            scatter_plot_of_peak(results_task_)
            # plot_benefits_of_each_dp(results_task_)
        finalize_fig()
    save_fig('f7_each_point')
    plt.show()















