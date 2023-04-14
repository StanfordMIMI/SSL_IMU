import numpy as np
from ssl_main.const import FONT_DICT, LINE_WIDTH_THICK
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from figures.PaperFigures import save_fig, load_da_data, results_dict_to_pd_profiles


def plot_map_with_number(ax, data_, x_ticks, y_ticks, title):
    data_ = data_.T
    mav_val = np.max(np.abs(data_))
    cmap = plt.colormaps.get_cmap('RdBu')
    im = ax.imshow(data_, interpolation='nearest', cmap=cmap, vmax=mav_val, vmin=-mav_val)
    # Add text to the matrix to display the values
    for i in range(len(data_)):
        for j in range(len(data_[i])):
            ax.text(j, i, round(data_[i, j], 3), ha='center', va='center', color='black')
    # Set the x and y axis labels and tick marks
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    plt.setp(ax.get_xticklabels(), ha='center')
    # Add a title and colorbar
    ax.set_title(title)
    return im


def plot_map_with_number_all_four(data_all, x_ticks, y_ticks, title_list):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), height_ratios=[4, 1])
    min_val, max_val = np.min(data_all), np.max(data_all)
    for i_subplot, (ax, data_, title_) in enumerate(zip(axes.flatten(), data_all, title_list)):
        data_ = data_.T
        cmap = colormaps.get_cmap('RdBu')
        if 'no SSL' in title_:
            data_ = data_[:1]
            y_ticks = ['']

        im = ax.imshow(data_, interpolation='nearest', cmap=cmap, vmax=max_val, vmin=min_val)
        # Add text to the matrix to display the values
        for i in range(len(data_)):
            for j in range(len(data_[i])):
                ax.text(j, i, round(data_[i, j], 2), ha='center', va='center', color='black')
        ax.set_xticks(np.arange(len(x_ticks)))
        ax.set_yticks(np.arange(len(y_ticks)))
        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)
        plt.setp(ax.get_xticklabels(), ha='center')
        ax.set_title(title_)
        # if i_subplot == 1:
        #     fig.colorbar(im, ax=ax)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # plt.tight_layout(rect=[0., 0., 1., 1.])


def init_fig():
    fig, ax = plt.subplots()
    return fig, ax


def finalize_fig(fig, ax, im):
    fig.colorbar(im, ax=ax)
    # plt.tight_layout(rect=[0., 0., 1., 1.])
    # save_fig('f9')


data_path = 'results/t01_acc0.95'
test_name = '/walking_knee_moment_output'     # hw_running_VALR   walking_knee_moment_output  Carmargo_output
rc('font', family='Arial')
colors = [np.array([125, 172, 80]) / 255, np.array([130, 130, 130]) / 255]

if __name__ == "__main__":
    metric = 'r2'
    results_task = load_da_data(data_path + test_name + '.h5')

    result_df = results_dict_to_pd_profiles(results_task, 1)
    result_df['percent_of_masking'] = result_df['mask_patch_num'] / (128 / result_df['patch_len'])

    patch_len_list = np.sort(list(set(result_df['patch_len'])))
    percent_of_masking_list = np.sort(list(set(result_df['percent_of_masking'])))
    percent_of_masking_list_str = [str(round(i * 100, 1)) + '%' for i in percent_of_masking_list]
    print(patch_len_list)
    result_mean_map = np.zeros([4, len(patch_len_list), len(percent_of_masking_list)])
    for i_patch, patch_len in enumerate(patch_len_list):
        for i_percent, percent_of_masking in enumerate(percent_of_masking_list):
            data_cond = result_df[(result_df['patch_len'] == patch_len) & (result_df['percent_of_masking'] == percent_of_masking)]
            data_cond_0 = data_cond[~data_cond['only_linear']&data_cond['use_ssl']]
            result_mean_map[0, i_patch, i_percent] = np.mean(data_cond_0[metric])
            data_cond_1 = data_cond[data_cond['only_linear']&data_cond['use_ssl']]
            result_mean_map[1, i_patch, i_percent] = np.mean(data_cond_1[metric])
            data_cond_2 = data_cond[~data_cond['only_linear']&~data_cond['use_ssl']]
            result_mean_map[2, i_patch, i_percent] = np.mean(data_cond_2[metric])
            data_cond_3 = data_cond[data_cond['only_linear']&~data_cond['use_ssl']]
            result_mean_map[3, i_patch, i_percent] = np.mean(data_cond_3[metric])

    title = data_path.split('_')[1]
    plot_map_with_number_all_four(result_mean_map, patch_len_list, percent_of_masking_list_str,
                                  ['SSL - Fine-tuning', 'SSL - Linear', 'no SSL - Fine-tuning', 'no SSL - Linear'])
    plt.title(title)

    # fig, ax = init_fig()
    # im = plot_map_with_number(ax, result_mean_map[0] - result_mean_map[2], patch_len_list, percent_of_masking_list_str, 'fine tuning')
    # finalize_fig(fig, ax, im)
    # fig, ax = init_fig()
    # plot_map_with_number(ax, result_mean_map[1] - result_mean_map[3], patch_len_list, percent_of_masking_list_str, 'linear prob')
    # finalize_fig(fig, ax, im)

    plt.show()

