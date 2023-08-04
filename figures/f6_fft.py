import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from ssl_main.const import LINE_WIDTH, FONT_DICT, LINE_WIDTH_THICK
from figures.PaperFigures import save_fig, load_da_data, format_axis
from ssl_main.config import RESULTS_PATH


def plot_spectrum(results_task):
    def format_ticks():
        ax.set_xlim(10, 50)
        ax.set_xticks(range(10, 51, 10))
        ax.set_xticklabels(range(10, 51, 10), fontdict=FONT_DICT)
        ax.set_xlabel('Frequency (Hz)', fontdict=FONT_DICT)

        ax.set_ylabel('vGRF Error (BW)', fontdict=FONT_DICT)
        if i_da == 0 or i_da == 1:
            ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1.], fontdict=FONT_DICT)
        elif i_da == 2:
            ax.set_yticks([0., 0.4, 0.8, 1.2, 1.6, 2.])
            ax.set_yticklabels([0., 0.4, 0.8, 1.2, 1.6, 2.], fontdict=FONT_DICT)

    ax = plt.gca()
    lines_handle, fill_handle = [], []
    for i_line, data_key_ in enumerate(results_task.items()):
        key_, data_ = data_key_
        # params = dict([param_tuple.split('_') for param_tuple in key_.split(', ')])
        output_data = list(data_.values())[2]
        output_num = int(output_data.shape[1] / 2)

        # step_spectrum_true, step_spectrum_pred = [], []
        step_spectrum_diff = []
        fft_fre = np.fft.fftfreq(n=output_data.shape[2], d=1/100)
        for i_step in range(output_data.shape[0]):
            # step_spectrum_true.append(np.abs(np.fft.fft(output_data[i_step, i_output])))
            # step_spectrum_pred.append(np.abs(np.fft.fft(output_data[i_step, i_output+output_num])))

            step_spectrum_diff.append(np.abs(np.fft.fft(output_data[i_step, 0])) -
                                      np.abs(np.fft.fft(output_data[i_step, 0+output_num])))

        # step_spectrum_true = np.mean(np.array(step_spectrum_true), axis=0)
        # step_spectrum_pred = np.mean(np.array(step_spectrum_pred), axis=0)
        step_spectrum_diff_mean = np.mean(np.abs(np.array(step_spectrum_diff)), axis=0)
        step_spectrum_diff_std = np.std(np.abs(np.array(step_spectrum_diff)), axis=0)

        # ax.plot(np.mean(np.array(np.abs(step_spectrum_diff)), axis=0))
        start_loc, end_loc = 12, 64
        lines_handle.append(ax.plot(fft_fre[start_loc:end_loc], step_spectrum_diff_mean[start_loc:end_loc], color=colors[i_line])[0])
        ax.fill_between(fft_fre[start_loc:end_loc], (step_spectrum_diff_mean - step_spectrum_diff_std)[start_loc:end_loc],
                        (step_spectrum_diff_mean + step_spectrum_diff_std)[start_loc:end_loc], facecolor=colors[i_line],
                        alpha=0.4, label='_nolegend_')
        fill_handle.append(ax.fill(np.NaN, np.NaN, color=colors[i_line], alpha=0.5)[0])
        plt.title(test_names_print[i_da], fontdict=FONT_DICT)
        format_ticks()
        format_axis(ax)
    return lines_handle, fill_handle


def finalize_fig(lines_handle, fill_handle):
    plt.tight_layout(rect=[0., 0., 1., .92], w_pad=2)
    plt.legend([(lines_handle[0], fill_handle[0]), (lines_handle[1], fill_handle[1])],
               ['Baseline', 'Self-Supervised Learning'], fontsize=FONT_DICT['fontsize'], ncol=2,
               frameon=False, bbox_to_anchor=(0.7, 1.28))


colors = [np.array(x) / 255 for x in [[110, 110, 110], [3, 136, 170]]]


if __name__ == "__main__":
    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]
    test_folder = '2023_07_17_11_12_25_SSL_AMASS'
    test_names_print = ('Task 1 - Overground Walking', 'Task 2 - Treadmill Walking', 'Task 3 - Drop Landing')
    data_path = RESULTS_PATH + test_folder + '/'
    plt.figure(figsize=(12, 4))

    for i_da, da_name in enumerate(da_names):
        plt.subplot(1, 3, i_da + 1)
        results_task = load_da_data(data_path + da_name + '.h5')
        results_task_ = {key_: value_ for key_, value_ in results_task.items()
                         if 'PatchLen_8' in key_ and 'MaskPatchNum_6' in key_ and 'LinearProb_False' in key_ and
                         'ratio_1' in key_}
        lines_handle, fill_handle = plot_spectrum(results_task_)
        # plot_example_windows(axes[1, i_da], results_task_)

    finalize_fig(lines_handle, fill_handle)
    save_fig('f6_fft')
    plt.show()















