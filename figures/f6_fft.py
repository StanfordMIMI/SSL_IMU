import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from figures.PaperFigures import save_fig, load_da_data, results_to_pd_summary
from ssl_main.config import RESULTS_PATH


def plot_spectrum(ax, results_task):
    def format_ticks():
        ax.tick_params(bottom=False)
        ax.set_xticks(range(10, 51, 10))
        ax.set_xticklabels(range(10, 51, 10))
        ax.set_xlabel('Frequency (Hz)')

        ax.set_ylabel('vGRF Error (BW)')
        # ax.set_ylim(0, 1)
        # ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1])
        # ax.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1])

    for i_line, data_key_ in enumerate(results_task.items()):
        key_, data_ = data_key_
        # params = dict([param_tuple.split('_') for param_tuple in key_.split(', ')])
        output_data = list(data_.values())[2]
        output_num = int(output_data.shape[1] / 2)

        # step_spectrum_true, step_spectrum_pred = [], []
        step_spectrum_diff = []
        fft_fre = np.fft.fftfreq(n=output_data.shape[2], d=1/100)
        for i_output in range(output_num):
            for i_step in range(output_data.shape[0]):
                # step_spectrum_true.append(np.abs(np.fft.fft(output_data[i_step, i_output])))
                # step_spectrum_pred.append(np.abs(np.fft.fft(output_data[i_step, i_output+output_num])))

                step_spectrum_diff.append(np.abs(np.fft.fft(output_data[i_step, i_output])) -
                                          np.abs(np.fft.fft(output_data[i_step, i_output+output_num])))

        # step_spectrum_true = np.mean(np.array(step_spectrum_true), axis=0)
        # step_spectrum_pred = np.mean(np.array(step_spectrum_pred), axis=0)
        step_spectrum_diff_mean = np.mean(np.abs(np.array(step_spectrum_diff)), axis=0)
        step_spectrum_diff_std = np.std(np.abs(np.array(step_spectrum_diff)), axis=0)

        # ax.plot(np.mean(np.array(np.abs(step_spectrum_diff)), axis=0))
        start_loc, end_loc = 12, 64
        ax.plot(fft_fre[start_loc:end_loc], step_spectrum_diff_mean[start_loc:end_loc], color=f'C{1 - i_line}')
        ax.fill_between(fft_fre[start_loc:end_loc], (step_spectrum_diff_mean - step_spectrum_diff_std)[start_loc:end_loc],
                        (step_spectrum_diff_mean + step_spectrum_diff_std)[start_loc:end_loc], facecolor=f'C{1 - i_line}', alpha=0.4, label='_nolegend_')

        ax.title.set_text(test_names_print[i_da])
    format_ticks()
    plt.tight_layout(rect=[0., 0., 1., 1.], w_pad=1)


def init_fig():
    fig, ax = plt.subplots()
    return fig, ax


colors = [np.array(x) / 255 for x in [[2, 83, 100], [180, 180, 180]]]


if __name__ == "__main__":
    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]
    test_folder = '2023_05_23_18_48_32_SSL_COMBINED'
    test_names_print = ('Task 1 -\nOverground Walking', 'Task 2 -\nTreadmill Walking', 'Task 3 -\nDrop Jump')
    data_path = RESULTS_PATH + test_folder + '/'

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
    for i_da, da_name in enumerate(da_names):
        results_task = load_da_data(data_path + da_name + '.h5')
        results_task_ = {key_: value_ for key_, value_ in results_task.items()
                         if 'PatchLen_8' in key_ and 'MaskPatchNum_6' in key_ and 'LinearProb_False' in key_}
        plot_spectrum(axes[i_da], results_task_)
        # plot_example_windows(axes[1, i_da], results_task_)

    plt.legend(['Baseline', 'AMASS'])

    plt.show()















