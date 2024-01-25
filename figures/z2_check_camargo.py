from ssl_main.const import FONT_DICT
from figures.PaperFigures import format_axis
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from figures.PaperFigures import save_fig, load_da_data


def finalize_fig(bars):
    def format_ticks():
        # ax.set_ylabel(r'$R^2$ - vGRF Profile', fontdict=FONT_DICT)
        ax.set_ylabel('Correlation Coefficient - vGRF Profile', fontdict=FONT_DICT)
        x_range = (-1, 14)
        ax.set_xlim(x_range[0], x_range[1])
        ax.tick_params(bottom=False)
        ax.set_xticks(np.arange(1.5, x_range[1], 5))
        ax.set_xticklabels(['Task 1 -/nOverground Walking', 'Task 2 -/nTreadmill Walking',
                            'Task 3 -/nDrop Jump'], fontdict=FONT_DICT)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1], fontdict=FONT_DICT)

    ax = plt.gca()
    ylim_ori = ax.get_ylim()
    format_ticks()
    format_axis()
    plt.legend(bars[:4], ['MoVi', 'AMASS', 'Combined', 'Baseline'], bbox_to_anchor=(0.65, 1.2),
               ncol=2, fontsize=FONT_DICT['fontsize'], frameon=False)
    plt.tight_layout(rect=[0., 0., 1., 1.01])
    plt.show()


colors = [np.array(x) / 255 for x in [[3, 166, 200], [2, 83, 100], [153, 181, 210], [180, 180, 180]]]


if __name__ == "__main__":
    metric = 'correlation'
    patch_len = 8
    mask_patch_num = 6

    rc('font', family='Arial')
    bars = []

    da_name = 'Camargo_levelground_output'

    results_task, results_columns = load_da_data('D:/Local/results/2023_07_12_17_44_04_ssl_hyper/' + da_name + '.h5')
    results_task = {key_: value_ for key_, value_ in results_task.items()
                    if f'PatchLen_{patch_len}' in key_ and
                    f'MaskPatchNum_{mask_patch_num}' in key_ and
                    'LinearProb_False' in key_ and
                    f'UseSsl_False' in key_}
    results_values = list(results_task.values())[0]
    for sub_, result_ in results_values.items():
        plt.figure()
        plt.plot(result_[:, 0].ravel(), label='true')
        plt.plot(result_[:, 1].ravel(), label='pred')
        plt.title(sub_)
    plt.show()



