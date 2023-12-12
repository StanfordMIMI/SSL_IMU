from ssl_main.const import LINE_WIDTH, FONT_DICT, LINE_WIDTH_THICK
from ssl_main.const import GRF_ML_AP_V
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from figures.PaperFigures import save_fig, load_da_data, results_to_pd_summary, format_errorbar_cap
from ssl_main.config import RESULTS_PATH
from scipy.stats import ttest_rel
import prettytable as pt


def print_anova_with_bonferroni(combo_results, combo_names):
    """ Equals to doing paired t-tests. """
    tb = pt.PrettyTable()
    tb.field_names = combo_names
    for i_combo, combo_a in enumerate(combo_names):
        p_val_row = []
        for j_combo, combo_b in enumerate(combo_names):
            if i_combo == j_combo:
                p_val = 1
            else:
                p_val = round(ttest_rel(combo_results[i_combo], combo_results[j_combo]).pvalue * 6, 3)
            p_val_row.append(p_val)
        tb.add_row(p_val_row)
    tb.add_column('', combo_names, align="l")
    print(tb)


if __name__ == "__main__":
    metric = 'correlation'
    patch_len = 1
    mask_patch_num = 16

    model_all = ['baseline', '2023_12_10_22_28_movi_all_three', '2023_12_10_22_28_amass_all_three', '2023_12_10_22_28_combined_all_three']
    model_name_print = ['Baseline', 'Proposed SSL - real IMU', 'Proposed SSL - synthetic IMU', 'Proposed SSL - synthetic and real IMU']

    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]
    test_names_print = ('Task 1 - Overground Walking', 'Task 2 - Treadmill Walking', 'Task 3 - Drop Landing')

    tab_str = ''

    for i_model, model_folder in enumerate(model_all):
        tab_str += model_name_print[i_model] + ' & '
        combos_for_statistic_test = []
        for i_da, da_name in enumerate(da_names[:]):
            if model_folder == 'baseline':
                model_folder_str = model_all[-2]
                use_ssl = False
            else:
                model_folder_str = model_folder
                use_ssl = True

            data_path = RESULTS_PATH + model_folder_str + '/'
            results_task, results_columns = load_da_data(data_path + da_name + '.h5')
            results_task = {key_: value_ for key_, value_ in results_task.items()
                            if f'PatchLen_{patch_len}' in key_ and
                            f'MaskPatchNum_{mask_patch_num}' in key_ and
                            'LinearProb_False' in key_ and
                            'ratio_1' in key_
                            }
            full_size_task = [value_ for key_, value_ in results_task.items() if 'ratio_1' in key_][0]
            win_number_total = np.sum([data_.shape[0] for _, data_ in full_size_task.items()])

            result_df = results_to_pd_summary(results_task, results_columns)
            result_df = result_df[result_df['UseSsl'] == use_ssl]

            for i_axis, axis in enumerate(GRF_ML_AP_V[da_name[:-7]]):
                results_list = result_df[f'{axis}_{metric}']
                # tab_str += '{} ({}) & '.format(np.round(np.mean(results_list), 2), np.round(np.std(results_list), 2))
                tab_str += '{:.2f} & '.format(np.round(np.mean(results_list), 2))

            tab_str = tab_str + '& '
        tab_str = tab_str[:-4] + '\\\\' + '\n'
    print(tab_str)













