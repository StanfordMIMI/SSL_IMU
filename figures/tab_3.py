from ssl_main.const import GRF_ML_AP_V
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
from figures.PaperFigures import save_fig, load_da_data, results_to_pd_summary, format_errorbar_cap
from ssl_main.config import RESULTS_PATH


def print_tab_3():
    model_names_print = ['Baseline', 'Proposed SSL\\tnote{b}']
    model_folder_str = models[1]

    tab_str = ''

    for i_da, da_name in enumerate(da_names):
        tab_str += '\midrule\n' + '\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}' + da_names_print[i_da] + '\end{tabular}} & '

        data_path = RESULTS_PATH + model_folder_str + '/'
        results_task, results_columns = load_da_data(data_path + da_name + '.h5')
        results_task = {key_: value_ for key_, value_ in results_task.items()
                        if f'PatchLen_{patch_len}' in key_ and
                        f'MaskPatchNum_{mask_patch_num}' in key_ and
                        'LinearProb_False' in key_ and
                        'ratio_1' in key_
                        }

        for i_model, (use_ssl, model_name) in enumerate(zip([False, True], model_names_print)):
            tab_str += model_name + ' & '
            result_df = results_to_pd_summary(results_task, results_columns)
            result_df = result_df[result_df['UseSsl'] == use_ssl]

            for i_axis, axis in enumerate(GRF_ML_AP_V[da_name[:-7]]):
                results_list = result_df[f'{axis}_{metric}']
                tab_str += '${:.2f} \pm {:.2f}$ & '.format(np.mean(results_list), np.std(results_list))

            tab_str = tab_str[:-3] + ' \\\\\n ' + ' & '
            if i_model == 0:
                tab_str = tab_str[:-5] + ' [0.15cm] \n  & '
        tab_str = tab_str[:-3]
    print(tab_str)


def print_accuracy_decrease():

    for i_model, use_ssl in enumerate([False, True]):
        diff = []
        for i_da, da_name in enumerate(da_names):
            model_results = {model_folder_str: [] for model_folder_str in models}
            for i_model, model_folder_str in enumerate(models):

                data_path = RESULTS_PATH + model_folder_str + '/'
                results_task, results_columns = load_da_data(data_path + da_name + '.h5')
                results_task = {key_: value_ for key_, value_ in results_task.items()
                                if f'PatchLen_{patch_len}' in key_ and
                                f'MaskPatchNum_{mask_patch_num}' in key_ and
                                'LinearProb_False' in key_ and
                                'ratio_1' in key_
                                }

                result_df = results_to_pd_summary(results_task, results_columns)
                result_df = result_df[result_df['UseSsl'] == use_ssl]

                for i_axis, axis in enumerate(GRF_ML_AP_V[da_name[:-7]]):
                    results_list = result_df[f'{axis}_{metric}']
                    model_results[model_folder_str].append(np.mean(results_list))

            diff.extend([model_results[models[0]][i] - model_results[models[1]][i] for i in range(len(model_results[models[0]]))])
        print('{:.2f}'.format(min(diff)), '{:.2f}'.format(max(diff)))


if __name__ == "__main__":
    metric = 'correlation'
    patch_len = 1
    mask_patch_num = 16

    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]
    da_names_print = ('Overground\\\\ Walking', 'Treadmill\\\\ Walking', 'Drop\\\\ Landing')
    models = ['2023_12_12_23_02_amass_da_ratio', '2023_12_12_23_07_amass_foot_only']

    print_accuracy_decrease()
    print_tab_3()





