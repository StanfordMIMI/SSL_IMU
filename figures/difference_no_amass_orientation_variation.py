from ssl_main.const import GRF_ML_AP_V, vgrf_names
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
from figures.PaperFigures import load_da_data, results_to_pd_summary
from ssl_main.config import RESULTS_PATH


def print_amass_no_variation_difference():
    metric = 'correlation'
    patch_len = 1
    mask_patch_num = 16

    model_all = ['2023_12_13_23_05_amass_PatchLen', '2023_12_13_22_32_amass_amass_no_variation']
    model_name_print = ['With Orientation Variation', 'No Orientation Variation']

    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]
    da_names_print = ('Overground Walking', 'Treadmill Walking', 'Drop Landing')
    tab_str = ''

    for i_da, da_name in enumerate(da_names[:]):
        tab_str += f'{da_names_print[i_da]}\n'
        for i_model, model_folder in enumerate(model_all):
            model_folder_str = model_folder
            tab_str += f'{model_name_print[i_model]}\t'
            data_path = RESULTS_PATH + model_folder_str + '/'
            results_task, results_columns = load_da_data(data_path + da_name + '.h5')
            results_task = {key_: value_ for key_, value_ in results_task.items()
                            if f'PatchLen_{patch_len}' in key_ and
                            f'MaskPatchNum_{mask_patch_num}' in key_ and
                            'LinearProb_False' in key_ and
                            'ratio_1' in key_ and
                            'UseSsl_True' in key_
                            }

            result_df = results_to_pd_summary(results_task, results_columns)

            for i_axis, axis in enumerate(GRF_ML_AP_V[da_name[:-7]]):
                results_list = result_df[f'{axis}_{metric}']
                tab_str += '{} +- {} \t'.format(np.round(np.mean(results_list), 3), np.round(np.std(results_list), 3))
            tab_str += '\n'
    print(tab_str)


def print_rmse_rrmse():
    patch_len = 1
    mask_patch_num = 16

    model_folder = '2023_12_12_23_02_amass_da_ratio'

    da_names = [element + '_output' for element in ['Camargo_levelground', 'walking_knee_moment', 'sun_drop_jump']]
    da_names_print = ('Overground Walking', 'Treadmill Walking', 'Drop Landing')
    tab_str = ''

    for i_da, da_name in enumerate(da_names[:]):
        tab_str += f'{da_names_print[i_da]}\n'

        model_folder_str = model_folder
        data_path = RESULTS_PATH + model_folder_str + '/'
        results_task, results_columns = load_da_data(data_path + da_name + '.h5')
        results_task = {key_: value_ for key_, value_ in results_task.items()
                        if f'PatchLen_{patch_len}' in key_ and
                        f'MaskPatchNum_{mask_patch_num}' in key_ and
                        'LinearProb_False' in key_ and
                        'ratio_1' in key_ and
                        'UseSsl_True' in key_
                        }

        result_df = results_to_pd_summary(results_task, results_columns)

        results_list = result_df[f'{vgrf_names[i_da]}_rmse']
        tab_str += '${:.2f} \pm {:.2f} N/kg$ and '.format(np.mean(results_list), np.std(results_list))

        results_list = result_df[f'{vgrf_names[i_da]}_r_rmse']
        tab_str += '${:.2f} \pm {:.2f} \%$\t'.format(np.mean(results_list), np.std(results_list))
        tab_str += '\n'
    print(tab_str)


if __name__ == "__main__":
    print_amass_no_variation_difference()
    print_rmse_rrmse()









