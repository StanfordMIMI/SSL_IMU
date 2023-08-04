import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import numpy as np
import matplotlib.pyplot as plt
from figures.PaperFigures import save_fig, load_da_data
from ssl_main.config import RESULTS_PATH
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
import pandas as pd


def results_to_pd_summary(result_all_tests, result_field_id, block_swing_phase=True):
    result_list = []
    for test_name, test_results in result_all_tests.items():
        param_tuples = [param_tuple.split('_') for param_tuple in test_name.split(', ')]
        for param_tuple in param_tuples:
            if param_tuple[-1] in ['True', 'False']:
                param_tuple[-1] = param_tuple[-1] == 'True'
            else:
                param_tuple[-1] = float(param_tuple[-1])
        rmse, r2 = [], []
        for sub_name, subject_data in test_results.items():
            data_true, data_pred = subject_data[:, result_field_id].ravel(), subject_data[:, result_field_id+int(subject_data.shape[1]/2)].ravel()
            if block_swing_phase:
                stance_phase_loc = np.where(np.abs(data_true) > 0.02)[0]
                data_true, data_pred = data_true[stance_phase_loc], data_pred[stance_phase_loc]
            rmse.append(np.sqrt(mse(data_true, data_pred)))
            r2.append(r2_score(data_true, data_pred))
        result_list.append([param_tuple[-1] for param_tuple in param_tuples] + [np.mean(rmse), np.mean(r2)])
    result_df = pd.DataFrame(result_list, columns=[param_tuple[0] for param_tuple in param_tuples] + ['rmse', 'r2'])
    return result_df


if __name__ == "__main__":
    da_name = 'Camargo_levelground_output'
    test_folder = '2023_07_16_18_21_09_tune_hyper'
    data_path = RESULTS_PATH + test_folder + '/'
    metric = 'r2'

    with PdfPages(data_path + f'f9_{test_folder}.pdf') as pdf:
        results_task = load_da_data(data_path + da_name + '.h5')
        hyper_list = [param_tuple.split('_')[0] for param_tuple in list(results_task.keys())[0].split(', ')][3:]
        result_summary = results_to_pd_summary(results_task, 0)
        result_summary = result_summary[(result_summary['LinearProb'] == False) & (result_summary['UseSsl'] == True)]

        for hyper in hyper_list:
            hyper_values = np.sort(result_summary[hyper].unique())
            hyper_lines = []
            for hyper_value in hyper_values:
                df_current_hyper = result_summary[result_summary[hyper] == hyper_value]
                hyper_lines.append(df_current_hyper[metric].values)

            plt.figure()
            for line_ in np.array(hyper_lines).T:
                plt.plot(hyper_values, line_)
            plt.xticks(hyper_values)
            plt.xlabel(hyper)
            plt.ylabel(metric)
            pdf.savefig()









