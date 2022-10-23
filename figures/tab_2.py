import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PaperFigures import load_da_data, results_dict_to_pd


def get_one_metric(result_df, only_linear, use_ssl, ratio, metric):
    df = result_df[(result_df['only_linear'] == only_linear) &
                   (result_df['use_ssl'] == use_ssl) &
                   (result_df['ratio'] == ratio)]
    return df[metric]


def get_mean_std_of_one_metric(**args):
    metrics = get_one_metric(**args)
    return np.mean(metrics), np.std(metrics)


def print_one_task(task_dir, config, print_str):
    results_task = load_da_data(task_dir)
    result_df = results_dict_to_pd(results_task)
    for metric in [['correlation', '', 2], ['r_rmse', '\%', 1]]:
        mean_, std_ = get_mean_std_of_one_metric(result_df=result_df, only_linear=config['only_linear'],
                                                   use_ssl=config['use_ssl'], ratio=1, metric=metric[0])
        mean_, std_ = np.round(mean_, metric[2]), np.round(std_, metric[2])
        print_str += '${} \pm {}{}$ & '.format(mean_, std_, metric[1])
    return print_str


configs = [{'only_linear': False, 'use_ssl': True, 'start_str': '\multirow{2}{*}{Fine-tuning}    & SSL & '},
           {'only_linear': False, 'use_ssl': False, 'start_str': '                               & no SSL & '},
           {'only_linear': True, 'use_ssl': True, 'start_str': '\multirow{2}{*}{Linear} & SSL & '},
           {'only_linear': True, 'use_ssl': False, 'start_str': '                               & no SSL & '}]


if __name__ == '__main__':
    test_path = 'D:\ssl_training_results\\2022-10-23 23_17_08'
    print_str = ''
    for config in configs:
        print_str += '\n' + config['start_str']
        for task_name in ['\\hw_running_VALR', '\\Carmargo_peak_fy', '\\walking_knee_moment_KFM']:
            print_str = print_one_task(test_path + task_name + '.h5', config, print_str)
        print_str = print_str[:-2] + '\\\\'
    print(print_str)












