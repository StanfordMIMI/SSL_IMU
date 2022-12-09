import numpy as np
from PaperFigures import load_da_data, results_dict_to_pd
from scipy.stats import ttest_rel
import sys


def get_one_metric(result_df, only_linear, use_ssl, ratio, metric):
    df = result_df[(result_df['only_linear'] == only_linear) &
                   (result_df['use_ssl'] == use_ssl) &
                   (result_df['ratio'] == ratio)]
    return df[metric]


def get_mean_std_of_one_metric(**args):
    metrics = get_one_metric(**args)
    return np.mean(metrics), np.std(metrics)


def print_one_task(task_dir, config, signi, task_name, print_str):
    results_task = load_da_data(task_dir)
    result_df = results_dict_to_pd(results_task)
    for metric in metrics_and_attributes:
        mean_, std_ = get_mean_std_of_one_metric(result_df=result_df, only_linear=config['only_linear'],
                                                   use_ssl=config['use_ssl'], ratio=1, metric=metric[0])
        mean_, std_ = str(np.round(mean_, metric[2])), str(np.round(std_, metric[2]))
        if mean_[-metric[2]] == '.': mean_ += '0'
        if std_[-metric[2]] == '.': std_ += '0'
        # signi_sign = ''
        # if signi[task_name][metric[0]][config['only_linear']] < 0.05 and config['use_ssl']:
        #     signi_sign = '*'
        # print_str += '${} \pm {}{}$ {} & '.format(mean_, std_, metric[1], signi_sign)
        print_str += '${} \pm {}{}$ & '.format(mean_, std_, metric[1])
    return print_str


configs = [{'only_linear': False, 'use_ssl': True, 'start_str': '\multirow{2}{*}{Fine-tuning}    & Self-Supervised Encoders & '},
           {'only_linear': False, 'use_ssl': False, 'start_str': '                               & Initial Encoders & '},
           {'only_linear': True, 'use_ssl': True, 'start_str': '\multirow{2}{*}{Linear} & Self-Supervised Encoders & '},
           {'only_linear': True, 'use_ssl': False, 'start_str': '                               & Initial Encoders & '}]
task_names = ['/hw_running_VALR', '/Carmargo_peak_fy', '/walking_knee_moment_KFM']
metrics_and_attributes = [['correlation', '', 2], ['r_rmse', '\%', 1]]

if __name__ == '__main__':
    test_path = sys.path[0] + '/results/2022-12-05 21_03_16'
    signi = {}
    for task_name in task_names:
        results_task = load_da_data(test_path + task_name + '.h5')
        result_df = results_dict_to_pd(results_task)
        signi[task_name] = {metric_attribute[0]: {protocol: {}} for metric_attribute in metrics_and_attributes for protocol in [False, True]}
        for metric_attribute in metrics_and_attributes:
            for linear_protocol in [False, True]:
                ssl_results = get_one_metric(result_df, False, True, 1, metric_attribute[0])
                no_ssl_results = get_one_metric(result_df, False, False, 1, metric_attribute[0])
                p_val = ttest_rel(ssl_results, no_ssl_results).pvalue
                signi[task_name][metric_attribute[0]][linear_protocol] = p_val

    print_str = ''
    for i_config, config in enumerate(configs):
        print_str += '\n' + config['start_str']
        for task_name in task_names:
            print_str = print_one_task(test_path + task_name + '.h5', config, signi, task_name, print_str)
        print_str = print_str[:-2] + '\\\\'
        if i_config == 1:
            print_str += '\n\midrule'
    print(print_str)












