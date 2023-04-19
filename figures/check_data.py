import matplotlib.pyplot as plt
from PaperFigures import load_da_data


data_path = 'D:\ssl_training_results\\2022-10-24 16_30_35_all_test'
test_name = '\\Camargo_peak_fy'
if __name__ == "__main__":
    # !!! 加一组线，KAM dataset for SSL
    metric = 'correlation'
    results_task = load_da_data(data_path + test_name + '.h5')['linear_protocol_False, use_ssl_True, ratio_1.0']
    for i_sub in range(0, 5):
        if 'sub_'+str(i_sub) not in results_task.keys(): continue
        data_ = results_task['sub_'+str(i_sub)]
        plt.figure()
        plt.plot(data_[:, 0], data_[:, 1], '.')
        plt.plot([0, 2], [0, 2], 'black')
    plt.show()
    x=1


