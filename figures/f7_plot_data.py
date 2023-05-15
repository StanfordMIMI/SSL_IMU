import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
from figures.PaperFigures import save_fig, load_da_data, results_to_pd_summary
from ssl_main.config import RESULTS_PATH


if __name__ == "__main__":
    # hw_running_VALR   walking_knee_moment_output  Camargo_100_output   sun_drop_jump_output
    test_name = '/Camargo_100_output'
    test_folder = '2023_05_01_12_54_17_comprehensive_da'
    data_path = RESULTS_PATH + test_folder
    results_task = load_da_data(data_path + test_name + '.h5')

    for key_, value_ in results_task.items():
        plt.figure()
        plt.title(key_)
        plt.plot(value_['sub_1'][:, 0].ravel())
        plt.plot(value_['sub_1'][:, 3].ravel())
        plt.figure()
        plt.title(key_)
        plt.plot(value_['sub_1'][:, 1].ravel())
        plt.plot(value_['sub_1'][:, 4].ravel())
        plt.figure()
        plt.title(key_)
        plt.plot(value_['sub_1'][:, 2].ravel())
        plt.plot(value_['sub_1'][:, 5].ravel())
    plt.show()

