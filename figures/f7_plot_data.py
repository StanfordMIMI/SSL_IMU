import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
from figures.PaperFigures import save_fig, load_da_data, results_dict_to_pd_profiles
from ssl_main.config import RESULTS_PATH


if __name__ == "__main__":
    # hw_running_VALR   walking_knee_moment_output  Camargo_output   sun_drop_jump_output
    test_name = '/Camargo_100_output'
    test_folder = 't03_test_camargo_filtered'
    data_path = RESULTS_PATH + test_folder
    results_task = load_da_data(data_path + test_name + '.h5')

    for key_, value_ in results_task.items():
        plt.figure()
        plt.title(key_)
        plt.plot(value_['sub_0'][0, 1])
        plt.plot(value_['sub_0'][0, 4])
    plt.show()

