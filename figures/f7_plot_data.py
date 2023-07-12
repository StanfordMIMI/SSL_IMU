import os, sys
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
from figures.PaperFigures import save_fig, load_da_data, results_to_pd_summary
from ssl_main.config import RESULTS_PATH


if __name__ == "__main__":
    # hw_running_VALR   walking_knee_moment_output  Camargo_100_output   sun_drop_jump_output
    test_name = '/Camargo_levelground_output'
    test_folder = 't04_da_opencap'
    data_path = RESULTS_PATH + test_folder
    results_task = load_da_data(data_path + test_name + '.h5')

    for key_, value_ in list(results_task.items()):
        plt.figure()
        plt.title(key_)
        sub_id = np.concatenate([np.full(value_sub.shape[0] * 128, 0.1 * i) for i, value_sub in enumerate(list(value_.values()))], axis=0)
        plt.plot(np.concatenate(list(value_.values()), axis=0)[:, 0].ravel())
        plt.plot(np.concatenate(list(value_.values()), axis=0)[:, 1].ravel())
        plt.plot(sub_id)

    plt.show()

