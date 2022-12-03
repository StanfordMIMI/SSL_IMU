import random
import h5py
import numpy as np
import os
from a1_ssl import ssl_task_Carmargo, DOWNSTREAM_TASK_3
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_embedding_separate_by_sensor(embedding_name, mod_to_process, num_to_select=100, all_subject=True):
    random.seed(0)
    with h5py.File(os.path.join(data_path, 'embedding_similarity_between_segments', test_name + '.h5'), 'r') as hf:
        if not all_subject:
            sub_selected = list(hf[embedding_name][mod_to_process].keys())[0]
            mod_data = hf[embedding_name][mod_to_process][sub_selected]
        else:
            mod_data = np.concatenate([sub_data for _, sub_data in hf[embedding_name][mod_to_process].items()], axis=0)

        num_total = mod_data.shape[0]
        rows_to_select = np.sort(random.sample(range(num_total), num_to_select))
        mod_data = mod_data[rows_to_select]
        num_of_segment = int(mod_data.shape[1] / emb_len)
        # num_of_wins = mod_data.shape[0]
        mod_data_reshaped = np.zeros([0, emb_len])
        for i_segment in range(num_of_segment):
            mod_data_reshaped = np.concatenate(
                [mod_data_reshaped, mod_data[:, i_segment * emb_len:(i_segment + 1) * emb_len]], axis=0)

    return mod_data_reshaped, num_of_segment


def draw_tsne(mod_data_reshaped, num_of_segment, title=''):
    mod_transformed = TSNE(random_state=0).fit_transform(mod_data_reshaped)
    num_of_wins = int(mod_transformed.shape[0] / num_of_segment)
    plt.figure()
    plt.title(title)
    for i_segment in range(num_of_segment):
        segment_data = mod_transformed[i_segment*num_of_wins:(i_segment+1)*num_of_wins]
        plt.scatter(segment_data[:, 0], segment_data[:, 1], s=5,
                    label=DOWNSTREAM_TASK_3['imu_segments'][i_segment])
    plt.legend()


emb_len = 128
data_path = 'D:\ssl_training_results\\2022-11-29 07_54_40'
test_name = 'walking_knee_moment'
mod_to_process = 'mod_acc'

mod_data_reshaped, num_of_segment = load_embedding_separate_by_sensor('no_ssl', mod_to_process)
draw_tsne(mod_data_reshaped, num_of_segment, 'no ssl')
mod_data_reshaped, num_of_segment = load_embedding_separate_by_sensor('use_ssl', mod_to_process)
draw_tsne(mod_data_reshaped, num_of_segment, 'use ssl')

plt.show()


