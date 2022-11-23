import random

import h5py
import json
from const import DATA_PATH, RESULTS_PATH
from utils import preprocess_modality, define_channel_names
from torch.nn import functional as F
import torch
import numpy as np
import os
from a1_ssl import ssl_task_Carmargo, DOWNSTREAM_TASK_3
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def calculate_similarity(emb1, emb2):
    emb1 = F.normalize(emb1, p=2, dim=-1)
    emb2 = F.normalize(emb2, p=2, dim=-1)
    similarity = torch.matmul(emb1, emb2.transpose(1, 2))
    return similarity / temperature


def load_embeddings(embedding_name, num_to_select=200):
    with h5py.File(os.path.join(data_path, 'embedding_similarity_between_segments', test_name + '.h5'), 'r') as hf:
        random.seed(10)
        num_total = hf[embedding_name]['mod_acc'].shape[0]
        rows_to_select = np.sort(random.sample(range(num_total), num_to_select))
        mod_acc = hf[embedding_name]['mod_acc'][rows_to_select]
        mod_gyr = hf[embedding_name]['mod_gyr'][rows_to_select]
        # mod_acc = hf[embedding_name]['mod_acc'][:100]     # !!!
        # mod_gyr = hf[embedding_name]['mod_gyr'][:100]
    return mod_acc, mod_gyr


def draw_tsne(mod_data, title=''):
    emb_len = int(mod_data.shape[1] / num_of_segment)
    num_of_wins = mod_data.shape[0]
    mod_data_reshaped = np.zeros([0, emb_len])
    for i_segment in range(num_of_segment):
        mod_data_reshaped = np.concatenate([mod_data_reshaped, mod_data[:, i_segment*emb_len:(i_segment+1)*emb_len]], axis=0)
    mod_transformed = TSNE(random_state=0).fit_transform(mod_data_reshaped)
    plt.figure()
    plt.title(title)
    for i_segment in range(num_of_segment):
        segment_data = mod_transformed[i_segment*num_of_wins:(i_segment+1)*num_of_wins]
        plt.scatter(segment_data[:, 0], segment_data[:, 1], label=DOWNSTREAM_TASK_3['imu_segments'][i_segment])
    plt.legend()


temperature = 0.1
num_of_segment = 8

data_path = 'D:\ssl_training_results\\2022-11-18 22_49_15'
test_name = 'hw_running_VALR'

mod_acc_no_ssl, mod_gyr_no_ssl = load_embeddings('no_ssl')
draw_tsne(mod_gyr_no_ssl)
mod_acc_post_ssl, mod_gyr_post_ssl = load_embeddings('use_ssl')
draw_tsne(mod_gyr_post_ssl)
# post_mat = draw_tsne('post_ssl')

plt.show()





















