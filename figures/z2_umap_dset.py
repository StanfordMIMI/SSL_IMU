import random
import h5py
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import umap


imu_segments = ['R_FOOT', 'R_SHANK', 'R_THIGH', 'L_SHANK', 'L_THIGH', 'L_FOOT', 'WAIST', 'CHEST']


def load_ssl_data():
    with h5py.File(os.path.join(data_path, 'Combined_walking_knee_moment.h5'), 'r') as hf:
        ssl_data = {dset: dset_data[:] for dset, dset_data in hf.items()}
    return ssl_data


def random_select_samples(dset_data, num_to_select=100, column_to_show=24):
    random.seed(0)
    num_total = dset_data.shape[0]
    rows_to_select = np.sort(random.sample(range(num_total), num_to_select))
    ssl_data = dset_data[rows_to_select, :].reshape([num_to_select, -1])

    return ssl_data


def draw_umap(list_of_array, label, title=''):
    array_concatenated = np.concatenate(list_of_array, axis=0)
    array_transformed = umap.UMAP(n_neighbors=15, random_state=1).fit_transform(array_concatenated)
    plt.figure()
    plt.title(title)
    current_index = 0
    for i_cate in range(len(list_of_array)):
        # if i_cate == 8:
        #     x=1
        segment_data = array_transformed[current_index:current_index+list_of_array[i_cate].shape[0]]
        current_index = current_index+list_of_array[i_cate].shape[0]
        plt.scatter(segment_data[:, 0], segment_data[:, 1], s=5, label=label[i_cate])
    plt.legend()


data_path = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed'


ssl_data = load_ssl_data()
dset_data_list = []
labels = []
for dset, dset_data in ssl_data.items():
    # print(dset)
    if dset in ['dset0', 'dset4', 'dset7']:
        continue
    dset_data_list.append(random_select_samples(dset_data))
    labels.append(dset)
draw_umap(dset_data_list, labels, 'training datasets')

plt.show()


