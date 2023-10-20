import random
import h5py
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from matplotlib import transforms


imu_segments = ['R_FOOT', 'R_SHANK', 'R_THIGH', 'L_SHANK', 'L_THIGH', 'L_FOOT', 'PELVIS', 'TRUNK']


def load_ssl_data():
    with h5py.File(os.path.join(data_path, 'Combined_walking_knee_moment.h5'), 'r') as hf:
        ssl_data = {dset: dset_data[:] for dset, dset_data in hf.items()}
    return ssl_data


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def random_select_samples(dset_data, num_to_select=100):
    random.seed(0)
    num_total = dset_data.shape[0]
    rows_to_select = np.sort(random.sample(range(num_total), num_to_select))
    ssl_data = dset_data[rows_to_select, :].reshape([num_to_select, -1])

    return ssl_data


def draw_pca(list_of_array, label, title=''):
    array_concatenated = np.concatenate(list_of_array, axis=0)
    array_transformed = PCA(n_components=2, random_state=1).fit_transform(array_concatenated)
    # plt.figure()
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.title(title)
    ax = plt.gca()
    current_index = 0
    for i_cate in range(len(list_of_array)):
        segment_data = array_transformed[current_index:current_index+list_of_array[i_cate].shape[0]]
        current_index = current_index+list_of_array[i_cate].shape[0]
        confidence_ellipse(segment_data[:, 0], segment_data[:, 1], ax, n_std=1, edgecolor='red')
        plt.scatter(segment_data[:, 0], segment_data[:, 1], s=5, label=label[i_cate])
    plt.legend()


data_path = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed'


ssl_data = load_ssl_data()
dset_data_list = []
labels = []
for dset, dset_data in ssl_data.items():
    if dset in ['dset0', 'dset4', 'dset7', 'dset8']:
        continue
    dset_data_list.append(random_select_samples(dset_data))
    labels.append(dset)
draw_pca(dset_data_list, labels, 'training datasets')

plt.show()


