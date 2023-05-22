import h5py
from ssl_main.const import RESULTS_PATH
from torch.nn import functional as F
import torch
import os
import matplotlib.pyplot as plt


def calculate_similarity(emb1, emb2):
    emb1 = F.normalize(emb1, p=2, dim=-1)
    emb2 = F.normalize(emb2, p=2, dim=-1)
    similarity = torch.matmul(emb1, emb2.transpose(1, 2))
    return similarity / temperature


def load_embeddings(embedding_name):
    with h5py.File(os.path.join(embedding_path, embedding_name + '.h5'), 'r') as hf:        # self.ssl_task['ssl_file_name'] + '.h5'
        # self.ssl_data = {sub_: sub_data[:] for sub_, sub_data in hf.items()}
        # self.ssl_columns = json.loads(hf.attrs['columns'])
        mod_acc = hf[embedding_name]['mod_acc'][:]
        mod_gyr = hf[embedding_name]['mod_gyr'][:]
    return mod_acc, mod_gyr


def compute_similarity_map_between_segments(embedding_name, num_of_segment=17):
    mod_acc, mod_gyr = load_embeddings(embedding_name)
    mod_acc, mod_gyr = torch.from_numpy(mod_acc), torch.from_numpy(mod_gyr)

    # num_of_windows = int(mod_acc.shape[0] / num_of_segment)
    sim_mat = []
    for i_sample in range(0, mod_acc.shape[0], num_of_segment):
        mod_acc_win, mod_gyr_win = mod_acc[i_sample:i_sample+num_of_segment], mod_gyr[i_sample:i_sample+num_of_segment]
        sim_one_win = calculate_similarity(torch.unsqueeze(mod_acc_win, 0), torch.unsqueeze(mod_gyr_win, 0))[0]
        # sim_one_win_acc_only = calculate_similarity(torch.unsqueeze(mod_acc_win, 0), torch.unsqueeze(mod_acc_win, 0))[0]
        # sim_one_win_gyr_only = calculate_similarity(torch.unsqueeze(mod_gyr_win, 0), torch.unsqueeze(mod_gyr_win, 0))[0]
        sim_mat.append(sim_one_win)
    mat = torch.logsumexp(torch.stack(sim_mat), dim=0).numpy()
    return mat


def show_sub_mat(mat, segments_to_show):
    # 对角线，用灰色替代
    index = [imu_segments.index(segment) for segment in segments_to_show]
    plt.figure()
    plt.imshow(mat[index, :][:, index])
    ax = plt.gca()
    ax.set_xticks(range(len(segments_to_show)))
    ax.set_yticks(range(len(segments_to_show)))
    ax.set_xticklabels(segments_to_show, rotation=90)
    ax.set_yticklabels(segments_to_show)
    plt.tight_layout()


temperature = 0.1
num_of_segment = 17
imu_segments = ssl_task_Camargo['imu_segments']

embedding_path = os.path.join(RESULTS_PATH, 'embedding_similarity_between_segments')
# pre_mat = compute_similarity_map_between_segments('pre_ssl')
post_mat = compute_similarity_map_between_segments('post_ssl')

# 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot'
# 'RightArm', 'RightForeArm'， 'RightHand'
# 'Hip', 'Spine1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Head',
#     'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'
show_sub_mat(post_mat, ['RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot'])
plt.show()





















