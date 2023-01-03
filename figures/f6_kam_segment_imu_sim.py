import random
import h5py
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap


DOWNSTREAM_TASK_3 = {'_mods': ['acc', 'gyr'], 'remove_trial_type': [], 'dataset': 'walking_knee_moment',
                     'output': 'KFM', 'imu_segments': ['R_FOOT', 'R_SHANK', 'R_THIGH', 'L_SHANK', 'L_THIGH', 'L_FOOT', 'WAIST', 'CHEST'], 'ssl_model': 'MoVi_'}

ssl_task_Carmargo = {'ssl_file_name': 'MoVi_Carmargo', 'imu_segments': [
    # 'Hip', 'Spine1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot']}
    'Hip', 'Spine1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']}

def load_mod_data(test_name, embedding_name, mod_to_process, all_subject=True):
    with h5py.File(os.path.join(data_path, 'embedding_similarity_between_segments', test_name + '.h5'), 'r') as hf:
        if not all_subject:
            sub_selected = list(hf[embedding_name][mod_to_process].keys())[0]
            mod_data = np.array(hf[embedding_name][mod_to_process][sub_selected])
        else:
            mod_data = np.concatenate([sub_data for _, sub_data in hf[embedding_name][mod_to_process].items()], axis=0)

    return mod_data


def load_mod_data_random_from_each_subject(test_name, embedding_name, mod_to_process, num_to_select=100, rows_to_select_subject=None):
    with h5py.File(os.path.join(data_path, 'embedding_similarity_between_segments', test_name + '.h5'), 'r') as hf:
        # mod_data = np.concatenate([sub_data for _, sub_data in hf[embedding_name][mod_to_process].items()], axis=0)
        mod_data, rows_to_select = [], []
        for _, sub_data in hf[embedding_name][mod_to_process].items():
            num_total = sub_data.shape[0]
            rows_to_select_subject = np.sort(random.sample(range(num_total), num_to_select))
            mod_data.append(sub_data[rows_to_select_subject])
        num_of_category = len(mod_data)
        mod_data = np.concatenate(mod_data, axis=0)
    return mod_data, num_of_category, rows_to_select


def random_select_categorize_to_segments(mod_data, num_to_select=100, rows_to_select=None):
    if rows_to_select is None:
        random.seed(0)
        num_total = mod_data.shape[0]
        rows_to_select = np.sort(random.sample(range(num_total), num_to_select))
    mod_data = mod_data[rows_to_select]

    num_of_category = num_of_segment = int(mod_data.shape[1] / emb_len)
    mod_data_reshaped = np.zeros([0, emb_len])
    for i_segment in range(num_of_segment):
        mod_data_reshaped = np.concatenate(
            [mod_data_reshaped, mod_data[:, i_segment * emb_len:(i_segment + 1) * emb_len]], axis=0)

    return mod_data_reshaped, num_of_category, rows_to_select


def random_select_categorize_to_trials(mod_data, trial_id_data, num_to_select=100, rows_to_select=None):
    trial_id_set = list(set(trial_id_data))
    if rows_to_select is None:
        rows_to_select = []
        for trial_id in trial_id_set:
            trial_loc = np.where(trial_id_data == trial_id)[0]
            rows_to_select_trial = np.random.choice(trial_loc, num_to_select)
            rows_to_select.append(rows_to_select_trial)
        rows_to_select = np.concatenate(rows_to_select, axis=0)
    mod_data = mod_data[rows_to_select]

    num_of_category = len(trial_id_set)
    # mod_data_reshaped = mod_data[:, :emb_len]       # Trunk IMU
    mod_data_reshaped = mod_data[:, emb_len:]       # Pelvis IMU
    return mod_data_reshaped, num_of_category, rows_to_select


def draw_tsne(mod_data_reshaped, num_of_category, label, title=''):
    # mod_transformed = TSNE(random_state=0).fit_transform(mod_data_reshaped)
    mod_transformed = umap.UMAP(n_neighbors=100, random_state=1).fit_transform(mod_data_reshaped)

    num_of_wins = int(mod_transformed.shape[0] / num_of_category)
    plt.figure()
    plt.title(title)
    for i_cate in range(num_of_category):
        segment_data = mod_transformed[i_cate*num_of_wins:(i_cate+1)*num_of_wins]
        plt.scatter(segment_data[:, 0], segment_data[:, 1], s=5, label=label[i_cate])
    plt.legend()


emb_len = 128
# data_path = sys.path[0] + '/results/' + '2022-12-09 21_29_35'
data_path = sys.path[0] + '/results/2022-12-23 09_54_56'
mod_to_process = 'mod_acc'


# """ UMAP of subjects """
# mod_data, num_of_category, rows_to_select = load_mod_data_random_from_each_subject('Carmargo', 'no_ssl', mod_to_process)
# draw_tsne(mod_data, num_of_category, range(num_of_category), 'no ssl')
# mod_data, num_of_category, _ = load_mod_data_random_from_each_subject('Carmargo', 'use_ssl', mod_to_process)
# draw_tsne(mod_data, num_of_category, range(num_of_category), 'use ssl')

# """ UMAP of ground conditions """
# trial_id_data = load_mod_data('Carmargo', 'no_ssl', 'info')
# mod_data = load_mod_data('Carmargo', 'no_ssl', mod_to_process)
# mod_data_reshaped, num_of_category, rows_to_select = random_select_categorize_to_trials(mod_data, trial_id_data)
# draw_tsne(mod_data_reshaped, num_of_category, ['LevelGround', 'Stair', 'Ramp'], 'no ssl')
# mod_data = load_mod_data('Carmargo', 'use_ssl', mod_to_process)
# mod_data_reshaped, num_of_category, _ = random_select_categorize_to_trials(mod_data, trial_id_data, rows_to_select=rows_to_select)
# draw_tsne(mod_data_reshaped, num_of_category, ['LevelGround', 'Stair', 'Ramp'], 'use ssl')

""" UMAP of segments """
mod_data = load_mod_data('walking_knee_moment', 'no_ssl', mod_to_process, all_subject=False)
mod_data_reshaped, num_of_segment, rows_to_select = random_select_categorize_to_segments(mod_data)
draw_tsne(mod_data_reshaped, num_of_segment, DOWNSTREAM_TASK_3['imu_segments'], 'no ssl')
mod_data = load_mod_data('walking_knee_moment', 'use_ssl', mod_to_process, all_subject=False)
mod_data_reshaped, num_of_segment, _ = random_select_categorize_to_segments(mod_data, rows_to_select=rows_to_select)
draw_tsne(mod_data_reshaped, num_of_segment, DOWNSTREAM_TASK_3['imu_segments'], 'use ssl')

# """ TSNE of MoVi """
# mod_data = load_mod_data('MoVi_walking_knee_moment', 'no_ssl', mod_to_process)
# mod_data_reshaped, num_of_segment, rows_to_select = random_select_categorize_to_segments(mod_data)
# draw_tsne(mod_data_reshaped, num_of_segment, ssl_task_Carmargo['imu_segments'], 'no ssl')
# mod_data = load_mod_data('MoVi_walking_knee_moment', 'use_ssl', mod_to_process)
# mod_data_reshaped, num_of_segment, _ = random_select_categorize_to_segments(mod_data, rows_to_select=rows_to_select)
# draw_tsne(mod_data_reshaped, num_of_segment, ssl_task_Carmargo['imu_segments'], 'use ssl')

plt.show()


