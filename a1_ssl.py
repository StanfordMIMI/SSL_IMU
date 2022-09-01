import copy
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import random
from scipy.stats import pearsonr
import datetime
from customized_logger import logger as logging, add_file_handler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error as mse
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset
from model import nce_loss, ImuTransformerEmbedding, ImuFcnnEmbedding, ImuRnnEmbedding, SslContrastiveNet, \
    CnnEmbedding, LinearRegressNet, SslReconstructNet, ImuResnetEmbedding
import time
from types import SimpleNamespace
import prettytable as pt
from utils import find_peak_max, fix_seed, DataStruct, data_filter
from const import DICT_SUBJECT_ID, DATA_PATH, TRIAL_TYPES, GRAVITY, IMU_SAMPLE_RATE, EMG_SAMPLE_RATE, \
    DICT_TRIAL_TYPE_ID, IMU_SEGMENT_LIST
import json


def prepare_dl(data_list, batch_size, shuffle):
    data_list_torch = [torch.from_numpy(data).float() for data in data_list]
    ds = TensorDataset(*data_list_torch)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl


def set_dtype_and_model(device, model):
    if device == 'cuda':
        dtype = torch.cuda.FloatTensor
        model.cuda()
    else:
        dtype = torch.FloatTensor
    return dtype, model


def get_non_zero_max(data_):
    zero_loc = (data_ == 0.).all(axis=1).reshape([data_.shape[0], 1, -1])
    zeros_to_exclude = np.concatenate([zero_loc for i in range(data_.shape[1])], axis=1)
    data_[zeros_to_exclude] = np.nan
    max_vals = np.nanmax(data_[:, :, 40:], axis=2)      # !!!
    max_vals[np.isnan(max_vals)] = 0.
    return max_vals


class FrameworkSSL:
    def __init__(self, config):
        self.config = SimpleNamespace(**config)
        logging.info('Loading ' + DATA_PATH + self.config.file_name + '.h5')
        with h5py.File(DATA_PATH + self.config.file_name + '.h5', 'r') as hf:
            if self.config.use_step_num is not None:
                self.data = {sub_: sub_data[:self.config.use_step_num] for sub_, sub_data in hf.items()}
            else:
                self.data = {sub_: sub_data[:] for sub_, sub_data in hf.items()}
            # self.data = np.concatenate(self.data, axis=0)
            self.columns = json.loads(hf.attrs['columns'])
        self.result_dir = os.path.join('D://SSL_training_results', self.result_folder())
        os.makedirs(os.path.join(self.result_dir), exist_ok=True)
        add_file_handler(logging, os.path.join(self.result_dir, 'training_log.txt'))

        self.mod_channel_num = [len(GROUPS_OF_DATA[mod]) for mod in da_task['input_mods']]
        self.emb_nets = torch.nn.ModuleList([self.config.emb_net(1, self.config.emb_output_dim, mod + ' embedding') for mod in da_task['input_mods']])
        self.regress_net = LinearRegressNet(self.emb_nets, self.mod_channel_num, 1)
        self.regress_net_init_state = self.regress_net.state_dict()

        # num_params = sum(param.numel() for param in model.embnet_imu.rnn_layer.parameters())
        # print(num_params)

        self._base_scalar = StandardScaler
        self._data_scalar = {}
        fix_seed()

    def reset_regress_net_to_init_state(self):
        self.regress_net.load_state_dict(self.regress_net_init_state)

    def reset_regress_net_to_post_ssl_state(self):
        self.regress_net.load_state_dict(self.regress_net_post_ssl_state)

    @staticmethod
    def result_folder():
        folder_name = str(datetime.datetime.now())[:-7]
        for item in ['.', ':']:
            folder_name = folder_name.replace(item, '_')
        return folder_name

    @staticmethod
    def select_data_by_list_of_values(data, column_index, the_val_list):
        vals = data[:, column_index, 0]
        selected_loc = [np.where(vals == the_val)[0] for the_val in the_val_list]
        selected_loc = np.concatenate(selected_loc, axis=0)
        data_selected = data[selected_loc]
        return data_selected

    @staticmethod
    def select_data_by_has_nonzero_element(data, column_index, search_start_percent=0., search_end_percent=1.):
        seq_len = data.shape[2]
        search_start = int(search_start_percent * seq_len)
        search_end = int(search_end_percent * seq_len)
        vals = data[:, column_index, search_start:search_end]
        vals = np.sum(vals, axis=1)
        selected_rows = np.where(vals != 0)[0]
        return data[selected_rows]

    def preprocess(self, train_sub_ids: List[str], validate_sub_ids: List[str],
                   test_sub_ids: List[str]):
        """
        train_sub_ids: a list of subject id for model training
        validate_sub_ids: a list of subject id for model validation
        test_sub_ids: a list of subject id for model testing
        """
        train_data = np.concatenate([self.data[sub] for sub in train_sub_ids], axis=0)
        vali_data = np.concatenate([self.data[sub] for sub in validate_sub_ids], axis=0)
        test_data = np.concatenate([self.data[sub] for sub in test_sub_ids], axis=0)

        # SSL preprocess
        train_data_ssl = self.preprocess_modality(train_data, 'fit_transform')
        logging.info('SSL training with subject ids: {}. Number of steps: {}'.format(train_sub_ids, list(train_data_ssl.values())[0].shape[0]))
        vali_data_ssl = self.preprocess_modality(vali_data, 'transform')
        logging.info('SSL validation with subject ids: {}. Number of steps: {}'.format(validate_sub_ids, list(vali_data_ssl.values())[0].shape[0]))
        test_data_ssl = self.preprocess_modality(test_data, 'transform')
        logging.info('SSL testing with subject ids: {}. Number of steps: {}'.format(test_sub_ids, list(test_data_ssl.values())[0].shape[0]))
        self.train_data_ssl, self.vali_data_ssl, self.test_data_ssl = train_data_ssl, vali_data_ssl, test_data_ssl

        # downstream_preprocess
        for data, data_name, norm_method in zip([train_data, vali_data, test_data], ['train', 'vali', 'test'],
                                                ['fit_transform', 'transform', 'transform']):
            type_to_exclude = [DICT_TRIAL_TYPE_ID[type] for type in da_task['remove_trial_type']]
            data_selected = self.select_data_by_list_of_values(data, self.columns.index('trial_type_id'), [i for i in range(4) if i not in type_to_exclude])
            data_selected = self.select_data_by_has_nonzero_element(data_selected, self.columns.index(da_task['output']), 0.4, 0.6)
            logging.info('Downstream {}. Number of steps: {}'.format(data_name, data_selected.shape[0]))
            data_ds = self.preprocess_modality(data_selected, 'transform')
            output_selected = data_selected[:, self.columns.index(da_task['output'])]
            output_loc = np.argmax(np.abs(output_selected), axis=1)
            output_ = np.array([output_selected[i_row, loc] for i_row, loc in enumerate(output_loc)]).reshape([-1, 1])
            data_ds[da_task['name']] = self.normalize_data(output_, da_task['name'], norm_method, 'by_all_columns')
            da_task[data_name] = data_ds

    def preprocess_modality(self, data_, norm_method):
        processed_data = {}
        for group_name, cols in GROUPS_OF_DATA.items():
            col_loc = [self.columns.index(col) for col in cols]
            group_data = data_[:, col_loc, :]
            group_data = self.normalize_data(group_data, group_name, norm_method, 'by_all_columns')
            processed_data[group_name] = group_data
        return processed_data

    @staticmethod
    def show_params(params):
        plt.figure()
        for i, param in enumerate(params):
            plt.plot(param.cpu().detach().numpy(), [i for _ in param], '.', markersize=1)

    @staticmethod
    def print_table(results):
        tb = pt.PrettyTable()
        for test_result in results:
            tb.field_names = test_result.keys()
            tb.add_row([np.round(np.mean(value), 3) if isinstance(value, (np.ndarray, float)) else value
                        for value in test_result.values()])
        logging.info(tb)

    def regressibility(self, only_linear_layer):
        def convert_batch_data(batch_data):
            xb = [data_.float().type(dtype) for data_ in batch_data[:-2]]
            yb = batch_data[-2].float().type(dtype)
            lens = batch_data[-1].float()
            return xb, yb, lens

        def train_batch(model, train_dl, optimizer, loss_fn):
            model.train()
            for i_batch, batch_data in enumerate(train_dl):
                n = random.randint(1, 100)
                optimizer.zero_grad()
                xb, yb, lens = convert_batch_data(batch_data)
                y_pred = model(xb, lens)
                loss = loss_fn(yb, y_pred)
                wandb.log({'linear batch loss': loss.item()})
                loss.backward()
                optimizer.step()

        def eval_during_training(model, dl, loss_fn, use_batch_num=5):
            model.eval()
            loss = []
            with torch.no_grad():
                for i_batch, batch_data in enumerate(dl):
                    if i_batch > use_batch_num:
                        continue
                    xb, yb, lens = convert_batch_data(batch_data)
                    y_pred = model(xb, lens)
                    loss.append(loss_fn(yb, y_pred).item())
            return np.mean(loss)

        def evaluate_after_training(test_dl):
            model.eval()
            with torch.no_grad():
                y_pred_list, y_true_list = [], []
                for i_batch, batch_data in enumerate(test_dl):
                    xb, yb, lens = convert_batch_data(batch_data)
                    y_pred_list.append(model(xb, lens).detach().cpu())
                    y_true_list.append(yb.detach().cpu())
                y_pred = torch.cat(y_pred_list).numpy()
                y_true = torch.cat(y_true_list).numpy()
            return y_pred, y_true

        train_data, test_data = da_task['train'], da_task['test']
        train_input_data = [train_data[mod] for mod in da_task['input_mods']]
        train_output_data = train_data[da_task['name']]
        train_step_lens = self._get_step_len(train_input_data[0])
        train_dl = prepare_dl([*train_input_data, train_output_data, train_step_lens], int(self.config.batch_size_linear), shuffle=True)
        test_input_data = [test_data[mod] for mod in da_task['input_mods']]
        test_output_data = test_data[da_task['name']]
        test_step_lens = self._get_step_len(test_input_data[0])
        test_dl = prepare_dl([*test_input_data, test_output_data, test_step_lens], int(self.config.batch_size_linear), shuffle=False)

        model = self.regress_net
        dtype, model = set_dtype_and_model(self.config.device, model)
        if only_linear_layer:
            logging.info('Regressing, only linear')
            param_to_train = model.linear.parameters()
            lr = 1e-3
        else:
            logging.info('Regressing, all params')
            param_to_train = model.parameters()
            lr = 1e-4

        optimizer = torch.optim.Adam(param_to_train, lr, weight_decay=1e-5)
        epoch_end_time = time.time()

        for i_epoch in range(self.config.epoch_regress):
            train_loss = eval_during_training(model, train_dl, torch.nn.MSELoss())
            test_loss = eval_during_training(model, test_dl, torch.nn.MSELoss())
            wandb.log({'linear train loss': train_loss, 'linear test loss': test_loss})     # , 'linear lr': scheduler.get_last_lr()[0]
            logging.info(f'| Linear Regressibility | epoch{i_epoch:3d}/{self.config.epoch_regress:3d} | time: {time.time() - epoch_end_time:5.2f}s |'
                         f' train loss {train_loss:5.3f} | test loss {test_loss:5.3f}')
            epoch_end_time = time.time()
            train_batch(model, train_dl, optimizer, torch.nn.MSELoss())

        y_pred, y_true = evaluate_after_training(test_dl)

        plt.figure()
        plt.title('Test')
        plt.plot(y_true.ravel(), y_pred.ravel(), '.')
        plt.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], color='black')
        plt.xlabel('True')
        plt.ylabel('Predicted')

        all_scores = FrameworkSSL.get_scores(y_true, y_pred, [da_task['output']], test_step_lens)
        all_scores = [{'subject': 'all', **scores} for scores in all_scores]
        self.print_table(all_scores)

        return y_pred, model

    def ssl_training(self, ssl_task_config):
        def train_batch(model, train_dl, optimizer, loss_fn):
            model.train()
            for i_batch, x in enumerate(train_dl):
                n = random.randint(1, 100)
                optimizer.zero_grad()
                xb_mod_a = x[0].float().type(dtype)
                xb_mod_b = x[1].float().type(dtype)
                lens = x[2].float()
                emb_a, emb_b = model(xb_mod_a, xb_mod_b, lens)
                loss = loss_fn(emb_a, emb_b)
                wandb.log({'ssl batch loss': loss.item()})
                loss.backward()
                optimizer.step()

        def eval_during_training(model, dl, loss_fn, use_batch_num=5):
            model.eval()
            with torch.no_grad():
                validation_loss = []
                for i_batch, x in enumerate(dl):
                    if i_batch > use_batch_num:
                        continue
                    xb_mod_a = x[0].float().type(dtype)
                    xb_mod_b = x[1].float().type(dtype)
                    lens = x[2].float()
                    emb_a, emb_b = model(xb_mod_a, xb_mod_b, lens)
                    validation_loss.append(loss_fn(emb_a, emb_b).item())
            return np.mean(validation_loss)

        mod_a, mod_b = ssl_task_config['mod_a'], ssl_task_config['mod_b']
        train_data, vali_data = self.train_data_ssl, self.vali_data_ssl
        train_step_lens = self._get_step_len(train_data[mod_a])
        vali_step_lens = self._get_step_len(vali_data[mod_a])
        # emb_nets = torch.nn.ModuleList([self.config.emb_net(1, self.config.emb_output_dim, mod + ' embedding') for mod in da_task['input_mods']])
        model = SslContrastiveNet(self.emb_nets[0], self.emb_nets[1], self.config.common_space_dim, [train_data[mod_a].shape[1], train_data[mod_b].shape[1]])
        wandb.watch(model, ssl_task_config['ssl_loss_fn'], log='all', log_freq=20)

        train_dl = prepare_dl([train_data[mod_a], train_data[mod_b], train_step_lens], int(self.config.batch_size_ssl), shuffle=True)
        vali_dl = prepare_dl([vali_data[mod_a], vali_data[mod_b], vali_step_lens], int(self.config.batch_size_ssl), shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr_ssl, weight_decay=1e-5)

        # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=int(self.config.epoch_ssl/4))
        dtype, model = set_dtype_and_model(self.config.device, model)
        # current_best_loss = eval_during_training(model, vali_dl, training_task['ssl_loss_fn'])
        epoch_end_time = time.time()
        for i_epoch in range(self.config.epoch_ssl):
            train_loss = eval_during_training(model, train_dl, ssl_task_config['ssl_loss_fn'])
            test_loss = eval_during_training(model, vali_dl, ssl_task_config['ssl_loss_fn'])
            logging.info(f'| SSL | epoch{i_epoch:3d}/{self.config.epoch_ssl:3d} | time: {time.time() - epoch_end_time:5.2f}s | '
                         f'train loss {train_loss:5.4f} | test loss {test_loss:5.4f}')
            epoch_end_time = time.time()
            train_batch(model, train_dl, optimizer, ssl_task_config['ssl_loss_fn'])
            wandb.log({'ssl train loss': train_loss, 'ssl test loss': test_loss})
            # scheduler.step()
        self.regress_net_post_ssl_state = copy.deepcopy(self.regress_net.state_dict())
        return {'model': model}

    def save_model_and_results(self, ):
        pass

    def normalize_data(self, data, name, method, scalar_mode):
        if method == 'fit_transform':
            self._data_scalar[name] = self._base_scalar()
        assert (scalar_mode in ['by_all_columns'])
        input_data = data.copy()
        original_shape = input_data.shape
        if len(input_data.shape) == 3:
            zero_loc = (input_data == 0.).all(axis=1)
            for i in range(input_data.shape[1]):
                input_data[:, i][zero_loc] = np.nan
            input_data = input_data.reshape([-1, 1])
        else:
            input_data[(input_data == 0.).all(axis=1), :] = np.nan
        scaled_data = getattr(self._data_scalar[name], method)(input_data)
        scaled_data = scaled_data.reshape(original_shape)
        scaled_data[np.isnan(scaled_data)] = 0.
        return scaled_data

    @staticmethod
    def down_sample_data(data, new_len):
        ratio = int(data.shape[1] / new_len)
        new_data = data[:, ::ratio, :]
        if new_data.shape[1] != new_len:
            raise RuntimeError('Check ori_len and new_len')
        return new_data

    def hyperparam_tuning(self, model, x_test):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def get_profile_scores(y_true, y_pred, y_fields, weight=None):
        def get_column_score(arr_true, arr_pred, w):
            r2, rmse, cor_value = [np.zeros(arr_true.shape[0]) for _ in range(3)]
            for i in range(arr_true.shape[0]):
                arr_true_i = arr_true[i, w[i, :]]
                arr_pred_i = arr_pred[i, w[i, :]]
                r2[i] = r2_score(arr_true_i, arr_pred_i)
                rmse[i] = np.sqrt(mse(arr_true_i, arr_pred_i))
                cor_value[i] = pearsonr(arr_true_i, arr_pred_i)[0]
            return {'r2': np.mean(r2), 'rmse': np.mean(rmse), 'cor_value': np.mean(cor_value)}

        scores = []
        for col, field in enumerate(y_fields):
            y_true_one_field = y_true[:, :, col]
            y_pred_one_field = y_pred[:, :, col]
            if weight is None:
                weight_one_field = np.full(y_true_one_field.shape, True)
            score_one_field = {'field': field}
            score_one_field.update(get_column_score(y_true_one_field, y_pred_one_field, weight_one_field))
            scores.append(score_one_field)
        return scores

    @staticmethod
    def get_scores(y_true, y_pred, y_fields, lens):
        scores = []
        for col, field in enumerate(y_fields):
            if len(y_true.shape) == 2:
                r2 = r2_score(y_true[:, col], y_pred[:, col])
                rmse = np.sqrt(mse(y_true[:, col], y_pred[:, col]))
                cor_value = pearsonr(y_true[:, col], y_pred[:, col])[0]
            else:
                r2, rmse, cor_value = [np.zeros(y_true.shape[0]) for _ in range(3)]
                for i_step in range(y_true.shape[0]):
                    y_true_one_step = y_true[i_step, :lens[i_step], col]
                    y_pred_one_step = y_pred[i_step, :lens[i_step], col]
                    r2[i_step] = r2_score(y_true_one_step, y_pred_one_step)
                    rmse[i_step] = np.sqrt(mse(y_true_one_step, y_pred_one_step))
                    cor_value[i_step] = pearsonr(y_true_one_step, y_pred_one_step)[0]
            score_one_field = {'field': field, 'r2': r2, 'rmse': rmse, 'cor_value': cor_value}
            scores.append(score_one_field)
        return scores

    @staticmethod
    def _get_step_len(data, feature_col=[0, 1, 2]):
        """
        :param data: Numpy array, 3d (step, sample, feature)
        :param feature_col: int, feature column id for step length detection. Different id would probably return
               the same results
        :return:
        """
        data_the_feature = data[:, feature_col, :]
        zero_loc = data_the_feature == 0.
        data_len = np.sum(~zero_loc, axis=2)
        data_len = np.max(data_len, axis=1)
        return data_len


PARAMS_TRIED = ['ramp', 'treadmill_speed', 'peak_knee_extension_angle']

MODALITIES = ['acc', 'gyr', 'emg']
GROUPS_OF_DATA = {
    'acc': [segment + '_Accel_' + axis for segment in ['trunk', 'shank'] for axis in ['X', 'Y', 'Z']],
    'gyr': [segment + '_Gyro_' + axis for segment in ['trunk', 'shank'] for axis in ['X', 'Y', 'Z']],
    'emg': ['gastrocmed', 'tibialisanterior', 'soleus']}
# EMG_LIST = ['gastrocmed', 'tibialisanterior', 'soleus', 'vastusmedialis', 'vastuslateralis', 'rectusfemoris',
#             'bicepsfemoris', 'semitendinosus', 'gracilis', 'gluteusmedius']


DOWNSTREAM_TASK_0 = {'name': 'peak_fy', 'input_mods': ['acc', 'acc'], 'remove_trial_type': ['Treadmill', 'Stair', 'Ramp'],
                     'output': 'peak_fy', 'processes': [], 'sensors': ['trunk IMU', 'shank IMU']}
DOWNSTREAM_TASK_1 = {'name': 'peak_knee_angle_r_moment', 'input_mods': ['acc', 'gyr'], 'remove_trial_type': ['Treadmill'],
                     'output': 'peak_knee_angle_r_moment', 'processes': [], 'sensors': ['trunk IMU', 'shank IMU']}
DOWNSTREAM_TASK_2 = {'name': 'peak_knee_angle_r', 'input_mods': ['acc', 'gyr'], 'remove_trial_type': [],
                     'output': 'peak_knee_angle_r', 'processes': []}
da_task = DOWNSTREAM_TASK_0
SSL_CONTRASTIVE_TASK = {'name': 'contrastive', 'ssl_loss_fn': nce_loss, 'mod_a': 'acc', 'mod_b': 'acc'}

# 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r'
# 'hip_flexion_r_moment', 'hip_adduction_r_moment', 'hip_rotation_r_moment', 'knee_angle_r_moment', 'ankle_angle_r_moment'
config = {'epoch_ssl': 30, 'epoch_regress': 30, 'batch_size_ssl': 2048, 'batch_size_linear': 128, 'lr_ssl': 1e-4,
          'emb_output_dim': 32, 'common_space_dim': 128, 'device': 'cuda', 'use_step_num': None,
          'emb_net': CnnEmbedding, 'file_name': 'UnivariantWinTest'}
wandb.init(project="IMU_EMG_SSL", config=config, name='linear prompt, ramp angle')


if __name__ == '__main__':
    # logging.info("Current commit is {}".format(execute_cmd("git rev-parse HEAD")))
    train_set = [
        'AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB17',
        'AB18', 'AB19', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28',
        'AB30'
    ]
    test_set = ['AB06',
                'AB07', 'AB08', 'AB09'
                ]
    ssl_framework = FrameworkSSL(config)
    ssl_framework.preprocess(train_set, test_set, test_set)

    model = ssl_framework.ssl_training(SSL_CONTRASTIVE_TASK)
    y_pred, model = ssl_framework.regressibility(only_linear_layer=False)
    ssl_framework.reset_regress_net_to_post_ssl_state()
    y_pred, model = ssl_framework.regressibility(only_linear_layer=True)
    ssl_framework.reset_regress_net_to_init_state()
    y_pred, model = ssl_framework.regressibility(only_linear_layer=True)
    ssl_framework.reset_regress_net_to_init_state()
    y_pred, model = ssl_framework.regressibility(only_linear_layer=False)

    plt.show()



