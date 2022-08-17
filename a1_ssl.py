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
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from model import nce_loss, ImuTransformerEmbedding, ImuFcnnEmbedding, ImuRnnEmbedding, SslContrastiveNet, \
    CnnEmbedding, LinearRegressNet, SslReconstructNet, ImuResnetEmbedding
import time
from types import SimpleNamespace
import prettytable as pt
from utils import get_data_by_merging_data_struct, fix_seed, DataStruct, data_filter
from const import DICT_SUBJECT_ID, DATA_PATH, AMBULATIONS, GRAVITY, IMU_SAMPLE_RATE, EMG_SAMPLE_RATE
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
            self.data = [subject_data[:] for _, subject_data in hf.items()][0]
            self.columns = json.loads(hf.attrs['columns'])
        self.result_dir = os.path.join('D://SSL_training_results', self.result_folder())
        os.makedirs(os.path.join(self.result_dir), exist_ok=True)
        add_file_handler(logging, os.path.join(self.result_dir, 'training_log.txt'))

        emb_net = self.config.emb_net
        self.emb_nets = {mod: emb_net(1, self.config.emb_output_dim, mod+' embedding') for mod in MODALITIES}
        # num_params = sum(param.numel() for param in model.embnet_imu.rnn_layer.parameters())
        # print(num_params)
        # self.linear_regress_net = LinearRegressNet(self.emb_net_acc, self.emb_net_gyr, len(OUTPUT_LIST))
        self.init_states = {mod: self.emb_nets[mod].state_dict() for mod in MODALITIES}

        self._base_scalar = StandardScaler
        self._data_scalar = {}
        fix_seed()

    def reset_emb_net_to_init_state(self):
        [self.emb_nets[mod].load_state_dict(self.init_states[mod]) for mod in MODALITIES]

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

    def preprocess(self, train_sub_ids: List[str], validate_sub_ids: List[str],
                   test_sub_ids: List[str]):
        """
        train_sub_ids: a list of subject id for model training
        validate_sub_ids: a list of subject id for model validation
        test_sub_ids: a list of subject id for model testing
        """
        train_data = self.select_data_by_list_of_values(self.data, self.columns.index('sub_id'), [DICT_SUBJECT_ID[sub] for sub in train_sub_ids])
        train_data_grouped = self.preprocess_input_data(train_data, 'fit_transform')
        logging.info('Train the model with subject ids: {}. Number of steps: {}'.format(train_sub_ids, list(train_data_grouped.values())[0].shape[0]))

        vali_data = self.select_data_by_list_of_values(self.data, self.columns.index('sub_id'), [DICT_SUBJECT_ID[sub] for sub in validate_sub_ids])
        vali_data_grouped = self.preprocess_input_data(vali_data, 'transform')
        logging.info('Validate the model with subject ids: {}. Number of steps: {}'.format(validate_sub_ids, list(vali_data_grouped.values())[0].shape[0]))

        test_data = self.select_data_by_list_of_values(self.data, self.columns.index('sub_id'), [DICT_SUBJECT_ID[sub] for sub in test_sub_ids])
        test_data_grouped = self.preprocess_input_data(test_data, 'transform')
        logging.info('Test the model with subject ids: {}. Number of steps: {}'.format(test_sub_ids, list(test_data_grouped.values())[0].shape[0]))
        self.train_data, self.vali_data, self.test_data = train_data_grouped, vali_data_grouped, test_data_grouped

        # for i in range(6):
        # plt.figure()
        # col_loc = self.columns.index(GROUPS_OF_DATA['acc'][0])
        # plt.plot(train_data[:10, col_loc, :].ravel())
        # for i in range(1):
        #     plt.figure()
        #     # plt.plot(test_data_grouped['gyr'][:10, i, :].ravel())
        #     plt.plot(test_data_grouped[DOWNSTREAM_TASKS[0]['name']].ravel())
        #     plt.grid()
        # plt.show()

    def preprocess_input_data(self, data_, norm_method):
        processed_data = {}
        for group_name, cols in GROUPS_OF_DATA.items():
            col_loc = [self.columns.index(col) for col in cols]
            group_data = data_[:, col_loc, :]
            group_data = self.normalize_data(group_data, group_name, norm_method, 'by_all_columns')  # TODO: Try by_each_column
            processed_data[group_name] = group_data

        for task in DOWNSTREAM_TASKS:
            col_loc = [self.columns.index(col) for col in task['output']]
            group_data = data_[:, col_loc, :]
            for process in task['processes']:
                group_data = process(group_data)
            group_data = self.normalize_data(group_data, task['name'], norm_method, 'by_all_columns')
            processed_data[task['name']] = group_data
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

    def regressibility(self, downstream_task, only_linear_layer):
        def convert_batch_data(batch_data):
            xb = [data_.float().type(dtype) for data_ in batch_data[:-2]]
            yb = batch_data[-2].float().type(dtype)
            lens = batch_data[-1].float()
            return xb, yb, lens

        def train_batch(model, train_dl, optimizer, loss_fn, use_ratio):
            model.train()
            for i_batch, batch_data in enumerate(train_dl):
                n = random.randint(1, 100)
                if n > use_ratio:
                    continue  # increase the speed of epoch
                optimizer.zero_grad()
                xb, yb, lens = convert_batch_data(batch_data)
                y_pred = model(xb, lens)
                loss = loss_fn(yb, y_pred)
                wandb.log({'linear batch loss': loss.item()})
                loss.backward()
                optimizer.step()

        def eval_during_training(model, dl, loss_fn, use_ratio_for_monitoring=5):
            model.eval()
            loss = []
            with torch.no_grad():
                for batch_data in dl:
                    n = random.randint(1, 100)
                    if len(dl) > 10 and n > use_ratio_for_monitoring:
                        continue  # increase the speed of epoch
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

        train_data, test_data = self.train_data, self.test_data
        train_input_data = [train_data[mod] for mod in downstream_task['input_mods']]
        train_output_data = train_data[downstream_task['name']]
        train_step_lens = self._get_step_len(train_input_data[0])
        train_dl = prepare_dl([*train_input_data, train_output_data, train_step_lens], int(self.config.batch_size_linear), shuffle=True)
        test_input_data = [test_data[mod] for mod in downstream_task['input_mods']]
        test_output_data = test_data[downstream_task['name']]
        test_step_lens = self._get_step_len(test_input_data[0])
        test_dl = prepare_dl([*test_input_data, test_output_data, test_step_lens], int(self.config.batch_size_linear), shuffle=False)

        emb_nets = torch.nn.ModuleList([self.emb_nets[mod] for mod in downstream_task['input_mods']])
        mod_channel_num = [mod_data.shape[1] for mod_data in train_input_data]
        model = LinearRegressNet(emb_nets, mod_channel_num, len(downstream_task['output']))
        wandb.watch(model, torch.nn.MSELoss(), log='all', log_freq=20)

        # model = self.linear_regress_net
        dtype, model = set_dtype_and_model(self.config.device, model)
        if only_linear_layer:
            logging.info('Regressing, only linear')
            param_to_train = model.linear.parameters()
        else:
            logging.info('Regressing, all params')
            param_to_train = model.parameters()

        optimizer = torch.optim.Adam(param_to_train, lr=self.config.lr_linear, weight_decay=1e-5)
        # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=int(self.config.epoch_linear/4))
        epoch_end_time = time.time()
        for i_epoch in range(self.config.epoch_linear):
            train_loss = eval_during_training(model, train_dl, torch.nn.MSELoss(), use_ratio_for_monitoring=2)
            test_loss = eval_during_training(model, test_dl, torch.nn.MSELoss(), use_ratio_for_monitoring=5)
            wandb.log({'linear train loss': train_loss, 'linear test loss': test_loss})     # , 'linear lr': scheduler.get_last_lr()[0]
            logging.info(f'| Linear Regressibility | epoch {i_epoch+1:3d} | time: {time.time() - epoch_end_time:5.2f}s | '
                         f'train loss {train_loss:5.3f} | test loss {test_loss:5.3f}')
            epoch_end_time = time.time()
            train_batch(model, train_dl, optimizer, torch.nn.MSELoss(), self.config.use_ratio)
            # scheduler.step()

        y_pred, y_true = evaluate_after_training(test_dl)

        plt.figure()
        plt.title('Test')
        plt.plot(y_true.ravel(), y_pred.ravel(), '.')
        plt.xlabel('True')
        plt.ylabel('Predicted')

        all_scores = FrameworkSSL.get_scores(y_true, y_pred, downstream_task['output'], test_step_lens)
        all_scores = [{'subject': 'all', **scores} for scores in all_scores]
        self.print_table(all_scores)

        return y_pred, model

    def ssl_training(self, ssl_task_config):
        def train_batch(model, train_dl, optimizer, loss_fn, use_ratio):
            model.train()
            for i_batch, x in enumerate(train_dl):
                n = random.randint(1, 100)
                if n > use_ratio:
                    continue  # increase the speed of epoch
                optimizer.zero_grad()
                xb_mod_a = x[0].float().type(dtype)
                xb_mod_b = x[1].float().type(dtype)
                lens = x[2].float()
                emb_a, emb_b = model(xb_mod_a, xb_mod_b, lens)
                loss = loss_fn(emb_a, emb_b)
                wandb.log({'ssl batch loss': loss.item()})
                loss.backward()
                optimizer.step()

        def eval_during_training(model, dl, loss_fn, use_ratio_for_monitoring=20):
            model.eval()
            with torch.no_grad():
                validation_loss = []
                for x in dl:
                    n = random.randint(1, 100)
                    if len(dl) > 10 and n > use_ratio_for_monitoring:
                        continue  # increase the speed of epoch
                    xb_mod_a = x[0].float().type(dtype)
                    xb_mod_b = x[1].float().type(dtype)
                    lens = x[2].float()
                    emb_a, emb_b = model(xb_mod_a, xb_mod_b, lens)
                    validation_loss.append(loss_fn(emb_a, emb_b).item())
            return np.mean(validation_loss)

        mod_a, mod_b = ssl_task_config['mod_a'], ssl_task_config['mod_b']
        train_data, vali_data = self.train_data, self.vali_data
        train_step_lens = self._get_step_len(train_data[mod_a])
        vali_step_lens = self._get_step_len(vali_data[mod_a])
        model = SslContrastiveNet(self.emb_nets[mod_a], self.emb_nets[mod_b], self.config.common_space_dim)

        train_dl = prepare_dl([train_data[mod_a], train_data[mod_b], train_step_lens], int(self.config.batch_size_ssl), shuffle=True)
        vali_dl = prepare_dl([vali_data[mod_a], vali_data[mod_b], vali_step_lens], int(self.config.batch_size_ssl), shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr_ssl, weight_decay=1e-5)

        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=int(self.config.epoch_ssl/4))
        dtype, model = set_dtype_and_model(self.config.device, model)
        # current_best_loss = eval_during_training(model, vali_dl, training_task['ssl_loss_fn'])
        epoch_end_time = time.time()
        for i_epoch in range(self.config.epoch_ssl):
            train_loss = eval_during_training(model, train_dl, ssl_task_config['ssl_loss_fn'], use_ratio_for_monitoring=1)
            test_loss = eval_during_training(model, vali_dl, ssl_task_config['ssl_loss_fn'], use_ratio_for_monitoring=5)
            logging.info(f'| SSL | epoch {i_epoch:3d} | time: {time.time() - epoch_end_time:5.2f}s | '
                         f'train loss {train_loss:5.4f} | test loss {test_loss:5.4f}')
            epoch_end_time = time.time()
            train_batch(model, train_dl, optimizer, ssl_task_config['ssl_loss_fn'], self.config.use_ratio)
            wandb.log({'ssl train loss': train_loss, 'ssl test loss': test_loss, 'ssl lr': scheduler.get_last_lr()[0]})
            scheduler.step()
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


MODALITIES = ['acc', 'gyr', 'emg']
GROUPS_OF_DATA = {
    'acc': [segment + '_Accel_' + axis for segment in ['thigh', 'foot'] for axis in ['X', 'Y', 'Z']],
    'gyr': [segment + '_Gyro_' + axis for segment in ['thigh', 'foot'] for axis in ['X', 'Y', 'Z']],
    'emg': ['gastrocmed', 'tibialisanterior', 'soleus']}
# EMG_LIST = ['gastrocmed', 'tibialisanterior', 'soleus', 'vastusmedialis', 'vastuslateralis', 'rectusfemoris',
#             'bicepsfemoris', 'semitendinosus', 'gracilis', 'gluteusmedius']

SSL_CONTRASTIVE_TASK = {'ssl_loss_fn': nce_loss, 'mod_a': 'acc', 'mod_b': 'gyr'}
DOWNSTREAM_TASK_1 = {'name': 'peak_angle', 'input_mods': ['acc', 'gyr'],
                     'output': ['knee_angle_r'], 'processes': [get_non_zero_max]}
DOWNSTREAM_TASKS = [DOWNSTREAM_TASK_1]

# 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r'
# 'hip_flexion_r_moment', 'hip_adduction_r_moment', 'hip_rotation_r_moment', 'knee_angle_r_moment', 'ankle_angle_r_moment'
config = {'epoch_ssl': 5, 'epoch_linear': 5, 'batch_size_ssl': 512, 'batch_size_linear': 128, 'lr_ssl': 1e-4, 'lr_linear': 1e-4,
          'emb_output_dim': 64, 'common_space_dim': 64 * 6, 'device': 'cuda', 'remove_trial_type': [], 'use_ratio': 100,        # !!! fix common_space_dim
          'emb_net': CnnEmbedding, 'file_name': 'StepWin'}
wandb.init(project="IMU_EMG_SSL", config=config, name='new framework')


if __name__ == '__main__':
    # logging.info("Current commit is {}".format(execute_cmd("git rev-parse HEAD")))
    train_set = [
        'AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB17',
        'AB18', 'AB19', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28',
        'AB30'
    ]
    test_set = ['AB06', 'AB07', 'AB08', 'AB09']
    ssl_framework = FrameworkSSL(config)
    ssl_framework.preprocess(train_set, test_set, test_set)

    model = ssl_framework.ssl_training(SSL_CONTRASTIVE_TASK)
    y_pred, model = ssl_framework.regressibility(DOWNSTREAM_TASK_1, only_linear_layer=True)
    ssl_framework.reset_emb_net_to_init_state()
    y_pred, model = ssl_framework.regressibility(DOWNSTREAM_TASK_1, only_linear_layer=True)
    ssl_framework.reset_emb_net_to_init_state()
    y_pred, model = ssl_framework.regressibility(DOWNSTREAM_TASK_1, only_linear_layer=False)

    plt.show()



