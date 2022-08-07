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
import ast
from sklearn.metrics import r2_score, mean_squared_error as mse
import torch
from torch.nn import functional as F
import wandb
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from model import nce_loss, ImuTransformerEmbedding, ImuFcnnEmbedding, ImuRnnEmbedding, SslContrastiveNet, \
    ImuCnnEmbedding, LinearRegressNet, SslReconstructNet
import time
from types import SimpleNamespace
import prettytable as pt
from utils import get_data_by_merging_data_struct, fix_seed, DataStruct, data_filter
from const import IMU_SEGMENT_LIST, DATA_PATH, AMBULATIONS, GRAVITY, IMU_SAMPLE_RATE, EMG_SAMPLE_RATE
from scipy.signal import find_peaks


def interpo_data(tensor, interpo_len, lens):
    for i_step in range(tensor.shape[0]):
        tensor = tensor.transpose(1, 2)
        tensor[i_step:i_step+1, :, :interpo_len] = F.interpolate(
            tensor[i_step:i_step+1, :, :int(lens[i_step])], interpo_len, mode='linear', align_corners=True)
        tensor = tensor.transpose(1, 2)
    return tensor[:, :interpo_len, :]


class FrameworkSSL:
    def __init__(self, data_reader, emb_net_imu, emb_net_emg, config):
        self.data_reader = data_reader
        self.data = data_reader.transform_to_step_data()
        self.result_dir = os.path.join('D://SSL_training_results', self.result_folder())
        os.makedirs(os.path.join(self.result_dir), exist_ok=True)
        add_file_handler(logging, os.path.join(self.result_dir, 'training_log.txt'))

        self.config = SimpleNamespace(**config)
        imu_dim, emg_dim = len(IMU_LIST), len(EMG_LIST)
        self.emb_net_imu = emb_net_imu(imu_dim, self.config.emb_output_dim, 'IMU Embedding')
        self.emb_net_emg = emb_net_emg(emg_dim, self.config.emb_output_dim, 'EMG Embedding')
        self.linear_regress_net = LinearRegressNet(self.emb_net_imu, self.emb_net_emg, len(OUTPUT_LIST))
        self.init_state_emb_net_imu = self.emb_net_imu.state_dict()
        self.init_state_emb_net_emg = self.emb_net_emg.state_dict()
        self.init_state_linear_regress = self.linear_regress_net.state_dict()

        self._base_scalar = StandardScaler
        self._data_scalar = {}
        fix_seed()

    def reset_emb_net_to_init_state(self):
        self.emb_net_imu.load_state_dict(self.init_state_emb_net_imu)
        self.emb_net_emg.load_state_dict(self.init_state_emb_net_emg)
        self.linear_regress_net.load_state_dict(self.init_state_linear_regress)

    @staticmethod
    def result_folder():
        folder_name = str(datetime.datetime.now())[:-7]
        for item in ['.', ':']:
            folder_name = folder_name.replace(item, '_')
        return folder_name

    def preprocess_train_evaluation(self, train_sub_ids: List[str], validate_sub_ids: List[str],
                                    test_sub_ids: List[str]):
        """
        train_sub_ids: a list of subject id for model training
        validate_sub_ids: a list of subject id for model validation
        test_sub_ids: a list of subject id for model testing
        """
        train_data = get_data_by_merging_data_struct([self.data[id] for id in train_sub_ids])
        train_data = self.preprocess_data(train_data, 'fit_transform')
        logging.info('Train the model with subject ids: {}. Number of steps: {}'.format(train_sub_ids, train_data['IMU'].shape[0]))

        vali_data = get_data_by_merging_data_struct([self.data[id] for id in validate_sub_ids])
        vali_data = self.preprocess_data(vali_data, 'transform')
        logging.info('Validate the model with subject ids: {}. Number of steps: {}'.format(validate_sub_ids, vali_data['IMU'].shape[0]))

        test_data = get_data_by_merging_data_struct([self.data[id] for id in test_sub_ids])
        test_data = self.preprocess_data(test_data, 'transform')
        logging.info('Test the model with subject ids: {}. Number of steps: {}'.format(test_sub_ids, test_data['IMU'].shape[0]))

        # self.save_model_and_results(test_data['y'], y_pred, OUTPUT_LIST, model, test_sub_ids)

        model = self.ssl_training(train_data, vali_data)
        y_pred, model = self.regressibility(train_data, test_data, only_linear_layer=True)
        self.reset_emb_net_to_init_state()
        y_pred, model = self.regressibility(train_data, test_data, only_linear_layer=True)
        self.reset_emb_net_to_init_state()
        y_pred, model = self.regressibility(train_data, test_data, only_linear_layer=False)

        plt.show()
        plt.close('all')

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

    def regressibility(self, train_data, test_data, only_linear_layer, verbose=False):
        def prepare_data(data, step_lens, batch_size):
            x_imu = torch.from_numpy(data['IMU']).float()
            x_emg = torch.from_numpy(data['EMG']).float()
            y_true = torch.from_numpy(data['y']).float()
            step_lens = torch.from_numpy(step_lens)
            ds = TensorDataset(x_imu, x_emg, step_lens, y_true)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
            return dl

        def convert_batch_data(batch_data):
            xb_imu = batch_data[0].float().type(dtype)
            xb_emg = batch_data[1].float().type(dtype)
            lens = batch_data[2].float()
            y_true = batch_data[3].float().type(dtype)
            return xb_imu, xb_emg, lens, y_true

        def train(model, train_dl, optimizer, loss_fn, use_ratio):
            model.train()
            for i_batch, batch_data in enumerate(train_dl):
                n = random.randint(1, 100)
                if n > use_ratio:
                    continue  # increase the speed of epoch
                optimizer.zero_grad()
                xb_imu, xb_emg, lens, y_true = convert_batch_data(batch_data)
                y_pred = model(xb_imu, xb_emg, lens)
                loss = loss_fn(y_pred, y_true)
                loss.backward()
                if verbose:
                    print(f'| epoch {i_epoch:3d} | {i_batch:5d}/{len(train_dl):4d} batches | loss {loss.item():5.2f}')
                optimizer.step()

        def eval_during_training(model, dl, loss_fn):
            model.eval()
            with torch.no_grad():
                loss, y_pred_list, y_true_list = [], [], []
                for batch_data in dl:
                    xb_imu, xb_emg, lens, y_true = convert_batch_data(batch_data)
                    y_pred = model(xb_imu, xb_emg, lens)
                    loss.append(loss_fn(y_true, y_pred).item())
                    y_pred_list.append(y_pred.detach().cpu())
                    y_true_list.append(y_true.detach().cpu())
            return np.mean(loss), y_pred_list, y_true_list

        def evaluate_after_training(test_dl):
            model.eval()
            with torch.no_grad():
                y_pred_list, y_true_list = [], []
                for i_batch, batch_data in enumerate(test_dl):
                    xb_imu, xb_emg, lens, y_true = convert_batch_data(batch_data)
                    y_pred_list.append(model(xb_imu, xb_emg, lens).detach().cpu())
                    y_true_list.append(y_true.detach().cpu())
                y_pred = torch.cat(y_pred_list).numpy()
                y_true = torch.cat(y_true_list).numpy()
            return y_pred, y_true

        logging.info('Linear regressibility test')
        train_step_lens = self._get_step_len(train_data['IMU'])
        train_dl = prepare_data(train_data, train_step_lens, int(self.config.batch_size_linear))
        test_step_lens = self._get_step_len(test_data['IMU'])
        test_dl = prepare_data(test_data, test_step_lens, int(self.config.batch_size_linear))

        model = self.linear_regress_net
        # num_params = sum(param.numel() for param in model.embnet_imu.rnn_layer.parameters())
        # print(num_params)
        # num_params = sum(param.numel() for param in model.embnet_imu.linear.parameters())
        # print(num_params)

        if self.config.device == 'cuda':
            model.cuda()
        if only_linear_layer:
            param_to_train = model.linear.parameters()
        else:
            param_to_train = model.parameters()
        optimizer = torch.optim.Adam(param_to_train, lr=self.config.lr_linear, weight_decay=1e-5)

        epoch_end_time = time.time()
        dtype = self.config.dtype
        for i_epoch in range(self.config.epoch_linear):
            train_loss, y_pred_train, y_true_train = eval_during_training(model, train_dl, torch.nn.MSELoss())
            test_loss, y_pred_test, y_true_test = eval_during_training(model, test_dl, torch.nn.MSELoss())

            if i_epoch in [self.config.epoch_linear-1]:
                plt.figure()
                plt.title('Train')
                plt.plot(torch.cat(y_true_train).transpose(1, 2).transpose(0, 1).numpy()[:, :10, :].ravel())
                plt.plot(torch.cat(y_pred_train).transpose(1, 2).transpose(0, 1).numpy()[:, :10, :].ravel())

                plt.figure()
                plt.title('Test')
                plt.plot(torch.cat(y_true_test).transpose(1, 2).transpose(0, 1).numpy()[:, :10, :].ravel())
                plt.plot(torch.cat(y_pred_test).transpose(1, 2).transpose(0, 1).numpy()[:, :10, :].ravel())
            if verbose:
                print('-' * 80)
            logging.info(f'| Linear Regressibility | epoch {i_epoch:3d} | time: {time.time() - epoch_end_time:5.2f}s | '
                         f'train loss {train_loss:5.3f} | test loss {test_loss:5.3f}')
            if verbose:
                print('-' * 80)
            epoch_end_time = time.time()

            # old_model = copy.deepcopy(model)
            train(model, train_dl, optimizer, torch.nn.MSELoss(), self.config.use_ratio)

        # plt.show()

        y_pred, y_true = evaluate_after_training(test_dl)
        all_scores = FrameworkSSL.get_scores(y_true, y_pred, OUTPUT_LIST, test_step_lens)
        all_scores = [{'subject': 'all', **scores} for scores in all_scores]
        self.print_table(all_scores)

        return y_pred, model

    def ssl_training(self, train_data, vali_data):
        def prepare_data(train_step_lens, vali_step_lens, batch_size):
            x_train_imu = torch.from_numpy(train_data['IMU']).float()
            x_train_emg = torch.from_numpy(train_data['EMG']).float()
            train_step_lens = torch.from_numpy(train_step_lens)
            train_ds = TensorDataset(x_train_imu, x_train_emg, train_step_lens)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            x_vali_imu = torch.from_numpy(vali_data['IMU']).float()
            x_vali_emg = torch.from_numpy(vali_data['EMG']).float()
            vali_step_lens = torch.from_numpy(vali_step_lens)
            vali_ds = TensorDataset(x_vali_imu, x_vali_emg, vali_step_lens)
            vali_dl = DataLoader(vali_ds, batch_size=batch_size)
            return train_dl, vali_dl

        def train(model, train_dl, optimizer, loss_fn, use_ratio):
            model.train()
            for i_batch, x in enumerate(train_dl):
                n = random.randint(1, 100)
                if n > use_ratio:
                    continue  # increase the speed of epoch
                optimizer.zero_grad()
                xb_imu = x[0].float().type(dtype)
                xb_emg = x[1].float().type(dtype)
                lens = x[2].float()
                emg_pred = model(xb_imu, lens)
                loss = loss_fn(emg_pred, xb_emg)
                if i_batch % 5 == 0:
                    train_loss, _, _ = eval_during_training(model, train_dl, self.config.ssl_loss_fn)
                    test_loss, y_pred_list, y_true_list = eval_during_training(model, vali_dl, self.config.ssl_loss_fn)
                    wandb.log({'train loss': train_loss, 'test loss': test_loss})
                    if i_epoch == 0:
                        plt.figure()
                        for i in range(3):
                            plt.plot(y_pred_list[0].numpy()[0, :, i], '--', color='C'+str(i))
                            plt.plot(y_true_list[0].numpy()[0, :, i], color='C'+str(i))
                        plt.savefig(self.result_dir + '/emg_batch{}.png'.format(i_batch))
                        plt.close()
                    model.train()
                loss.backward()
                optimizer.step()

        def eval_during_training(model, dl, loss_fn):
            model.eval()
            with torch.no_grad():
                validation_loss, y_pred_list, y_true_list = [], [], []
                for x in dl:
                    xb_imu = x[0].float().type(dtype)
                    xb_emg = x[1].float().type(dtype)
                    lens = x[2].float()
                    emb_imu = model(xb_imu, lens)
                    validation_loss.append(loss_fn(emb_imu, xb_emg).item())
                    y_pred_list.append(emb_imu.detach().cpu())
                    y_true_list.append(xb_emg.detach().cpu())
            return np.mean(validation_loss), y_pred_list, y_true_list

        train_step_lens = self._get_step_len(train_data['IMU'])
        vali_step_lens = self._get_step_len(vali_data['IMU'])

        model = SslReconstructNet(self.emb_net_imu, len(EMG_LIST))
        wandb.watch(model, self.config.ssl_loss_fn, log='all', log_freq=10)
        if self.config.device == 'cuda':
            model.cuda()

        train_dl, vali_dl = prepare_data(train_step_lens, vali_step_lens, int(self.config.batch_size_ssl))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr_ssl, weight_decay=1e-5)

        # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=int(self.config.epoch_ssl/4))

        dtype = self.config.dtype
        current_best_loss, _, _ = eval_during_training(model, vali_dl, self.config.ssl_loss_fn)
        best_model = copy.deepcopy(model)
        epoch_end_time = time.time()
        for i_epoch in range(self.config.epoch_ssl):

            train_loss, _, _ = eval_during_training(model, train_dl, self.config.ssl_loss_fn)
            test_loss, _, _ = eval_during_training(model, vali_dl, self.config.ssl_loss_fn)
            logging.info(f'| SSL | epoch {i_epoch:3d} | time: {time.time() - epoch_end_time:5.2f}s | '
                         f'train loss {train_loss:5.4f} | test loss {test_loss:5.4f}')
            epoch_end_time = time.time()
            train(model, train_dl, optimizer, self.config.ssl_loss_fn, self.config.use_ratio)

            # wandb.log({'train loss': train_loss, 'test loss': test_loss})

            if test_loss < current_best_loss:
                current_best_loss = test_loss
                best_model = copy.deepcopy(model)
            # scheduler.step()
        model = best_model
        return {'model': model}

    def save_model_and_results(self, ):
        pass

    def normalize_data(self, data, name, method, scalar_mode):
        if method == 'fit_transform':
            self._data_scalar[name] = self._base_scalar()
        assert (scalar_mode in ['by_each_column', 'by_all_columns'])
        input_data = data.copy()
        original_shape = input_data.shape
        if len(input_data.shape) == 3:
            target_shape = [-1, input_data.shape[2]] if scalar_mode == 'by_each_column' else [-1, 1]
            input_data[(input_data == 0.).all(axis=2), :] = np.nan
            input_data = input_data.reshape(target_shape)
        else:
            input_data[(input_data == 0.).all(axis=1), :] = np.nan
        scaled_data = getattr(self._data_scalar[name], method)(input_data)
        scaled_data = scaled_data.reshape(original_shape)
        scaled_data[np.isnan(scaled_data)] = 0.
        return scaled_data

    def preprocess_data(self, data, method):        # TODO: Try by_each_column
        if len(self.data_reader.input_col_loc_acc):
            data['IMU'][:, :, self.data_reader.input_col_loc_acc] = self.normalize_data(
                data['IMU'][:, :, self.data_reader.input_col_loc_acc], 'acc', method, 'by_all_columns')
        if len(self.data_reader.input_col_loc_gyr):
            data['IMU'][:, :, self.data_reader.input_col_loc_gyr] = self.normalize_data(
                data['IMU'][:, :, self.data_reader.input_col_loc_gyr], 'gyr', method, 'by_all_columns')
        data['EMG'] = self.down_sample_data(data['EMG'], data['IMU'].shape[1])
        data['EMG'] = self.normalize_data(data['EMG'], 'EMG', method, 'by_each_column')

        data['y'] = self.normalize_data(data['y'], 'y', method, 'by_each_column')

        # interpolation
        if config['interpo_len'] is not None:
            step_lens = self._get_step_len(data['IMU'])
            base_len = data['IMU'].shape[1]
            for key_ in ['IMU', 'EMG', 'y']:
                ratio = int(data[key_].shape[1] / base_len)
                step_lens_modal = [x * ratio for x in step_lens]
                y_tensor = interpo_data(torch.from_numpy(data[key_]), config['interpo_len'], step_lens_modal)
                data[key_] = y_tensor.numpy()
        return data

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
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    @staticmethod
    def _get_step_len(data, feature_col=[0, 1, 2]):
        """
        :param data: Numpy array, 3d (step, sample, feature)
        :param feature_col: int, feature column id for step length detection. Different id would probably return
               the same results
        :return:
        """
        data_the_feature = data[:, :, feature_col]
        zero_loc = data_the_feature == 0.
        data_len = np.sum(~zero_loc, axis=1)
        data_len = np.max(data_len, axis=1)
        return data_len


class DatasetLoader:
    def __init__(self, subject_list):
        self.sample_rate = IMU_SAMPLE_RATE
        self.step_len_max, self.step_len_min = int(256*self.sample_rate/200), int(40*self.sample_rate/200)
        self.columns = self.load_columns()
        self.input_col_imu = IMU_LIST
        self.input_col_emg = EMG_LIST
        self.output_col = OUTPUT_LIST
        self.input_col_loc_acc = [IMU_LIST.index(x) for x in self.input_col_imu if 'Accel' in x]
        self.input_col_loc_gyr = [IMU_LIST.index(x) for x in self.input_col_imu if 'Gyro' in x]
        self.subject_list = subject_list
        self.data_contin = {}
        for subject in subject_list:
            logging.info('Loading data of {}.'.format(subject))
            with h5py.File(DATA_PATH + subject + '.h5', 'r') as hf:
                data_trials = {}
                [data_trials.update({trial: [trial_data['data_200'][:], trial_data['data_1000'][:]]})
                    for trial, trial_data in hf.items() if len(data_trials.keys()) < config['max_trial_num'] and
                 trial.split('_')[0].lower() not in config['remove_trial_type']]

                for trial, trial_data in data_trials.items():
                    condition = trial.split('_')[0]
                    input_col_loc_emg = [self.columns[condition]['1000'].index(x) for x in EMG_LIST]
                    emg_data = trial_data[1][:, input_col_loc_emg]
                    trial_data[1][:, input_col_loc_emg] = data_filter(np.abs(emg_data), 20, EMG_SAMPLE_RATE)
            self.data_contin[subject] = data_trials
        self.trials = list(data_trials.keys())

    @staticmethod
    def interpo_extreme_large_data(data, data_loc, thd):
        for axis in data_loc:
            data_axis = data[:, axis]
            ok = np.abs(data_axis) < thd
            if (~ok).any():
                xp = ok.ravel().nonzero()[0]
                fp = data_axis[ok]
                x = (~ok).ravel().nonzero()[0]
                data_axis[~ok] = np.interp(x, xp, fp)
                data[:, axis] = data_axis

    @staticmethod
    def are_kinematics_correct(data_200, columns, kinematic_range, subject, trial):
        if np.min(data_200[:, columns]) < kinematic_range[0] or np.max(data_200[:, columns]) > kinematic_range[1]:
            return False
        return True

    @staticmethod
    def load_columns():
        columns = {ambulation: {} for ambulation in AMBULATIONS}
        for ambulation in AMBULATIONS:
            for frequency in ['200', '1000']:
                columns[ambulation][frequency] = list(np.array(ast.literal_eval(open(
                    DATA_PATH + ambulation + '_' + frequency + '_columns.txt').read()), dtype=object))
        return columns

    def transform_to_step_data(self):
        self.data = {}
        logging.info('Transform to step data.')
        for subject in self.subject_list:
            data_struct = DataStruct(len(self.input_col_imu), len(self.input_col_emg), len(self.output_col), self.step_len_max)
            for trial in self.trials:
                if trial not in self.data_contin[subject].keys():
                    continue
                trial_type = trial.split('_')[0]
                input_col_loc_imu = [self.columns[trial_type]['200'].index(x) for x in self.input_col_imu]
                input_col_loc_emg = [self.columns[trial_type]['1000'].index(x) for x in self.input_col_emg]
                output_col_loc = [self.columns[trial_type]['200'].index(x) for x in self.output_col]
                acc_loc = [self.columns[trial_type]['200'].index('foot_Accel_' + axis) for axis in ['X', 'Y', 'Z']]
                gyr_loc = [self.columns[trial_type]['200'].index('foot_Gyro_' + axis) for axis in ['X', 'Y', 'Z']]
                trial_data_200, trial_data_1000 = self.data_contin[subject][trial]
                kinematic_col_loc = [self.columns[trial_type]['200'].index(col) for col in
                                     ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r']]
                if not self.are_kinematics_correct(trial_data_200, kinematic_col_loc, (-180, 180), subject, trial):
                    continue
                self.interpo_extreme_large_data(trial_data_200, acc_loc+gyr_loc, 15)
                acc_data, gyr_data = trial_data_200[:, acc_loc], trial_data_200[:, gyr_loc]
                strike_list, off_list, gyr_y = self.get_walking_strike_off(acc_data, gyr_data, 10)
                steps_200, steps_1000 = self.strike_off_to_step_and_remove_incorrect_step(
                    gyr_y, strike_list, off_list, self.step_len_max, self.step_len_min)
                for step_200, step_1000 in zip(steps_200, steps_1000):
                    data_x_200 = trial_data_200[step_200[0]:step_200[1], input_col_loc_imu]       # 遇到 out of xxx，写exception 跳过该步
                    data_x_1000 = trial_data_1000[step_1000[0]:step_1000[1], input_col_loc_emg]
                    data_y_200 = trial_data_200[step_200[0]:step_200[1], output_col_loc]
                    data_struct.add_new_step(data_x_200, data_x_1000, data_y_200)
            self.data[subject] = data_struct
        return self.data

    @staticmethod
    def strike_off_to_step_and_remove_incorrect_step(gyr_y, strike_list, off_list, step_len_max, step_len_min):
        steps_200 = []
        if config['from_strike_to_off']:
            event_1, event_2 = np.array(strike_list), np.array(off_list)
            event_1_height_thd, event_2_height_thd = 0, 4
        else:
            event_2, event_1 = np.array(strike_list), np.array(off_list)
            event_2_height_thd, event_1_height_thd = 0, 4
        for i_event_1 in range(len(event_1)):
            potential_event_2 = event_2[event_1[i_event_1] + step_len_min < event_2]
            potential_event_2 = potential_event_2[potential_event_2 < event_1[i_event_1] + step_len_max - SAMPLES_BEFORE_STEP - SAMPLES_AFTER_STEP]
            if len(potential_event_2) == 1:
                if gyr_y[event_1[i_event_1]] > event_1_height_thd and gyr_y[potential_event_2] > event_2_height_thd:
                    steps_200.append([event_1[i_event_1] - SAMPLES_BEFORE_STEP, potential_event_2[0] + SAMPLES_AFTER_STEP])
        steps_1000 = [[5 * step[0], 5 * step[1]] for step in steps_200]
        return steps_200, steps_1000

    @staticmethod
    def initalize_steps_and_stance_phase(data_df, strike_list, off_list, sample_after_thd=10):
        """The name "stance phase" is not accurate. It starts from gyr < thd + sample_after_thd sample,
         ends in the middle of the stance"""

        gyr_all = np.deg2rad(data_df[['gyr_x', 'gyr_y', 'gyr_z']])
        gyr_magnitude = np.linalg.norm(gyr_all, axis=1)

        imu_sample_rate = IMU_SAMPLE_RATE
        stance_phase_sample_thd_lower = 0.3 * imu_sample_rate
        stance_phase_sample_thd_higher = 1 * imu_sample_rate
        data_len = data_df.shape[0]
        strike_array, off_array = np.array(strike_list), np.array(off_list)
        strike_num = len(strike_array)
        steps = []
        stance_phase_flag = np.zeros([data_len], dtype=bool)
        abandoned_step_num = 0
        last_off = 0
        for i_strike in range(strike_num):
            strike = strike_array[i_strike]
            offs_near_strike = off_array[max(0, i_strike - 70): i_strike + 70]
            off = offs_near_strike[offs_near_strike > strike + stance_phase_sample_thd_lower]
            off = off[off < strike + stance_phase_sample_thd_higher]
            if len(off) == 1:  # stance phase detected
                if strike < last_off:
                    continue
                off = off[0]
                steps.append([int(strike), int(off)])
                flag_start = strike + 20
                flag_end = int(round((strike + off) / 2))
                for i_sample in range(strike, off):
                    if all(gyr_magnitude[i_sample:i_sample + 5] < 1.7):
                        flag_start = i_sample + sample_after_thd
                        break
                stance_phase_flag[flag_start:flag_end] = True
                last_off = off
            else:
                abandoned_step_num += 1
        return steps, stance_phase_flag

    def get_walking_strike_off(self, acc_data, gyr_data, cut_off_fre_strike_off=None, verbose=False):
        """ Reliable algorithm used in TNSRE first submission"""
        gyr_thd = 2.6
        acc_thd = 1.2 / GRAVITY
        max_distance = self.sample_rate * 2  # distance from stationary phase should be smaller than 2 seconds
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        gyr_magnitude = np.linalg.norm(gyr_data, axis=1)
        gyr_y = gyr_data[:, 1]
        data_len = gyr_data.shape[0]

        if cut_off_fre_strike_off is not None:
            acc_magnitude = data_filter(acc_magnitude, cut_off_fre_strike_off, self.sample_rate, filter_order=2)
            gyr_magnitude = data_filter(gyr_magnitude, cut_off_fre_strike_off, self.sample_rate, filter_order=2)
            gyr_y = data_filter(gyr_y, cut_off_fre_strike_off, self.sample_rate, filter_order=2)

        acc_magnitude = acc_magnitude - 1       # remove the gravity

        stationary_flag = self.__find_stationary_phase(
            gyr_magnitude, acc_magnitude, acc_thd, gyr_thd)

        strike_list, off_list = [], []
        i_sample = 0

        while i_sample < data_len:
            # step 0, go to the next stationary phase
            if not stationary_flag[i_sample]:
                i_sample += 1
            else:
                front_crossing, back_crossing = self.__find_zero_crossing(gyr_y, gyr_thd, i_sample)

                if not back_crossing:  # if back zero crossing not found
                    break
                if not front_crossing:  # if front zero crossing not found
                    i_sample = back_crossing
                    continue

                the_strike = self.find_peak_max(gyr_y[front_crossing:i_sample], height=0)
                the_off = self.find_peak_max(gyr_y[i_sample:back_crossing], height=0)

                if the_strike is not None and i_sample - (the_strike + front_crossing) < max_distance:
                    strike_list.append(the_strike + front_crossing)
                if the_off is not None and the_off < max_distance:
                    off_list.append(the_off + i_sample)
                i_sample = back_crossing
        if verbose:
            plt.figure()
            plt.plot(stationary_flag)
            plt.plot(gyr_y)
            plt.plot(strike_list, gyr_y[strike_list], 'g*')
            plt.plot(off_list, gyr_y[off_list], 'r*')

        return strike_list, off_list, gyr_y

    @staticmethod
    def __find_stationary_phase(gyr_magnitude, acc_magnitude, foot_stationary_acc_thd, foot_stationary_gyr_thd):
        data_len = gyr_magnitude.shape[0]
        stationary_flag, stationary_flag_temp = np.zeros(gyr_magnitude.shape), np.zeros(gyr_magnitude.shape)
        stationary_flag_temp[
            (acc_magnitude < foot_stationary_acc_thd) & (abs(gyr_magnitude) < foot_stationary_gyr_thd)] = 1
        for i_sample in range(data_len):
            if stationary_flag_temp[i_sample - 12:i_sample + 12].all():
                stationary_flag[i_sample] = 1
        return stationary_flag

    def __find_zero_crossing(self, gyr_x, foot_stationary_gyr_thd, i_sample):
        """
        Detected as a zero crossing if the value is lower than negative threshold.
        :return:
        """
        max_search_range = self.sample_rate * 3  # search 3 second front data at most
        front_crossing, back_crossing = False, False
        for j_sample in range(i_sample, max(0, i_sample - max_search_range), -1):
            if gyr_x[j_sample] < - foot_stationary_gyr_thd:
                front_crossing = j_sample
                break
        for j_sample in range(i_sample+1, gyr_x.shape[0]):
            if gyr_x[j_sample] < - foot_stationary_gyr_thd:
                back_crossing = j_sample
                break
        return front_crossing, back_crossing

    @staticmethod
    def find_peak_max(data_clip, height, width=None, prominence=None):
        """
        find the maximum peak
        :return:
        """
        peaks, properties = find_peaks(data_clip, width=width, height=height, prominence=prominence)
        if len(peaks) == 0:
            return None
        peak_heights = properties['peak_heights']
        max_index = np.argmax(peak_heights)
        return peaks[max_index]


SUB_LIST = [
    'AB06', 'AB07', 'AB08', 'AB09', 'AB10', 'AB11', 'AB12', 'AB13',
    'AB14', 'AB15', 'AB16', 'AB17', 'AB18', 'AB19', 'AB21', 'AB23',
    'AB24', 'AB25',
    'AB27', 'AB28', 'AB30'
]
IMU_LIST = [segment + sensor + axis for segment in ['thigh', 'foot'] for sensor in ['_Accel_', '_Gyro_'] for axis in ['X', 'Y', 'Z']]
# IMU_LIST = [segment + sensor + axis for segment in ['foot', 'shank', 'thigh', 'trunk'] for sensor in ['_Accel_', '_Gyro_'] for axis in ['X', 'Y', 'Z']]
EMG_LIST = ['gastrocmed', 'tibialisanterior', 'soleus']
# EMG_LIST = ['gastrocmed', 'tibialisanterior', 'soleus', 'vastusmedialis', 'vastuslateralis', 'rectusfemoris',
#             'bicepsfemoris', 'semitendinosus', 'gracilis', 'gluteusmedius']
OUTPUT_LIST = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
# 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r'
# 'hip_flexion_r_moment', 'hip_adduction_r_moment', 'hip_rotation_r_moment', 'knee_angle_r_moment', 'ankle_angle_r_moment'
SAMPLES_BEFORE_STEP, SAMPLES_AFTER_STEP = 0, 0
config = {'epoch_ssl': 25, 'epoch_linear': 20, 'batch_size_ssl': 512, 'batch_size_linear': 128, 'lr_ssl': 1e-4, 'lr_linear': 1e-4,
          'emb_output_dim': 256, 'common_space_dim': 1, 'device': 'cuda', 'dtype': torch.FloatTensor,
          'interpo_len': None, 'remove_trial_type': [], 'use_ratio': 100,
          'from_strike_to_off': True, 'ssl_loss_fn': torch.nn.MSELoss(), 'max_trial_num': 2000}
wandb.init(
    project="IMU_EMG_SSL", notes="tweak baseline", tags=["baseline", "paper1"], config=config, name='SSL via EMG reconstruction worked')
if config['device'] == 'cuda':
    config['dtype'] = torch.cuda.FloatTensor


if __name__ == '__main__':
    # logging.info("Current commit is {}".format(execute_cmd("git rev-parse HEAD")))
    test_set = ['AB25', 'AB27', 'AB28', 'AB30']
    train_set = [item for item in SUB_LIST if item not in test_set]
    data_reader = DatasetLoader(train_set + [sub for sub in test_set if sub not in train_set])
    ssl_framework = FrameworkSSL(data_reader, ImuRnnEmbedding, ImuRnnEmbedding, config)
    ssl_framework.preprocess_train_evaluation(train_set, test_set, test_set)



