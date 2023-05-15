import argparse
import copy
import os
import h5py
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import List
from customized_logger import logger as logging, add_file_handler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import wandb
from model import RegressNet, transformer, SslReconstructNet, mse_loss_masked, \
    show_reconstructed_signal
import time
from types import SimpleNamespace
from utils import prepare_dl, set_dtype_and_model, fix_seed, normalize_data, result_folder, define_channel_names, \
    preprocess_modality, get_scores, print_table, get_step_len
from const import DICT_TRIAL_TYPE_ID, RESULTS_PATH, CAMARGO_SUB_HEIGHT_WEIGHT, \
    GRAVITY, DSET_SUBS_FOR_SSL_TEST, DSET_SUBS_FOR_SSL_TRAINING, \
    SUB_ID_ALL_DATASETS, _mods, STANDARD_IMU_SEQUENCE
from config import DATA_PATH
import json
import pytorch_warmup as warmup
import matplotlib
# matplotlib.use('WebAgg')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class FrameworkDownstream:
    def __init__(self, config, da_task):
        self.config = SimpleNamespace(**config)
        self.output_columns = da_task['output_columns']
        self.da_use_ratios = da_task['da_use_ratios']
        self._data_scalar = {'base_scalar': StandardScaler}
        self.da_task = da_task
        fix_seed()

        mask_input_channel = [False if imu in self.da_task['imu_segments'] else True for imu in STANDARD_IMU_SEQUENCE]
        self.emb_net = self.config.emb_net(len(da_task['imu_segments'])*6, self.config.nlayers, self.config.nhead,
                                           self.config.FeedForwardDim, mask_input_channel, mask_patch_num=0, patch_len=self.config.PatchLen)
        self.regress_net = RegressNet(self.emb_net, len(STANDARD_IMU_SEQUENCE)+len(STANDARD_IMU_SEQUENCE), len(self.output_columns))
        _, self.regress_net = set_dtype_and_model(self.config.device, self.regress_net)
        self.regress_net_init_state = copy.deepcopy(self.regress_net.state_dict())

        post_ssl_emb_net = self.load_post_ssl_emb_net()
        self.regress_net.emb_net.load_state_dict(post_ssl_emb_net)
        self.regress_net_post_ssl_state = copy.deepcopy(self.regress_net.state_dict())

    def load_post_ssl_emb_net(self):
        emb_path = os.path.join(self.config.result_dir, 'emb_' + self.config.FileNameAppendix[2:] + '.pth')
        return torch.load(emb_path)

    def load_and_process(self, test_name, train_sub_ids: List[str], validate_sub_ids: List[str], test_sub_ids: List[str]):
        if test_name == 'hw_running':
            self.load_and_process_hw_running(train_sub_ids, validate_sub_ids, test_sub_ids)
        elif 'Camargo' in test_name:
            self.load_and_process_camargo(train_sub_ids, validate_sub_ids, test_sub_ids)
        elif test_name in ['walking_knee_moment', 'filtered_walking_knee_moment']:
            self.load_and_process_kam(train_sub_ids, validate_sub_ids, test_sub_ids)
        elif test_name == 'sun_drop_jump':
            self.load_and_process_sun(train_sub_ids, validate_sub_ids, test_sub_ids)
        elif 'opencap' in test_name:
            self.load_and_process_opencap(train_sub_ids, validate_sub_ids, test_sub_ids)

    def load_and_process_camargo(self, train_sub_ids: List[str], validate_sub_ids: List[str],
                                 test_sub_ids: List[str]):
        """
        train_sub_ids: a list of subject id for model training
        validate_sub_ids: a list of subject id for model validation
        test_sub_ids: a list of subject id for model testing
        """
        type_to_exclude = [DICT_TRIAL_TYPE_ID[type] for type in self.da_task['remove_trial_type']]
        self.set_data = {}
        with h5py.File(DATA_PATH + self.da_task['dataset'] + '.h5', 'r') as hf:
            self.data_columns = json.loads(hf.attrs['columns'])
            for set_sub_ids, data_name in zip([train_sub_ids, validate_sub_ids, test_sub_ids], ['train', 'vali', 'test']):
                current_set_data_list = []
                for sub_id in set_sub_ids:
                    sub_data = hf[sub_id][:, :, :]
                    has_grf_index = np.where(np.max(sub_data[:, self.data_columns.index('fz')], axis=1) > 10)
                    sub_data = sub_data[has_grf_index]
                    sub_weight = CAMARGO_SUB_HEIGHT_WEIGHT[sub_id][1] * GRAVITY
                    force_col_loc = [self.data_columns.index(x) for x in ['fx', 'fy', 'fz']]
                    sub_data[:, force_col_loc] = sub_data[:, force_col_loc] / sub_weight
                    current_set_data_list.append(sub_data)
                current_set_data = np.concatenate(current_set_data_list, axis=0)
                """ [step, feature, time] """
                # use rand noise to replace reduced IMUs
                rand_noise = np.random.normal(size=(current_set_data.shape[0], 6, current_set_data.shape[2]))
                current_set_data = np.concatenate([current_set_data, rand_noise], axis=1)
                # # Only keep grf > 0 rows if using overground walking data
                # current_set_data_ = current_set_data[np.all(np.abs(current_set_data[:, self.data_columns.index('fz')]) > 0.01, axis=1)]
                self.set_data[data_name] = current_set_data
            self.data_columns.extend(['rand_noise' + sensor + axis for sensor in ['_Accel_', '_Gyro_'] for axis in ['X', 'Y', 'Z']])

    def load_and_process_kam(self, train_sub_ids: List[str], validate_sub_ids: List[str],
                             test_sub_ids: List[str]):
        self.set_data = {}
        with h5py.File(DATA_PATH + self.da_task['dataset'] + '.h5', 'r') as hf:
            self.data_columns = json.loads(hf.attrs['columns'])
            for set_sub_ids, data_name in zip([train_sub_ids, validate_sub_ids, test_sub_ids], ['train', 'vali', 'test']):
                current_set_data_list = [hf[sub_] for sub_ in set_sub_ids]
                current_set_data = np.concatenate(current_set_data_list, axis=0).transpose([0, 2, 1])
                """ [step, feature, time] """
                self.set_data[data_name] = current_set_data
                # self.check_dataset_imu_orientation(self.set_data[data_name], self.data_columns, self.da_task['imu_segments'])

    def load_and_process_sun(self, train_sub_ids: List[str], validate_sub_ids: List[str],
                             test_sub_ids: List[str]):
        self.set_data = {}
        with h5py.File(DATA_PATH + self.da_task['dataset'] + '.h5', 'r') as hf:
            self.data_columns = list(hf[train_sub_ids[0]].attrs['columns'])
            if 'sub_id' not in self.data_columns:
                self.data_columns.append('sub_id')
            for set_sub_ids, data_name in zip([train_sub_ids, validate_sub_ids, test_sub_ids], ['train', 'vali', 'test']):
                current_set_data_list = []
                for sub_ in set_sub_ids:
                    current_sub_data_list = [hf[sub_][('0' + str(trial + 1))[-2:]] for trial in range(30)
                                             if ('0' + str(trial + 1))[-2:] in hf[sub_]]
                    current_sub_data = np.stack(current_sub_data_list, axis=0).transpose([0, 2, 1])
                    sub_id_np = np.full([current_sub_data.shape[0], 1, current_sub_data.shape[2]], int(sub_[2:4]))
                    current_set_data_list.append(np.concatenate([current_sub_data, sub_id_np], axis=1))
                """ [step, feature, time] """
                current_set_data = np.concatenate(current_set_data_list, axis=0)
                """ padding to 128 samples"""
                current_set_data = np.concatenate([current_set_data, np.zeros([*current_set_data.shape[:2], 128-80])], axis=2)
                self.set_data[data_name] = current_set_data
                # self.check_dataset_imu_orientation(self.set_data[data_name], self.data_columns, self.da_task['imu_segments'])

    def load_and_process_opencap(self, train_sub_ids: List[str], validate_sub_ids: List[str], test_sub_ids: List[str]):
        self.set_data = {}
        with h5py.File(DATA_PATH + self.da_task['dataset'] + '.h5', 'r') as hf:
            self.data_columns = json.loads(hf.attrs['columns'])
            for set_sub_ids, data_name in zip([train_sub_ids, validate_sub_ids, test_sub_ids], ['train', 'vali', 'test']):
                current_set_data_list = [hf[sub_] for sub_ in set_sub_ids]
                current_set_data = np.concatenate(current_set_data_list, axis=0)
                """ [step, feature, time] """
                self.set_data[data_name] = current_set_data

    @staticmethod
    def check_dataset_imu_orientation(data_, columns_, imu_list):
        for imu in imu_list:
            plt.figure()
            plt.title(imu)
            col_loc = [columns_.index(imu + '_Gyro_' + axis) for axis in ['X', 'Y', 'Z']]
            plt.plot(data_[0, col_loc, :].T)
        plt.show()

    def load_and_process_hw_running(self, train_sub_ids: List[str], validate_sub_ids: List[str], test_sub_ids: List[str]):
        self.set_data = {}
        with h5py.File(DATA_PATH + self.da_task['dataset'] + '.h5', 'r') as hf:
            self.data_columns = json.loads(hf.attrs['columns'])
            if 'output_processed' not in self.data_columns:
                self.data_columns.append('output_processed')
            for set_sub_ids, data_name in zip([train_sub_ids, validate_sub_ids, test_sub_ids], ['train', 'vali', 'test']):
                current_set_data_list = [hf[sub_] for sub_ in set_sub_ids]
                current_set_data = np.concatenate(current_set_data_list, axis=0).transpose([0, 2, 1])
                """ [step, feature, time] """
                output_raw = current_set_data[:, self.data_columns.index('output')]
                output_loc = np.argmax(np.abs(output_raw), axis=1)
                output_ = np.array([output_raw[i_row, loc] for i_row, loc in enumerate(output_loc)]).reshape([-1, 1])
                output_processed = np.repeat(output_[:, np.newaxis], current_set_data.shape[2], axis=1).reshape([-1, 1, current_set_data.shape[2]])
                current_set_data = np.concatenate([current_set_data, output_processed], axis=1)
                self.set_data[data_name] = current_set_data

    def sample_and_normalize_data(self):
        for data_name, norm_method in zip(['train', 'vali', 'test'], ['fit_transform', 'transform', 'transform']):
            current_set_data = self.set_data[data_name]
            sampled_rows = np.sort(random.sample(range(current_set_data.shape[0]), int(self.config.da_use_ratio*current_set_data.shape[0])))
            sampled_data = copy.deepcopy(current_set_data[sampled_rows])
            logging.info('Downstream {}. Number of steps: {}'.format(data_name, sampled_data.shape[0]))
            data_ds = preprocess_modality(self.data_columns, self._data_scalar, sampled_data, define_channel_names(self.da_task), norm_method)
            output_data = sampled_data[:, [self.data_columns.index(x) for x in self.output_columns]]
            # output_data = self.norm_output_by_subject_weight(sampled_data)
            data_ds['output'] = normalize_data(self._data_scalar, output_data, 'output', norm_method, 'by_each_column')
            data_ds['sub_id'] = sampled_data[:, self.data_columns.index('sub_id'), 0]
            if 'trial_type_id' in self.data_columns:
                data_ds['trial_type_id'] = sampled_data[:, self.data_columns.index('trial_type_id'), 0]
            else:
                data_ds['trial_type_id'] = np.zeros([sampled_data.shape[0]])
            self.da_task[data_name] = data_ds

    def inverse_normalize_output(self, y_true, y_pred):
        y_true = normalize_data(self._data_scalar, y_true, 'output', 'inverse_transform', 'by_each_column')
        y_pred = normalize_data(self._data_scalar, y_pred, 'output', 'inverse_transform', 'by_each_column')
        return y_true, y_pred

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

    def save_model_and_results(self, test_name, y_true, y_pred, sub_ids, model):
        os.makedirs(os.path.join(self.config.result_dir, 'da_models', self.da_task['dataset']), exist_ok=True)
        copied_model = copy.deepcopy(model)
        torch.save(copied_model.state_dict(), os.path.join(self.config.result_dir, 'da_models', self.da_task['dataset'], test_name + '.pth'))
        self.save_results(test_name, y_true, y_pred, sub_ids)

    def save_results(self, test_name, y_true, y_pred, sub_ids):
        sub_id_set = list(set(sub_ids))
        results = np.concatenate([y_true, y_pred], axis=1)
        columns = ['y_true', 'y_pred']
        with h5py.File(os.path.join(self.config.result_dir, self.da_task['dataset'] + '_' + 'output' + '.h5'), 'a') as hf:
            grp = hf.require_group(test_name)
            for i_sub in sub_id_set:
                results_sub = results[sub_ids == i_sub]
                grp.require_dataset('sub_' + str(int(i_sub)), shape=results_sub.shape, data=results_sub, dtype='float32')
                grp.attrs['columns'] = json.dumps(columns)

    def set_regress_net_to_post_linear_init_head_first(self, use_ssl):
        model_name = 'LinearProb_True, UseSsl_' + str(use_ssl) + ', ratio_' + str(self.config.da_use_ratio) + self.config.FileNameAppendix
        regress_net_post_linear_head_init = torch.load(os.path.join(
            self.config.result_dir, 'da_models', self.da_task['dataset'], model_name + '.pth'))
        self.regress_net.load_state_dict(regress_net_post_linear_head_init)

    def set_regress_net_to_init_state(self):
        self.regress_net.load_state_dict(self.regress_net_init_state)

    def set_regress_net_to_post_ssl_state(self):
        self.regress_net.load_state_dict(self.regress_net_post_ssl_state)

    def regressibility(self, linear_protocol, use_ssl, show_fig=False, verbose=True):
        def convert_batch_data(batch_data):
            xb = [data_.float().type(dtype) for data_ in batch_data[:-2]]
            yb = batch_data[-2].float().type(dtype)
            lens = batch_data[-1].float()
            return xb, yb, lens

        def train_batch(model, train_dl, optimizer, loss_fn, i_optimize):
            model.train()
            for i_batch, batch_data in enumerate(train_dl):
                optimizer.zero_grad()
                xb, yb, lens = convert_batch_data(batch_data)
                y_pred, _ = model(xb, lens)
                loss = loss_fn(yb, y_pred)
                if self.config.log_with_wandb:
                    wandb.log({'linear batch loss': loss.item(), 'lr da': optimizer.param_groups[0]['lr']})
                loss.backward()
                optimizer.step()
                i_optimize += 1

                # plt.figure()
                # plt.plot(xb[1].detach().cpu().numpy()[:, 18, :].ravel())
                # plt.plot(yb.detach().cpu().numpy()[:, 1, :].ravel())
                # plt.show()

                with warmup_scheduler.dampening():
                    scheduler.step()
            return i_optimize

        def eval_during_training(model, dl, loss_fn, use_batch_num=5):
            model.eval()
            loss = []
            with torch.no_grad():
                for i_batch, batch_data in enumerate(dl):
                    if i_batch >= use_batch_num:
                        return np.mean(loss)
                    xb, yb, lens = convert_batch_data(batch_data)
                    y_pred, _ = model(xb, lens)
                    loss.append(loss_fn(yb, y_pred).item())
            return np.mean(loss)

        def evaluate_after_training(test_dl):
            model.eval()
            with torch.no_grad():
                y_pred_list, y_true_list = [], []
                for i_batch, batch_data in enumerate(test_dl):
                    xb, yb, lens = convert_batch_data(batch_data)
                    y_true_list.append(yb.detach().cpu())
                    y_pred_batch, mod_outputs_batch = model(xb, lens)
                    y_pred_list.append(y_pred_batch.detach().cpu())
                y_true = torch.cat(y_true_list).numpy()
                y_pred = torch.cat(y_pred_list).numpy()
            return y_true, y_pred

        def record_intermediate_results(i_optimize):
            y_true, y_pred = evaluate_after_training(test_dl)
            y_true, y_pred = self.inverse_normalize_output(y_true, y_pred)
            intermediate_result_name = 'LinearProb_' + str(linear_protocol) + ', UseSsl_' + str(use_ssl) + \
                                       ', ratio_' + str(self.config.da_use_ratio) + ', i_optimize_' + str(i_optimize)
            self.save_results(intermediate_result_name, y_true, y_pred, test_data['sub_id'])

        self.config = SimpleNamespace(**config)
        test_name = 'LinearProb_' + str(linear_protocol) + ', UseSsl_' + str(use_ssl) + \
                    ', ratio_' + str(self.config.da_use_ratio) + self.config.FileNameAppendix
        logging.info('Regressing, ' + test_name)
        if linear_protocol:
            if use_ssl:
                self.set_regress_net_to_post_ssl_state()
            else:
                self.set_regress_net_to_init_state()
        else:
            self.set_regress_net_to_post_linear_init_head_first(use_ssl)
        self.sample_and_normalize_data()
        train_data, test_data = self.da_task['train'], self.da_task['test']
        train_input_data = [train_data[mod] for mod in self.da_task['_mods']]
        train_output_data = train_data['output']
        train_step_lens = get_step_len(train_input_data[0])
        train_dl = prepare_dl([*train_input_data, train_output_data, train_step_lens], int(self.config.batch_size_linear), shuffle=True)
        test_input_data = [test_data[mod] for mod in self.da_task['_mods']]
        test_output_data = test_data['output']
        test_step_lens = get_step_len(test_input_data[0])
        test_dl = prepare_dl([*test_input_data, test_output_data, test_step_lens], int(self.config.batch_size_linear), shuffle=False)

        model = self.regress_net
        dtype, model = set_dtype_and_model(self.config.device, model)
        if linear_protocol:
            lr_ = 3e-3
            param_to_train = model.linear.parameters()
        else:
            lr_ = 3e-5
            param_to_train = model.parameters()

        optimizer = torch.optim.AdamW(param_to_train, lr_, weight_decay=1e-5)
        epoch_end_time = time.time()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.NumGradDeDa)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=int(self.config.NumGradDeDa/5))

        epoch = int(np.ceil(self.config.NumGradDeDa / len(train_dl)))
        i_optimize = 0
        for i_epoch in range(epoch):
            if verbose:
                train_loss = eval_during_training(model, train_dl, torch.nn.MSELoss())
                test_loss = eval_during_training(model, test_dl, torch.nn.MSELoss())
                if self.config.log_with_wandb:
                    wandb.log({'linear train loss': train_loss, 'linear test loss': test_loss})
                if epoch < 2 or i_epoch % int(epoch / 2) == 0 or i_epoch == epoch - 1:
                    logging.info(f'| Regressibility | epoch{i_epoch:3d}/{epoch:3d} | time: {time.time() - epoch_end_time:5.2f}s |'
                                 f' train loss {train_loss:5.3f} | test loss {test_loss:5.3f}')
                epoch_end_time = time.time()

                if i_epoch in [0, int(epoch/2), epoch-1] and self.config.log_with_wandb:
                    model.eval()
                    with torch.no_grad():
                        data_ = list(enumerate(test_dl))[0][1]
                        xb_mods = [xb_mod.float().type(dtype) for xb_mod in data_[:-2]]
                        outputs, _ = model(xb_mods, None)
                        show_reconstructed_signal(data_[-2], outputs, f'epoch {i_epoch}',
                                                  fig_group=f"use_ssl {use_ssl}, linear {linear_protocol}",
                                                  channel_names=self.output_columns)

            i_optimize = train_batch(model, train_dl, optimizer, torch.nn.MSELoss(), i_optimize)

        y_true, y_pred = evaluate_after_training(test_dl)
        y_true, y_pred = self.inverse_normalize_output(y_true, y_pred)
        self.save_model_and_results(test_name, y_true, y_pred, test_data['sub_id'], self.regress_net)
        if show_fig:
            plt.figure()
            for i_output in range(y_true.shape[1]):
                plt.title('test set')
                plt.plot(y_true[:, i_output].ravel(), '-', color='C'+str(i_output), label=self.output_columns)
                plt.plot(y_pred[:, i_output].ravel(), '--', color='C'+str(i_output), label=self.output_columns)
                plt.legend()
            plt.show()
        if verbose:
            all_scores = get_scores(y_true, y_pred, self.output_columns, test_step_lens)
            all_scores = [{'subject': 'all', **scores} for scores in all_scores]
            print_table(all_scores)

        return y_pred, model


class FrameworkSSL:
    def __init__(self, config, ssl_task):
        self.config = SimpleNamespace(**config)
        self.ssl_task = ssl_task
        self.ssl_data_dict, self.ssl_columns_dict = {}, {}
        for ssl_file_name in ssl_task['ssl_file_names']:
            with h5py.File(DATA_PATH + ssl_file_name + '.h5', 'r') as hf:
                data_dict = {sub_: sub_data[:int(self.config.ssl_use_ratio * hf[sub_].shape[0]), :, :] for sub_, sub_data in hf.items()}
                if ssl_file_name in ['walking_knee_moment', 'filtered_walking_knee_moment']:
                    data_dict = {sub_: data_.transpose([0, 2, 1]) for sub_, data_ in data_dict.items()}     # [step, feature, time]
                self.ssl_data_dict[ssl_file_name] = data_dict
                self.ssl_columns_dict[ssl_file_name] = json.loads(hf.attrs['columns'])
        os.makedirs(os.path.join(self.config.result_dir), exist_ok=True)

        self.emb_net = self.config.emb_net(len(ssl_task['imu_segments'])*6, self.config.nlayers, self.config.nhead,
                                           self.config.FeedForwardDim, [False for _ in range(8)], self.config.MaskPatchNum, self.config.PatchLen)
        logging.info('# of trainable parameters: {}'.format(sum(p.numel() for p in self.emb_net.transformer_encoder.parameters() if p.requires_grad)))
        self._data_scalar = {'base_scalar': StandardScaler}
        fix_seed()

    def preprocess(self, ssl_file_name):
        train_sub_ids = DSET_SUBS_FOR_SSL_TRAINING[ssl_file_name]
        validate_sub_ids = test_sub_ids = DSET_SUBS_FOR_SSL_TEST[ssl_file_name]
        data_, columns_ = self.ssl_data_dict[ssl_file_name], self.ssl_columns_dict[ssl_file_name]
        if train_sub_ids == ['except test']:
            train_data = np.concatenate([data_[sub] for sub in list(data_.keys()) if sub not in test_sub_ids], axis=0)
        else:
            train_data = np.concatenate([data_[sub] for sub in train_sub_ids], axis=0)
        vali_data = np.concatenate([data_[sub] for sub in validate_sub_ids], axis=0)
        test_data = np.concatenate([data_[sub] for sub in test_sub_ids], axis=0)

        # SSL preprocess
        train_data_ssl = preprocess_modality(columns_, self._data_scalar, train_data, define_channel_names(self.ssl_task), 'fit_transform')
        logging.info('SSL training with dataset {} subject ids: {}. Number of steps: {}'.format(ssl_file_name, train_sub_ids, list(train_data_ssl.values())[0].shape[0]))
        vali_data_ssl = preprocess_modality(columns_, self._data_scalar, vali_data, define_channel_names(self.ssl_task), 'transform')
        # logging.info('Validation with subject ids: {}. Number of steps: {}'.format(validate_sub_ids, list(vali_data_ssl.values())[0].shape[0]))
        test_data_ssl = preprocess_modality(columns_, self._data_scalar, test_data, define_channel_names(self.ssl_task), 'transform')
        # logging.info('Testing with subject ids: {}. Number of steps: {}'.format(test_sub_ids, list(test_data_ssl.values())[0].shape[0]))
        return train_data_ssl, vali_data_ssl, test_data_ssl

    @staticmethod
    def show_params(params):
        plt.figure()
        for i, param in enumerate(params):
            plt.plot(param.cpu().detach().numpy(), [i for _ in param], '.', markersize=1)

    def ssl_training(self, config):
        def train_batch(model, train_dl, optimizer):
            model.train()
            for i_batch, x in enumerate(train_dl):
                optimizer.zero_grad()
                xb_mods = [xb_mod.float().type(dtype) for xb_mod in x[:-1]]
                lens = x[2].float()
                loss, _, _ = model(xb_mods, lens)
                if self.config.log_with_wandb:
                    wandb.log({'ssl batch loss': loss.item(), 'lr ssl': optimizer.param_groups[0]['lr']})
                loss.backward()
                optimizer.step()
                with warmup_scheduler.dampening():
                    scheduler.step()

        def eval_during_training(model, dl, use_batch_num=5):
            model.eval()
            with torch.no_grad():
                validation_loss = []
                mod_output_all = []
                for i_batch, x in enumerate(dl):
                    if i_batch >= use_batch_num:
                        continue
                    xb_mods = [xb_mod.float().type(dtype) for xb_mod in x[:-1]]
                    lens = x[2].float()
                    loss, mod_outputs, _ = model(xb_mods, lens)
                    validation_loss.append(loss.item())
                    mod_output_all.append(mod_outputs)
            return np.mean(validation_loss), mod_output_all

        train_data, vali_data = self.train_data_ssl, self.vali_data_ssl
        train_step_lens = get_step_len(train_data[_mods[0]])
        vali_step_lens = get_step_len(vali_data[_mods[0]])
        model = SslReconstructNet(self.emb_net, config['ssl_loss_fn'])
        if self.config.log_with_wandb:
            wandb.watch(model, config['ssl_loss_fn'], log='all', log_freq=20)

        train_dl = prepare_dl([train_data[mod] for mod in _mods] + [train_step_lens], int(self.config.batch_size_ssl), shuffle=True, drop_last=True)
        vali_dl = prepare_dl([vali_data[mod] for mod in _mods] + [vali_step_lens], int(self.config.batch_size_ssl), shuffle=False, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr_ssl, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.NumGradDeSsl)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=int(self.config.NumGradDeSsl/5))
        dtype, model = set_dtype_and_model(self.config.device, model)
        epoch_end_time = time.time()
        epoch = int(np.ceil(self.config.NumGradDeSsl / len(train_dl)))
        _, mod_output_all = eval_during_training(model, vali_dl, np.nan)
        for i_epoch in range(epoch):
            train_loss, _ = eval_during_training(model, train_dl)
            test_loss, _ = eval_during_training(model, vali_dl)

            if epoch < 5 or i_epoch % int(epoch / 5) == 0 or i_epoch == epoch - 1:
                logging.info(f'| SSL | epoch{i_epoch:3d}/{epoch:3d} | time: {time.time() - epoch_end_time:5.2f}s | '
                             f'train loss {train_loss:5.4f} | test loss {test_loss:5.4f}')
            epoch_end_time = time.time()
            train_batch(model, train_dl, optimizer)
            if self.config.log_with_wandb:
                wandb.log({'ssl train loss': train_loss, 'ssl test loss': test_loss})

                # if i_epoch in list(range(0, epoch, int(epoch/6))) + [epoch-1]:
                if i_epoch in [0, epoch-1]:
                    model.eval()
                    with torch.no_grad():
                        x = list(enumerate(vali_dl))[0]
                        xb_mods = [xb_mod.float().type(dtype) for xb_mod in x[1][:-1]]
                        loss, mod_outputs, mask_indices = model(xb_mods, None)
                        show_reconstructed_signal(torch.concat(xb_mods, dim=1), mod_outputs,
                                                  'epoch ' + str(i_epoch) + self.config.FileNameAppendix, mask_indices)
        self.save_emb_net_post_ssl(model)
        return {'model': model}

    def save_emb_net_post_ssl(self, model):
        emb_net_post_ssl_state = model.emb_net.state_dict()
        save_path = os.path.join(self.config.result_dir, 'emb_' + self.config.FileNameAppendix[2:] + '.pth')
        torch.save(emb_net_post_ssl_state, save_path)

    def run_ssl(self):
        sets_of_data_all_dsets = []
        for ssl_file_name in self.ssl_task['ssl_file_names']:
            sets_of_data_all_dsets.append(self.preprocess(ssl_file_name))
        self.train_data_ssl = {mod: np.concatenate([sets_of_data[0][mod] for sets_of_data in sets_of_data_all_dsets], axis=0) for mod in ['acc', 'gyr']}
        self.vali_data_ssl = {mod: np.concatenate([sets_of_data[1][mod] for sets_of_data in sets_of_data_all_dsets], axis=0) for mod in ['acc', 'gyr']}
        self.test_data_ssl = {mod: np.concatenate([sets_of_data[2][mod] for sets_of_data in sets_of_data_all_dsets], axis=0) for mod in ['acc', 'gyr']}
        self.ssl_training(config)


def parse_config(config):
    parser = argparse.ArgumentParser(description='TODO', argument_default=argparse.SUPPRESS)
    parser.add_argument('--NumGradDeSsl', type=int)
    config.update(vars(parser.parse_args()))
    return config


def run_da(da_frameworks, fold_num=5):
    for da_framework in da_frameworks:
        dataset_name = da_framework.da_task['dataset']
        sub_ids = SUB_ID_ALL_DATASETS[dataset_name]
        test_sets = np.array_split(sub_ids, fold_num)
        for i_fold, test_set in enumerate(test_sets):
            test_set = list(test_set)
            train_set = [id for id in sub_ids if id not in test_set]
            logging.info(dataset_name + ', cross validation fold {}'.format(i_fold))
            da_framework.load_and_process(dataset_name, train_set, test_set, test_set)
            for ratio in da_framework.da_use_ratios:
                config['da_use_ratio'] = ratio
                da_framework.regressibility(linear_protocol=True, use_ssl=True)
                da_framework.regressibility(linear_protocol=True, use_ssl=False)
                da_framework.regressibility(linear_protocol=False, use_ssl=True)
                da_framework.regressibility(linear_protocol=False, use_ssl=False)
                plt.close("all")


DOWNSTREAM_0 = {'_mods': _mods, 'remove_trial_type': [], 'dataset': 'walking_knee_moment', 'output_columns': ['KFM'],
                'da_use_ratios': [0.1], 'imu_segments': STANDARD_IMU_SEQUENCE}
DOWNSTREAM_1 = {'_mods': _mods, 'remove_trial_type': [], 'dataset': 'Camargo_100', 'output_columns': ['knee_angle_r_moment'],
                'da_use_ratios': [0.1], 'imu_segments': ['CHEST', 'rand_noise', 'rand_noise', 'rand_noise', 'R_SHANK', 'rand_noise', 'rand_noise', 'rand_noise']}
DOWNSTREAM_2 = {'_mods': _mods, 'remove_trial_type': [], 'dataset': 'sun_drop_jump', 'output_columns': ['R_KNEE_MOMENT_X'],
                'da_use_ratios': [1.], 'imu_segments': STANDARD_IMU_SEQUENCE}
DOWNSTREAM_3 = {'_mods': _mods, 'remove_trial_type': [], 'dataset': 'opencap_dj', 'output_columns': ['knee_angle_r_moment'],
                'da_use_ratios': [1.], 'imu_segments': STANDARD_IMU_SEQUENCE}
DOWNSTREAM_4 = {'_mods': _mods, 'remove_trial_type': [], 'dataset': 'opencap_squat', 'output_columns': ['knee_angle_r_moment'],
                'da_use_ratios': [1.], 'imu_segments': STANDARD_IMU_SEQUENCE}
DOWNSTREAM_5 = {'_mods': _mods, 'remove_trial_type': [], 'dataset': 'opencap_sts', 'output_columns': ['knee_angle_r_moment'],
                'da_use_ratios': [1.], 'imu_segments': STANDARD_IMU_SEQUENCE}

# log_array = [round(10**x, 3) for x in np.linspace(-2, 0, 11)]

SSL_KAM = {'ssl_file_names': ['filtered_walking_knee_moment'], 'imu_segments': STANDARD_IMU_SEQUENCE}
SSL_KAM_MOVI = {'ssl_file_names': ['filtered_walking_knee_moment', 'MoVi'], 'imu_segments': STANDARD_IMU_SEQUENCE}
SSL_KAM_AMASS = {'ssl_file_names': ['filtered_walking_knee_moment', 'amass'], 'imu_segments': STANDARD_IMU_SEQUENCE}
# SSL_KAM_AMASS_TEN_PERCENT = {'ssl_file_names': ['filtered_walking_knee_moment', 'amass'], 'imu_segments': STANDARD_IMU_SEQUENCE}

config = {'NumGradDeSsl': 3e4, 'NumGradDeDa': 2e2, 'ssl_use_ratio': 1, 'log_with_wandb': True,
# config = {'NumGradDeSsl': 1e1, 'NumGradDeDa': 1e1, 'ssl_use_ratio': 0.1, 'log_with_wandb': False,
          'batch_size_ssl': 64, 'batch_size_linear': 32, 'lr_ssl': 1e-4, 'FeedForwardDim': 512, 'nlayers': 6, 'nhead': 48,
          'device': 'cuda', 'ssl_loss_fn': mse_loss_masked, 'emb_net': transformer}

test_name = 'SSL_KAM_AMASS'
test_info = 'SSL_KAM_AMASS'

# config['result_dir'] = os.path.join('../figures/results/t04_da_opencap')      # local
# config['result_dir'] = os.path.join('../../results/2023_05_08_da')      # cluster
config['result_dir'] = os.path.join(RESULTS_PATH, result_folder() + "_" + test_name)
config = parse_config(config)

if config['log_with_wandb']:
    wandb.init(project="IMU_SSL", config=config, name=test_name)

if __name__ == '__main__':
    os.makedirs(os.path.join(config['result_dir']), exist_ok=True)
    add_file_handler(logging, os.path.join(config['result_dir'], 'training_log.txt'))

    # independent_hyper = ('nlayers', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    independent_hyper = ('NumGradDeSsl', [config['NumGradDeSsl']])
    # {1: [16, 32, 48, 64, 80, 96], 2: [8, 16, 24, 32, 40, 48], 4: [4, 8, 12, 16, 20, 24], 8: [2, 4, 6, 8, 10, 12]}
    coupled_hypers = (['PatchLen', 'MaskPatchNum'], {4: [8, 16], 8: [4, 8]})
    # coupled_hypers = (['PatchLen', 'MaskPatchNum'],  {8: [6]})

    for indep_hyper_val in independent_hyper[1]:
        for coupled_hyper_val_1, coupled_hyper_val_list_2 in coupled_hypers[1].items():
            for coupled_hyper_val_2 in coupled_hyper_val_list_2:
                config[independent_hyper[0]] = indep_hyper_val
                config[coupled_hypers[0][0]] = coupled_hyper_val_1
                config[coupled_hypers[0][1]] = coupled_hyper_val_2
                config['FileNameAppendix'] = f', {independent_hyper[0]}_{indep_hyper_val},' \
                                             f' {coupled_hypers[0][0]}_{coupled_hyper_val_1},' \
                                             f' {coupled_hypers[0][1]}_{coupled_hyper_val_2}'
                logging.info(test_info)
                logging.info(config['FileNameAppendix'])
                logging.info(config)

                FrameworkSSL(config, SSL_KAM_AMASS).run_ssl()
                da_frameworks = [FrameworkDownstream(config, da_task) for da_task in
                                 [DOWNSTREAM_0, DOWNSTREAM_1, DOWNSTREAM_2, DOWNSTREAM_3, DOWNSTREAM_4, DOWNSTREAM_5]]
                run_da(da_frameworks, fold_num=5)

                plt.show()


"""
[Insights]
1. Transformer complexity matters. Longer patch length corresponds to larger models. Improvements might be from larger complexity.

[TODO]
0. Improvement on each window
1. !!! Camargo check incorrect trials and remove them
2. Robustness as a downstream task, e.g. against one biased sensor
3. 
4. flash attention
5. Compare against 1000+ rand init

"""

