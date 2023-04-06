import argparse
import configparser
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
from model import resnet18, SslContrastiveNet, \
    SslContrastiveNet, RegressNet, resnet50, transformer, SslReconstructNet, \
    mse_loss_masked, mse_loss
import time
from types import SimpleNamespace
from utils import prepare_dl, set_dtype_and_model, fix_seed, normalize_data, result_folder, define_channel_names, \
    preprocess_modality, get_scores, print_table, get_step_len, save_multi_image
from const import DICT_TRIAL_TYPE_ID, RESULTS_PATH, \
    CAMARGO_SUB_HEIGHT_WEIGHT, \
    GRAVITY, train_sub_carmargo, test_sub_carmargo, test_sub_hw, train_sub_hw, train_sub_kam, test_sub_kam, \
    SUB_ID_ALL_DATASETS, _mods
from config import DATA_PATH
import json
import pytorch_warmup as warmup
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class FrameworkDownstream:
    def __init__(self, config, da_task):
        self.config = SimpleNamespace(**config)
        self.output_columns = da_task['output_columns']
        self._data_scalar = {'base_scalar': StandardScaler}
        self.da_task = da_task
        fix_seed()

        self.emb_net = self.config.emb_net(len(da_task['imu_segments'])*6, self.config.emb_output_dim, mask_patch_num=0, patch_len=self.config.patch_len)
        self.regress_net = RegressNet(self.emb_net, len(define_channel_names(da_task)['acc'])+len(define_channel_names(da_task)['gyr']), len(self.output_columns))
        _, self.regress_net = set_dtype_and_model(self.config.device, self.regress_net)
        self.regress_net_init_state = copy.deepcopy(self.regress_net.state_dict())

        post_ssl_emb_net = self.load_post_ssl_emb_net()
        self.regress_net.emb_net.load_state_dict(post_ssl_emb_net)
        self.regress_net_post_ssl_state = copy.deepcopy(self.regress_net.state_dict())

        # num_params = sum(param.numel() for param in model.embnet_imu.rnn_layer.parameters())
        # print(num_params)

    def load_post_ssl_emb_net(self):
        emb_path = os.path.join(RESULTS_PATH, self.da_task['ssl_model'] + self.da_task['dataset'] + '_' +
                                self.config.emb_net.__name__ + '.pth')
        return torch.load(emb_path)

    def load_and_process(self, test_name, train_sub_ids: List[str], validate_sub_ids: List[str], test_sub_ids: List[str]):
        if test_name == 'hw_running':
            self.load_and_process_hw_running(train_sub_ids, validate_sub_ids, test_sub_ids)
        elif 'Carmargo' in test_name:
            self.load_and_process_camargo(train_sub_ids, validate_sub_ids, test_sub_ids)
        elif test_name == 'walking_knee_moment':
            self.load_and_process_kam(train_sub_ids, validate_sub_ids, test_sub_ids)

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
                    sub_data = hf[sub_id][::100, :, :]      #
                    has_grf_index = np.where(np.max(sub_data[:, self.data_columns.index('fz')], axis=1) > 10)
                    sub_data = sub_data[has_grf_index]
                    sub_weight = CAMARGO_SUB_HEIGHT_WEIGHT[sub_id][1] * GRAVITY
                    force_col_loc = [self.data_columns.index(x) for x in ['fx', 'fy', 'fz']]
                    sub_data[:, force_col_loc] = sub_data[:, force_col_loc] / sub_weight
                    current_set_data_list.append(sub_data)
                current_set_data = np.concatenate(current_set_data_list, axis=0)
                """ [step, feature, time] """
                self.set_data[data_name] = current_set_data

    def load_and_process_kam(self, train_sub_ids: List[str], validate_sub_ids: List[str],
                             test_sub_ids: List[str]):
        self.set_data = {}
        with h5py.File(DATA_PATH + self.da_task['dataset'] + '.h5', 'r') as hf:
            self.data_columns = json.loads(hf.attrs['columns'])
            for set_sub_ids, data_name in zip([train_sub_ids, validate_sub_ids, test_sub_ids], ['train', 'vali', 'test']):
                current_set_data_list = [hf[sub_][:, :128] for sub_ in set_sub_ids]     # only keep 128 time steps
                current_set_data = np.concatenate(current_set_data_list, axis=0).transpose([0, 2, 1])
                """ [step, feature, time] """
                self.set_data[data_name] = current_set_data

    def load_and_process_sun(self, train_sub_ids: List[str], validate_sub_ids: List[str],
                             test_sub_ids: List[str]):
        self.set_data = {}
        with h5py.File(DATA_PATH + self.da_task['dataset'] + '.h5', 'r') as hf:
            self.data_columns = list(hf[train_sub_ids[0]].attrs['columns'])
            if 'sub_id' not in self.data_columns:
                self.data_columns.append('sub_id')
                self.data_columns.append('output_processed')
            for set_sub_ids, data_name in zip([train_sub_ids, validate_sub_ids, test_sub_ids], ['train', 'vali', 'test']):
                current_set_data_list = []
                for sub_ in set_sub_ids:
                    current_sub_data_list = [hf[sub_][('0' + str(trial + 1))[-2:]] for trial in range(30)
                                             if ('0' + str(trial + 1))[-2:] in hf[sub_]]
                    current_sub_data = np.stack(current_sub_data_list, axis=0).transpose([0, 2, 1])
                    sub_id_np = np.full([current_sub_data.shape[0], 1, current_sub_data.shape[2]], int(sub_[2:4]))
                    current_set_data_list.append(np.concatenate([current_sub_data, sub_id_np], axis=1))
                current_set_data = np.concatenate(current_set_data_list, axis=0)

                step_lens = get_step_len(current_set_data)
                output_selected = current_set_data[:, self.data_columns.index('output')]
                output_loc = np.zeros([output_selected.shape[0]])
                for i_step, step_len in enumerate(step_lens):
                    output_loc[i_step] = np.argmax(output_selected[i_step, :int(.5 * step_len)])
                output_loc = output_loc.astype(int)
                output_ = np.array([output_selected[i_row, loc] for i_row, loc in enumerate(output_loc)]).reshape([-1, 1])
                output_processed = np.repeat(output_[:, np.newaxis], current_set_data.shape[2], axis=1).reshape([-1, 1, current_set_data.shape[2]])

                current_set_data = np.concatenate([current_set_data, output_processed], axis=1)
                self.set_data[data_name] = current_set_data

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
            if data_name == 'train':
                sampled_rows = np.sort(random.sample(range(current_set_data.shape[0]), int(self.config.da_use_ratio*current_set_data.shape[0])))
                sampled_data = copy.deepcopy(current_set_data[sampled_rows])
            else:
                sampled_data = copy.deepcopy(current_set_data)
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

    # def norm_output_by_subject_weight(self, sampled_data):
    #     output_data = sampled_data[:, [self.data_columns.index(x) for x in self.output_columns.keys()]]
    #     for i_output, (output_name, output_norm_col) in enumerate(self.output_columns.items()):
    #         if output_norm_col is not None:
    #             weight_data = sampled_data[:, self.data_columns.index(output_norm_col)]
    #             output_data[:, i_output] = np.divide(output_data[:, i_output], weight_data, out=np.zeros_like(output_data[:, i_output]), where=weight_data!=0)
    #     return output_data

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

    def save_embeddings(self, model, test_dl, embedding_name):
        dtype, model = set_dtype_and_model(self.config.device, model)
        mod_output_all, sub_id_all, trial_id_all = [], [], []
        model.eval()
        with torch.no_grad():
            for i_batch, batch_data in enumerate(test_dl):
                xb = [data_.float().type(dtype) for data_ in batch_data[:2]]
                lens = batch_data[3].float()
                _, mod_outputs_batch = model(xb, lens)
                mod_output_all.append(mod_outputs_batch)
                sub_ids = batch_data[4].int()
                sub_id_all.append(sub_ids)
                trial_ids = batch_data[5].int()
                trial_id_all.append(trial_ids)

        ssl_general_model_path = os.path.join(self.config.result_dir, 'embedding_similarity_between_segments')
        os.makedirs(ssl_general_model_path, exist_ok=True)
        mod_acc, mod_gyr = [], []
        for mod_outputs in mod_output_all:
            mod_acc.append(mod_outputs[0])
            mod_gyr.append(mod_outputs[1])
        mod_acc, mod_gyr = torch.concat(mod_acc, dim=0), torch.concat(mod_gyr, dim=0)
        with h5py.File(os.path.join(ssl_general_model_path, self.da_task['dataset'] + '.h5'), 'a') as hf:
            grp = hf.require_group(embedding_name)
            grp_acc, grp_gyr, grp_info = grp.require_group('mod_acc'), grp.require_group('mod_gyr'), grp.require_group('info')
            mod_acc, mod_gyr = mod_acc.detach().cpu().numpy(), mod_gyr.detach().cpu().numpy()
            sub_id_all = torch.concat(sub_id_all, dim=0).cpu().numpy()
            trial_id_all = torch.concat(trial_id_all, dim=0).cpu().numpy()
            subject_id_set = list(set(sub_id_all))
            for i_sub in subject_id_set:
                sub_data_loc = np.where(sub_id_all == i_sub)[0]
                mod_acc_sub, mod_gyr_sub, trial_id_sub = mod_acc[sub_data_loc], mod_gyr[sub_data_loc], trial_id_all[sub_data_loc]
                grp_acc.require_dataset('sub_'+str(i_sub), shape=mod_acc_sub.shape, data=mod_acc_sub, dtype='float32')
                grp_gyr.require_dataset('sub_'+str(i_sub), shape=mod_gyr_sub.shape, data=mod_gyr_sub, dtype='float32')
                grp_info.require_dataset('sub_'+str(i_sub), shape=trial_id_sub.shape, data=trial_id_sub, dtype='float32')

    def save_model_and_results(self, test_name, y_true, y_pred, sub_ids, model):
        os.makedirs(os.path.join(self.config.result_dir, 'test_models', self.da_task['dataset']), exist_ok=True)
        copied_model = copy.deepcopy(model)
        torch.save(copied_model.state_dict(), os.path.join(self.config.result_dir, 'test_models', self.da_task['dataset'], test_name + '.pth'))
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
        model_name = 'linear_protocol_True, use_ssl_' + str(use_ssl) + ', ratio_' + str(self.config.da_use_ratio) + self.config.file_name_appendix
        regress_net_post_linear_head_init = torch.load(os.path.join(
            self.config.result_dir, 'test_models', self.da_task['dataset'], model_name + '.pth'))
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
                # if i_optimize in [int(10 ** x) for x in np.linspace(1, 3, 81)]:
                #     record_intermediate_results(i_optimize)
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
            intermediate_result_name = 'linear_protocol_' + str(linear_protocol) + ', use_ssl_' + str(use_ssl) + \
                                       ', ratio_' + str(self.config.da_use_ratio) + ', i_optimize_' + str(i_optimize)
            self.save_results(intermediate_result_name, y_true, y_pred, test_data['sub_id'])

        self.config = SimpleNamespace(**config)
        test_name = 'linear_protocol_' + str(linear_protocol) + ', use_ssl_' + str(use_ssl) + \
                    ', ratio_' + str(self.config.da_use_ratio) + self.config.file_name_appendix
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_gradient_de_da)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=int(self.config.num_gradient_de_da/5))

        epoch = int(np.ceil(self.config.num_gradient_de_da / len(train_dl)))
        i_optimize = 0
        for i_epoch in range(epoch):
            if verbose:
                train_loss = eval_during_training(model, train_dl, torch.nn.MSELoss())
                test_loss = eval_during_training(model, test_dl, torch.nn.MSELoss())
                if self.config.log_with_wandb:
                    wandb.log({'linear train loss': train_loss, 'linear test loss': test_loss})
                logging.info(f'| Regressibility | epoch{i_epoch:3d}/{epoch:3d} | time: {time.time() - epoch_end_time:5.2f}s |'
                             f' train loss {train_loss:5.3f} | test loss {test_loss:5.3f}')
                epoch_end_time = time.time()
            i_optimize = train_batch(model, train_dl, optimizer, torch.nn.MSELoss(), i_optimize)

        y_true, y_pred = evaluate_after_training(test_dl)
        y_true, y_pred = self.inverse_normalize_output(y_true, y_pred)
        self.save_model_and_results(test_name, y_true, y_pred, test_data['sub_id'], self.regress_net)
        if show_fig:
            plt.figure()
            for i_output in range(y_true.shape[1]):
                plt.title('test set')
                plt.plot(y_true[:10, i_output].ravel(), '-', color='C'+str(i_output), label=list(self.output_columns.keys())[i_output])
                plt.plot(y_pred[:10, i_output].ravel(), '--', color='C'+str(i_output), label=list(self.output_columns.keys())[i_output])
                plt.legend()
        if verbose:
            all_scores = get_scores(y_true, y_pred, self.output_columns, test_step_lens)
            all_scores = [{'subject': 'all', **scores} for scores in all_scores]
            print_table(all_scores)

        return y_pred, model

    def load_data_and_save_ssl_embedding(self, da_use_ratios):
        config['da_use_ratio'] = da_use_ratios[0]
        self.config = SimpleNamespace(**config)
        self.sample_and_normalize_data()

        test_data = self.da_task['test']
        test_input_data = [test_data[mod] for mod in self.da_task['_mods']]
        test_output_data = test_data['output']
        test_step_lens = get_step_len(test_input_data[0])
        test_dl = prepare_dl([*test_input_data, test_output_data, test_step_lens, test_data['sub_id'],
                              test_data['trial_type_id']], 1024, shuffle=False)

        self.set_regress_net_to_post_ssl_state()
        self.save_embeddings(copy.deepcopy(self.regress_net), test_dl, 'use_ssl')

        self.set_regress_net_to_init_state()
        self.save_embeddings(copy.deepcopy(self.regress_net), test_dl, 'no_ssl')


class FrameworkSSL:
    def __init__(self, config, ssl_task):
        self.config = SimpleNamespace(**config)
        self.ssl_task = ssl_task
        with h5py.File(DATA_PATH + ssl_task['ssl_file_name'] + '.h5', 'r') as hf:
            self.ssl_data = {sub_: sub_data[:int(self.config.ssl_use_ratio * hf[sub_].shape[0]), :, :]
                             for sub_, sub_data in hf.items()}
            self.ssl_columns = json.loads(hf.attrs['columns'])
        os.makedirs(os.path.join(self.config.result_dir), exist_ok=True)

        if 'walking_knee_moment' == ssl_task['ssl_file_name']:
            self.ssl_data = {sub_: data_.transpose([0, 2, 1]) for sub_, data_ in self.ssl_data.items()}
            """ [step, feature, time] """

        self.emb_net = self.config.emb_net(len(ssl_task['imu_segments'])*6, self.config.emb_output_dim, self.config.mask_patch_num, self.config.patch_len)
        logging.info('# of trainable parameters: {}'.format(sum(p.numel() for p in self.emb_net.transformer_encoder.parameters() if p.requires_grad)))
        self._data_scalar = {'base_scalar': StandardScaler}
        fix_seed()

    def preprocess(self, train_sub_ids: List[str], validate_sub_ids: List[str], test_sub_ids: List[str]):
        """
        train_sub_ids: a list of subject id for model training
        validate_sub_ids: a list of subject id for model validation
        test_sub_ids: a list of subject id for model testing
        """
        if train_sub_ids == ['except test']:
            train_data = np.concatenate([self.ssl_data[sub] for sub in list(self.ssl_data.keys()) if sub not in test_sub_ids], axis=0)
        else:
            train_data = np.concatenate([self.ssl_data[sub] for sub in train_sub_ids], axis=0)
        vali_data = np.concatenate([self.ssl_data[sub] for sub in validate_sub_ids], axis=0)
        test_data = np.concatenate([self.ssl_data[sub] for sub in test_sub_ids], axis=0)

        # SSL preprocess
        train_data_ssl = preprocess_modality(self.ssl_columns, self._data_scalar, train_data, define_channel_names(self.ssl_task), 'fit_transform')
        logging.info('SSL training with subject ids: {}. Number of steps: {}'.format(train_sub_ids, list(train_data_ssl.values())[0].shape[0]))
        vali_data_ssl = preprocess_modality(self.ssl_columns, self._data_scalar, vali_data, define_channel_names(self.ssl_task), 'transform')
        logging.info('SSL validation with subject ids: {}. Number of steps: {}'.format(validate_sub_ids, list(vali_data_ssl.values())[0].shape[0]))
        test_data_ssl = preprocess_modality(self.ssl_columns, self._data_scalar, test_data, define_channel_names(self.ssl_task), 'transform')
        logging.info('SSL testing with subject ids: {}. Number of steps: {}'.format(test_sub_ids, list(test_data_ssl.values())[0].shape[0]))
        self.train_data_ssl, self.vali_data_ssl, self.test_data_ssl = train_data_ssl, vali_data_ssl, test_data_ssl

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
                loss, _ = model(xb_mods, lens)
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
                    loss, mod_outputs = model(xb_mods, lens)
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

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_gradient_de_ssl)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=int(self.config.num_gradient_de_ssl/5))
        dtype, model = set_dtype_and_model(self.config.device, model)
        epoch_end_time = time.time()
        epoch = int(np.ceil(self.config.num_gradient_de_ssl / len(train_dl)))
        _, mod_output_all = eval_during_training(model, vali_dl, np.nan)
        if self.config.save_emb: self.save_embeddings(mod_output_all, 'no_ssl')
        for i_epoch in range(epoch):
            train_loss, _ = eval_during_training(model, train_dl)
            test_loss, _ = eval_during_training(model, vali_dl)

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
                        # x = list(enumerate(train_dl))[i_fig_num]
                        # xb_mods = [xb_mod.float().type(dtype) for xb_mod in x[1][:-1]]
                        # model.show_reconstructed_signal(xb_mods, None, f'Train_{i_epoch}')

                        x = list(enumerate(vali_dl))[0]
                        xb_mods = [xb_mod.float().type(dtype) for xb_mod in x[1][:-1]]
                        model.show_reconstructed_signal(xb_mods, None, self.config.file_name_appendix)
        self.save_emb_net_post_ssl(model)
        if self.config.save_emb:
            _, mod_output_all = eval_during_training(model, vali_dl, np.nan)
            self.save_embeddings(mod_output_all, 'use_ssl')
        return {'model': model}

    def save_emb_net_post_ssl(self, model):
        emb_net_post_ssl_state = model.emb_net.state_dict()
        save_path = os.path.join(RESULTS_PATH, self.ssl_task['ssl_file_name'] + '_' + self.config.emb_net.__name__ + '.pth')
        torch.save(emb_net_post_ssl_state, save_path)

    def save_embeddings(self, mod_output_all, embedding_name):
        ssl_general_model_path = os.path.join(self.config.result_dir, 'embedding_similarity_between_segments')
        os.makedirs(ssl_general_model_path, exist_ok=True)
        mod_acc, mod_gyr = [], []
        for mod_outputs in mod_output_all:
            mod_acc.append(mod_outputs[0])
            mod_gyr.append(mod_outputs[1])
        mod_acc, mod_gyr = torch.concat(mod_acc, dim=0), torch.concat(mod_gyr, dim=0)
        with h5py.File(os.path.join(ssl_general_model_path, self.ssl_task['ssl_file_name'] + '.h5'), 'a') as hf:
            grp = hf.require_group(embedding_name)
            mod_acc, mod_gyr = mod_acc.cpu().numpy(), mod_gyr.cpu().numpy()
            num_of_sensors = len(ssl_task_Carmargo['imu_segments'])
            mod_acc, mod_gyr = np.reshape(mod_acc, [-1, num_of_sensors*mod_acc.shape[1]]), np.reshape(mod_gyr, [-1, num_of_sensors*mod_acc.shape[1]])
            grp_acc, grp_gyr = grp.require_group('mod_acc'), grp.require_group('mod_gyr')
            grp_acc.require_dataset('sub_0', shape=mod_acc.shape, data=mod_acc, dtype='float32')
            grp_gyr.require_dataset('sub_0', shape=mod_gyr.shape, data=mod_gyr, dtype='float32')

    def save_ssl_general_net(self, model, model_name):
        ssl_general_model_path = os.path.join(RESULTS_PATH, 'embedding_similarity_between_segments', self.ssl_task['ssl_file_name'])
        os.makedirs(ssl_general_model_path, exist_ok=True)
        save_path = os.path.join(ssl_general_model_path, model_name + '.pth')
        torch.save(model, save_path)

    def hyperparam_tuning(self, model, x_test):
        raise RuntimeError('Method not implemented')


def parse_config(config):
    parser = argparse.ArgumentParser(description='TODO', argument_default=argparse.SUPPRESS)
    parser.add_argument('--num_gradient_de_ssl', type=int)
    config.update(vars(parser.parse_args()))
    return config


def run_ssl(ssl_task):
    ssl_framework = FrameworkSSL(config, ssl_task)
    if 'MoVi' in ssl_task['ssl_file_name']:
        ssl_framework.preprocess(train_sub_movi, test_sub_movi, test_sub_movi)
    elif ssl_task['ssl_file_name'] == 'Carmargo':
        ssl_framework.preprocess(train_sub_carmargo+test_sub_carmargo, test_sub_carmargo, test_sub_carmargo)
    elif ssl_task['ssl_file_name'] == 'walking_knee_moment':
        ssl_framework.preprocess(train_sub_kam+test_sub_kam, test_sub_kam, test_sub_kam)
    elif ssl_task['ssl_file_name'] == 'hw_running':
        ssl_framework.preprocess(train_sub_hw+test_sub_hw, test_sub_hw, test_sub_hw)
    elif 'Combined' in ssl_task['ssl_file_name']:
        ssl_framework.preprocess(train_sub_combined_dataset, test_sub_combined_dataset, test_sub_combined_dataset)
    ssl_framework.ssl_training(config)


def run_da(da_framework, da_use_ratios=[1.]):
    for ratio in da_use_ratios:
        config['da_use_ratio'] = ratio
        da_framework.regressibility(linear_protocol=True, use_ssl=True)
        da_framework.regressibility(linear_protocol=True, use_ssl=False)
        da_framework.regressibility(linear_protocol=False, use_ssl=True)
        da_framework.regressibility(linear_protocol=False, use_ssl=False)


def run_cross_vali(da_framework, da_use_ratios, fold_num=5, only_test_one_fold=False):
    dataset_name = da_framework.da_task['dataset']
    sub_ids = SUB_ID_ALL_DATASETS[dataset_name]
    test_sets = np.array_split(sub_ids, fold_num)
    if only_test_one_fold: test_sets = test_sets[-1:]
    for i_fold, test_set in enumerate(test_sets):
        test_set = list(test_set)
        train_set = [id for id in sub_ids if id not in test_set]
        logging.info(dataset_name + ', cross validation fold {}'.format(i_fold))
        da_framework.load_and_process(dataset_name, train_set, test_set, test_set)
        if config['save_emb']: da_framework.load_data_and_save_ssl_embedding(da_use_ratios)
        run_da(da_framework, da_use_ratios)


DOWNSTREAM_TASK_0 = {'_mods': _mods, 'remove_trial_type': ['Treadmill', 'Stair', 'Ramp'], 'dataset': 'Carmargo',
                     'imu_segments': ['trunk', 'shank'], 'ssl_model': 'MoVi_'}
DOWNSTREAM_TASK_1 = {'_mods': _mods, 'remove_trial_type': ['LevelGround', 'Stair', 'Ramp'], 'dataset': 'Carmargo',
                     'imu_segments': ['trunk', 'shank'], 'ssl_model': 'MoVi_'}
DOWNSTREAM_TASK_2 = {'_mods': _mods, 'remove_trial_type': ['Treadmill'], 'dataset': 'Carmargo', 'output_columns': ['fx', 'fy', 'fz'],
                     'imu_segments': ['trunk', 'shank'], 'ssl_model': 'Combined_'}
DOWNSTREAM_TASK_3 = {'_mods': _mods, 'remove_trial_type': [], 'dataset': 'walking_knee_moment', 'output_columns': ['KFM', 'KAM', 'EXT_KM_Z'],
                     'imu_segments': ['R_FOOT', 'R_SHANK', 'R_THIGH', 'L_SHANK', 'L_THIGH', 'L_FOOT', 'WAIST', 'CHEST'], 'ssl_model': ''}
DOWNSTREAM_TASK_4 = {'_mods': _mods, 'remove_trial_type': [], 'dataset': 'sun_drop_jump',
                     'imu_segments': ['R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_FOOT', 'L_SHANK', 'L_THIGH'], 'ssl_model': 'MoVi_'}
DOWNSTREAM_TASK_5 = {'_mods': _mods, 'remove_trial_type': [], 'dataset': 'sun_drop_jump',
                     'imu_segments': ['R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_FOOT', 'L_SHANK', 'L_THIGH'], 'ssl_model': 'MoVi_'}
DOWNSTREAM_TASK_6 = {'_mods': _mods, 'remove_trial_type': [], 'dataset': 'hw_running',
                     'imu_segments': ['l_shank'], 'max_pooling_to_downsample': True, 'ssl_model': 'MoVi_'}

ssl_task_Carmargo = {'ssl_file_name': 'Combined_Carmargo', 'imu_segments': ['R_SHANK', 'R_THIGH']}
    # 'Hip', 'Spine1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Head',
    # 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']}
ssl_task_sun_drop_jump = copy.deepcopy(ssl_task_Carmargo)
ssl_task_sun_drop_jump.update({'ssl_file_name': 'MoVi_sun_drop_jump'})
ssl_task_kam = copy.deepcopy(ssl_task_Carmargo)
ssl_task_kam.update({'ssl_file_name': 'walking_knee_moment', 'imu_segments': [
    'R_FOOT', 'R_SHANK', 'R_THIGH', 'L_SHANK', 'L_THIGH', 'L_FOOT', 'WAIST', 'CHEST']})
ssl_task_hw_running = copy.deepcopy(ssl_task_Carmargo)
ssl_task_hw_running.update({'ssl_file_name': 'MoVi_hw_running'})

config = {'num_gradient_de_ssl': 2e3, 'num_gradient_de_da': 5e2, 'ssl_use_ratio': 1, 'log_with_wandb': True,
# config = {'num_gradient_de_ssl': 1e2, 'num_gradient_de_da': 1e1, 'ssl_use_ratio': 0.02, 'log_with_wandb': False,
          'batch_size_ssl': 64, 'batch_size_linear': 32, 'lr_ssl': 1e-3, 'emb_output_dim': 16,
          'device': 'cuda', 'result_dir': os.path.join(RESULTS_PATH, result_folder()), 'save_emb': False,
          'ssl_loss_fn': mse_loss_masked, 'emb_net': transformer}
config = parse_config(config)

if config['log_with_wandb']:
    wandb.init(project="IMU_SSL", config=config, name='nlayers=6, nhead=8, dim_feedforward=512')
train_sub_movi = ['sub_' + str(i+1) for i in range(0, 80)]
test_sub_movi = ['sub_' + str(i+1) for i in range(80, 88)]
train_sub_combined_dataset = ['except test']        # except test
test_sub_combined_dataset = ['dset0'] + ['dset' + str(i) for i in range(6, 9)]


"""
[Insights]
1. Transformer complexity matters. Longer patch length corresponds to larger models. Improvements might be from larger complexity.

[TODO]
0. Create a dataset of 17/21 IMUs with 3s windows
1. Load data dynamically

[TOTEST]
2. Robustness as a downstream task
3. Upload testing set to google drive

3. Compare against 1000+ rand init
4. Use joint contact force as downstream application, since AddBiom cannot provide this
"""


if __name__ == '__main__':
    os.makedirs(os.path.join(config['result_dir']), exist_ok=True)
    add_file_handler(logging, os.path.join(config['result_dir'], 'training_log.txt'))
    logging.info(config)

    patch_len_and_mask_patch_num = {1: [16, 32, 48, 64, 80, 96], 2: [8, 16, 24, 32, 40, 48], 4: [4, 8, 12, 16, 20, 24], 8: [2, 4, 6, 8, 10, 12]}
    for patch_len, mask_patch_num_list in patch_len_and_mask_patch_num.items():
        for mask_patch_num in mask_patch_num_list:
            config['mask_patch_num'] = mask_patch_num
            config['patch_len'] = patch_len
            config['file_name_appendix'] = f', masking_{mask_patch_num}, patchlen_{patch_len}'

            logging.info(f"Masking patch number: {mask_patch_num}; patch len {patch_len}")

            # run_ssl(ssl_task_hw_running)
            # run_ssl(ssl_task_Carmargo, mask_patch_num)
            run_ssl(ssl_task_kam)

            # da_framework = FrameworkDownstream(config, DOWNSTREAM_TASK_6)
            # run_cross_vali(da_framework, da_use_ratios=[1])

            # da_framework = FrameworkDownstream(config, DOWNSTREAM_TASK_2)
            # run_cross_vali(da_framework, da_use_ratios=[1])

            # log_array = [round(10**x, 3) for x in np.linspace(-2, 0, 11)]
            da_framework = FrameworkDownstream(config, DOWNSTREAM_TASK_3)
            run_cross_vali(da_framework, da_use_ratios=[0.1], fold_num=5)

            plt.show()
            plt.close("all")

