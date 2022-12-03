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
from model import nce_loss, CnnEmbedding, RegressNet, SslGeneralNet, nce_loss_groups_of_positive
import time
from types import SimpleNamespace
from utils import prepare_dl, set_dtype_and_model, fix_seed, normalize_data, result_folder, define_channel_names, \
    preprocess_modality, get_scores, print_table, get_step_len
from const import DATA_PATH, DICT_TRIAL_TYPE_ID, IMU_CARMARGO_SEGMENT_LIST, RESULTS_PATH, CAMARGO_SUB_HEIGHT_WEIGHT, \
    GRAVITY
from const import train_sub_carmargo, test_sub_carmargo, test_sub_hw, train_sub_hw, train_sub_kam, test_sub_kam, SUB_ID_ALL_DATASETS
import json
import pytorch_warmup as warmup
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class FrameworkDownstream:
    def __init__(self, config, da_task):
        self.config = SimpleNamespace(**config)
        self._data_scalar = {'base_scalar': StandardScaler()}
        self.da_task = da_task
        fix_seed()

        self.emb_nets = {mod: self.config.emb_net(3, self.config.emb_output_dim, mod + ' embedding') for mod in _mods}
        self.regress_net = RegressNet(torch.nn.ModuleList([self.emb_nets[mod] for mod in da_task['_mods']]),
                                      [len(define_channel_names(da_task)[mod]) for mod in da_task['_mods']], 1)
        _, self.regress_net = set_dtype_and_model(self.config.device, self.regress_net)
        self.regress_net_init_state = copy.deepcopy(self.regress_net.state_dict())

        post_ssl_emb_net = self.load_post_ssl_emb_net()
        [self.regress_net.emb_nets[i_mod].load_state_dict(post_ssl_emb_net[mod]) for i_mod, mod in enumerate(da_task['_mods'])]
        self.regress_net_post_ssl_state = copy.deepcopy(self.regress_net.state_dict())

        # num_params = sum(param.numel() for param in model.embnet_imu.rnn_layer.parameters())
        # print(num_params)

    def load_post_ssl_emb_net(self):
        emb_path = os.path.join(RESULTS_PATH, self.da_task['ssl_model'] + self.da_task['dataset'] + '.pth')
        return torch.load(emb_path)

    def load_and_process(self, test_name, train_sub_ids: List[str], validate_sub_ids: List[str], test_sub_ids: List[str]):
        if test_name == 'hw_running':
            self.load_and_process_hw_running(train_sub_ids, validate_sub_ids, test_sub_ids)
        elif 'Carmargo' in test_name:
            self.load_and_process_carmargo(train_sub_ids, validate_sub_ids, test_sub_ids)
        elif test_name == 'walking_knee_moment':
            self.load_and_process_kam(train_sub_ids, validate_sub_ids, test_sub_ids)

    def load_and_process_carmargo(self, train_sub_ids: List[str], validate_sub_ids: List[str],
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
            if 'output_processed' not in self.data_columns:
                self.data_columns.append('output_processed')
            for set_sub_ids, data_name in zip([train_sub_ids, validate_sub_ids, test_sub_ids], ['train', 'vali', 'test']):
                current_set_data_list, current_set_output_list = [], []
                for sub_id in set_sub_ids:
                    sub_data = hf[sub_id][:, :, :]
                    sub_weight = CAMARGO_SUB_HEIGHT_WEIGHT[sub_id][1] * GRAVITY
                    data_selected = self.select_data_by_list_of_values(sub_data, self.data_columns.index('trial_type_id'), [i for i in range(4) if i not in type_to_exclude])
                    data_selected = self.select_data_by_has_nonzero_element(data_selected, self.data_columns.index(self.da_task['output']), 3/8, 5/8)
                    output_raw = data_selected[:, self.data_columns.index(self.da_task['output'])]
                    output_loc = np.argmax(np.abs(output_raw[:, int(3/8 * output_raw.shape[1]): int(5/8 * output_raw.shape[1])]), axis=1) + int(3/8 * output_raw.shape[1])

                    # if data_name == 'vali':
                    #     plt.figure()
                    #     for i_step, step_data in enumerate(data_selected):
                    #         plt.plot(step_data[self.data_columns.index('fy')])
                    #         plt.plot(output_loc[i_step], output_raw[i_step, output_loc[i_step]], '*')
                    #     plt.show()

                    output_processed = np.array([output_raw[i_row, loc] for i_row, loc in enumerate(output_loc)]).reshape([-1, 1]) / sub_weight
                    output_processed = np.repeat(output_processed[:, np.newaxis], data_selected.shape[2], axis=1).reshape([-1, 1, data_selected.shape[2]])
                    data_selected = np.concatenate([data_selected, output_processed], axis=1)
                    current_set_data_list.append(data_selected)
                self.set_data[data_name] = np.concatenate(current_set_data_list, axis=0)

    def load_and_process_kam(self, train_sub_ids: List[str], validate_sub_ids: List[str],
                             test_sub_ids: List[str]):
        self.set_data = {}
        with h5py.File(DATA_PATH + self.da_task['dataset'] + '.h5', 'r') as hf:
            self.data_columns = json.loads(hf.attrs['columns'])
            if 'output_processed' not in self.data_columns:
                self.data_columns.append('output_processed')
            for set_sub_ids, data_name in zip([train_sub_ids, validate_sub_ids, test_sub_ids], ['train', 'vali', 'test']):
                current_set_data_list = [hf[sub_][:, :128] for sub_ in set_sub_ids]     # only keep 128 time steps
                current_set_data = np.concatenate(current_set_data_list, axis=0).transpose([0, 2, 1])
                """ [step, feature, time] """

                step_lens = get_step_len(current_set_data, feature_col=[1, 2])
                output_selected = current_set_data[:, self.data_columns.index(self.da_task['output'])]
                output_loc = np.zeros([output_selected.shape[0]])
                for i_step, step_len in enumerate(step_lens):
                    output_loc[i_step] = np.argmax(output_selected[i_step, 20:int(.5 * step_len)])+20
                output_loc = output_loc.astype(int)
                output_ = np.array([output_selected[i_row, loc] for i_row, loc in enumerate(output_loc)]).reshape([-1, 1])
                #
                # plt.figure()
                # for i_step, step_len in enumerate(step_lens):
                #     plt.plot(output_selected[i_step])
                #     plt.plot(output_loc[i_step], output_selected[i_step, output_loc[i_step]], '*')
                # plt.show()

                output_processed = np.repeat(output_[:, np.newaxis], current_set_data.shape[2], axis=1).reshape([-1, 1, current_set_data.shape[2]])
                current_set_data = np.concatenate([current_set_data, output_processed], axis=1)
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
                output_selected = current_set_data[:, self.data_columns.index(self.da_task['output'])]
                output_loc = np.zeros([output_selected.shape[0]])
                for i_step, step_len in enumerate(step_lens):
                    output_loc[i_step] = np.argmax(output_selected[i_step, :int(.5 * step_len)])
                output_loc = output_loc.astype(int)
                output_ = np.array([output_selected[i_row, loc] for i_row, loc in enumerate(output_loc)]).reshape([-1, 1])
                output_processed = np.repeat(output_[:, np.newaxis], current_set_data.shape[2], axis=1).reshape([-1, 1, current_set_data.shape[2]])

                # plt.figure()
                # for i_step, step_len in enumerate(step_lens):
                #     plt.plot(output_selected[i_step])
                #     plt.plot(output_loc[i_step], output_selected[i_step, output_loc[i_step]], '*')

                # plt.figure()
                # for i_step, step_len in enumerate(step_lens):
                #     plt.plot(current_set_data[i_step, self.data_columns.index('R_FOOT_Gyro_X')], 'r')
                # plt.show()

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
                output_raw = current_set_data[:, self.data_columns.index(self.da_task['output'])]
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
            data_ds[self.da_task['output']] = normalize_data(self._data_scalar, sampled_data[:, self.data_columns.index('output_processed'), 0:1],
                                                             self.da_task['output'], norm_method, 'by_all_columns')
            data_ds['sub_id'] = sampled_data[:, self.data_columns.index('sub_id'), 0]
            self.da_task[data_name] = data_ds

    def inverse_normalize_output(self, y_true, y_pred):
        y_true = normalize_data(self._data_scalar, y_true, self.da_task['output'], 'inverse_transform', 'by_all_columns')
        y_pred = normalize_data(self._data_scalar, y_pred, self.da_task['output'], 'inverse_transform', 'by_all_columns')
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
        mod_output_all, sub_id_all = [], []
        model.eval()
        with torch.no_grad():
            for i_batch, batch_data in enumerate(test_dl):
                xb = [data_.float().type(dtype) for data_ in batch_data[:-3]]
                lens = batch_data[-2].float()
                _, mod_outputs_batch = model(xb, lens)
                mod_output_all.append(mod_outputs_batch)
                sub_ids = batch_data[-1].int()
                sub_id_all.append(sub_ids)

        ssl_general_model_path = os.path.join(self.config.result_dir, 'embedding_similarity_between_segments')
        os.makedirs(ssl_general_model_path, exist_ok=True)
        # test_name = 'linear_protocol_' + str(linear_protocol) + ', use_ssl_' + str(use_ssl) + \
        #             ', ratio_' + str(self.config.da_use_ratio) + ', i_optimize_' + str(i_optimize)
        mod_acc, mod_gyr = [], []
        for mod_outputs in mod_output_all:
            mod_acc.append(mod_outputs[0])
            mod_gyr.append(mod_outputs[1])
        mod_acc, mod_gyr = torch.concat(mod_acc, dim=0), torch.concat(mod_gyr, dim=0)
        with h5py.File(os.path.join(ssl_general_model_path, self.da_task['dataset'] + '.h5'), 'a') as hf:
            grp = hf.require_group(embedding_name)
            grp_acc, grp_gyr = grp.require_group('mod_acc'), grp.require_group('mod_gyr')
            mod_acc, mod_gyr = mod_acc.detach().cpu().numpy(), mod_gyr.detach().cpu().numpy()
            sub_id_all = torch.concat(sub_id_all, dim=0).cpu().numpy()
            subject_id_set = list(set(sub_id_all))
            for i_sub in subject_id_set:
                sub_data_loc = np.where(sub_id_all == i_sub)[0]
                mod_acc_sub, mod_gyr_sub = mod_acc[sub_data_loc], mod_gyr[sub_data_loc]
                grp_acc.require_dataset('sub_'+str(i_sub), shape=mod_acc_sub.shape, data=mod_acc_sub, dtype='float32')
                grp_gyr.require_dataset('sub_'+str(i_sub), shape=mod_gyr_sub.shape, data=mod_gyr_sub, dtype='float32')

    def save_model_and_results(self, test_name, y_true, y_pred, sub_ids, model):
        os.makedirs(os.path.join(self.config.result_dir, 'test_models', self.da_task['dataset']), exist_ok=True)
        copied_model = copy.deepcopy(model)
        torch.save(copied_model.state_dict(), os.path.join(self.config.result_dir, 'test_models', self.da_task['dataset'], test_name + '.pth'))
        self.save_results(test_name, y_true, y_pred, sub_ids)

    def save_results(self, test_name, y_true, y_pred, sub_ids):
        sub_id_set = list(set(sub_ids))
        results = np.concatenate([y_true, y_pred], axis=1)
        columns = ['y_true', 'y_pred']
        with h5py.File(os.path.join(self.config.result_dir, self.da_task['dataset'] + '_' + self.da_task['output'] + '.h5'), 'a') as hf:
            grp = hf.require_group(test_name)
            for i_sub in sub_id_set:
                results_sub = results[sub_ids == i_sub]
                grp.require_dataset('sub_' + str(int(i_sub)), shape=results_sub.shape, data=results_sub, dtype='float32')
                grp.attrs['columns'] = json.dumps(columns)

    def set_regress_net_to_post_linear_init_head_first(self, use_ssl):
        model_name = 'linear_protocol_True, use_ssl_' + str(use_ssl) + ', ratio_' + str(self.config.da_use_ratio)
        regress_net_post_linear_head_init = torch.load(os.path.join(self.config.result_dir, 'test_models', self.da_task['dataset'], model_name + '.pth'))
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
                    if i_batch > use_batch_num:
                        continue
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
                    ', ratio_' + str(self.config.da_use_ratio)
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
        train_output_data = train_data[self.da_task['output']]
        train_step_lens = get_step_len(train_input_data[0])
        train_dl = prepare_dl([*train_input_data, train_output_data, train_step_lens], int(self.config.batch_size_linear), shuffle=True)
        test_input_data = [test_data[mod] for mod in self.da_task['_mods']]
        test_output_data = test_data[self.da_task['output']]
        test_step_lens = get_step_len(test_input_data[0])
        test_dl = prepare_dl([*test_input_data, test_output_data, test_step_lens], int(self.config.batch_size_linear), shuffle=False)

        model = self.regress_net
        dtype, model = set_dtype_and_model(self.config.device, model)
        if linear_protocol:
            param_to_train = model.linear.parameters()
        else:
            param_to_train = model.parameters()

        optimizer = torch.optim.AdamW(param_to_train, self.config.lr_regress, weight_decay=1e-5)
        epoch_end_time = time.time()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_gradient_de_ssl)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=int(self.config.num_gradient_de_da/5))

        epoch = int(np.ceil(self.config.num_gradient_de_da / len(train_dl)))
        i_optimize = 0
        for i_epoch in range(epoch):
            i_optimize = train_batch(model, train_dl, optimizer, torch.nn.MSELoss(), i_optimize)
            if verbose:
                train_loss = eval_during_training(model, train_dl, torch.nn.MSELoss())
                test_loss = eval_during_training(model, test_dl, torch.nn.MSELoss())
                if self.config.log_with_wandb:
                    wandb.log({'linear train loss': train_loss, 'linear test loss': test_loss})
                logging.info(f'| Regressibility | epoch{i_epoch:3d}/{epoch:3d} | time: {time.time() - epoch_end_time:5.2f}s |'
                             f' train loss {train_loss:5.3f} | test loss {test_loss:5.3f}')
                epoch_end_time = time.time()

        y_true, y_pred = evaluate_after_training(test_dl)
        y_true, y_pred = self.inverse_normalize_output(y_true, y_pred)
        self.save_model_and_results(test_name, y_true, y_pred, test_data['sub_id'], self.regress_net)
        if show_fig:
            plt.figure()
            plt.title('Test')
            plt.plot(y_true.ravel(), y_pred.ravel(), '.')
            plt.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], color='black')
            plt.xlabel('True')
            plt.ylabel('Predicted')
        if verbose:
            all_scores = get_scores(y_true, y_pred, [self.da_task['output']], test_step_lens)
            all_scores = [{'subject': 'all', **scores} for scores in all_scores]
            print_table(all_scores)

        return y_pred, model

    def load_data_and_save_ssl_embedding(self, da_use_ratios):
        config['da_use_ratio'] = da_use_ratios[0]
        self.config = SimpleNamespace(**config)
        self.sample_and_normalize_data()

        test_data = self.da_task['test']
        test_input_data = [test_data[mod] for mod in self.da_task['_mods']]
        test_output_data = test_data[self.da_task['output']]
        test_step_lens = get_step_len(test_input_data[0])
        test_data_sub_id = test_data['sub_id']
        test_dl = prepare_dl([*test_input_data, test_output_data, test_step_lens, test_data_sub_id],
                             1024, shuffle=False)

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

        self.emb_nets = {mod: self.config.emb_net(3, self.config.emb_output_dim, mod + ' embedding') for mod in _mods}
        self._data_scalar = {'base_scalar': StandardScaler()}
        fix_seed()

    def preprocess(self, train_sub_ids: List[str], validate_sub_ids: List[str], test_sub_ids: List[str]):
        """
        train_sub_ids: a list of subject id for model training
        validate_sub_ids: a list of subject id for model validation
        test_sub_ids: a list of subject id for model testing
        """
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
                    if i_batch > use_batch_num:
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
        model = SslGeneralNet(torch.nn.ModuleList([self.emb_nets[mod] for mod in _mods]),
                              self.config.common_space_dim, config['ssl_loss_fn'])
        # self.save_ssl_general_net(model, 'pre_ssl')
        if self.config.log_with_wandb:
            wandb.watch(model, config['ssl_loss_fn'], log='all', log_freq=20)

        train_dl = prepare_dl([train_data[mod] for mod in _mods] + [train_step_lens], int(self.config.batch_size_ssl), shuffle=True)
        vali_dl = prepare_dl([vali_data[mod] for mod in _mods] + [vali_step_lens], int(self.config.batch_size_ssl), shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr_ssl, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_gradient_de_ssl)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=int(self.config.num_gradient_de_ssl/5))
        dtype, model = set_dtype_and_model(self.config.device, model)
        epoch_end_time = time.time()
        epoch = int(np.ceil(self.config.num_gradient_de_ssl / len(train_dl)))
        _, mod_output_all = eval_during_training(model, vali_dl, np.nan)
        self.save_embeddings(mod_output_all, 'pre_ssl')
        for i_epoch in range(epoch):
            train_loss, _ = eval_during_training(model, train_dl)
            test_loss, _ = eval_during_training(model, vali_dl)

            logging.info(f'| SSL | epoch{i_epoch:3d}/{epoch:3d} | time: {time.time() - epoch_end_time:5.2f}s | '
                         f'train loss {train_loss:5.4f} | test loss {test_loss:5.4f}')
            epoch_end_time = time.time()
            train_batch(model, train_dl, optimizer)
            if self.config.log_with_wandb:
                wandb.log({'ssl train loss': train_loss, 'ssl test loss': test_loss})
        self.save_emb_net_post_ssl(model)
        _, mod_output_all = eval_during_training(model, vali_dl, np.nan)
        self.save_embeddings(mod_output_all, 'post_ssl')
        return {'model': model}

    def save_emb_net_post_ssl(self, model):
        emb_nets_post_ssl_state = copy.deepcopy({mod: model.emb_nets[i_mod].state_dict() for i_mod, mod in enumerate(_mods)})
        save_path = os.path.join(RESULTS_PATH, self.ssl_task['ssl_file_name'] + '.pth')
        torch.save(emb_nets_post_ssl_state, save_path)

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
            grp.require_dataset('mod_acc', shape=mod_acc.shape, data=mod_acc, dtype='float32')
            grp.require_dataset('mod_gyr', shape=mod_gyr.shape, data=mod_gyr, dtype='float32')

    def save_ssl_general_net(self, model, model_name):
        ssl_general_model_path = os.path.join(RESULTS_PATH, 'embedding_similarity_between_segments', self.ssl_task['ssl_file_name'])
        os.makedirs(ssl_general_model_path, exist_ok=True)
        save_path = os.path.join(ssl_general_model_path, model_name + '.pth')
        torch.save(model, save_path)

    def hyperparam_tuning(self, model, x_test):
        raise RuntimeError('Method not implemented')


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
        da_framework.load_data_and_save_ssl_embedding(da_use_ratios)     # !!!
        run_da(da_framework, da_use_ratios)


PARAMS_TRIED = ['ramp', 'treadmill_speed', 'peak_knee_extension_angle']
_mods = ['acc', 'gyr']
DOWNSTREAM_TASK_0 = {'_mods': ['acc', 'gyr'], 'remove_trial_type': ['Treadmill', 'Stair', 'Ramp'], 'dataset': 'Carmargo',
                     'output': 'peak_fy', 'imu_segments': ['trunk', 'shank'], 'ssl_model': 'MoVi_'}
DOWNSTREAM_TASK_1 = {'_mods': ['acc', 'gyr'], 'remove_trial_type': ['LevelGround', 'Stair', 'Ramp'], 'dataset': 'Carmargo',
                     'output': 'peak_fy', 'imu_segments': ['trunk', 'shank'], 'ssl_model': 'MoVi_'}
DOWNSTREAM_TASK_2 = {'_mods': ['acc', 'gyr'], 'remove_trial_type': ['Treadmill'], 'dataset': 'Carmargo',
                     'output': 'peak_fy', 'imu_segments': ['trunk', 'shank'], 'ssl_model': 'MoVi_'}
DOWNSTREAM_TASK_3 = {'_mods': ['acc', 'gyr'], 'remove_trial_type': [], 'dataset': 'walking_knee_moment',
                     'output': 'KFM', 'imu_segments': ['R_FOOT', 'R_SHANK', 'R_THIGH', 'L_SHANK', 'L_THIGH', 'L_FOOT', 'WAIST', 'CHEST'], 'ssl_model': 'MoVi_'}
DOWNSTREAM_TASK_4 = {'_mods': ['acc', 'gyr'], 'remove_trial_type': [], 'dataset': 'sun_drop_jump', 'output': 'R_KNEE_MOMENT_X',
                     'imu_segments': ['R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_FOOT', 'L_SHANK', 'L_THIGH'], 'ssl_model': 'MoVi_'}
DOWNSTREAM_TASK_5 = {'_mods': ['acc', 'gyr'], 'remove_trial_type': [], 'dataset': 'sun_drop_jump', 'output': 'R_GRF_Z',
                     'imu_segments': ['R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_FOOT', 'L_SHANK', 'L_THIGH'], 'ssl_model': 'MoVi_'}
DOWNSTREAM_TASK_6 = {'_mods': ['acc', 'gyr'], 'remove_trial_type': [], 'dataset': 'hw_running',     # batch size = 64, lr = 1e-3
                     'output': 'VALR', 'imu_segments': ['l_shank'], 'max_pooling_to_downsample': True, 'ssl_model': 'MoVi_'}

# run_ssl(ssl_task_Carmargo)
ssl_task_Carmargo = {'ssl_file_name': 'MoVi_Carmargo', 'imu_segments': [
    # 'Hip', 'Spine1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot']}
    'Hip', 'Spine1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']}
ssl_task_sun_drop_jump = copy.deepcopy(ssl_task_Carmargo)
ssl_task_sun_drop_jump.update({'ssl_file_name': 'MoVi_sun_drop_jump'})
ssl_task_kam = copy.deepcopy(ssl_task_Carmargo)
ssl_task_kam.update({'ssl_file_name': 'MoVi_walking_knee_moment'})
ssl_task_hw_running = copy.deepcopy(ssl_task_Carmargo)
ssl_task_hw_running.update({'ssl_file_name': 'MoVi_hw_running'})

# # !!! fast version
config = {'num_gradient_de_ssl': 1e4, 'num_gradient_de_da': 1e3, 'batch_size_ssl': 256, 'batch_size_linear': 256,
          'lr_ssl': 1e-4, 'lr_regress': 1e-3, 'emb_net': CnnEmbedding, 'emb_output_dim': 128, 'common_space_dim': 128,
# config = {'num_gradient_de_ssl': 1e4, 'num_gradient_de_da': 1000, 'batch_size_ssl': 256, 'batch_size_linear': 256,
#           'lr_ssl': 1e-4, 'lr_regress': 1e-3, 'emb_net': CnnEmbedding, 'emb_output_dim': 128, 'common_space_dim': 128,
          'device': 'cuda', 'result_dir': os.path.join(RESULTS_PATH, result_folder()),
          'log_with_wandb': False, 'ssl_loss_fn': nce_loss_groups_of_positive, 'ssl_use_ratio': 1}
# clip_loss   nce_loss   nce_loss_hard_negative
if config['log_with_wandb']:
    wandb.init(project="IMU_EMG_SSL", config=config, name='')
train_sub_movi = ['sub_' + str(i+1) for i in range(0, 80)]
test_sub_movi = ['sub_' + str(i+1) for i in range(80, 88)]

if __name__ == '__main__':
    os.makedirs(os.path.join(config['result_dir']), exist_ok=True)
    add_file_handler(logging, os.path.join(config['result_dir'], 'training_log.txt'))
    logging.info('Test no overlapping')

    run_ssl(ssl_task_hw_running)
    run_ssl(ssl_task_Carmargo)
    run_ssl(ssl_task_kam)

    da_framework = FrameworkDownstream(config, DOWNSTREAM_TASK_6)
    run_cross_vali(da_framework, da_use_ratios=[1])

    da_framework = FrameworkDownstream(config, DOWNSTREAM_TASK_2)
    run_cross_vali(da_framework, da_use_ratios=[1.])
    #
    # log_array = [round(10**x, 3) for x in np.linspace(-2, 0, 11)]
    da_framework = FrameworkDownstream(config, DOWNSTREAM_TASK_3)
    run_cross_vali(da_framework, da_use_ratios=[1])

    plt.show()

    # run_ssl(ssl_task_sun_drop_jump)

    # da_framework = FrameworkDownstream(config, DOWNSTREAM_TASK_5)
    # da_framework.load_and_process_sun(train_sub_sun, test_sub_sun, test_sub_sun)
    # run_da(da_framework, da_use_ratios=[1.])

    # TEST SSL subject number
    # for ssl_sub_num in range(60, 81, 10):
    #     train_sub_movi = ['sub_' + str(i+1) for i in random.sample(range(0, 80), ssl_sub_num)]
    #     run_ssl(ssl_task_Carmargo)
    #     da_framework = FrameworkDownstream(config, DOWNSTREAM_TASK_2)
    #     run_cross_vali(da_framework, da_use_ratios=[1])

