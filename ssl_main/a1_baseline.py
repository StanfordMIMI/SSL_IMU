import argparse
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from customized_logger import logger as logging, add_file_handler
from sklearn.preprocessing import StandardScaler
import torch
import wandb
from model import RegressNet, transformer, mse_loss_masked, show_reconstructed_signal
import time
from types import SimpleNamespace
from utils import prepare_dl, set_dtype_and_model, fix_seed, normalize_data, result_folder, define_channel_names, \
    preprocess_modality
from const import RESULTS_PATH, DSET_SUBS_FOR_SSL_TEST, DSET_SUBS_FOR_SSL_TRAINING, \
    SUB_ID_ALL_DATASETS, STANDARD_IMU_SEQUENCE
from config import DATA_PATH
import json
import pytorch_warmup as warmup
from a1_ssl import FrameworkDownstream, DOWNSTREAM_0, DOWNSTREAM_1, DOWNSTREAM_2
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class FrameworkBaselinePretraining:
    def __init__(self, config, ssl_task):
        self.config = SimpleNamespace(**config)
        self.ssl_task = ssl_task
        self.output_columns = ssl_task['output_columns']

        self.ssl_data_dict, self.ssl_columns_dict = {}, {}
        with h5py.File(DATA_PATH + ssl_task['ssl_file_name'] + '.h5', 'r') as hf:
            logging.info('{} size, {}'.format(ssl_task['ssl_file_name'], sum([hf[sub_].shape[0] for sub_, sub_data in hf.items()])))
            data_dict = {sub_: sub_data[:int(self.config.ssl_use_ratio * hf[sub_].shape[0]), :, :] for sub_, sub_data in hf.items()}
            if ssl_task['ssl_file_name'] in ['walking_knee_moment', 'filtered_walking_knee_moment']:
                data_dict = {sub_: data_.transpose([0, 2, 1]) for sub_, data_ in data_dict.items()}     # [step, feature, time]
            self.ssl_data_dict[ssl_task['ssl_file_name']] = data_dict
            self.ssl_columns_dict[ssl_task['ssl_file_name']] = json.loads(hf.attrs['columns'])

        # if ssl_task['ssl_file_name'] == 'amass':
        #     pos_acc = 0
        #     self.ssl_columns_dict[ssl_task['ssl_file_name']] += ['pos_acc_x', 'pos_acc_y', 'pos_acc_z']

        os.makedirs(os.path.join(self.config.result_dir), exist_ok=True)

        self.emb_net = self.config.emb_net(len(ssl_task['imu_segments'])*6, self.config.nlayers, self.config.nhead,
                                           self.config.FeedForwardDim, [False for _ in range(8)], self.config.MaskPatchNum, self.config.PatchLen)
        logging.info('# of trainable parameters: {}'.format(sum(p.numel() for p in self.emb_net.transformer.parameters() if p.requires_grad)))
        self._data_scalar = {'base_scalar': StandardScaler}
        fix_seed()

    def sample_and_normalize_data(self):
        for data_name, norm_method in zip(['train', 'vali', 'test'], ['fit_transform', 'transform', 'transform']):
            current_set_data = self.set_data[data_name]
            logging.info('Downstream {}. Number of steps: {}'.format(data_name, current_set_data.shape[0]))
            data_ds = preprocess_modality(self.data_columns, self._data_scalar, current_set_data, define_channel_names(self.da_task), norm_method)
            output_data = current_set_data[:, [self.data_columns.index(x) for x in self.output_columns]]
            data_ds['output'] = normalize_data(self._data_scalar, output_data, 'output', norm_method, 'by_each_column').astype(np.float32)
            data_ds['sub_id'] = current_set_data[:, self.data_columns.index('sub_id'), 0]
            if 'trial_type_id' in self.data_columns:
                data_ds['trial_type_id'] = current_set_data[:, self.data_columns.index('trial_type_id'), 0]
            else:
                data_ds['trial_type_id'] = np.zeros([current_set_data.shape[0]])
            self.da_task[data_name] = data_ds

    def preprocess(self, ssl_file_name):
        train_sub_ids = DSET_SUBS_FOR_SSL_TRAINING[ssl_file_name]
        validate_sub_ids = test_sub_ids = DSET_SUBS_FOR_SSL_TEST[ssl_file_name]
        data_, columns_ = self.ssl_data_dict[ssl_file_name], self.ssl_columns_dict[ssl_file_name]
        train_sub_ids_print = []
        if train_sub_ids == ['except test']:
            train_data = np.concatenate([data_[sub] for sub in list(data_.keys()) if sub not in test_sub_ids], axis=0)
            train_sub_ids_print = [sub for sub in list(data_.keys()) if sub not in test_sub_ids]
        elif train_sub_ids == ['all']:
            train_data = np.concatenate([data_[sub] for sub in list(data_.keys())], axis=0)
            train_sub_ids_print = [sub for sub in list(data_.keys())]
        else:
            train_data = np.concatenate([data_[sub] for sub in train_sub_ids], axis=0)
        vali_data = np.concatenate([data_[sub] for sub in validate_sub_ids], axis=0)
        test_data = np.concatenate([data_[sub] for sub in test_sub_ids], axis=0)

        channel_names_dict = define_channel_names(self.ssl_task)
        channel_names_dict['output'] = self.output_columns

        # SSL preprocess
        train_data_ssl = preprocess_modality(columns_, self._data_scalar, train_data, channel_names_dict, 'fit_transform')
        logging.info('Baseline Pre-training with dataset {} subject ids: {}. Number of steps: {}'.format(ssl_file_name, train_sub_ids_print, list(train_data_ssl.values())[0].shape[0]))
        vali_data_ssl = preprocess_modality(columns_, self._data_scalar, vali_data, channel_names_dict, 'transform')
        test_data_ssl = preprocess_modality(columns_, self._data_scalar, test_data, channel_names_dict, 'transform')
        logging.info('Baseline test with dataset {} subject ids: {}. Number of steps: {}'.format(ssl_file_name, test_sub_ids, list(test_data_ssl.values())[0].shape[0]))
        return train_data_ssl, vali_data_ssl, test_data_ssl

    def ssl_training(self, config):
        def train_batch(model, train_dl, optimizer, loss_fn):
            model.train()
            for i_batch, x in enumerate(train_dl):
                optimizer.zero_grad()

                xb_mods = [xb_mod.float().type(dtype) for xb_mod in x[:-1]]
                output, _ = model(xb_mods, True)
                loss = loss_fn(x[-1].float().type(dtype), output)

                if self.config.log_with_wandb:
                    wandb.log({'ssl batch loss': loss.item(), 'lr ssl': optimizer.param_groups[0]['lr']})
                loss.backward()
                optimizer.step()
                with warmup_scheduler.dampening():
                    scheduler.step()

        def eval_during_training(model, dl, loss_fn, use_batch_num=5):
            model.eval()
            with torch.no_grad():
                validation_loss = []
                for i_batch, x in enumerate(dl):
                    if i_batch >= use_batch_num:
                        continue
                    xb_mods = [xb_mod.float().type(dtype) for xb_mod in x[:-1]]
                    output, _ = model(xb_mods, True)
                    loss = loss_fn(x[-1].float().type(dtype), output)

                    validation_loss.append(loss.item())
            return np.mean(validation_loss)

        train_data, vali_data = self.train_data_ssl, self.vali_data_ssl
        model = RegressNet(self.emb_net, len(STANDARD_IMU_SEQUENCE) * 2, len(self.output_columns))
        if self.config.log_with_wandb:
            wandb.watch(model, config['ssl_loss_fn'], log='all', log_freq=20)

        train_dl = prepare_dl([np.concatenate([train_data['acc'], train_data['gyr']], axis=1), train_data['output']],
                              int(self.config.BatchSizeSsl), shuffle=True, drop_last=True)
        vali_dl = prepare_dl([np.concatenate([vali_data['acc'], vali_data['gyr']], axis=1), vali_data['output']],
                              int(self.config.BatchSizeSsl), shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.LrSsl)

        # self.loss_fun = torch.nn.MSELoss()
        dtype, model = set_dtype_and_model(self.config.device, model)
        epoch_end_time = time.time()
        epoch = int(np.ceil(self.config.NumGradDeSsl / len(train_dl)))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch*len(train_dl))
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=int(epoch*len(train_dl)/5))

        loss_fn = torch.nn.MSELoss()
        for i_epoch in range(epoch):

            if epoch < 5 or i_epoch % int(epoch / 5) == 0 or i_epoch == epoch - 1:
                train_loss = eval_during_training(model, train_dl, loss_fn)
                test_loss = eval_during_training(model, vali_dl, loss_fn)
                logging.info(f'| SSL | epoch{i_epoch:3d}/{epoch:3d} | time: {time.time() - epoch_end_time:5.2f}s | '
                             f'train loss {train_loss:5.4f} | test loss {test_loss:5.4f}')
                if self.config.log_with_wandb:
                    wandb.log({'ssl train loss': train_loss, 'ssl test loss': test_loss})
                    model.eval()
                    with torch.no_grad():
                        x = list(enumerate(vali_dl))[0]
                        xb_mods = [xb_mod.float().type(dtype) for xb_mod in x[1][:-1]]
                        loss, mod_outputs, mask_indices = model(xb_mods, None)
                        show_reconstructed_signal(torch.concat(xb_mods, dim=1), mod_outputs,
                                                  'epoch ' + str(i_epoch) + self.config.FileNameAppendix, mask_indices)
            epoch_end_time = time.time()
            train_batch(model, train_dl, optimizer, loss_fn)
        self.save_emb_net_post_ssl(model)
        return {'model': model}

    def save_emb_net_post_ssl(self, model):
        emb_net_post_ssl_state = model.emb_net.state_dict()
        save_path = os.path.join(self.config.result_dir, 'emb_' + self.config.FileNameAppendix[2:] + '.pth')
        torch.save(emb_net_post_ssl_state, save_path)

    def run_pretraining(self):
        sets_of_data_all_dsets = []
        logging.info('Start SSL data preprocessing')
        sets_of_data_all_dsets.append(self.preprocess(self.ssl_task['ssl_file_name']))
        self.train_data_ssl = {mod: np.concatenate([sets_of_data[0][mod] for sets_of_data in sets_of_data_all_dsets], axis=0).astype(np.float32) for mod in ['acc', 'gyr', 'output']}
        self.vali_data_ssl = {mod: np.concatenate([sets_of_data[1][mod] for sets_of_data in sets_of_data_all_dsets], axis=0).astype(np.float32) for mod in ['acc', 'gyr', 'output']}
        self.test_data_ssl = {mod: np.concatenate([sets_of_data[2][mod] for sets_of_data in sets_of_data_all_dsets], axis=0).astype(np.float32) for mod in ['acc', 'gyr', 'output']}
        self.ssl_training(config)


def parse_config(config):
    parser = argparse.ArgumentParser(description='TODO', argument_default=argparse.SUPPRESS)
    parser.add_argument('--ssl_dset', type=str, default='motion_transfer')
    config.update(vars(parser.parse_args()))
    return config


def run_da(da_frameworks, fold_num=5, run_baseline=True):
    for da_framework in da_frameworks:
        dataset_name = da_framework.da_task['dataset']
        sub_ids = SUB_ID_ALL_DATASETS[dataset_name]
        np.random.shuffle(sub_ids)
        test_sets = np.array_split(sub_ids, fold_num)
        for i_fold, test_set in enumerate(test_sets):
            test_set = list(test_set)
            train_set = [id for id in sub_ids if id not in test_set]
            logging.info(dataset_name + ', cross validation fold {}'.format(i_fold))
            da_framework.load_and_process(dataset_name, train_set, test_set, test_set)
            for ratio in da_framework.da_use_ratios:
                config['da_use_ratio'] = ratio
                da_framework.config = SimpleNamespace(**config)
                da_framework.sample_and_normalize_data()
                da_framework.regressibility(linear_protocol=True, use_ssl=True)
                if run_baseline:
                    da_framework.regressibility(linear_protocol=True, use_ssl=False)
                da_framework.regressibility(linear_protocol=False, use_ssl=True)
                if run_baseline:
                    da_framework.regressibility(linear_protocol=False, use_ssl=False)
                plt.close("all")


MOTION_TRANSFER = {'ssl_file_name': 'hw_running', 'output_columns': ['fx', 'fy', 'fz'],
                   'imu_segments': ['TRUNK', 'PELVIS', 'L_THIGH', 'L_THIGH', 'L_SHANK', 'L_SHANK', 'L_FOOT', 'L_FOOT']}

TASK_TRANSFER = {'ssl_file_name': 'amass', 'output_columns': ['Pelvis_global_acc_X', 'Pelvis_global_acc_Y', 'Pelvis_global_acc_Z'],
                 'imu_segments': STANDARD_IMU_SEQUENCE}

PRETRAIN_CONFIGS = {'motion_transfer': MOTION_TRANSFER, 'task_transfer': TASK_TRANSFER}

config = {'NumGradDeSsl': 5e4, 'NumGradDeDa': 3e2, 'ssl_use_ratio': 1, 'log_with_wandb': False,
# config = {'NumGradDeSsl': 1e1, 'NumGradDeDa': 3e2, 'ssl_use_ratio': 0.1, 'log_with_wandb': False,
          'BatchSizeSsl': 64, 'BatchSizeLinear': 64, 'LrSsl': 1e-4, 'LrDa': 1e-4, 'FeedForwardDim': 512,
          'nlayers': 6, 'nhead': 8, 'device': 'cuda', 'ssl_loss_fn': mse_loss_masked, 'emb_net': transformer}

test_name = 'PretrainingBaseline'
test_info = ''

config = parse_config(config)
config['result_dir'] = os.path.join(RESULTS_PATH, result_folder() + config['ssl_dset'] + "_" + test_name)

if config['log_with_wandb']:
    wandb.init(project="IMU_SSL", config=config, name=test_name)

if __name__ == '__main__':
    os.makedirs(os.path.join(config['result_dir']), exist_ok=True)
    add_file_handler(logging, os.path.join(config['result_dir'], 'training_log.txt'))
    logging.info(test_name + '\t' + test_info)

    coupled_hypers = (['PatchLen', 'MaskPatchNum'],  {1: [16]})
    independent_hyper = ('NumGradDeSsl', [config['NumGradDeSsl']])

    for indep_hyper_val in independent_hyper[1]:
        for coupled_hyper_val_1, coupled_hyper_val_list_2 in coupled_hypers[1].items():
            for coupled_hyper_val_2 in coupled_hyper_val_list_2:
                config[independent_hyper[0]] = indep_hyper_val
                config[coupled_hypers[0][0]] = coupled_hyper_val_1
                config[coupled_hypers[0][1]] = coupled_hyper_val_2
                config['FileNameAppendix'] = f', {independent_hyper[0]}_{indep_hyper_val},' \
                                             f' {coupled_hypers[0][0]}_{coupled_hyper_val_1},' \
                                             f' {coupled_hypers[0][1]}_{coupled_hyper_val_2}'
                logging.info(config['FileNameAppendix'])
                logging.info(config)

                FrameworkBaselinePretraining(config, PRETRAIN_CONFIGS[config['ssl_dset']]).run_pretraining()
                da_frameworks = [FrameworkDownstream(config, da_task) for da_task in
                                 [DOWNSTREAM_0, DOWNSTREAM_1, DOWNSTREAM_2]]
                run_da(da_frameworks, fold_num=5)

                plt.show()












