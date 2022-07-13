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
from torch.utils.data import DataLoader, TensorDataset
from model import nce_loss, ImuTestTransformerEmbedding, \
    LinearTestNet, ImuTestRnnEmbedding, EmgTestRnnEmbedding, SslTestNet, EmgTestTransformerEmbedding
import time
from types import SimpleNamespace
import prettytable as pt
from utils import get_data_by_merging_data_struct, fix_seed, DataStruct, data_filter
from const import IMU_SEGMENT_LIST, DATA_PATH, AMBULATIONS, GRAVITY
from scipy.signal import find_peaks


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
        self.linear_regress_net = LinearTestNet(self.emb_net_imu, self.emb_net_emg, len(OUTPUT_LIST))
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
        logging.info('Train the model with subject ids: {}'.format(train_sub_ids))
        train_data = get_data_by_merging_data_struct([self.data[id] for id in train_sub_ids])
        train_data = self.preprocess_data(train_data, 'fit_transform')
        train_data['IMU'], train_data['EMG'] = self.unison_shuffled_copies(train_data['IMU'], train_data['EMG'])

        logging.info('Validate the model with subject ids: {}'.format(validate_sub_ids))
        vali_data = get_data_by_merging_data_struct([self.data[id] for id in validate_sub_ids])
        vali_data = self.preprocess_data(vali_data, 'transform')

        logging.info('Test the model with subject ids: {}'.format(test_sub_ids))
        test_data = get_data_by_merging_data_struct([self.data[id] for id in test_sub_ids])
        test_data = self.preprocess_data(test_data, 'transform')

        y_pred, model = self.linear_regressibility(train_data, test_data)
        # self.save_model_and_results(test_data['y'], y_pred, OUTPUT_LIST, model, test_sub_ids)

        self.reset_emb_net_to_init_state()
        model = self.ssl_training(train_data, vali_data)
        y_pred, model = self.linear_regressibility(train_data, test_data)

        plt.show()
        plt.close('all')

    @staticmethod
    def print_table(results):
        tb = pt.PrettyTable()
        for test_result in results:
            tb.field_names = test_result.keys()
            tb.add_row([np.round(np.mean(value), 3) if isinstance(value, (np.ndarray, float)) else value
                        for value in test_result.values()])
        logging.info(tb)

    def linear_regressibility(self, train_data, test_data):
        def prepare_data(data, step_lens, batch_size):
            x_imu = torch.from_numpy(data['IMU']).float()
            x_emg = torch.from_numpy(data['EMG']).float()
            y_true = torch.from_numpy(data['y']).float()
            step_lens = torch.from_numpy(step_lens)
            ds = TensorDataset(x_imu, x_emg, step_lens, y_true)
            dl = DataLoader(ds, batch_size=batch_size)
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
                loss_fn(y_pred, y_true).backward()
                optimizer.step()

        def eval_during_training(model, dl, loss_fn, ratio_of_data_from_dl=100):
            model.eval()
            with torch.no_grad():
                loss, y_pred_list = [], []
                for batch_data in dl:
                    n = random.randint(1, 100)
                    if n > ratio_of_data_from_dl:
                        continue  # increase the speed of epoch
                    xb_imu, xb_emg, lens, y_true = convert_batch_data(batch_data)
                    y_pred = model(xb_imu, xb_emg, lens)
                    loss.append(loss_fn(y_true, y_pred).item())
                    y_pred_list.append(y_pred.detach().cpu())
            return np.mean(loss), y_pred_list

        def evaluate_after_training(test_dl):
            model.eval()
            with torch.no_grad():
                y_pred_list = []
                for i_batch, batch_data in enumerate(test_dl):
                    xb_imu, xb_emg, lens, y_true = convert_batch_data(batch_data)
                    y_pred_list.append(model(xb_imu, xb_emg, lens).detach().cpu())
                y_pred = torch.cat(y_pred_list)
            return y_pred.numpy()

        logging.info('Linear regressibility test')
        train_step_lens = self._get_step_len(train_data['IMU'])
        train_dl = prepare_data(train_data, train_step_lens, int(self.config.batch_size))
        test_step_lens = self._get_step_len(test_data['IMU'])
        test_dl = prepare_data(test_data, test_step_lens, int(self.config.batch_size))

        model = self.linear_regress_net

        if self.config.device is 'cuda':
            model.cuda()
        optimizer = torch.optim.Adam(model.linear.parameters(), lr=self.config.lr_linear)      # !!! model.linear.parameters

        logging.info('\tEpoch | Train_set_Loss | Test_set_Loss | Duration\t\t')
        epoch_end_time = time.time()
        dtype = self.config.dtype
        for i_epoch in range(self.config.epoch):
            test_loss, y_pred_test = eval_during_training(model, test_dl, torch.nn.MSELoss(), 100)
            train_loss, y_pred_train = eval_during_training(model, train_dl, torch.nn.MSELoss(), 100)

            if i_epoch in [self.config.epoch-1]:
                plt.figure()
                plt.title('Train')
                plt.plot(train_data['y'].ravel())
                plt.plot(torch.cat(y_pred_train).numpy().ravel())

                plt.figure()
                plt.title('Test')
                plt.plot(test_data['y'].ravel())
                plt.plot(torch.cat(y_pred_test).numpy().ravel())

            # plt.figure()
            # plt.plot(torch.cat(y_pred_test)[:, 0], test_data['y'][:, 0], '.', color='b')
            # print(model.embnet_imu.conv_1.weight[0])

            logging.info("\t{:3}\t{:12.3f}\t{:12.3f}\t{:13.2f}s\t\t".format(i_epoch, train_loss, test_loss, time.time() - epoch_end_time))
            epoch_end_time = time.time()
            train(model, train_dl, optimizer, torch.nn.MSELoss(), self.config.use_ratio)
        # plt.show()

        y_pred = evaluate_after_training(test_dl)
        all_scores = FrameworkSSL.get_peak_scores(test_data['y'], y_pred, OUTPUT_LIST, test_step_lens)
        all_scores = [{'subject': 'all', **scores} for scores in all_scores]
        self.print_table(all_scores)

        # print(np.concatenate([test_data['y'], y_pred], axis=1)[:20])

        return y_pred, model

    def ssl_training(self, train_data, vali_data):
        def prepare_data(train_step_lens, vali_step_lens, batch_size):
            x_train_imu = torch.from_numpy(train_data['IMU']).float()
            x_train_emg = torch.from_numpy(train_data['EMG']).float()
            train_step_lens = torch.from_numpy(train_step_lens)
            train_ds = TensorDataset(x_train_imu, x_train_emg, train_step_lens)
            train_dl = DataLoader(train_ds, batch_size=batch_size)

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
                emb_imu, emb_emg, temp = model(xb_imu, xb_emg, lens)
                loss_fn(emb_imu, emb_emg).backward()
                optimizer.step()

                # # FOR DEBUG
                # if i_batch == 0:
                #     plt.figure()
                #     plt.plot(temp[0, :, :3].cpu().detach().numpy())

        def eval_during_training(model, dl, loss_fn):
            model.eval()
            with torch.no_grad():
                validation_loss = []
                for x in dl:
                    xb_imu = x[0].float().type(dtype)
                    xb_emg = x[1].float().type(dtype)
                    lens = x[2].float()
                    emb_imu, emb_emg, _ = model(xb_imu, xb_emg, lens)       # !!!
                    validation_loss.append(loss_fn(emb_imu, emb_emg).item())
            return np.mean(validation_loss)

        train_step_lens = self._get_step_len(train_data['IMU'])
        vali_step_lens = self._get_step_len(vali_data['IMU'])

        model = SslTestNet(self.emb_net_imu, self.emb_net_emg, self.config.common_space_dim)
        if self.config.device is 'cuda':
            model.cuda()

        train_dl, vali_dl = prepare_data(train_step_lens, vali_step_lens, int(self.config.batch_size))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr_ssl)
        logging.info('\tEpoch | Validation_set_Loss | Duration\t\t')
        epoch_end_time = time.time()
        dtype = self.config.dtype
        for i_epoch in range(self.config.epoch):
            vali_loss = eval_during_training(model, vali_dl, self.config.loss_fn)
            logging.info("\t{:3}\t{:15.3f}\t{:13.2f}s\t\t".format(i_epoch, vali_loss, time.time() - epoch_end_time))
            epoch_end_time = time.time()
            train(model, train_dl, optimizer, self.config.loss_fn, self.config.use_ratio)

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
        data['IMU'][:, :, self.data_reader.input_col_loc_acc] = self.normalize_data(
            data['IMU'][:, :, self.data_reader.input_col_loc_acc], 'acc', method, 'by_all_columns')
        data['IMU'][:, :, self.data_reader.input_col_loc_gyr] = self.normalize_data(
            data['IMU'][:, :, self.data_reader.input_col_loc_gyr], 'gyr', method, 'by_all_columns')
        data['EMG'] = self.normalize_data(data['EMG'], 'EMG', method, 'by_all_columns')

        # data['y'][:, :, 0] = -data['y'][:, :, 0]
        # data['y'][(data['y'] == 0.).all(axis=2), :] = np.nan
        # data['y'] = np.nanmax(data['y'], axis=1)
        # data['y'][np.isnan(data['y'])] = 0.
        # data['y'] = self.normalize_data(data['y'], 'y', method, 'by_each_column')
        data['y'] = self.normalize_data(data['y'], 'y', method, 'by_each_column')

        # For DEBUG
        # plt.figure()
        # [plt.plot(data['IMU'][i, :, 0]) for i in range(data['IMU'].shape[0])]
        # plt.figure()
        # [plt.plot(data['EMG'][i, :, 0]) for i in range(data['EMG'].shape[0])]
        # plt.show()
        return data

    @staticmethod
    def build_linear_classifier(x_train, y_train, x_validation=None, y_validation=None, validation_weight=None):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def predict(model, x_test):
        raise RuntimeError('Method not implemented')

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
    def get_peak_scores(y_true, y_pred, y_fields, lens):
        scores = []
        for col, field in enumerate(y_fields):
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
        self.sample_rate = 200
        self.step_len_max, self.step_len_min = int(1.5*self.sample_rate), int(0.3*self.sample_rate)
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
                 for trial, trial_data in hf.items() if len(data_trials.keys()) < config['max_trial_num']]
                # if config['trial_num'] is 'all':
                #     data_trials = {trial: [trial_data['data_200'][:], trial_data['data_1000'][:]]
                #                    for trial, trial_data in hf.items() if i_trial < config['trial_num']}
            self.data_contin[subject] = data_trials
        self.trials = list(data_trials.keys())

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
                trial_type = trial.split('_')[0]
                input_col_loc_imu = [self.columns[trial_type]['200'].index(x) for x in self.input_col_imu]
                input_col_loc_emg = [self.columns[trial_type]['1000'].index(x) for x in self.input_col_emg]
                output_col_loc = [self.columns[trial_type]['200'].index(x) for x in self.output_col]
                acc_loc = [self.columns[trial_type]['200'].index('foot_Accel_' + axis) for axis in ['X', 'Y', 'Z']]
                gyr_loc = [self.columns[trial_type]['200'].index('foot_Gyro_' + axis) for axis in ['X', 'Y', 'Z']]
                trial_data_200, trial_data_1000 = self.data_contin[subject][trial]
                acc_data, gyr_data = trial_data_200[:, acc_loc], trial_data_200[:, gyr_loc]
                strike_list, off_list = self.get_walking_strike_off(acc_data, gyr_data, 10)
                steps_200, steps_1000 = self.strike_off_to_step_and_remove_incorrect_step(
                    strike_list, off_list, self.step_len_max, self.step_len_min)
                for step_200, step_1000 in zip(steps_200, steps_1000):
                    data_x_200 = trial_data_200[step_200[0]:step_200[1], input_col_loc_imu]       # 遇到 out of xxx，写exception 跳过该步
                    data_x_1000 = trial_data_1000[step_1000[0]:step_1000[1], input_col_loc_emg]
                    data_y_200 = trial_data_200[step_200[0]:step_200[1], output_col_loc]
                    data_struct.add_new_step(data_x_200, data_x_1000, data_y_200)
            self.data[subject] = data_struct
        return self.data

    @staticmethod
    def strike_off_to_step_and_remove_incorrect_step(strike_list, off_list, step_len_max, step_len_min):
        steps_200 = []
        strike_np, off_np = np.array(strike_list), np.array(off_list)
        for i_strike in range(len(strike_np)):
            potential_offs = off_np[strike_np[i_strike] + step_len_min < off_np]
            potential_offs = potential_offs[potential_offs < strike_np[i_strike] + step_len_max - SAMPLES_BEFORE_STRIKE - SAMPLES_AFTER_OFF]
            if len(potential_offs) > 0:
                if i_strike != len(strike_np) - 1 and potential_offs[0] < strike_list[i_strike+1]:
                    steps_200.append([strike_np[i_strike] - SAMPLES_BEFORE_STRIKE, potential_offs[0] + SAMPLES_AFTER_OFF])
        steps_1000 = [[5 * step[0], 5 * step[1]] for step in steps_200]
        return steps_200, steps_1000

    @staticmethod
    def initalize_steps_and_stance_phase(data_df, strike_list, off_list, sample_after_thd=10):
        """The name "stance phase" is not accurate. It starts from gyr < thd + sample_after_thd sample,
         ends in the middle of the stance"""

        gyr_all = np.deg2rad(data_df[['gyr_x', 'gyr_y', 'gyr_z']])
        gyr_magnitude = np.linalg.norm(gyr_all, axis=1)

        imu_sample_rate = 100
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

        if cut_off_fre_strike_off is not None:
            acc_data = data_filter(acc_data, cut_off_fre_strike_off, self.sample_rate, filter_order=2)
            gyr_data = data_filter(gyr_data, cut_off_fre_strike_off, self.sample_rate, filter_order=2)

        gyr_y = gyr_data[:, 1]
        data_len = gyr_data.shape[0]

        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        gyr_magnitude = np.linalg.norm(gyr_data, axis=1)
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

        return strike_list, off_list

    @staticmethod
    def __find_stationary_phase(gyr_magnitude, acc_magnitude, foot_stationary_acc_thd, foot_stationary_gyr_thd):
        """ Old function, require 10 continuous setps """
        data_len = gyr_magnitude.shape[0]
        stationary_flag, stationary_flag_temp = np.zeros(gyr_magnitude.shape), np.zeros(gyr_magnitude.shape)
        stationary_flag_temp[
            (acc_magnitude < foot_stationary_acc_thd) & (abs(gyr_magnitude) < foot_stationary_gyr_thd)] = 1
        for i_sample in range(data_len):
            if stationary_flag_temp[i_sample - 5:i_sample + 5].all():
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


IMU_LIST = [segment + sensor + axis for segment in IMU_SEGMENT_LIST for sensor in ['_Accel_', '_Gyro_'] for axis in ['X', 'Y', 'Z']]
EMG_LIST = ['gastrocmed', 'tibialisanterior', 'soleus', 'vastusmedialis', 'vastuslateralis', 'rectusfemoris',
            'bicepsfemoris', 'semitendinosus', 'gracilis', 'gluteusmedius']     # , 'rightexternaloblique'
OUTPUT_LIST = ['ankle_angle_r']
SAMPLES_BEFORE_STRIKE, SAMPLES_AFTER_OFF = 0, 0
config = {'epoch': 6, 'batch_size': 16, 'lr_ssl': 1e-2, 'lr_linear': 1e-3, 'use_ratio': 100, 'emb_output_dim': 32, 'common_space_dim': 128,
          'device': 'cuda', 'dtype': torch.FloatTensor, 'loss_fn': nce_loss, 'max_trial_num': 2}
if config['device'] is 'cuda':
    config['dtype'] = torch.cuda.FloatTensor


if __name__ == '__main__':
    # logging.info("Current commit is {}".format(execute_cmd("git rev-parse HEAD")))
    data_reader = DatasetLoader(['AB25', 'AB27', 'AB28', 'AB30'])
    ssl_framework = FrameworkSSL(data_reader, ImuTestTransformerEmbedding, EmgTestTransformerEmbedding, config)
    ssl_framework.preprocess_train_evaluation(['AB27', 'AB28', 'AB30'], ['AB25'], ['AB25'])




