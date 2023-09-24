import numpy as np
import h5py
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import functional as F
import torch
import json
import copy
import math
import pytorch_warmup as warmup
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error as mse
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler


CAMARGO_SUB_HEIGHT_WEIGHT = {
    'AB06': [1.80, 74.8], 'AB07': [1.65, 55.3], 'AB09': [1.63, 63.5], 'AB10': [1.75, 83.9], 'AB11': [1.75, 77.1],
    'AB12': [1.74, 86.2], 'AB13': [1.73, 59.0], 'AB14': [1.52, 58.4], 'AB15': [1.78, 96.2], 'AB16': [1.65, 55.8],
    'AB17': [1.68, 61.2], 'AB19': [1.70, 68.0], 'AB21': [1.57, 58.1], 'AB23': [1.80, 76.8], 'AB24': [1.73, 72.6],
    'AB25': [1.63, 52.2], 'AB27': [1.70, 68.0], 'AB28': [1.69, 62.1], 'AB30': [1.77, 77.0]
}
GRAVITY = 9.81


class RegressNet(nn.Module):
    def __init__(self, emb_net, output_dim):
        super(RegressNet, self).__init__()
        self.emb_net = emb_net
        self.output_dim = output_dim
        self.linear = nn.Linear(emb_net.embedding_dim, output_dim * emb_net.patch_len)

    def forward(self, x):
        batch_size, _, seq_len = x[0].shape
        mod_all = torch.concat(x, dim=1)
        mod_outputs = self.emb_net(mod_all)
        output = self.linear(mod_outputs).view([*mod_outputs.shape[:2], -1, self.output_dim]).flatten(1, 2).transpose(1, 2)
        return output, mod_outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout: float = 0.):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(max_len * 2) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        try:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        except RuntimeError:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :-1]

        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


# class TransformerBase(nn.Module):
#     def __init__(self, imu_to_use, device='cuda'):
#         super().__init__()
#         imu_to_use_mapped, mask_input_imu_index = imu_name_mapping(imu_to_use)
#         x_dim = 48
#         patch_len = 8
#         patch_step_len = 8
#         self.patch_len = patch_len
#         self.patch_step_len = patch_step_len
#         self.pad_len = 0
#         self.x_dim = x_dim
#         self.d_model = int(self.x_dim * patch_len)
#         encoder_layers = TransformerEncoderLayer(self.d_model, 48, 512, batch_first=True)
#
#         self.transformer_encoder = TransformerEncoder(encoder_layers, 6)
#         self.device = device
#         self.ssl_mask_emb = nn.Parameter(torch.zeros([1, x_dim * patch_len]).uniform_() - 0.5)
#         self.reduced_imu_mask_emb = nn.Parameter(torch.zeros([48, 128]).uniform_() - 0.5)
#         self.mask_input_imu_index = mask_input_imu_index
#         self.patch_num = int(128 / self.patch_len)
#         self.pos_encoding = PositionalEncoding(x_dim * patch_len, self.patch_num)
#
#     def forward(self, sequence):
#         sequence = self.apply_reduced_imu_set_mask(sequence)
#         sequence_patch = self.divide_into_patches(sequence)
#         sequence_patch = self.pos_encoding(sequence_patch)
#         sequence_patch = self.transformer_encoder(sequence_patch)
#
#         output = self.flat_patches(sequence_patch)
#         return output
#
#     def divide_into_patches(self, sequence):
#         sequence = F.pad(sequence, (0, 0, self.pad_len, self.pad_len))
#         sequence = sequence.unfold(2, self.patch_len, self.patch_step_len)
#         sequence = sequence.transpose(1, 2).flatten(start_dim=2)
#         return sequence
#
#     def flat_patches(self, sequence):
#         sequence = sequence.view([*sequence.shape[:2], -1, self.patch_len])
#         sequence = sequence.transpose(1, 2).flatten(start_dim=2)
#         return sequence
#
#     def apply_reduced_imu_set_mask(self, seq):
#         bs, patch_emb, length = seq.shape
#         mask_indices_np = np.full((bs, patch_emb), False)
#         for i_imu in range(8):
#             if i_imu in self.mask_input_imu_index:
#                 mask_indices_np[:, i_imu*3:(i_imu+1)*3] = True
#                 mask_indices_np[:, i_imu*3+24:(i_imu+1)*3+24] = True
#         mask_indices = torch.from_numpy(mask_indices_np).to(seq.device)
#         mask_indices = mask_indices.unsqueeze(-1).expand_as(seq)
#
#         tensor = torch.mul(seq, ~mask_indices) + torch.mul(self.reduced_imu_mask_emb, mask_indices)
#         return tensor

class TransformerEncoderOnly(nn.Module):
    def __init__(self, imu_to_use, device='cpu'):
        super().__init__()
        imu_to_use_mapped, mask_input_imu_index = imu_name_mapping(imu_to_use)
        x_dim = 48
        self.patch_step_len = 1
        self.patch_len = 1
        self.x_dim = x_dim
        self.pad_len = 0
        self.embedding_dim = 192
        self.input_to_embedding = nn.Linear(x_dim * self.patch_len, self.embedding_dim)
        self.d_model = int(self.embedding_dim)
        encoder_layers = TransformerEncoderLayer(self.d_model, 48, 512, batch_first=True)

        self.transformer = TransformerEncoder(encoder_layers, 6)
        self.device = device
        self.ssl_mask_emb = nn.Parameter(torch.zeros([1, x_dim * self.patch_len]).uniform_() - 0.5)
        self.reduced_imu_mask_emb = nn.Parameter(torch.zeros([48, 128]).uniform_() - 0.5)
        self.mask_input_imu_index = mask_input_imu_index
        self.patch_num = int(128 / self.patch_len)
        self.pos_encoding = PositionalEncoding(self.embedding_dim, self.patch_num)

    def forward(self, sequence):
        sequence = self.apply_reduced_imu_set_mask(sequence)
        sequence_patch = self.divide_into_patches(sequence)
        sequence_patch = self.input_to_embedding(sequence_patch)
        sequence_patch = self.pos_encoding(sequence_patch)
        output = self.transformer(sequence_patch)
        return output

    def divide_into_patches(self, sequence):
        sequence = F.pad(sequence, (0, 0, self.pad_len, self.pad_len))
        sequence = sequence.unfold(2, self.patch_len, self.patch_step_len)
        sequence = sequence.transpose(1, 2).flatten(start_dim=2)
        return sequence

    def apply_reduced_imu_set_mask(self, seq):
        bs, patch_emb, length = seq.shape
        mask_indices_np = np.full((bs, patch_emb), False)
        for i_imu in range(8):
            if i_imu in self.mask_input_imu_index:
                mask_indices_np[:, i_imu*3:(i_imu+1)*3] = True
                mask_indices_np[:, i_imu*3+24:(i_imu+1)*3+24] = True
        mask_indices = torch.from_numpy(mask_indices_np).to(seq.device)
        mask_indices = mask_indices.unsqueeze(-1).expand_as(seq)

        tensor = torch.mul(seq, ~mask_indices) + torch.mul(self.reduced_imu_mask_emb, mask_indices)
        return tensor


def load_data(data_file, imu_to_use, outputs, sub_for_fine_tuning, sub_for_testing):
    data_downstream_dict = {}
    data_scalar = {'base_scalar': StandardScaler}
    with h5py.File(data_file, 'r') as hf:
        data_columns = json.loads(hf.attrs['columns'])
        data_columns.extend(['msk' + sensor + axis for sensor in ['_Accel_', '_Gyro_'] for axis in ['X', 'Y', 'Z']])

        for set_sub_ids, set_name, norm_method in zip([sub_for_fine_tuning, sub_for_testing], ['tuning', 'test'],
                                                       ['fit_transform', 'transform']):
            current_set_data_list = []
            for sub_id in set_sub_ids:
                sub_data = hf[sub_id][:, :, :]
                sub_weight = CAMARGO_SUB_HEIGHT_WEIGHT[sub_id][1] * GRAVITY
                force_col_loc = [data_columns.index(x) for x in ['fx', 'fy', 'fz']]
                sub_data[:, force_col_loc] = sub_data[:, force_col_loc] / sub_weight
                current_set_data_list.append(sub_data)
            current_set_data = np.concatenate(current_set_data_list, axis=0)
            """ [step, feature, time] """
            # use rand noise to replace reduced IMUs
            rand_noise = np.random.normal(size=(current_set_data.shape[0], 6, current_set_data.shape[2]))
            current_set_data = np.concatenate([current_set_data, rand_noise], axis=1)

            imu_to_use_mapped, _ = imu_name_mapping(imu_to_use)
            channel_names = {
                'acc': [segment + '_Accel_' + axis for segment in imu_to_use_mapped for axis in ['X', 'Y', 'Z']],
                'gyr': [segment + '_Gyro_' + axis for segment in imu_to_use_mapped for axis in ['X', 'Y', 'Z']]}

            sampled_data = copy.deepcopy(current_set_data)
            set_data, data_scalar = preprocess_modality(data_columns, data_scalar, sampled_data, channel_names, norm_method)

            output_data = sampled_data[:, [data_columns.index(x) for x in outputs]]
            set_data['output'] = normalize_data(data_scalar, output_data, 'output', norm_method, 'by_each_column')
            set_data['sub_id'] = sampled_data[:, data_columns.index('sub_id'), 0]
            if 'trial_type_id' in data_columns:
                set_data['trial_type_id'] = sampled_data[:, data_columns.index('trial_type_id'), 0]
            else:
                set_data['trial_type_id'] = np.zeros([sampled_data.shape[0]])
            data_downstream_dict[set_name] = set_data
    return data_downstream_dict, data_scalar


def imu_name_mapping(imu_to_use):
    standard_list = ('CHEST', 'WAIST', 'R_THIGH', 'L_THIGH', 'R_SHANK', 'L_SHANK', 'R_FOOT', 'L_FOOT')
    imu_to_use_mapped = []
    mask_input_imu_index = []
    for i_imu, imu in enumerate(standard_list):
        if imu in imu_to_use:
            imu_to_use_mapped.append(imu)
        else:
            imu_to_use_mapped.append('msk')
            mask_input_imu_index.append(i_imu)
    return imu_to_use_mapped, mask_input_imu_index


def normalize_data(data_scalar, data, name, method, scalar_mode, with_mean=False):
    if method == 'fit_transform':
        data_scalar[name] = copy.deepcopy(data_scalar['base_scalar'](with_mean=with_mean))
    assert (scalar_mode in ['by_each_column', 'by_all_columns'])

    input_data = data.copy()
    size, channel, length = input_data.shape
    target_shape = [-1, input_data.shape[1]] if scalar_mode == 'by_each_column' else [-1, 1]
    zero_loc = (input_data == 0.).all(axis=1)
    for i in range(input_data.shape[1]):
        input_data[:, i][zero_loc] = np.nan
    input_data = input_data.transpose(0, 2, 1).reshape(target_shape) if scalar_mode == 'by_each_column' else input_data.reshape(target_shape)
    scaled_data = getattr(data_scalar[name], method)(input_data)
    scaled_data = scaled_data.reshape(size, length, channel).transpose(0, 2, 1) if scalar_mode == 'by_each_column' else scaled_data.reshape(size, channel, length)
    scaled_data[np.isnan(scaled_data)] = 0.
    return scaled_data


def preprocess_modality(data_columns, data_scalar, data_, channel_names, norm_method):
    processed_data = {}
    for group_name, cols in channel_names.items():
        col_loc = [data_columns.index(col) for col in cols]
        group_data = data_[:, col_loc, :]
        group_data = normalize_data(data_scalar, group_data, group_name, norm_method, 'by_all_columns')
        processed_data[group_name] = group_data
    return processed_data, data_scalar


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
                y_true_one_step = y_true[i_step, col, :lens[i_step]]
                y_pred_one_step = y_pred[i_step, col, :lens[i_step]]
                r2[i_step] = r2_score(y_true_one_step, y_pred_one_step)
                rmse[i_step] = np.sqrt(mse(y_true_one_step, y_pred_one_step))
                cor_value[i_step] = pearsonr(y_true_one_step, y_pred_one_step)[0]
        score_one_field = {'field': field, 'r2': r2, 'rmse': rmse, 'cor_value': cor_value}
        scores.append(score_one_field)
    return scores


def inverse_normalize_output(y_true, y_pred):
    y_true = normalize_data(data_scalar, y_true, 'output', 'inverse_transform', 'by_each_column')
    y_pred = normalize_data(data_scalar, y_pred, 'output', 'inverse_transform', 'by_each_column')
    return y_true, y_pred


def prepare_dl(data_list, batch_size, shuffle, drop_last=False):
    data_list_torch = [torch.from_numpy(data).float() for data in data_list]
    ds = TensorDataset(*data_list_torch)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dl


def set_dtype_and_model(device, model):
    if device == 'cuda':
        dtype = torch.cuda.FloatTensor
        model.cuda()
    else:
        dtype = torch.FloatTensor
    return dtype, model


def model_fine_tuning(linear_protocol, regress_net, data_downstream_dict, phase_name, show_fig=False, verbose=True):
    def convert_batch_data(batch_data):
        xb = [data_.float().type(dtype) for data_ in batch_data[:-1]]
        yb = batch_data[-1].float().type(dtype)
        return xb, yb

    def train_batch(regress_net, train_dl, optimizer, loss_fn, i_optimize):
        regress_net.train()
        for i_batch, batch_data in enumerate(train_dl):
            optimizer.zero_grad()
            xb, yb = convert_batch_data(batch_data)
            y_pred, _ = regress_net(xb)
            loss = loss_fn(yb, y_pred)
            loss.backward()
            optimizer.step()
            i_optimize += 1
            with warmup_scheduler.dampening():
                scheduler.step()
        return i_optimize

    def eval_during_training(regress_net, dl, loss_fn, use_batch_num=5):
        regress_net.eval()
        loss = []
        with torch.no_grad():
            for i_batch, batch_data in enumerate(dl):
                if i_batch >= use_batch_num:
                    return np.mean(loss)
                xb, yb = convert_batch_data(batch_data)
                y_pred, _ = regress_net(xb)
                loss.append(loss_fn(yb, y_pred).item())
        return np.mean(loss)

    def concat_data_after_training(test_dl):
        regress_net.eval()
        with torch.no_grad():
            y_pred_list, y_true_list = [], []
            for i_batch, batch_data in enumerate(test_dl):
                xb, yb = convert_batch_data(batch_data)
                y_true_list.append(yb.detach().cpu())
                y_pred_batch, mod_outputs_batch = regress_net(xb)
                y_pred_list.append(y_pred_batch.detach().cpu())
            y_true = torch.cat(y_true_list).numpy()
            y_pred = torch.cat(y_pred_list).numpy()
        return y_true, y_pred

    train_data, test_data = data_downstream_dict['tuning'], data_downstream_dict['test']
    train_input_data = [train_data[mod] for mod in ['acc', 'gyr']]
    train_output_data = train_data['output']
    # train_step_lens = get_step_len(train_input_data[0])
    train_dl = prepare_dl([*train_input_data, train_output_data], 64, shuffle=True)
    test_input_data = [test_data[mod] for mod in ['acc', 'gyr']]
    test_output_data = test_data['output']
    # test_step_lens = get_step_len(test_input_data[0])
    test_dl = prepare_dl([*test_input_data, test_output_data], 64, shuffle=False)
    dtype, regress_net = set_dtype_and_model('cpu', regress_net)

    if linear_protocol:
        lr_ = 1e-3
        param_to_train = regress_net.linear.parameters()
    else:
        lr_ = 1e-4
        param_to_train = regress_net.parameters()

    optimizer = torch.optim.AdamW(param_to_train, lr_)
    epoch_end_time = time.time()
    num_of_back_propagation = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_back_propagation)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=int(num_of_back_propagation/5))

    i_optimize = 0
    for i_epoch in range(num_of_back_propagation):
        if verbose:
            train_loss = eval_during_training(regress_net, train_dl, torch.nn.MSELoss())
            test_loss = eval_during_training(regress_net, test_dl, torch.nn.MSELoss())
            if num_of_back_propagation < 2 or i_epoch % int(num_of_back_propagation / 2) == 0 or i_epoch == num_of_back_propagation - 1:
                print(f'| {phase_name} | num of backpropagation{i_epoch+1:3d}/{num_of_back_propagation:3d} |'
                      f' time: {time.time() - epoch_end_time:5.2f}s |'
                      f' train loss {train_loss:5.3f} | test loss {test_loss:5.3f}')
            epoch_end_time = time.time()

        i_optimize = train_batch(regress_net, train_dl, optimizer, torch.nn.MSELoss(), i_optimize)

    if linear_protocol:
        return regress_net

    y_true, y_pred = concat_data_after_training(test_dl)
    y_true, y_pred = inverse_normalize_output(y_true, y_pred)

    plt.figure()
    for i_output in range(y_true.shape[1]):
        plt.plot(y_true[:, i_output].ravel(), '-', color='C'+str(i_output), label='Ground Truth')
        plt.plot(y_pred[:, i_output].ravel(), '--', color='C'+str(i_output), label='Estimation')
        plt.xlabel('Time Step')
        plt.ylabel('Ground Reaction Force (Body Weight)')
        plt.legend()
    plt.show()


if __name__ == "__main__":
    imu_to_use = ['CHEST', 'R_THIGH']
    sub_for_fine_tuning = list(CAMARGO_SUB_HEIGHT_WEIGHT.keys())[:-3]
    sub_for_testing = list(CAMARGO_SUB_HEIGHT_WEIGHT.keys())[-3:]
    outputs = ['fy']
    emb_net = TransformerEncoderOnly(imu_to_use)
    emb_net.load_state_dict(torch.load('pretrained_model_weights.pth'))
    regress_net = RegressNet(emb_net, len(outputs))
    set_data, data_scalar = load_data('Camargo_levelground.h5', imu_to_use, outputs, sub_for_fine_tuning, sub_for_testing)
    # First train the last linear layer to preserve representation from pre-training
    regress_net = model_fine_tuning(True, regress_net, set_data, 'Train the last layer')
    model_fine_tuning(False, regress_net, set_data, 'Fine tune the entire model', True)






