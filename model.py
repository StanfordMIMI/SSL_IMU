import copy

from torch import nn, Tensor
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from torch.nn import TransformerEncoder
import math
import numpy as np
from utils import fix_seed, off_diagonal
import matplotlib.pyplot as plt
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices


fix_seed()


def vic_loss(mod_outputs, temperature):
    if len(mod_outputs) == 2:
        combos = [[0, 1]]
    elif len(mod_outputs) == 3:
        combos = [[0, 1], [1, 2], [0, 2]]
    else:
        raise ValueError('Only 2 or 3 embs are allowed.')
    repr_loss = sum([F.mse_loss(mod_outputs[mod_a], mod_outputs[mod_b]) for mod_a, mod_b in combos])

    xs = [mod_output - mod_output.mean(dim=0) for mod_output in mod_outputs]
    mod_stds = [torch.sqrt(x.var(dim=0) + 1e-15) for x in xs]
    std_loss = sum([torch.mean(F.relu(1 - mod_std)) for mod_std in mod_stds]) / len(mod_outputs)

    B, C = mod_outputs[0].shape
    cov_xs = [(x.T @ x) / (B - 1) for x in xs]
    cov_loss = sum([off_diagonal(cov_x).pow_(2).sum().div(C) for cov_x in cov_xs])
    loss = (1 * repr_loss + 1 * std_loss
            + 0.01 * cov_loss
            )
    return loss


def nce_loss(mod_outputs, temperature):
    def _calculate_similarity(emb1, emb2):
        # make each vector unit length
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        similarity = torch.matmul(emb1, emb2.transpose(1, 2))
        return similarity / temperature

    if len(mod_outputs) == 2:
        combos = [[0, 1]]
    elif len(mod_outputs) == 3:
        combos = [[0, 1], [1, 2], [0, 2]]
    else:
        raise ValueError('Only 2 or 3 embs are allowed.')
    emb_pos = [torch.unsqueeze(emb, 1) for emb in mod_outputs]
    sim_pos = [_calculate_similarity(emb_pos[combo[0]], emb_pos[combo[1]]) for combo in combos]
    emb_all = [torch.unsqueeze(emb, 0) for emb in mod_outputs]
    sim_all = [_calculate_similarity(emb_all[combo[0]], emb_all[combo[1]]) for combo in combos]

    logsumexp_pos = torch.flatten(torch.mean(torch.stack(sim_pos), dim=0))
    logsumexp_all = [torch.logsumexp(sim_, dim=2) for sim_ in sim_all]
    logsumexp_all = torch.flatten(torch.mean(torch.stack(logsumexp_all), dim=0))
    loss = (logsumexp_all - logsumexp_pos).mean()
    return loss


def nce_loss_groups_of_positive(mod_outputs, temperature):
    def _calculate_similarity(emb1, emb2):
        # make each vector unit length
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        similarity = torch.matmul(emb1, emb2.transpose(1, 2))
        return similarity / temperature

    def get_group_negative_mask(matrix_size, group_len):
        negative_mask = torch.ones(matrix_size, dtype=bool)
        for i in range(group_num):
            negative_mask[i*group_len:(i+1)*group_len, i*group_len:(i+1)*group_len] = 0
        return negative_mask

    sim_pos = _calculate_similarity(torch.unsqueeze(mod_outputs[0], 1), torch.unsqueeze(mod_outputs[1], 1))
    logsumexp_pos = torch.flatten(torch.mean(sim_pos))

    sim_all = _calculate_similarity(torch.unsqueeze(mod_outputs[0], 0), torch.unsqueeze(mod_outputs[1], 0))[0]
    group_len = 17
    assert sim_all.shape[0] % group_len == 0
    group_num = int(sim_all.shape[0] / group_len)
    mask = get_group_negative_mask(sim_all.shape, group_len).to("cuda")
    sim_all = sim_all.masked_select(mask).view(-1, mask.shape[0] - group_len)
    logsumexp_all = torch.logsumexp(sim_all, dim=1)
    logsumexp_all = torch.flatten(torch.mean(logsumexp_all, dim=0))
    loss = logsumexp_all - logsumexp_pos
    return loss


# def nce_loss_groups_of_positive_2(mod_outputs, temperature):
#     def _calculate_similarity(emb1, emb2):
#         # make each vector unit length
#         emb1 = F.normalize(emb1, p=2, dim=-1)
#         emb2 = F.normalize(emb2, p=2, dim=-1)
#         similarity = torch.matmul(emb1, emb2.transpose(1, 2))
#         return similarity / temperature
#
#     def get_group_positive_mask(matrix_size, group_len):
#         negative_mask = torch.zeros(matrix_size, dtype=bool)
#         for i in range(group_num):
#             negative_mask[i*group_len:(i+1)*group_len, i*group_len:(i+1)*group_len] = 1
#         return negative_mask
#
#     sim_all = _calculate_similarity(torch.unsqueeze(mod_outputs[0], 0), torch.unsqueeze(mod_outputs[1], 0))[0]
#     group_len = 17
#     assert sim_all.shape[0] % group_len == 0
#     group_num = int(sim_all.shape[0] / group_len)
#     mask = get_group_positive_mask(sim_all.shape, group_len).to("cuda")
#     sim_pos = sim_all
#     sim_pos = sim_pos.masked_select(mask).view(-1, group_len)
#     logsumexp_pos = torch.logsumexp(sim_pos, dim=1)
#     logsumexp_all = torch.logsumexp(sim_all, dim=1)
#     logsumexp_pos = torch.flatten(torch.mean(logsumexp_pos, dim=0))
#     logsumexp_all = torch.flatten(torch.mean(logsumexp_all, dim=0))
#     loss = logsumexp_all - logsumexp_pos
#     return loss


def nce_loss_hard_negative(mod_outputs, temperature):
    def _calculate_similarity(emb1, emb2):
        # make each vector unit length
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        similarity = torch.matmul(emb1, emb2.transpose(1, 2))
        return similarity / temperature

    def get_negative_mask(batch_size):
        negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            # negative_mask[i, i + batch_size] = 0

        # negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    sim_pos = _calculate_similarity(torch.unsqueeze(mod_outputs[0], 1), torch.unsqueeze(mod_outputs[1], 1))
    sim_all = _calculate_similarity(torch.unsqueeze(mod_outputs[0], 0), torch.unsqueeze(mod_outputs[1], 0))

    pos = sim_pos[:, 0, 0].exp()
    neg = sim_all[0].exp()

    N = batch_size = mod_outputs[0].shape[0]
    mask = get_negative_mask(batch_size).to("cuda")
    neg = neg.masked_select(mask).view(batch_size, -1)

    beta, tau_plus = 1, .1
    reweight = (beta * neg) / neg.mean(dim=1).unsqueeze(1)
    # Neg = torch.max((-N * tau_plus * pos + (reweight * neg).sum(dim=1)) / (1 - tau_plus), torch.full([N], np.e ** (-1 / temperature)).cuda())

    loss = ((pos + (reweight * neg).sum(dim=1)).log() - pos.log()).mean()
    return loss


def clip_loss(mod_outputs, temperature):
    def _calculate_similarity(emb1, emb2):
        # make each vector unit length
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        similarity = torch.matmul(emb1, emb2.transpose(1, 2))
        return similarity / temperature

    sim_all = _calculate_similarity(torch.unsqueeze(mod_outputs[0], 0), torch.unsqueeze(mod_outputs[1], 0))[0]
    labels = torch.arange(sim_all.shape[1], device='cuda')
    loss_t = F.cross_entropy(sim_all, labels)
    loss_i = F.cross_entropy(sim_all.T, labels)
    return (loss_t + loss_i) / 2.


class SslGeneralNet(nn.Module):
    def __init__(self, emb_nets, common_space_dim, loss_fn):
        super(SslGeneralNet, self).__init__()
        self.emb_nets = emb_nets
        output_channel_nums = [embnet.output_dim for embnet in emb_nets]
        self.linear_proj_1 = nn.ModuleList([nn.Linear(output_channel_num, common_space_dim) for output_channel_num in output_channel_nums])
        self.linear_proj_2 = nn.ModuleList([nn.Linear(common_space_dim, common_space_dim) for _ in output_channel_nums])
        self.bn_proj_1 = nn.ModuleList([nn.BatchNorm1d(common_space_dim) for _ in output_channel_nums])
        # self.bn_proj_2 = nn.ModuleList([nn.BatchNorm1d(common_space_dim) for _ in output_channel_nums])       # !!!
        self.net_name = 'Combined Net'
        self.loss_fn = loss_fn
        # self.temperature = nn.Parameter(torch.from_numpy(np.full([1], 0.1)), requires_grad=True)
        self.temperature = 0.1

    def __str__(self):
        return self.net_name

    def set_scalars(self, scalars):
        self.scalars = scalars

    def forward(self, mods, lens):
        def reshape_and_emb(mod_, embnet, linear_proj_1, linear_proj_2, bn_proj_1):
            mod_ = mod_.view(-1, 3, *mod_.shape[2:])
            mod_ = mod_.transpose(1, 2)
            mod_, _ = embnet(mod_, lens)
            mod_ = F.relu(bn_proj_1(linear_proj_1(mod_)))
            mod_ = linear_proj_2(mod_)
            return mod_

        mod_outputs = []
        for i_mod, mod in enumerate(mods):
            mod_outputs.append(reshape_and_emb(
                mod, self.emb_nets[i_mod], self.linear_proj_1[i_mod], self.linear_proj_2[i_mod], self.bn_proj_1[i_mod]))
        loss = self.loss_fn(mod_outputs, self.temperature)
        return loss, mod_outputs


class RegressNet(nn.Module):
    def __init__(self, emb_nets, mod_channel_num, output_dim):
        super(RegressNet, self).__init__()
        self.emb_nets = emb_nets

        emb_output_dims = [net.output_dim * channel_num / 3 for net, channel_num in zip(self.emb_nets, mod_channel_num)]
        self.emb_output_dim = int(sum(emb_output_dims))
        self.output_dim = output_dim
        self.linear_1 = nn.Linear(self.emb_output_dim, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.linear_3 = nn.Linear(128, output_dim)
        self.linear = nn.ModuleList([self.linear_1, self.linear_2, self.linear_3])

    def forward(self, x, lens):
        batch_size = x[0].shape[0]
        mod_outputs = []
        for i_mod, x_mod in enumerate(x):
            x_mod = x_mod.view(-1, 3, *x_mod.shape[2:])
            x_mod = x_mod.transpose(1, 2)
            mod_output, _ = self.emb_nets[i_mod](x_mod, lens)
            mod_outputs.append(mod_output.reshape(batch_size, -1))

        output = torch.concat(mod_outputs, dim=1)
        output = F.relu(self.linear_1(output))
        output = F.relu(self.linear_2(output))
        output = self.linear_3(output)
        return output, mod_outputs


class ImuRnnEmbedding(nn.Module):
    def __init__(self, x_dim, output_dim, net_name, nlayer=2, linear_dim=32):
        super(ImuRnnEmbedding, self).__init__()
        self.net_name = net_name
        self.rnn_layer = nn.LSTM(x_dim, linear_dim, nlayer, bidirectional=True, bias=True)
        self.linear = nn.Linear(linear_dim*2, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.output_dim = output_dim
        self.ratio_to_base_fre = 1
        for name, param in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=10)

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):
        sequence = sequence.transpose(0, 1)
        total_len = sequence.shape[0]
        lens = lens * self.ratio_to_base_fre
        sequence = pack_padded_sequence(sequence, lens, enforce_sorted=False)
        sequence, (emb, _) = self.rnn_layer(sequence)
        sequence, _ = pad_packed_sequence(sequence, total_length=total_len)
        sequence = self.linear(sequence)
        sequence = sequence.transpose(0, 1)
        emb = emb[-2:]
        emb = emb.transpose(0, 1)
        emb = torch.flatten(emb, start_dim=1)
        emb = emb.transpose(0, 1)
        return sequence, emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        try:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        except RuntimeError:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :-1]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ImuTransformerEmbedding(nn.Module):
    def __init__(self, d_model, output_dim, net_name, nlayers=1, nhead=1, d_hid=20, dropout=0, device='cuda'):
        super().__init__()
        self.patch_len = 16
        self.patch_step_len = 8
        self.pad_len = int(self.patch_step_len / 2)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = int(d_model * (256 / self.patch_len))
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.output_dim = output_dim
        self.linear = nn.Linear(self.d_model, output_dim * self.patch_step_len)
        self.device = device

    def forward(self, sequence, lens):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        sequence = self.divide_into_patches(sequence)
        sequence = sequence.transpose(0, 1)
        output = self.transformer_encoder(sequence)
        output = self.linear(output)
        output = output.transpose(0, 1)
        output = self.flat_patches(output)
        return output, None

    def divide_into_patches(self, sequence):
        sequence = F.pad(sequence, (0, 0, self.pad_len, self.pad_len))
        sequence = sequence.unfold(1, self.patch_len, self.patch_step_len).flatten(start_dim=2)
        return sequence

    def flat_patches(self, sequence):
        shape_1 = sequence.shape[1] * self.patch_step_len
        sequence = sequence.reshape([sequence.shape[0], shape_1, -1])
        return sequence

    def generate_padding_mask(self, lens, seq_len):
        msk = np.ones([len(lens), seq_len], dtype=bool)
        for i, the_len in enumerate(lens):
            msk[i, :int(the_len)] = 0
        return torch.from_numpy(msk).to(self.device)


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=9, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return self.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride[0], padding=4)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=9, stride=stride[1], padding=4)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm1d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(extra_x + out)


class CnnEmbedding(nn.Module):
    def __init__(self, x_dim, output_dim, net_name):
        super(CnnEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(x_dim, 16, kernel_size=49, stride=2, padding=24)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.output_dim = output_dim
        self.net_name = net_name
        self.layer1 = nn.Sequential(RestNetBasicBlock(16, 16), RestNetBasicBlock(16, 16))
        self.layer2 = nn.Sequential(RestNetDownBlock(16, 32, [2, 1]), RestNetBasicBlock(32, 32))
        self.layer3 = nn.Sequential(RestNetDownBlock(32, 64, [2, 1]), RestNetBasicBlock(64, 64))
        self.layer4 = nn.Sequential(RestNetDownBlock(64, output_dim, [2, 1]), RestNetBasicBlock(output_dim, output_dim))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.net_name = 'CnnEmbedding'

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):
        out = sequence.transpose(-1, -2)
        # print(out.shape[1:])
        out = self.relu(self.bn1(self.conv1(out)))
        # print(out.shape[1:])
        out = self.maxpool(out)
        out = self.layer1(out)
        # print(out.shape[1:])
        out = self.layer2(out)
        # print(out.shape[1:])
        out = self.layer3(out)
        # print(out.shape[1:])
        out = self.layer4(out)
        # print(out.shape[1:])
        out = self.avgpool(out)
        # print(out.shape[1:])
        return out.squeeze(dim=-1), None




