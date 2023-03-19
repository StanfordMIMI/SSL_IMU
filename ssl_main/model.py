import copy
import random

import wandb
from const import _mods
from torch import nn, Tensor
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from torch.nn import TransformerEncoder
import math
import numpy as np
from utils import fix_seed
import matplotlib.pyplot as plt


fix_seed()


def mse_loss_masked(mod_outputs, mods, mask_indices):
    loss = torch.square(mod_outputs[mask_indices] - mods[mask_indices])
    return loss.mean()


def mse_loss(mod_outputs, mods, _):
    loss = torch.square(mod_outputs - mods)
    return loss.mean()


class SslReconstructNet(nn.Module):
    def __init__(self, emb_net, common_space_dim, loss_fn):
        super(SslReconstructNet, self).__init__()
        self.emb_net = emb_net
        # output_channel_num = emb_net.output_dim
        # self.linear_proj_1 = nn.ModuleList([nn.Linear(output_channel_num, common_space_dim) for _ in range(len(_mods))])
        # self.linear_proj_2 = nn.ModuleList([nn.Linear(common_space_dim, common_space_dim) for _ in range(len(_mods))])
        # self.bn_proj_1 = nn.ModuleList([nn.BatchNorm1d(common_space_dim) for _ in range(len(_mods))])
        self.loss_fn = loss_fn

    def forward(self, mods, lens):
        mod_all = torch.concat(mods, dim=1)
        mod_outputs, mask_indices = self.emb_net(mod_all, lens)
        loss = self.loss_fn(mod_outputs, mod_all, mask_indices)
        return loss, mod_outputs

    def show_reconstructed_signal(self, mods, lens, fig_title):
        mod_all = torch.concat(mods, dim=1)
        mod_outputs, mask_indices = self.emb_net(mod_all, lens)

        fig = plt.figure()
        max_vals, min_vals = [], []
        for i_color, (i_channel, label_) in enumerate(zip([0, -1], ['Acc channel 1', 'Gyr channel 1'])):       # one acc and one gyr channel
            true_data = mod_all.detach().cpu().numpy()[0, i_channel]
            pred_data = mod_outputs.detach().cpu().numpy()[0, i_channel]
            msk = mask_indices.detach().cpu().numpy()[0, i_channel]
            plt.plot(true_data, color=f'C{i_color}', label=label_+' - True')
            plt.plot(pred_data, '-.', color=f'C{i_color}', label=label_+' - Reconstructed')
            max_vals.append(max(np.max(true_data), np.max(pred_data)))
            min_vals.append(min(np.min(true_data), np.min(pred_data)))
        plt.fill_between(range(true_data.shape[0]), min(min_vals), max(max_vals), label='Masked Patch', where=msk, facecolor='gray', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        wandb.log({"img":[wandb.Image(fig, caption=fig_title)]})


class SslContrastiveNet(nn.Module):
    def __init__(self, emb_net, common_space_dim, loss_fn):
        super(SslContrastiveNet, self).__init__()
        self.emb_net = emb_net
        output_channel_num = emb_net.output_dim
        self.linear_proj_1 = nn.ModuleList([nn.Linear(output_channel_num, common_space_dim) for _ in range(len(_mods))])
        self.linear_proj_2 = nn.ModuleList([nn.Linear(common_space_dim, common_space_dim) for _ in range(len(_mods))])
        self.bn_proj_1 = nn.ModuleList([nn.BatchNorm1d(common_space_dim) for _ in range(len(_mods))])
        self.loss_fn = loss_fn
        self.temperature = 0.1

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
                mod, self.emb_net, self.linear_proj_1[i_mod], self.linear_proj_2[i_mod], self.bn_proj_1[i_mod]))
        loss = self.loss_fn(mod_outputs, self.temperature)
        return loss, mod_outputs


class RegressNet(nn.Module):
    def __init__(self, emb_net, mod_channel_num, output_dim):
        super(RegressNet, self).__init__()
        self.emb_net = emb_net
        self.emb_output_dim = mod_channel_num
        self.output_dim = output_dim

        self.linear = nn.Linear(self.emb_output_dim, output_dim)

        # self.linear = nn.Linear(self.emb_output_dim*128, output_dim*128)

    def forward(self, x, lens):
        batch_size, _, seq_len = x[0].shape
        mod_all = torch.concat(x, dim=1)
        mod_outputs, _ = self.emb_net(mod_all, lens)

        output = mod_outputs.transpose(1, 2)
        output = self.linear(output)
        output = output.transpose(1, 2)

        # output = torch.flatten(mod_outputs, start_dim=1)
        # output = self.linear(output)
        # output = output.view([batch_size, -1, seq_len])

        return output, mod_outputs


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


class TransformerBase(nn.Module):
    def __init__(self, x_dim, output_dim, mask_patch_num, nlayers, nhead, patch_len, patch_step_len,
                 device='cuda'):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step_len = patch_step_len
        self.pad_len = 0
        # self.d_model = 256
        # self.linear_expansion = nn.Linear(int(x_dim * patch_len), self.d_model)
        self.d_model = int(x_dim * patch_len)
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.output_dim = output_dim
        self.device = device
        self.mask_patch_num = mask_patch_num
        self.mask_emb = nn.Parameter(torch.zeros([x_dim * patch_len]).uniform_() - 0.5)

    def forward(self, sequence, lens):
        sequence = self.divide_into_patches(sequence)
        sequence, mask_indices = self.apply_mask(sequence)
        sequence = self.transformer_encoder(sequence)
        output = self.flat_patches(sequence)
        mask_indices = self.flat_patches(mask_indices)
        return output, mask_indices

    def divide_into_patches(self, sequence):
        sequence = F.pad(sequence, (0, 0, self.pad_len, self.pad_len))
        sequence = sequence.unfold(2, self.patch_len, self.patch_step_len)
        sequence = sequence.transpose(1, 2).flatten(start_dim=2)
        return sequence

    def flat_patches(self, sequence):
        sequence = sequence.view([*sequence.shape[:2], -1, self.patch_len])
        sequence = sequence.transpose(1, 2).flatten(start_dim=2)
        return sequence

    def apply_mask(self, seq):
        bs, length, patch_emb = seq.shape
        # mask_indices_np = np.random.choice(a=[True, False], size=(bs, length), p=[self.mask_patch_num, 1 - self.mask_patch_num])
        mask_indices_np = np.full((bs, length), False)
        for i_row in range(mask_indices_np.shape[0]):
            masked_loc = random.sample(range(length), self.mask_patch_num)
            mask_indices_np[i_row, masked_loc] = True

        mask_indices = torch.from_numpy(mask_indices_np).to(seq.device)

        for _ in range(mask_indices.dim(), seq.dim()):
            mask_indices = mask_indices.unsqueeze(-1)
        if mask_indices.size(-1) < seq.size(-1):
            mask_indices = mask_indices.expand_as(seq)
        # t1 = torch.mul(seq, ~mask_indices).detach().cpu().numpy()
        # t2 = torch.mul(self.mask_emb, mask_indices).detach().cpu().numpy()
        tensor = torch.mul(seq, ~mask_indices) + torch.mul(self.mask_emb, mask_indices)

        return tensor, mask_indices


def transformer(x_dim, output_dim, mask_patch_num=0):
    return TransformerBase(x_dim, output_dim, mask_patch_num=mask_patch_num, nlayers=6, nhead=8, patch_len=16, patch_step_len=16)


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_length):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=1, padding=4)
        self.ln1 = nn.LayerNorm(out_length)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=9, stride=1, padding=4)
        self.ln2 = nn.LayerNorm(out_length)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.ln1(output))
        output = self.conv2(output)
        output = self.ln2(output)
        return self.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, out_length):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride[0], padding=4)
        self.ln1 = nn.LayerNorm(out_length)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=9, stride=stride[1], padding=4)
        self.ln2 = nn.LayerNorm(out_length)
        self.extra = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.LayerNorm(out_length)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.ln1(output))        # !!!

        out = self.conv2(out)
        out = self.ln2(out)
        return self.relu(extra_x + out)


class ResNetBase(nn.Module):
    def __init__(self, x_dim, output_dim, net_name, layer_nums):
        super(ResNetBase, self).__init__()
        gate_size = 16
        kernel_depths = [16, 32, 64, output_dim]
        out_lengths = [64, 32, 16, 8, 4]
        self.conv1 = nn.Conv1d(x_dim, gate_size, kernel_size=49, stride=2, padding=24)
        self.ln1 = nn.LayerNorm(out_lengths[0])     # !!!
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.output_dim = output_dim
        self.net_name = net_name

        self.layers = nn.ModuleList()
        block_list = [RestNetBasicBlock(gate_size, kernel_depths[0], out_lengths[1])] + \
                     [RestNetBasicBlock(kernel_depths[0], kernel_depths[0], out_lengths[1]) for _ in range(layer_nums[0]-1)]
        self.layers.append(nn.Sequential(*block_list))
        for i_layer in range(1, 4):
            block_list = [RestNetDownBlock(kernel_depths[i_layer-1], kernel_depths[i_layer], [2, 1], out_lengths[i_layer+1])] +\
                         [RestNetBasicBlock(kernel_depths[i_layer], kernel_depths[i_layer], out_lengths[i_layer+1]) for _ in range(layer_nums[i_layer]-1)]
            self.layers.append(nn.Sequential(*block_list))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):
        out = sequence.transpose(-1, -2)
        # print(out.shape[1:])
        out = self.relu(self.ln1(self.conv1(out)))
        # print(out.shape[1:])
        out = self.maxpool(out)
        out = self.layers[0](out)
        # print(out.shape[1:])
        out = self.layers[1](out)
        # print(out.shape[1:])
        out = self.layers[2](out)
        # print(out.shape[1:])
        out = self.layers[3](out)
        # print(out.shape[1:])
        out = self.avgpool(out)
        # print(out.shape[1:])
        return out.squeeze(dim=-1), None


def resnet18(x_dim, output_dim):
    layer_nums = [2, 2, 2, 2]
    return ResNetBase(x_dim, output_dim, 'resnet18', layer_nums)


def resnet50(x_dim, output_dim):
    layer_nums = [3, 4, 6, 3]
    return ResNetBase(x_dim, output_dim, 'resnet50', layer_nums)


def resnet101(x_dim, output_dim):
    layer_nums = [3, 4, 32, 3]
    return ResNetBase(x_dim, output_dim, 'resnet50', layer_nums)
