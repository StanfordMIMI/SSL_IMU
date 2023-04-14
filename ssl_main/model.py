import copy
import random
import wandb
from const import _mods
from torch import nn, Tensor
import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import functional as F
import math
import numpy as np
from utils import fix_seed
import matplotlib.pyplot as plt


fix_seed()


def mse_loss_masked(mod_outputs, mods, mask_indices):
    loss = torch.square(mod_outputs[mask_indices] - mods[mask_indices])
    return loss.mean()


def mse_loss_masked_weight_acc_gyr(mod_outputs, mods, mask_indices):
    loss_mods = []
    for i_mod in range(2):
        loss_mod = torch.square(mod_outputs[:, 24*i_mod:24*(i_mod+1)][mask_indices[:, 24*i_mod:24*(i_mod+1)]] -
                            mods[:, 24*i_mod:24*(i_mod+1)][mask_indices[:, 24*i_mod:24*(i_mod+1)]])
        loss_mods.append(loss_mod.mean())
    loss = 0.05 * loss_mods[0] + 0.95 * loss_mods[1]
    return loss


def mse_loss(mod_outputs, mods, _):
    loss = torch.square(mod_outputs - mods)
    return loss.mean()


class SslReconstructNet(nn.Module):
    def __init__(self, emb_net, loss_fn):
        super(SslReconstructNet, self).__init__()
        self.emb_net = emb_net
        self.linear = nn.Linear(emb_net.x_dim, emb_net.x_dim)
        self.loss_fn = loss_fn

    def forward(self, mods, lens):
        mod_all = torch.concat(mods, dim=1)
        # mod_all = mod_all[:, 24:]     # !!!
        mod_outputs, mask_indices, mod_all_expanded = self.emb_net(mod_all, lens)
        mod_outputs = self.linear(mod_outputs.transpose(1, 2)).transpose(1, 2)
        loss = self.loss_fn(mod_outputs, mod_all, mask_indices)
        return loss, mod_outputs

    def show_reconstructed_signal(self, mods, lens, fig_title):
        mod_all = torch.concat(mods, dim=1)
        # mod_all = mod_all[:, 24:]     # !!!
        mod_outputs, mask_indices, _ = self.emb_net(mod_all, lens)
        mod_outputs = self.linear(mod_outputs.transpose(1, 2)).transpose(1, 2)

        for i_color, (i_channel, label_) in enumerate(zip([0, -1], ['Acc channel 1', 'Gyr channel 1'])):       # one acc and one gyr channel
            fig = plt.figure()
            true_data = mod_all.detach().cpu().numpy()[0, i_channel]
            pred_data = mod_outputs.detach().cpu().numpy()[0, i_channel]
            msk = mask_indices.detach().cpu().numpy()[0, i_channel]
            plt.plot(true_data, color=f'C{i_color}', label=label_+' - True')
            plt.plot(pred_data, '-.', color=f'C{i_color}', label=label_+' - Reconstructed')
            max_val = max(np.max(true_data), np.max(pred_data))
            min_val = min(np.min(true_data), np.min(pred_data))
            plt.fill_between(range(true_data.shape[0]), min_val, max_val, label='Masked', where=msk, facecolor='gray', alpha=0.3)
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
        self.emb_output_dim = emb_net.x_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(self.emb_output_dim, output_dim)

    def forward(self, x, lens):
        batch_size, _, seq_len = x[0].shape
        mod_all = torch.concat(x, dim=1)
        # mod_all = mod_all[:, 24:]            # !!!
        mod_outputs, _, _ = self.emb_net(mod_all, lens)

        output = mod_outputs.transpose(1, 2)
        output = self.linear(output)
        output = output.transpose(1, 2)
        return output, mod_outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0., patch_num: int = 16):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(patch_num).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(patch_num * 2) / d_model))
        pe = torch.zeros(patch_num, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        try:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        except RuntimeError:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :-1]

        pe = pe.transpose(0, 1)
        # plt.figure()
        # plt.imshow(pe[0])
        # plt.show()
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # plt.figure()
        # plt.imshow(x.detach().cpu().numpy()[0])
        # plt.figure()
        # plt.imshow(self.pe.detach().cpu().numpy()[0])

        x = x + self.pe     # [:, :x.shape[1], :]

        # plt.figure()
        # plt.imshow(x.detach().cpu().numpy()[0])
        # plt.show()
        return self.dropout(x)


class TransformerBase(nn.Module):
    def __init__(self, x_dim, output_dim, mask_patch_num, nlayers, nhead, dim_feedforward, patch_len, patch_step_len, device='cuda'):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step_len = patch_step_len
        self.pad_len = 0
        self.x_dim = x_dim
        self.d_model = int(self.x_dim * patch_len)
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, dim_feedforward, batch_first=True)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.output_dim = output_dim
        self.device = device
        self.mask_patch_num = mask_patch_num
        self.mask_emb = nn.Parameter(torch.zeros([x_dim, patch_len]).uniform_() - 0.5)
        self.patch_num = int(128 / self.patch_len)
        self.pos_encoding = PositionalEncoding(x_dim * patch_len, patch_num=self.patch_num)

    def forward(self, sequence, lens):
        sequence, mask_indices = self.apply_mask(sequence)
        sequence_patch = self.divide_into_patches(sequence)
        sequence_patch = self.pos_encoding(sequence_patch)
        sequence_patch = self.transformer_encoder(sequence_patch)

        output = self.flat_patches(sequence_patch)
        return output, mask_indices, sequence

    def divide_into_patches(self, sequence):
        sequence = F.pad(sequence, (0, 0, self.pad_len, self.pad_len))
        sequence = sequence.unfold(2, self.patch_len, self.patch_step_len)
        sequence = sequence.transpose(1, 2).flatten(start_dim=2)
        return sequence

    def flat_patches(self, sequence):
        sequence = sequence.view([*sequence.shape[:2], -1, self.patch_len])
        sequence = sequence.transpose(1, 2).flatten(start_dim=2)
        return sequence

    def apply_mask(self, seq, mask_type=1):
        """

        :param seq: [bs, embedding, time]
        :param mask_type: 0 for mask random samples, 1 for mask one entire patch, 2 for testing
        :return:
        """
        if mask_type == 0:
            pass
        elif mask_type == 1:
            bs, patch_emb, length = seq.shape
            mask_indices_np = np.full((bs, length), False)
            for i_row in range(mask_indices_np.shape[0]):
                len_to_mask = [self.mask_patch_num]     # [self.mask_patch_num]   [int(0.5*self.mask_patch_num), int(0.5*self.mask_patch_num)]
                for len_ in len_to_mask:
                    masked_loc = random.sample(range(self.patch_num - len_ + 1), 1)[0]
                    mask_indices_np[i_row, masked_loc*self.patch_len:(masked_loc+len_)*self.patch_len] = True
            mask_indices = torch.from_numpy(mask_indices_np).to(seq.device)
            mask_indices = mask_indices.unsqueeze(1).expand_as(seq)
            mask_emb_repeated = self.mask_emb.repeat(1, self.patch_num)
            tensor = torch.mul(seq, ~mask_indices) + torch.mul(mask_emb_repeated, mask_indices)

        elif mask_type == 2:
            pass
        return tensor, mask_indices


def transformer(x_dim, output_dim, mask_patch_num, patch_len):
    return TransformerBase(x_dim, output_dim, mask_patch_num=mask_patch_num, nlayers=6, nhead=48, dim_feedforward=512,
                           patch_len=patch_len, patch_step_len=patch_len)


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
        out = F.relu(self.ln1(output))

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
        self.ln1 = nn.LayerNorm(out_lengths[0])
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
