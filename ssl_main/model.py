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
import time

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
        self.linear = nn.Linear(emb_net.embedding_dim, emb_net.x_dim * emb_net.patch_len)
        self.loss_fn = loss_fn

    def forward(self, mods, lens, tgt=None):
        mod_all = torch.concat(mods, dim=1)
        if tgt is not None:
            tgt = torch.concat(tgt, dim=1)
        mod_outputs, mask_indices, mod_all_expanded = self.emb_net(mod_all, lens, tgt)
        mask_indices = mask_indices.view([*mod_outputs.shape[:2], -1, mod_all.shape[1]]).flatten(1, 2).transpose(1, 2)
        mod_outputs = self.linear(mod_outputs).view([*mod_outputs.shape[:2], -1, mod_all.shape[1]]).flatten(1, 2).transpose(1, 2)
        loss = self.loss_fn(mod_outputs, mod_all, mask_indices)
        return loss, mod_outputs, mask_indices


def show_reconstructed_signal(mod_all, mod_outputs, fig_title, mask_indices=None, fig_group='Img', channel_names=['Acc channel 1', 'Gyr channel 1']):
    for i_color, (i_channel, label_) in enumerate(zip([0, -1], channel_names)):       # one acc and one gyr channel
        fig = plt.figure()
        true_data = mod_all.detach().cpu().numpy()[0, i_channel]
        pred_data = mod_outputs.detach().cpu().numpy()[0, i_channel]
        plt.plot(true_data, color=f'C{i_color}', label=label_+' - True')
        plt.plot(pred_data, '-.', color=f'C{i_color}', label=label_+' - Reconstructed')
        max_val = max(np.max(true_data), np.max(pred_data))
        min_val = min(np.min(true_data), np.min(pred_data))
        if mask_indices is not None:
            msk = mask_indices.detach().cpu().numpy()[0, i_channel]
            plt.fill_between(range(true_data.shape[0]), min_val, max_val, label='Masked', where=msk, facecolor='gray', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        wandb.log({fig_group: [wandb.Image(fig, caption=fig_title)]})


class RegressNet(nn.Module):
    def __init__(self, emb_net, mod_channel_num, output_dim):
        super(RegressNet, self).__init__()
        self.emb_net = emb_net
        self.output_dim = output_dim
        self.linear = nn.Linear(emb_net.embedding_dim, output_dim * emb_net.patch_len)

    def forward(self, x, test_flag, tgt=None):
        batch_size, _, seq_len = x[0].shape
        mod_all = torch.concat(x, dim=1)
        # if tgt is not None:       # This is for an encoder-decoder TF
        #     tgt = tgt.repeat(1, mod_all.shape[1] // tgt.shape[1], 1)
        mod_outputs, _, _ = self.emb_net(mod_all, test_flag, tgt)
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


class TransformerEncoderOnly(nn.Module):
    def __init__(self, x_dim, mask_input_channel, mask_patch_num, nlayers, nhead, dim_feedforward,
                 patch_len, patch_step_len, device='cuda'):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step_len = patch_step_len
        self.pad_len = 0
        self.x_dim = x_dim
        self.embedding_dim = 192
        self.input_to_embedding = nn.Linear(x_dim * patch_len, self.embedding_dim)
        self.d_model = int(self.embedding_dim)
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, dim_feedforward, batch_first=True)

        self.transformer = TransformerEncoder(encoder_layers, nlayers)
        self.device = device
        self.mask_patch_num = mask_patch_num
        self.ssl_mask_emb = nn.Parameter(torch.zeros([1, x_dim * patch_len]).uniform_() - 0.5)
        self.reduced_imu_mask_emb = nn.Parameter(torch.zeros([48, 128]).uniform_() - 0.5)
        self.mask_input_channel = mask_input_channel
        self.patch_num = int(128 / self.patch_len)
        self.pos_encoding = PositionalEncoding(self.embedding_dim, self.patch_num)

    def forward(self, sequence, test_flag, _):
        sequence = self.apply_reduced_imu_set_mask(sequence)
        sequence_patch = self.divide_into_patches(sequence)
        sequence_patch, mask_indices = self.apply_ssl_mask(sequence_patch)
        sequence_patch = self.input_to_embedding(sequence_patch)
        sequence_patch = self.pos_encoding(sequence_patch)
        output = self.transformer(sequence_patch)
        return output, mask_indices, sequence

    def divide_into_patches(self, sequence):
        sequence = F.pad(sequence, (0, 0, self.pad_len, self.pad_len))
        sequence = sequence.unfold(2, self.patch_len, self.patch_step_len)
        sequence = sequence.transpose(1, 2).flatten(start_dim=2)
        return sequence

    def apply_reduced_imu_set_mask(self, seq):
        bs, patch_emb, length = seq.shape
        mask_indices_np = np.full((bs, patch_emb), False)
        for i_channel in range(8):
            if self.mask_input_channel[i_channel]:
                mask_indices_np[:, i_channel*3:(i_channel+1)*3] = True
                mask_indices_np[:, i_channel*3+24:(i_channel+1)*3+24] = True
        mask_indices = torch.from_numpy(mask_indices_np).to(seq.device)
        mask_indices = mask_indices.unsqueeze(-1).expand_as(seq)

        # [test]
        # ttt = torch.zeros(self.reduced_imu_mask_emb.shape).cuda()
        # tensor = torch.mul(seq, ~mask_indices) + torch.mul(ttt, mask_indices)
        # temp = tensor.detach().cpu().numpy()

        tensor = torch.mul(seq, ~mask_indices) + torch.mul(self.reduced_imu_mask_emb, mask_indices)
        return tensor

    def apply_ssl_mask(self, seq):
        """

        :param seq: [bs, embedding, time]
        :param mask_type: 0 for mask random samples, 1 for mask one entire patch, 2 for testing
        :return:
        """
        bs, patch_num, emb_dim = seq.shape
        mask_indices_np = np.full((bs, patch_num), False)
        for i_row in range(mask_indices_np.shape[0]):
            len_to_mask = [self.mask_patch_num]     # [self.mask_patch_num]   [int(0.5*self.mask_patch_num), int(0.5*self.mask_patch_num)]
            for len_ in len_to_mask:
                masked_loc = random.sample(range(self.patch_num - len_ + 1), 1)[0]
                mask_indices_np[i_row, masked_loc:masked_loc+len_] = True
        mask_indices = torch.from_numpy(mask_indices_np).to(seq.device)
        mask_indices = mask_indices.unsqueeze(2).expand_as(seq)
        tensor = torch.mul(seq, ~mask_indices) + torch.mul(self.ssl_mask_emb, mask_indices)
        return tensor, mask_indices


def transformer(x_dim, nlayers, nhead, dim_feedforward, mask_input_channel, mask_patch_num, patch_len):
    # return Transformer(x_dim, mask_input_channel=mask_input_channel,
    return TransformerEncoderOnly(x_dim, mask_input_channel=mask_input_channel,
                       mask_patch_num=mask_patch_num, nlayers=nlayers, nhead=nhead, dim_feedforward=dim_feedforward,
                       patch_len=patch_len, patch_step_len=patch_len)

