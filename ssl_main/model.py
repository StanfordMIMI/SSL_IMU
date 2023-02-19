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


fix_seed()


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
    group_len = 8
    assert sim_all.shape[0] % group_len == 0
    group_num = int(sim_all.shape[0] / group_len)
    mask = get_group_negative_mask(sim_all.shape, group_len).to("cuda")
    sim_all = sim_all.masked_select(mask).view(-1, mask.shape[0] - group_len)
    logsumexp_all = torch.logsumexp(sim_all, dim=1)
    logsumexp_all = torch.flatten(torch.mean(logsumexp_all, dim=0))
    loss = logsumexp_all - logsumexp_pos
    return loss


def nce_loss_each_segment_individually(mod_outputs, temperature):
    def _calculate_similarity(emb1, emb2):
        # make each vector unit length
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        similarity = torch.matmul(emb1, emb2.transpose(1, 2))
        return similarity / temperature

    group_len = 8
    losses = []
    for i_group in range(group_len):
        sub_mod_outputs = [mod_output[i_group::group_len] for mod_output in mod_outputs]
        sim_pos = _calculate_similarity(torch.unsqueeze(sub_mod_outputs[0], 1), torch.unsqueeze(sub_mod_outputs[1], 1))
        logsumexp_pos = torch.flatten(torch.mean(sim_pos))
        sim_all = _calculate_similarity(torch.unsqueeze(sub_mod_outputs[0], 0), torch.unsqueeze(sub_mod_outputs[1], 0))[0]

        logsumexp_all = torch.logsumexp(sim_all, dim=1)
        logsumexp_all = torch.flatten(torch.mean(logsumexp_all, dim=0))
        losses.append(logsumexp_all - logsumexp_pos)
    loss = [losses[0] + loss_ for loss_ in losses[1:]][0]
    return loss


def mse_loss(mod_outputs, mods):
    losses = [torch.square(mod - mod_output) for mod_output, mod in zip(mod_outputs, mods)]
    loss = [losses[0].mean() + loss_.mean() for loss_ in losses[1:]][0]
    return loss


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
        emb_output_dim = emb_net.output_dim * (mod_channel_num / 3)
        self.emb_output_dim = len(_mods) * 24
        self.output_dim = output_dim
        self.linear = nn.Linear(self.emb_output_dim, output_dim)

    def forward(self, x, lens):
        batch_size, _, seq_len = x[0].shape
        mod_outputs = []
        for i_mod, x_mod in enumerate(x):
            x_mod = x_mod.view(-1, 3, *x_mod.shape[2:])
            mod_output, _ = self.emb_net(x_mod, lens)
            mod_outputs.append(mod_output.view(batch_size, -1, seq_len))

        output = torch.concat(mod_outputs, dim=1)
        output = output.transpose(1, 2)
        output = self.linear(output)
        output = output.transpose(1, 2)
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


class SslGenerativeNet(SslContrastiveNet):
    def forward(self, mods, lens):
        def reshape_and_emb(mod_, embnet, linear_proj_1, linear_proj_2, bn_proj_1):
            mod_ = mod_.view(-1, 3, *mod_.shape[2:])
            mod_, _ = embnet(mod_, lens)
            return mod_
        mod_outputs = []
        for i_mod, mod in enumerate(mods):
            mod_outputs.append(reshape_and_emb(
                mod, self.emb_net, self.linear_proj_1[i_mod], self.linear_proj_2[i_mod], self.bn_proj_1[i_mod]))
        loss = self.loss_fn(mod_outputs, [mod_.view(-1, 3, *mod_.shape[2:]) for mod_ in mods])

        return loss, mod_outputs


class TransformerBase(nn.Module):
    def __init__(self, d_model, output_dim, nlayers, nhead, d_hid, dropout, patch_len, patch_step_len, device='cuda'):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step_len = patch_step_len
        self.pad_len = 0
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = patch_len*3
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.output_dim = output_dim
        # self.linear = nn.Linear(self.d_model, output_dim * self.patch_step_len)
        self.device = device

        self.mask_prob = 0.1
        # self.mask_length = self.patch_len
        self.mask_emb = nn.Parameter(torch.zeros([48]).uniform_()-0.5)

    def forward(self, sequence, lens):
        sequence = self.divide_into_patches(sequence)
        sequence, mask_indices = self.apply_mask(sequence)
        output = self.transformer_encoder(sequence)
        output = self.flat_patches(output)
        return output, None

    def divide_into_patches(self, sequence):
        sequence = F.pad(sequence, (0, 0, self.pad_len, self.pad_len))
        sequence = sequence.unfold(2, self.patch_len, self.patch_step_len)
        sequence = sequence.transpose(1, 2).flatten(start_dim=2)
        return sequence

    def flat_patches(self, sequence):
        sequence = sequence.view([*sequence.shape[:2], 3, -1])
        sequence = sequence.transpose(1, 2).flatten(start_dim=2)
        return sequence

    def apply_mask(self, seq):
        bs, patches, length = seq.shape
        mask_indices_np = np.random.choice(a=[True, False], size=(bs, patches), p=[self.mask_prob, 1 - self.mask_prob])
        mask_indices = torch.from_numpy(mask_indices_np).to(seq.device)

        for _ in range(mask_indices.dim(), seq.dim()):
            mask_indices = mask_indices.unsqueeze(-1)
        if mask_indices.size(-1) < seq.size(-1):
            mask_indices = mask_indices.expand_as(seq)
        # t1 = torch.mul(seq, ~mask_indices).detach().cpu().numpy()
        # t2 = torch.mul(self.mask_emb, mask_indices).detach().cpu().numpy()
        tensor = torch.mul(seq, ~mask_indices) + torch.mul(self.mask_emb, mask_indices)

        # masked_patch_list = random.sample(range(patches), int(patches * self.mask_prob))
        # mask = np.full(x.shape, False)
        # for masked_patch in masked_patch_list:
        #     mask[:, :, masked_patch] = True
        # mask_indices = torch.from_numpy(mask).to(x.device)
        return tensor, mask_indices


def transformer(x_dim, output_dim):
    return TransformerBase(x_dim, output_dim,       # , mask_prob=50, mask_length=50, mask_selection=50, mask_other=50
                           nlayers=4, nhead=4, d_hid=20, dropout=0, patch_len=16, patch_step_len=16)


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
