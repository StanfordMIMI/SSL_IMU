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

# TODO: for RNN, Transformer, FCNN, check if the input follows (batch, channel, time_step)

fix_seed()
temperature = 0.1


def nce_loss(emb1, emb2):
    def _calculate_similarity(emb1, emb2):
        # make each vector unit length
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)

        similarity = torch.matmul(emb1, emb2.transpose(1, 2))
        return similarity / temperature

    emb1_pos, emb2_pos = torch.unsqueeze(emb1, 1), torch.unsqueeze(emb2, 1)
    sim_pos = _calculate_similarity(emb1_pos, emb2_pos)
    emb1_all, emb2_all = torch.unsqueeze(emb1, 0), torch.unsqueeze(emb2, 0)
    sim_all = _calculate_similarity(emb1_all, emb2_all)

    logsumexp_pos = torch.flatten(sim_pos)
    logsumexp_all = torch.logsumexp(sim_all, dim=2)
    logsumexp_all = torch.flatten(logsumexp_all)
    loss = (logsumexp_all - logsumexp_pos).mean()
    return loss


class SslContrastiveNet(nn.Module):
    def __init__(self, embnet_acc, embnet_gyr, common_space_dim):
        super(SslContrastiveNet, self).__init__()
        self.embnet_acc = embnet_acc
        self.embnet_gyr = embnet_gyr
        emb_output_dim = embnet_acc.output_dim * 6
        self.linear_proj_imu_1 = nn.Linear(emb_output_dim, emb_output_dim)
        self.linear_proj_emg_1 = nn.Linear(emb_output_dim, emb_output_dim)
        self.linear_proj_imu_2 = nn.Linear(emb_output_dim, common_space_dim)
        self.linear_proj_emg_2 = nn.Linear(emb_output_dim, common_space_dim)
        self.bn_proj_imu_1 = nn.BatchNorm1d(emb_output_dim)
        self.bn_proj_emg_1 = nn.BatchNorm1d(emb_output_dim)
        self.bn_proj_imu_2 = nn.BatchNorm1d(common_space_dim)
        self.bn_proj_emg_2 = nn.BatchNorm1d(common_space_dim)
        self.net_name = 'Combined Net'

    def __str__(self):
        return self.net_name

    def set_scalars(self, scalars):
        self.scalars = scalars

    def forward(self, x_imu, x_emg, lens):
        # seq_imu, _ = self.embnet_imu(x_imu, lens)
        # seq_emg, _ = self.embnet_emg(x_emg, lens)
        # seq_imu = F.relu(self.bn_proj_imu_1(self.linear_proj_imu_1(seq_imu).transpose(1, 2)).transpose(1, 2))
        # seq_emg = F.relu(self.bn_proj_emg_1(self.linear_proj_emg_1(seq_emg).transpose(1, 2)).transpose(1, 2))
        # seq_imu = self.bn_proj_imu_2(self.linear_proj_imu_2(seq_imu).transpose(1, 2)).transpose(1, 2)
        # seq_emg = self.bn_proj_emg_2(self.linear_proj_emg_2(seq_emg).transpose(1, 2)).transpose(1, 2)

        batch_size = x_imu.shape[0]
        half_loc = int(x_imu.shape[2] / 2)
        x_acc = x_imu[:, :, :half_loc]
        x_gyr = x_imu[:, :, half_loc:]

        x_acc = x_acc.transpose(1, 2).unsqueeze(dim=-1)
        x_acc = x_acc.reshape(-1, *x_acc.shape[2:])
        acc_output, _ = self.embnet_acc(x_acc, lens)
        acc_output = acc_output.reshape(batch_size, -1)

        x_gyr = x_gyr.transpose(1, 2).unsqueeze(dim=-1)
        x_gyr = x_gyr.reshape(-1, *x_gyr.shape[2:])
        gyr_output, _ = self.embnet_gyr(x_gyr, lens)
        gyr_output = gyr_output.reshape(batch_size, -1)

        seq_imu = F.relu(self.bn_proj_imu_1(self.linear_proj_imu_1(acc_output)))
        seq_emg = F.relu(self.bn_proj_emg_1(self.linear_proj_emg_1(gyr_output)))
        seq_imu = self.bn_proj_imu_2(self.linear_proj_imu_2(seq_imu))
        seq_emg = self.bn_proj_emg_2(self.linear_proj_emg_2(seq_emg))
        return seq_imu, seq_emg


class SslReconstructNet(nn.Module):
    def __init__(self, embnet_acc, embnet_gyr, emg_dim):
        super(SslReconstructNet, self).__init__()
        self.embnet_acc = embnet_acc
        self.embnet_gyr = embnet_gyr
        emb_output_dim = embnet_acc.output_dim + embnet_gyr.output_dim
        self.linear_proj_1 = nn.Linear(emb_output_dim * 6, emg_dim)
        # self.bn_proj_1 = nn.BatchNorm1d(emg_dim)
        self.net_name = 'Combined Net'

    def __str__(self):
        return self.net_name

    def set_scalars(self, scalars):
        self.scalars = scalars

    def forward(self, x_imu, lens):
        batch_size = x_imu.shape[0]
        half_loc = int(x_imu.shape[2] / 2)
        x_acc = x_imu[:, :, :half_loc]
        x_gyr = x_imu[:, :, half_loc:]

        x_acc = x_acc.transpose(1, 2).unsqueeze(dim=-1)
        x_acc = x_acc.reshape(-1, *x_acc.shape[2:])
        acc_output, _ = self.embnet_acc(x_acc, lens)
        acc_output = acc_output.reshape(batch_size, -1)

        x_gyr = x_gyr.transpose(1, 2).unsqueeze(dim=-1)
        x_gyr = x_gyr.reshape(-1, *x_gyr.shape[2:])
        gyr_output, _ = self.embnet_gyr(x_gyr, lens)
        gyr_output = gyr_output.reshape(batch_size, -1)

        imu_output = torch.concat([acc_output, gyr_output], dim=1)
        output = self.linear_proj_1(imu_output)
        return output


class LinearRegressNet(nn.Module):
    def __init__(self, embnet_imu, _, output_dim):
        super(LinearRegressNet, self).__init__()
        self.embnet_acc = embnet_imu

        self.embnet_gyr = _        # !!!

        emb_output_dim = embnet_imu.output_dim
        self.emb_output_dim = emb_output_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(emb_output_dim * 12, output_dim)

    def forward(self, x_imu, x_emg, lens):
        batch_size = x_imu.shape[0]
        half_loc = int(x_imu.shape[2] / 2)
        x_acc = x_imu[:, :, :half_loc]
        x_gyr = x_imu[:, :, half_loc:]

        x_acc = x_acc.transpose(1, 2).unsqueeze(dim=-1)
        x_acc = x_acc.reshape(-1, *x_acc.shape[2:])
        acc_output, _ = self.embnet_acc(x_acc, lens)
        acc_output = acc_output.reshape(batch_size, -1)

        x_gyr = x_gyr.transpose(1, 2).unsqueeze(dim=-1)
        x_gyr = x_gyr.reshape(-1, *x_gyr.shape[2:])
        gyr_output, _ = self.embnet_gyr(x_gyr, lens)
        gyr_output = gyr_output.reshape(batch_size, -1)

        imu_output = torch.concat([acc_output, gyr_output], dim=1)
        output = self.linear(imu_output)
        return output


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


class ImuFcnnEmbedding(nn.Module):
    def __init__(self, x_dim, output_dim, net_name):
        super(ImuFcnnEmbedding, self).__init__()
        self.net_name = net_name
        self.seq_len = 256
        hid_dim = 256
        self.linear1 = nn.Linear(x_dim * self.seq_len, hid_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.linear2 = nn.Linear(hid_dim, output_dim * self.seq_len)
        self.bn2 = nn.BatchNorm1d(output_dim * self.seq_len)
        self.output_dim = output_dim
        self.ratio_to_base_fre = 1

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):
        original_shape = sequence.shape
        sequence = self.bn1(torch.sigmoid(self.linear1(sequence.view(sequence.shape[0], -1))))
        sequence = self.bn2(torch.sigmoid(self.linear2(sequence)))
        sequence = sequence.view(original_shape[0], original_shape[1], -1)
        return sequence, None


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


class ImuResnetEmbedding(nn.Module):
    def __init__(self, x_dim, output_dim, net_name):
        super(ImuResnetEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(x_dim, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.output_dim = output_dim
        self.net_name = net_name
        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64), RestNetBasicBlock(64, 64))
        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]), RestNetBasicBlock(128, 128))
        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]), RestNetBasicBlock(256, 256))
        # self.layer4 = nn.Sequential(RestNetDownBlock(256, output_dim, [2, 1]), RestNetBasicBlock(output_dim, output_dim))
        self.layer4_1 = RestNetDownBlock(256, output_dim, [2, 1])
        self.layer4_2 = RestNetBasicBlock(output_dim, output_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(512, output_dim)

    def forward(self, sequence, lens):
        out = sequence.transpose(1, 2)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4_2(self.layer4_1(out))
        out = self.avgpool(out)
        return out.squeeze(dim=-1), None


class CnnEmbedding(nn.Module):
    def __init__(self, x_dim, output_dim, net_name):
        super(CnnEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(x_dim, 8, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(8)
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.output_dim = output_dim
        self.net_name = net_name
        self.layer1 = nn.Sequential(RestNetBasicBlock(8, 8), RestNetBasicBlock(8, 8))
        self.layer2 = nn.Sequential(RestNetDownBlock(8, 16, [2, 1]), RestNetBasicBlock(16, 16))
        self.layer3 = nn.Sequential(RestNetDownBlock(16, 32, [2, 1]), RestNetBasicBlock(32, 32))
        self.layer4 = nn.Sequential(RestNetDownBlock(32, output_dim, [2, 1]), RestNetBasicBlock(output_dim, output_dim))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(512, output_dim)

    def forward(self, sequence, lens):
        out = sequence.transpose(-1, -2)
        out = self.conv1(out)
        # if sequence.shape[0] != 128:
        #     print(out.shape)
        out = self.bn1(out)
        # if sequence.shape[0] != 128:
        #     print(out.shape)
        out = self.layer1(out)
        # if sequence.shape[0] != 128:
        #     print(out.shape)
        out = self.layer2(out)
        # if sequence.shape[0] != 128:
        #     print(out.shape)
        out = self.layer3(out)
        # if sequence.shape[0] != 128:
        #     print(out.shape)
        out = self.layer4(out)
        # if sequence.shape[0] != 128:
        #     print(out.shape)
        out = self.avgpool(out)
        # if sequence.shape[0] != 128:
        #     print(round(out.abs().mean().item(), 4))
        return out.squeeze(dim=-1), None




