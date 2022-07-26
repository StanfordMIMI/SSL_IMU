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

interpo_len = 50
fix_seed()


def nce_loss(emb1, emb2):
    temperature = 0.1       # TODO: optimize this

    def _calculate_similarity(emb1, emb2):
        # make each vector unit length
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)

        # Similarities [B_1, B_2*L].
        similarity = torch.matmul(emb1, emb2.transpose(1, 2))

        # [B_1, B_2, L]
        similarity = torch.flatten(similarity)
        similarity /= temperature
        return similarity

    batch_size, feature_dim = emb1.shape  # B
    emb1_pos, emb2_pos = torch.unsqueeze(emb1, 1), torch.unsqueeze(emb2, 1)
    sim_pos = _calculate_similarity(emb1_pos, emb2_pos)
    emb1_all, emb2_all = torch.unsqueeze(emb1, 0), torch.unsqueeze(emb2, 0)
    sim_all = _calculate_similarity(emb1_all, emb2_all)

    # Compute the log sum exp (numerator) of the NCE loss.
    logsumexp_pos = torch.logsumexp(sim_pos, dim=0)
    # Compute the log sum exp (denominator) of the NCE loss.
    logsumexp_all = torch.logsumexp(sim_all, dim=0)
    # Compute the loss.
    loss = logsumexp_all - logsumexp_pos
    return loss / batch_size


class SslNet(nn.Module):
    def __init__(self, embnet_imu, embnet_emg, common_space_dim):
        super(SslNet, self).__init__()
        self.embnet_imu = embnet_imu
        self.embnet_emg = embnet_emg
        self.interpo_len = 50
        emb_output_dim = embnet_imu.output_dim
        self.linear_proj_imu = nn.Linear(emb_output_dim, common_space_dim)
        self.linear_proj_emg = nn.Linear(emb_output_dim, common_space_dim)
        self.bn_proj_imu = nn.BatchNorm1d(common_space_dim)
        self.bn_proj_emg = nn.BatchNorm1d(common_space_dim)
        self.net_name = 'Combined Net'

    def __str__(self):
        return self.net_name

    def set_scalars(self, scalars):
        self.scalars = scalars

    def forward(self, x_imu, x_emg, lens):
        seq_imu, _ = self.embnet_imu(x_imu, lens)
        seq_emg, _ = self.embnet_emg(x_emg, lens)

        seq_imu = torch.flatten(seq_imu, start_dim=1)
        seq_emg = torch.flatten(seq_emg, start_dim=1)
        seq_imu = self.bn_proj_imu(self.linear_proj_imu(seq_imu))
        seq_emg = self.bn_proj_emg(self.linear_proj_emg(seq_emg))
        return seq_imu, seq_emg


class LinearRegressNet(nn.Module):
    def __init__(self, embnet_imu, embnet_emg, output_dim):
        super(LinearRegressNet, self).__init__()
        self.embnet_imu = embnet_imu
        self.embnet_emg = embnet_emg
        emb_output_dim = embnet_imu.output_dim
        self.linear = nn.Linear(emb_output_dim * 2, output_dim, bias=False)

    def forward(self, x_imu, x_emg, lens):
        seq_imu, _ = self.embnet_imu(x_imu, lens)
        seq_emg, _ = self.embnet_emg(x_emg, lens)
        seq_combined = torch.cat([seq_imu, seq_emg], dim=2)
        output = self.linear(seq_combined)
        return output


class LinearTestNet(nn.Module):
    def __init__(self, embnet_imu, embnet_emg, output_dim):
        super(LinearTestNet, self).__init__()
        self.embnet_imu = embnet_imu
        self.embnet_emg = embnet_emg
        emb_output_dim = embnet_imu.output_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(emb_output_dim * 2, output_dim)

    def forward(self, x_imu, x_emg, lens):
        seq_imu, _ = self.embnet_imu(x_imu, lens)
        seq_emg, _ = self.embnet_emg(x_emg, lens)
        seq_combined = torch.cat([seq_imu, seq_emg], dim=2)
        seq_combined = torch.flatten(seq_combined, start_dim=1)
        output = self.linear(seq_combined)
        # output = output.view([-1, interpo_len, self.output_dim])
        return output


class ImuRnnEmbedding(nn.Module):
    def __init__(self, x_dim, output_dim, net_name, nlayer=2, linear_dim=64):
        super(ImuRnnEmbedding, self).__init__()
        self.net_name = net_name
        self.rnn_layer = nn.LSTM(x_dim, linear_dim, nlayer, bidirectional=True, bias=True)
        self.linear = nn.Linear(linear_dim*2, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.output_dim = output_dim
        self.ratio_to_base_fre = 1
        for name, param in self.rnn_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):
        sequence = sequence.transpose(0, 1)
        total_len = sequence.shape[0]
        lens = lens * self.ratio_to_base_fre
        sequence = pack_padded_sequence(sequence, lens, enforce_sorted=False)
        sequence, (emb, _) = self.rnn_layer(sequence)
        sequence, _ = pad_packed_sequence(sequence, total_length=total_len)
        sequence = self.down_sampling_via_pooling(sequence)
        sequence = F.relu(self.linear(sequence))
        sequence = sequence.transpose(0, 1).transpose(1, 2)
        sequence = self.bn(sequence)
        sequence = sequence.transpose(1, 2)
        emb = emb[-2:]
        emb = emb.transpose(0, 1)
        emb = torch.flatten(emb, start_dim=1)
        emb = emb.transpose(0, 1)
        return sequence, emb

    def down_sampling_via_pooling(self, sequence):
        return sequence


class EmgRnnEmbedding(ImuRnnEmbedding):
    def __init__(self, *args, **kwargs):
        super(EmgRnnEmbedding, self).__init__(*args, **kwargs)
        self.pooling = nn.AvgPool1d(5, 5)
        self.ratio_to_base_fre = 5

    def down_sampling_via_pooling(self, sequence):
        sequence = sequence.transpose(0, 2)
        sequence = self.pooling(sequence)
        sequence = sequence.transpose(0, 2)
        return sequence


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
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.output_dim = output_dim
        self.linear = nn.Linear(d_model, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.device = device

    def forward(self, sequence, lens):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        sequence = sequence.transpose(0, 1)
        # sequence = self.pos_encoder(sequence)
        msk = self.generate_padding_mask(lens, sequence.shape[0])
        output = self.transformer_encoder(sequence, src_key_padding_mask=msk)
        output = self.down_sampling_via_pooling(output)
        output = F.relu(self.linear(output))
        output = output.transpose(0, 1).transpose(1, 2)
        output = self.bn(output)
        output = output.transpose(1, 2)
        return output, None

    def generate_padding_mask(self, lens, seq_len):
        msk = np.ones([len(lens), seq_len], dtype=bool)
        for i, the_len in enumerate(lens):
            msk[i, :int(the_len)] = 0
        return torch.from_numpy(msk).to(self.device)

    def down_sampling_via_pooling(self, sequence):
        return sequence


class EmgTransformerEmbedding(ImuTransformerEmbedding):
    def __init__(self, *args, **kwargs):
        super(EmgTransformerEmbedding, self).__init__(*args, **kwargs)
        self.pooling = nn.AvgPool1d(5, 5)

    def down_sampling_via_pooling(self, sequence):
        sequence = sequence.transpose(0, 2)
        sequence = self.pooling(sequence)
        sequence = sequence.transpose(0, 2)
        return sequence


class ImuFcnnEmbedding(nn.Module):
    def __init__(self, x_dim, output_dim, net_name):
        super(ImuFcnnEmbedding, self).__init__()
        self.net_name = net_name
        self.interpo_len = 50
        hid_dim = 256
        self.linear1 = nn.Linear(x_dim * self.interpo_len, hid_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.linear2 = nn.Linear(hid_dim, output_dim * self.interpo_len)
        self.bn2 = nn.BatchNorm1d(output_dim * self.interpo_len)
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


class EmgFcnnEmbedding(ImuFcnnEmbedding):
    pass


"""Implementation 1"""


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
        return out, None


class ImuCnnEmbedding(nn.Module):
    def __init__(self, x_dim, output_dim, net_name):
        super(ImuCnnEmbedding, self).__init__()
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
        out = sequence.transpose(1, 2)
        out = self.conv1(out)
        if sequence.shape[0] != 128:
            print(round(out.abs().mean().item(), 4), end=' ')
        out = self.bn1(out)
        if sequence.shape[0] != 128:
            print(round(out.abs().mean().item(), 4), end=' ')
        out = self.layer1(out)
        if sequence.shape[0] != 128:
            print(round(out.abs().mean().item(), 4), end=' ')
        out = self.layer2(out)
        if sequence.shape[0] != 128:
            print(round(out.abs().mean().item(), 4), end=' ')
        out = self.layer3(out)
        if sequence.shape[0] != 128:
            print(round(out.abs().mean().item(), 4), end=' ')
        out = self.layer4(out)
        if sequence.shape[0] != 128:
            print(round(out.abs().mean().item(), 4), end=' ')
        out = self.avgpool(out)
        if sequence.shape[0] != 128:
            print(round(out.abs().mean().item(), 4))
        return out, None




