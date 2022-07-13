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
# from scipy.interpolate import interp1d


fix_seed()


def nce_loss(emb1, emb2):
    temperature = 0.1       # TODO: optimize this

    def _calculate_similarity(emb1, emb2):
        emb1 = F.normalize(emb1, p=2, dim=-1)        # TODO: try if removing this could lead to better results
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


class SslTestNet(nn.Module):
    def __init__(self, embnet_imu, embnet_emg, common_space_dim):
        super(SslTestNet, self).__init__()
        self.embnet_imu = embnet_imu
        self.embnet_emg = embnet_emg
        # emb_output_dim = embnet_imu.output_dim
        self.linear_proj_imu = nn.Linear(9600, common_space_dim)
        self.linear_proj_emg = nn.Linear(9600, common_space_dim)        # 9600
        self.net_name = 'Combined Net'

    def __str__(self):
        return self.net_name

    def set_scalars(self, scalars):
        self.scalars = scalars

    def forward(self, x_imu, x_emg, lens):
        def interpo_data(tensor):
            for i_step in range(tensor.shape[0]):
                tensor = tensor.transpose(1, 2)
                tensor[i_step:i_step+1, :, :new_seq_len] = F.interpolate(
                    tensor[i_step:i_step+1, :, :int(lens[i_step])], new_seq_len, mode='linear', align_corners=False)
                tensor = tensor.transpose(1, 2)
            return tensor[:, :new_seq_len, :]

        seq_imu, _ = self.embnet_imu(x_imu, lens)
        seq_emg, _ = self.embnet_emg(x_emg, lens)
        new_seq_len = 150
        seq_imu = interpo_data(seq_imu)
        seq_emg = interpo_data(seq_emg)

        # temp_imu = seq_imu          # !!!
        seq_imu = torch.flatten(seq_imu, start_dim=1)
        seq_emg = torch.flatten(seq_emg, start_dim=1)
        seq_imu = self.linear_proj_imu(seq_imu)
        seq_emg = self.linear_proj_imu(seq_emg)
        return seq_imu, seq_emg, None          # !!!


class LinearTestNet(nn.Module):
    def __init__(self, embnet_imu, embnet_emg, output_dim):
        super(LinearTestNet, self).__init__()
        self.embnet_imu = embnet_imu
        self.embnet_emg = embnet_emg
        emb_output_dim = embnet_imu.output_dim
        self.linear = nn.Linear(emb_output_dim * 2, output_dim)

    def forward(self, x_imu, x_emg, lens):
        seq_imu, _ = self.embnet_imu(x_imu, lens)
        seq_emg, _ = self.embnet_emg(x_emg, lens)
        seq_combined = torch.cat([seq_imu, seq_emg], dim=2)         # !!!
        output = self.linear(seq_combined)
        return output


class ImuTestRnnEmbedding(nn.Module):
    def __init__(self, x_dim, output_dim, net_name, nlayer=1):
        super(ImuTestRnnEmbedding, self).__init__()
        self.net_name = net_name
        self.rnn_layer = nn.LSTM(x_dim, output_dim, nlayer, bidirectional=True)
        self.output_dim = output_dim * 2
        for name, param in self.rnn_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):      # TODO: use a transformer to replace this
        sequence = sequence.transpose(0, 1)
        total_len = sequence.shape[0]
        sequence = pack_padded_sequence(sequence, lens, enforce_sorted=False)
        sequence, (emb, _) = self.rnn_layer(sequence)
        sequence, _ = pad_packed_sequence(sequence, total_length=total_len)
        sequence = self.down_sampling_via_pooling(sequence)
        emb = emb[-2:]
        emb = emb.transpose(0, 1)
        emb = torch.flatten(emb, start_dim=1)
        sequence = sequence.transpose(0, 1)
        emb = emb.transpose(0, 1)
        return sequence, emb

    def down_sampling_via_pooling(self, sequence):
        return sequence


class EmgTestRnnEmbedding(ImuTestRnnEmbedding):
    def __init__(self, *args, **kwargs):
        super(EmgTestRnnEmbedding, self).__init__(*args, **kwargs)
        self.pooling = nn.AvgPool1d(5, 5)

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


class ImuTestTransformerEmbedding(nn.Module):
    def __init__(self, d_model, output_dim, net_name, nlayers=1, nhead=1, d_hid=20, dropout=0, device='cuda'):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.output_dim = output_dim
        self.linear = nn.Linear(d_model, output_dim)
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
        src = self.pos_encoder(sequence)
        msk = self.generate_padding_mask(lens, sequence.shape[0])
        output = self.transformer_encoder(src, src_key_padding_mask=msk)
        output = self.linear(output)
        output = self.down_sampling_via_pooling(output)
        output = output.transpose(0, 1)
        return output, None

    # def generate_square_subsequent_mask(self, sz):
    #     """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    #     self.src_mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def generate_padding_mask(self, lens, seq_len):
        msk = np.ones([len(lens), seq_len], dtype=bool)
        for i, the_len in enumerate(lens):
            msk[i, :int(the_len)] = 0
        return torch.from_numpy(msk).to(self.device)

    def down_sampling_via_pooling(self, sequence):
        return sequence


class EmgTestTransformerEmbedding(ImuTestTransformerEmbedding):
    def __init__(self, *args, **kwargs):
        super(EmgTestTransformerEmbedding, self).__init__(*args, **kwargs)
        self.pooling = nn.AvgPool1d(5, 5)

    def down_sampling_via_pooling(self, sequence):
        sequence = sequence.transpose(0, 2)
        sequence = self.pooling(sequence)
        sequence = sequence.transpose(0, 2)
        return sequence









