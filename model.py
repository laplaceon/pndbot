import torch
import torch.nn as nn
import torch.nn.functional as F

class PumpDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=0)
        self.cnn2 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=4, stride=2, padding=0)
        self.linear1 = nn.Linear(in_features=126, out_features=2)

    def forward(self, x):
        out = F.relu(self.cnn1(x.permute(0, 2, 1)))
        out = F.relu(self.cnn2(out))
        out = torch.flatten(out, start_dim=1)
        return self.linear1(out)

class TradeSequenceEncoder(nn.Module):
    def __init__(self, d_m, n_rnn, heads, t_dim, kernel):
        super(TradeSequenceEncoder, self).__init__()
        self.n_rnn = n_rnn
        self.d_m = d_m
        # self.pos_encoder = nn.Embedding(seq_len - kernel + 1, t_dim)
        self.cnn = nn.Conv1d(in_channels=4, out_channels=t_dim, kernel_size=kernel, stride=1)
        # self.rnn = nn.LSTM(input_size=4, hidden_size=t_dim, num_layers=n_rnn)
        # self.rnn = nn.LSTM(input_size=4, hidden_size=t_dim, num_layers=n_rnn, dropout=0.1)
        enc_layer = nn.TransformerEncoderLayer(d_model=t_dim, nhead=heads, dim_feedforward=d_m)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_rnn)

    def forward(self, input):
        # CNN
        conv = self.cnn(input.permute(0, 2, 1))
        encoded = self.enc(conv.permute(2, 0, 1))

        # LSTM with transformer
        # out, _ = self.rnn(input.permute(1, 0, 2))
        # encoded = self.enc(out)

        # LSTM
        # out, _ = self.rnn(input.permute(1, 0, 2))
        # encoded = out[-1]

        return encoded

class PndModel(nn.Module):
    def __init__(self, n_rnn=1, nheads=4, d_m=32, t_dim=36, kernel=8):
        super(PndModel, self).__init__()
        self.encoder = TradeSequenceEncoder(d_m, n_rnn, nheads, t_dim, kernel)
        dec_layer = nn.TransformerDecoderLayer(d_model=4, nhead=nheads, dim_feedforward=d_m)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_rnn)
        self.classifiers = nn.ModuleList([nn.Linear(in_features=t_dim, out_features=2), nn.Linear(in_features=t_dim, out_features=4)])

    def forward(self, input, next):
        encoding = self.encoder(input)
        decoded = self.classifiers[1](encoding)
        return (self.classifiers[0](encoding.mean(0)), self.decoder(next.permute(1, 0, 2), decoded))
        # return (self.classifiers[0](encoding.mean(0)), None)
