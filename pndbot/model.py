import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

class PndModelMLP(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        self.linear1 = nn.Linear(2, 8)
        self.linear2 = nn.Linear(8, 2)
    
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out

class PndModelLSTM(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        self.lstm = nn.LSTM(input_size=4, hidden_size=64, num_layers=3)
        self.fc = nn.Linear(64, 2)
    
    def forward(self, x):
        out, _ = self.lstm(x.permute(1, 0, 2))
        out = self.fc(out[-1])
        return out

class PndModel(nn.Module):
    def __init__(
        self,
        h_dims=(7, 16, 16, 16),
        scales=(2, 2, 2),
        blocks_per_stages=(2, 2, 2)
    ):
        super().__init__()

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            encoder_stage = ResStage(in_channels, out_channels, 5, scales[i], blocks_per_stages[i])
            stages.append(encoder_stage)

        self.conv = nn.Sequential(*stages)
        self.lstm1 = nn.LSTM(input_size=h_dims[-1], hidden_size=h_dims[-1], num_layers=2)
        self.drop = nn.Dropout(p=0.1)

        self.linear1 = nn.Linear(in_features=h_dims[-1], out_features=2)

    def forward(self, x):
        h = self.conv(x.permute(0, 2, 1))
        out1, _ = self.lstm1(h.permute(2, 0, 1))
        out1 = self.linear1(self.drop(out1[-1]))
        
        return out1

class ResStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        scale,
        num_blocks
    ):
        super().__init__()

        self.downscale = nn.Conv1d(in_channels, out_channels, kernel_size=scale+1, stride=scale, padding=1)

        # self.attn = nn.MultiheadAttention(out_channels, 4, dropout=0.1, batch_first=True)

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    kernel,
                    groups=4
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.downscale(x)

        # out = out.permute(0, 2, 1)
        # attn, _ = self.attn(out, out, out)
        # out += attn
        # out.permute(0, 2, 1)

        out = self.blocks(out)
        return out

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        dilation=1,
        groups=8,
        activation=nn.ReLU()
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=kernel, dilation=dilation, padding="same"),
            nn.GroupNorm(groups, out_channels),
            activation
        )
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=kernel, dilation=dilation, padding="same"),
            nn.GroupNorm(groups, out_channels)
        )

        self.conv_res = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=dilation, padding="same")

        self.activation = activation

    def forward(self, x):
        residual = x
        h = self.conv1(x)
        h = self.conv2(h)

        return self.activation(h + residual)

class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        padding=1,
        bias=False
    ):
        super().__init__()

        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class PointwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=False
    ):
        super().__init__()

        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=1, bias=bias)

    def forward(self, x):
        return self.pointwise(x)

class DepthwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        padding=1,
        bias=False
    ):
        super().__init__()

        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.depthwise(x)