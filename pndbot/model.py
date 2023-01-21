import torch
import torch.nn as nn
import torch.nn.functional as F

class PndModel(nn.Module):
    def __init__(
        self,
        h_dims=(4, 6, 8),
        scales=(2, 2, 2),
        blocks_per_stages=(1, 1, 1),
        layers_per_blocks=(2, 2, 2)
    ):
        super().__init__()

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            encoder_stage = ResStage(in_channels, out_channels, 3, scales[i], blocks_per_stages[i], layers_per_blocks[i])
            stages.append(encoder_stage)

        self.conv = nn.Sequential(*stages)
        self.lstm = nn.LSTM(input_size=h_dims[-1], hidden_size=16, num_layers=2)

        self.linear1 = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        out = self.conv(x.permute(0, 2, 1))
        out, _ = self.lstm(out.permute(2, 0, 1))
        return self.linear1(out[-1])

class ResStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        scale,
        num_blocks,
        layers_per_block
    ):
        super().__init__()

        self.downscale = nn.Conv1d(in_channels, out_channels, kernel_size=scale+1, stride=scale, padding=1)

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    kernel,
                    layers_per_block,
                    groups=2
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.downscale(x)
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
        activation=nn.SiLU()
    ):
        super().__init__()

        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=kernel, dilation=dilation, padding="same")
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=kernel, dilation=dilation, padding="same")

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.conv_res = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=dilation, padding="same")

        self.activation = activation

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))

        return self.activation(h + self.conv_res(x))

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