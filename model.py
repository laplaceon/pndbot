import torch
import torch.nn as nn
import torch.nn.functional as F

class PumpDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=0)
        self.linear1 = nn.Linear(in_features=510, out_features=2)

    def forward(self, x):
        out = self.cnn(x.permute(0, 2, 1))
        out = torch.flatten(out, start_dim=1)
        return self.linear1(F.relu(out))
