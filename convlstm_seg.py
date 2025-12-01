import torch
import torch.nn as nn

from .sar_temporal_encoder import SARTemporalEncoder

class ConvLSTMSeg(nn.Module):
    def __init__(self, num_classes, sar_in_channels=2, sar_timesteps=12, base_channels=64):
        super().__init__()
        hidden_dims = (64, 128, base_channels * 2)
        self.encoder = SARTemporalEncoder(in_channels=sar_in_channels, hidden_dims=hidden_dims, kernel_size=3, num_steps=sar_timesteps)
        self.conv1 = nn.Conv2d(hidden_dims[-1], base_channels * 2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels * 2)
        self.conv2 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, sar_seq):
        feat_sar, _ = self.encoder(sar_seq)
        x = self.act(self.bn1(self.conv1(feat_sar)))
        x = self.act(self.bn2(self.conv2(x)))
        return self.head(x)
