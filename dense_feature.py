import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.act(self.bn(x)))
        return torch.cat([x, out], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers=4):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DenseFeatureAggregation(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.block1 = DenseBlock(in_channels, growth_rate, num_layers)
        self.block2 = DenseBlock(self.block1.out_channels, growth_rate, num_layers)
        self.block3 = DenseBlock(self.block2.out_channels, growth_rate, num_layers)
        self.proj = nn.Conv2d(self.block3.out_channels, in_channels, 1)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(b1) + b1
        b3 = self.block3(b2) + b2
        return self.proj(b3)
