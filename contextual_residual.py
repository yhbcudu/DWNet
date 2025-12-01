import torch
import torch.nn as nn

class MultiScaleContextExtractor(nn.Module):
    def __init__(self, channels, dilation_rates=(1, 2, 4)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d, bias=False)
            for d in dilation_rates
        ])
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = 0
        for conv in self.convs:
            out = out + conv(x)
        out = self.bn(out)
        return self.act(out)

class ResidualRefinementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)

class ContextualResidualFusion(nn.Module):
    def __init__(self, channels, dilation_rates=(1, 2, 4)):
        super().__init__()
        self.context = MultiScaleContextExtractor(channels, dilation_rates=dilation_rates)
        self.refine = ResidualRefinementBlock(channels)

    def forward(self, x):
        ctx = self.context(x)
        return self.refine(ctx + x)
