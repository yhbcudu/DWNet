import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class DSCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, stride=1)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + identity

class OpticalEncoder(nn.Module):
    def __init__(self, in_channels=4, base_channels=64):
        super().__init__()
        self.stage1 = DSCBlock(in_channels, base_channels, stride=2)
        self.stage2 = DSCBlock(base_channels, base_channels * 2, stride=2)
        self.stage3 = DSCBlock(base_channels * 2, base_channels * 4, stride=2)
        self.stage4 = DSCBlock(base_channels * 4, base_channels * 8, stride=2)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4}
