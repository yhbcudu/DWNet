import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

def _upsample_like(src, target):
    return F.interpolate(src, size=target.shape[2:], mode="bilinear", align_corners=False)

def _downsample_like(src, target):
    return F.adaptive_avg_pool2d(src, output_size=target.shape[2:])

class UNet3Plus(nn.Module):
    def __init__(self, in_channels=4, n_classes=6, base_channels=64):
        super().__init__()
        c = base_channels
        self.conv1 = ConvBlock(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(c * 4, c * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = ConvBlock(c * 8, c * 16)

        cat_channels = c
        self.stage4d = ConvBlock(cat_channels * 5, c * 8)
        self.stage3d = ConvBlock(cat_channels * 5, c * 4)
        self.stage2d = ConvBlock(cat_channels * 5, c * 2)
        self.stage1d = ConvBlock(cat_channels * 5, c)

        self.outc = nn.Conv2d(c, n_classes, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.conv5(self.pool4(x4))

        x1_4 = _downsample_like(x1, x4)
        x2_4 = _downsample_like(x2, x4)
        x3_4 = _downsample_like(x3, x4)
        x4_4 = x4
        x5_4 = _upsample_like(x5, x4)
        x4d = self.stage4d(torch.cat([x1_4, x2_4, x3_4, x4_4, x5_4], dim=1))

        x1_3 = _downsample_like(x1, x3)
        x2_3 = _downsample_like(x2, x3)
        x3_3 = x3
        x4_3 = _upsample_like(x4d, x3)
        x5_3 = _upsample_like(x5, x3)
        x3d = self.stage3d(torch.cat([x1_3, x2_3, x3_3, x4_3, x5_3], dim=1))

        x1_2 = _downsample_like(x1, x2)
        x2_2 = x2
        x3_2 = _upsample_like(x3d, x2)
        x4_2 = _upsample_like(x4d, x2)
        x5_2 = _upsample_like(x5, x2)
        x2d = self.stage2d(torch.cat([x1_2, x2_2, x3_2, x4_2, x5_2], dim=1))

        x1_1 = x1
        x2_1 = _upsample_like(x2d, x1)
        x3_1 = _upsample_like(x3d, x1)
        x4_1 = _upsample_like(x4d, x1)
        x5_1 = _upsample_like(x5, x1)
        x1d = self.stage1d(torch.cat([x1_1, x2_1, x3_1, x4_1, x5_1], dim=1))

        return self.outc(x1d)
