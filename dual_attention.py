import torch
import torch.nn as nn

class LocalSpatialAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.q_conv = nn.Conv2d(channels, channels // reduction, 3, padding=1)
        self.k_conv = nn.Conv2d(channels, channels // reduction, 3, padding=1)
        self.v_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.scale = (channels // reduction) ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q_conv(x).view(b, -1, h * w)
        k = self.k_conv(x).view(b, -1, h * w)
        v = self.v_conv(x).view(b, c, h * w)
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return out

class GlobalChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc2(torch.relu(self.fc1(y)))
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y

class DualAttention(nn.Module):
    def __init__(self, channels, reduction_spatial=4, reduction_channel=16):
        super().__init__()
        self.local_attn = LocalSpatialAttention(channels, reduction=reduction_spatial)
        self.global_attn = GlobalChannelAttention(channels, reduction=reduction_channel)
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        local_feat = self.local_attn(x)
        global_feat = self.global_attn(x)
        out = torch.cat([local_feat, global_feat], dim=1)
        return self.fuse(out)
