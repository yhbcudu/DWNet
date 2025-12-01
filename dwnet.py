import torch
import torch.nn as nn

from .optical_dsc_encoder import OpticalEncoder
from .sar_temporal_encoder import SARTemporalEncoder
from .enhanced_aff import EnhancedAFF

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return x

class D_WNet(nn.Module):
    def __init__(self, num_classes, optical_in_channels=4, sar_in_channels=2, sar_timesteps=12, base_channels=64):
        super().__init__()
        self.optical_encoder = OpticalEncoder(in_channels=optical_in_channels, base_channels=base_channels)
        sar_hidden_dims = (64, 128, base_channels * 8)
        self.sar_encoder = SARTemporalEncoder(in_channels=sar_in_channels, hidden_dims=sar_hidden_dims, kernel_size=3, num_steps=sar_timesteps)
        self.fusion = EnhancedAFF(in_channels_opt=base_channels * 8, in_channels_sar=base_channels * 8, fusion_channels=base_channels * 8, num_heads=4)
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, img_opt, img_sar):
        feats_opt = self.optical_encoder(img_opt)
        x1, x2, x3, x4 = feats_opt["x1"], feats_opt["x2"], feats_opt["x3"], feats_opt["x4"]
        feat_sar, _ = self.sar_encoder(img_sar)
        fusion = self.fusion(x4, feat_sar)
        d1 = self.up1(fusion, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        d4 = self.up4(d3)
        logits = self.head(d4)
        return logits
