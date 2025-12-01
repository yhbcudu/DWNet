import torch
import torch.nn as nn

from .optical_dsc_encoder import OpticalEncoder
from .sar_temporal_encoder import SARTemporalEncoder
from .dwnet import UpBlock

class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, feat_opt, feat_sar):
        b, c, h, w = feat_opt.shape
        n = h * w
        opt_flat = feat_opt.view(b, c, n).transpose(1, 2)
        sar_flat = feat_sar.view(b, c, n).transpose(1, 2)
        q = self.q_proj(opt_flat)
        k = self.k_proj(sar_flat)
        v = self.v_proj(sar_flat)
        out, _ = self.mha(q, k, v)
        out = self.out_proj(out).transpose(1, 2).view(b, c, h, w)
        return out

class CrossAttentionNet(nn.Module):
    def __init__(self, num_classes, optical_in_channels=4, sar_in_channels=2, sar_timesteps=12, base_channels=64):
        super().__init__()
        self.optical_encoder = OpticalEncoder(in_channels=optical_in_channels, base_channels=base_channels)
        sar_hidden_dims = (64, 128, base_channels * 8)
        self.sar_encoder = SARTemporalEncoder(in_channels=sar_in_channels, hidden_dims=sar_hidden_dims, kernel_size=3, num_steps=sar_timesteps)
        self.ca = CrossAttentionBlock(base_channels * 8, num_heads=4)
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, img_opt, img_sar):
        feats_opt = self.optical_encoder(img_opt)
        x1, x2, x3, x4 = feats_opt["x1"], feats_opt["x2"], feats_opt["x3"], feats_opt["x4"]
        feat_sar, _ = self.sar_encoder(img_sar)
        fusion = self.ca(x4, feat_sar)
        d1 = self.up1(fusion, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        d4 = self.up4(d3)
        return self.head(d4)
