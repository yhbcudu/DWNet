import torch
import torch.nn as nn

from .dense_feature_aggregation import DenseFeatureAggregation
from .dual_attention import DualAttention
from .multihead_subspace_attention import MultiHeadSubspaceAttention
from .contextual_residual_fusion import ContextualResidualFusion

class EnhancedAFF(nn.Module):
    def __init__(self, in_channels_opt, in_channels_sar, fusion_channels=256, num_heads=4):
        super().__init__()
        self.align_opt = nn.Conv2d(in_channels_opt, fusion_channels // 2, 1)
        self.align_sar = nn.Conv2d(in_channels_sar, fusion_channels // 2, 1)
        self.dense_agg = DenseFeatureAggregation(fusion_channels, growth_rate=32, num_layers=4)
        self.dual_attn = DualAttention(fusion_channels)
        self.subspace_attn = MultiHeadSubspaceAttention(fusion_channels, num_heads=num_heads)
        self.ctx_res_fusion = ContextualResidualFusion(fusion_channels)

    def forward(self, feat_opt, feat_sar):
        opt = self.align_opt(feat_opt)
        sar = self.align_sar(feat_sar)
        x = torch.cat([opt, sar], dim=1)
        x = self.dense_agg(x)
        x = self.dual_attn(x)
        x = self.subspace_attn(x)
        x = self.ctx_res_fusion(x)
        return x
