import torch
import torch.nn as nn

class KCCAFusionNet(nn.Module):
    def __init__(self, dim_opt, dim_sar, hidden_dim=128, num_classes=6):
        super().__init__()
        self.opt_proj = nn.Sequential(
            nn.Linear(dim_opt, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.sar_proj = nn.Sequential(
            nn.Linear(dim_sar, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feat_opt, feat_sar):
        z_opt = self.opt_proj(feat_opt)
        z_sar = self.sar_proj(feat_sar)
        z = torch.cat([z_opt, z_sar], dim=1)
        return self.classifier(z)
