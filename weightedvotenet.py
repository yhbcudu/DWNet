import torch
import torch.nn as nn

from .unet import UNet
from .convlstm_seg import ConvLSTMSeg

class WeightedVoteNet(nn.Module):
    def __init__(self, num_classes, optical_in_channels=4, sar_in_channels=2, sar_timesteps=12, base_channels=64):
        super().__init__()
        self.opt_net = UNet(n_channels=optical_in_channels, n_classes=num_classes, base_channels=base_channels)
        self.sar_net = ConvLSTMSeg(num_classes=num_classes, sar_in_channels=sar_in_channels, sar_timesteps=sar_timesteps, base_channels=base_channels)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, img_opt, img_sar):
        logits_opt = self.opt_net(img_opt)
        logits_sar = self.sar_net(img_sar)
        weights = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)
        a, b = weights[0], weights[1]
        return a * logits_opt + b * logits_sar
