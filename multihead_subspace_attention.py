import torch
import torch.nn as nn

class MultiHeadSubspaceAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        x_flat = x.view(b, c, n).transpose(1, 2)
        out, _ = self.mha(x_flat, x_flat, x_flat)
        out = self.proj(out).transpose(1, 2).view(b, c, h, w)
        return out
