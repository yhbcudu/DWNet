import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x, h_prev, c_prev):
        if h_prev is None:
            b, _, h, w = x.shape
            h_prev = torch.zeros(b, self.hidden_dim, h, w, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros_like(h_prev)
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=3):
        super().__init__()
        self.num_layers = len(hidden_dims)
        cells = []
        for i, h in enumerate(hidden_dims):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cells.append(ConvLSTMCell(in_dim, h, kernel_size=kernel_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        b, t, c, h, w = x.shape
        hs = [None] * self.num_layers
        cs = [None] * self.num_layers
        outputs = []
        for i in range(t):
            xt = x[:, i]
            for l in range(self.num_layers):
                cell = self.cells[l]
                hs[l], cs[l] = cell(xt, hs[l], cs[l])
                xt = hs[l]
            outputs.append(hs[-1])
        return outputs, hs[-1]

class TemporalAttention(nn.Module):
    def __init__(self, channels, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, features_list):
        b, c, h, w = features_list[0].shape
        t = len(features_list)
        feats = torch.stack(features_list, dim=1)
        gap = self.gap(feats.view(b * t, c, h, w)).view(b, t, c)
        scores = self.fc(gap)
        attn = torch.softmax(scores, dim=1)
        feats_flat = feats.view(b, t, c, h * w)
        attn_expanded = attn.view(b, t, 1, 1)
        fused = (feats_flat * attn_expanded).sum(dim=1).view(b, c, h, w)
        return fused, attn.squeeze(-1)

class SARTemporalEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_dims=(64, 128, 256), kernel_size=3, num_steps=12):
        super().__init__()
        self.convlstm = ConvLSTM(in_channels, list(hidden_dims), kernel_size=kernel_size)
        self.temporal_attn = TemporalAttention(hidden_dims[-1], num_steps=num_steps)

    def forward(self, x):
        outputs, _ = self.convlstm(x)
        fused, attn = self.temporal_attn(outputs)
        return fused, attn
