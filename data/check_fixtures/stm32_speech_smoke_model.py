import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(32, 192))
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        if x.ndim == 3:
            batch = x.shape[0]
            signal = x.mean(dim=(1, 2), keepdim=True)
        else:
            batch = x.shape[0]
            signal = x.mean(dim=1, keepdim=True).unsqueeze(-1)
        return self.bias.unsqueeze(0).expand(batch, -1, -1) + self.scale * signal
