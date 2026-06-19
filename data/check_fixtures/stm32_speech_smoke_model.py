import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, stride=8, padding=7),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=11, stride=5, padding=5),
            nn.ReLU(),
            nn.Conv1d(16, 24, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv1d(24, 32, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
        )
        self.classifier = nn.Conv1d(32, 79, kernel_size=1)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[1] != 1:
            x = x[:, :1, :]
        logits = self.classifier(self.encoder(x))
        return logits.transpose(1, 2)
