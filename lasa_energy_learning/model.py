import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        R = self.layers(x).reshape((-1, 2, 2))
        V = x[:, None, :] @ (0.05 * torch.eye(2)[None, ...] + R.transpose(1, 2) @ R) @ x[:, :, None]
        return V
        # return torch.exp(self.layers(x))
