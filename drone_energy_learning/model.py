import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 128, 1)
        self.conv2 = nn.Conv1d(128, 6, 1)
        self.encoder = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=128, activation='gelu')
        self.goal = torch.nn.Parameter(torch.randn(18))

    def forward(self, x):
        x_f = - x.flatten(1, 2)
        temp = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        temp = self.encoder(temp)
        temp = self.conv2(temp.transpose(1, 2)).transpose(1, 2)
        temp = temp.flatten(1, 2)
        R = temp[:, :, None] @ temp[:, None, :]
        P = (0.05 * torch.eye(x_f.shape[1])[None, ...] + R.transpose(1, 2) @ R)
        V = x_f[:, None, :] @ P @ x_f[:, :, None]
        return V
