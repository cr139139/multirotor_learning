from dataloader import drone_data
import torch
import torch.nn as nn
from model import Model

data = drone_data()

print(data.X_t0.shape)


encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=128, activation='gelu')
m = nn.Conv1d(6, 128, 1)

input = data.X_t0.float()
input_f = input.flatten(1, 2)
print((input_f[:, :, None] @ input_f[:, None, :]).shape)


input = m(input.transpose(1, 2)).transpose(1, 2)
input = encoder_layer(input)


model = Model()



print(model(data.X_t0.float()).shape)