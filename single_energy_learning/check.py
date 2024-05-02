import torch
import numpy as np
from model import Model

model = Model()
model.load_state_dict(torch.load('model.pth'))

state = np.ones((1, 1, 6))  # batch x n_drones x 6 (states)

input = torch.from_numpy(state).float()
input.requires_grad = True
output = model(input)[:, 0, 0]
gradient = torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output))[0]
gradient = gradient.cpu().detach().numpy()   # batch x n_drones x 6 (states)


print(gradient.shape)
