import pyLasaDataset as lasa
import torch
from model import MLP
from dataloader import lasa_data

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epoch = 2000
batch = 1024
epsilon1 = 1e-0
epsilon2 = 1e-0
epsilon3 = 1e-0
epsilon4 = 1e-0
alpha = 0.05

dataset = lasa_data()
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=1)

with torch.autograd.set_detect_anomaly(True):
    for epoch in range(epoch):
        for i, data in enumerate(trainloader, 0):
            X_t0, X_t1, X_goal, X_start = data
            X_t0, X_t1, X_goal, X_start = X_t0.float(), X_t1.float(), X_goal.float(), X_start.float()

            optimizer.zero_grad()

            y_t0 = model(X_t0)
            y_t1 = model(X_t1)
            y_goal = model(X_goal)
            y_start = model(X_start)

            loss = (
                    # epsilon1 * torch.maximum(-y_t0, torch.zeros_like(y_t0)).mean() +
                    # epsilon2 * torch.nn.functional.mse_loss(y_goal, torch.zeros_like(y_goal)) +
                    # epsilon2 * torch.nn.functional.mse_loss(y_start, torch.ones_like(y_start)) +
                    epsilon3 * torch.maximum(y_t1 - y_t0, torch.zeros_like(y_t0)).mean()
            )

            n_perturb = 10
            for j in range(n_perturb):
                y_small = model(X_t0 + torch.randn_like(X_t0) * j * 0.1)
                y_large = model(X_t0 + torch.randn_like(X_t0) * (j+1) * 0.1)
                loss += epsilon4 * torch.maximum(y_small - y_large, torch.zeros_like(y_t0)).mean() / n_perturb

            loss.backward()

            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch %5d, Loss %5d: %.10f' % (epoch + 1, i + 1, loss.item()))
        if epoch % 100 == 0:
            dataset.draw(model)
