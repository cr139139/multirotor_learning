import pyLasaDataset as lasa
import numpy as np
import torch

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class lasa_data:
    def __init__(self, data=lasa.DataSet.Line):
        demos = data.demos
        X_t0 = []
        X_t1 = []
        X_goal = []

        for demo in demos:
            n = demo.pos.shape[1]
            X_t0.append(demo.pos[:, :-1])
            X_t1.append(demo.pos[:, 1:])
            X_goal.append(demo.pos[:, n - 1:n].repeat(n - 1, axis=1))
        self.X_t0 = torch.from_numpy(np.concatenate(X_t0, axis=1).T)
        self.X_t1 = torch.from_numpy(np.concatenate(X_t1, axis=1).T)
        self.X_goal = torch.from_numpy(np.concatenate(X_goal, axis=1).T)

        x_draw = torch.linspace(self.X_t0[:, 0].min() - 5, self.X_t0[:, 0].max() + 5, 100)
        y_draw = torch.linspace(self.X_t0[:, 1].min() - 5, self.X_t0[:, 1].max() + 5, 100)
        self.x_draw, self.y_draw = torch.meshgrid(x_draw, y_draw)

    def draw(self, model):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        with torch.no_grad():
            x = self.x_draw.flatten()
            y = self.y_draw.flatten()
        xy = torch.stack([x, y]).T
        xy.requires_grad = True
        z = model(xy)
        z.backward(torch.ones_like(z))
        z = z.reshape(100, 100)
        xy_grad = xy.grad.reshape(100, 100, 2)

        ax.plot_surface(self.x_draw.cpu().detach().numpy(),
                        self.y_draw.cpu().detach().numpy(),
                        z.cpu().detach().numpy(), cmap='viridis')

        ax = fig.add_subplot(1, 2, 2)
        ax.quiver(self.x_draw.cpu().detach().numpy(),
                  self.y_draw.cpu().detach().numpy(),
                  -xy_grad[..., 0].cpu().detach().numpy(),
                  -xy_grad[..., 1].cpu().detach().numpy(), scale_units='inches')


        plt.show()

    def __len__(self):
        return len(self.X_t0)

    def __getitem__(self, idx):
        return self.X_t0[idx], self.X_t1[idx], self.X_goal[idx]
