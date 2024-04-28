import pyLasaDataset as lasa
import numpy as np
import torch

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class lasa_data:
    def __init__(self, data=lasa.DataSet.Line):
        self.demos = data.demos
        X_t0 = []
        X_t1 = []
        X_goal = []
        X_start = []

        for demo in self.demos:
            n = demo.pos.shape[1]
            X_t0.append(demo.pos[:, :-1])
            X_t1.append(demo.pos[:, 1:])
            X_goal.append(demo.pos[:, n - 1:n].repeat(n - 1, axis=1))
            X_start.append(demo.pos[:, 0:1].repeat(n - 1, axis=1))
        self.X_t0 = torch.from_numpy(np.concatenate(X_t0, axis=1).T)
        self.X_t1 = torch.from_numpy(np.concatenate(X_t1, axis=1).T)
        self.X_goal = torch.from_numpy(np.concatenate(X_goal, axis=1).T)
        self.X_start = torch.from_numpy(np.concatenate(X_start, axis=1).T)

        self.xmin = self.X_t0[:, 0].min() - 10
        self.xmax = self.X_t0[:, 0].max() + 10
        self.ymin = self.X_t0[:, 1].min() - 10
        self.ymax = self.X_t0[:, 1].max() + 10
        x_draw = torch.linspace(self.xmin, self.xmax, 100)
        y_draw = torch.linspace(self.ymin, self.ymax, 100)
        self.x_draw, self.y_draw = torch.meshgrid(x_draw, y_draw, indexing='xy')

    def draw(self, model):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        with torch.no_grad():
            x = self.x_draw.flatten()
            y = self.y_draw.flatten()

        xy = torch.stack([x, y]).T
        epsilon = 1e-0

        xy.requires_grad = True
        z = model(xy)
        z.backward(torch.ones_like(z))
        z = z.reshape(100, 100)
        xy_grad = xy.grad.reshape(100, 100, 2)
        # xy_grad /= torch.linalg.norm(xy_grad, dim=2, keepdims=True) + 1e-9
        # xy_grad *= 5

        # xy_dx = torch.stack([x + epsilon, y]).T
        # xy_dy = torch.stack([x, y + epsilon]).T
        # z_dx = model(xy_dx)
        # z_dy = model(xy_dy)
        # z_dx = z_dx.reshape(100, 100, 1)
        # z_dy = z_dy.reshape(100, 100, 1)
        # xy_grad = torch.cat([(z_dx - z) / epsilon, (z_dy - z) / epsilon], dim=2)

        ax.plot_surface(self.x_draw.cpu().detach().numpy(),
                        self.y_draw.cpu().detach().numpy(),
                        z.cpu().detach().numpy(), cmap='viridis')
        ax.set_xlabel(r'$X$')
        ax.set_ylabel(r'$Y$')
        ax.set_zlabel(r'$V(X,Y)$')
        ax.set_title('Lyapunov Function')

        ax = fig.add_subplot(1, 2, 2)
        M = np.arctan2(-xy_grad[..., 0].cpu().detach().numpy(), -xy_grad[..., 1].cpu().detach().numpy())
        ax.quiver(self.x_draw.cpu().detach().numpy(),
                  self.y_draw.cpu().detach().numpy(),
                  -xy_grad[..., 0].cpu().detach().numpy(),
                  -xy_grad[..., 1].cpu().detach().numpy(), M,
                  units='width', cmap='twilight')

        x_draw = np.linspace(self.xmin, self.xmax, 100)
        y_draw = np.linspace(self.ymin, self.ymax, 100)
        x_draw, y_draw = np.meshgrid(x_draw, y_draw, indexing='xy')
        ax.streamplot(x_draw,
                      y_draw,
                      -xy_grad[..., 0].cpu().detach().numpy(),
                      -xy_grad[..., 1].cpu().detach().numpy(), density=1.0, color='k')
        lines = []
        for demo in self.demos:
            lines.append(ax.plot(demo.pos[0], demo.pos[1], '-r', label='Demo Trajectories'))
        ax.legend(['Demo Trajectories'], loc='upper center')
        ax.set_aspect('equal')
        ax.set_title('Gradient of Lyapunov Function')
        plt.show()

    def __len__(self):
        return len(self.X_t0)

    def __getitem__(self, idx):
        return self.X_t0[idx], self.X_t1[idx], self.X_goal[idx], self.X_start[idx]
