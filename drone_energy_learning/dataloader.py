import numpy as np
import torch


class drone_data:
    def __init__(self):
        self.data_list = ['0.0', '1.0', '2.0', '-1.0', '-2.0']
        X_t0 = []
        X_t1 = []
        X_goal = []

        T = 40
        Nx = 13  # state dimension
        n = 3  # number of agents
        Nu = 4  # control dimension
        dt = 0.05  # time step

        goal = np.array([[[0., 0., 0., 0., 0., 0.]]]).repeat(T, 0).repeat(3, 1)
        print(goal.shape)


        for data in self.data_list:
            X = np.load("../jax_traj_gen/solution_X_center_y_" + data + ".npy")
            U = np.load("../jax_traj_gen/solution_U_center_y_" + data + ".npy")

            state = X.reshape((T + 1, n, Nx))
            control = U.reshape((T, Nu, n))

            state = state[:, :, [0, 1, 2, 7, 8, 9]]

            X_t0.append(state[:-1, :, :])
            X_t1.append(state[1:, :, :])
            X_goal.append(goal)

        self.X_t0 = torch.from_numpy(np.concatenate(X_t0, axis=0))
        self.X_t1 = torch.from_numpy(np.concatenate(X_t1, axis=0))
        self.X_goal = torch.from_numpy(np.concatenate(X_goal, axis=0))

    def __len__(self):
        return len(self.X_t0)

    def __getitem__(self, idx):
        return self.X_t0[idx], self.X_t1[idx], self.X_goal[idx]
