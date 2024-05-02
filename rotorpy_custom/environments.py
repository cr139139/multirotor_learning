import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import time as clk
from rotorpy.world import World

from rotorpy.utils.postprocessing import unpack_sim_data

import os

from simulate import simulate
from rotorpy.utils.animate import animate
from jax import numpy as jnp


def matrices_from_quaternions(Q):
    Q = np.asarray(Q)

    w = Q[..., 3]
    x = Q[..., 0]
    y = Q[..., 1]
    z = Q[..., 2]

    x2 = 2.0 * x * x
    y2 = 2.0 * y * y
    z2 = 2.0 * z * z
    xy = 2.0 * x * y
    xz = 2.0 * x * z
    yz = 2.0 * y * z
    xw = 2.0 * x * w
    yw = 2.0 * y * w
    zw = 2.0 * z * w

    out = np.empty(w.shape + (3, 3))

    out[..., 0, 0] = 1.0 - y2 - z2
    out[..., 0, 1] = xy - zw
    out[..., 0, 2] = xz + yw
    out[..., 1, 0] = xy + zw
    out[..., 1, 1] = 1.0 - x2 - z2
    out[..., 1, 2] = yz - xw
    out[..., 2, 0] = xz - yw
    out[..., 2, 1] = yz + xw
    out[..., 2, 2] = 1.0 - x2 - y2

    return out


class Environment():
    """
    Sandbox represents an instance of the simulation environment containing a unique vehicle, 
    controller, trajectory generator, wind profile. 

    """

    def __init__(self, vehicles,  # vehicle object, must be specified.
                 controllers,  # controller object, must be specified.
                 trajectories,  # trajectory object, must be specified.
                 world=None,  # The world object
                 sim_rate=100,  # The update frequency of the simulator in Hz
                 safety_margin=0.25,  # The radius of the safety region around the robot.
                 ):

        self.sim_rate = sim_rate
        self.vehicles = vehicles
        self.controllers = controllers
        self.trajectories = trajectories

        self.safety_margin = safety_margin

        if world is None:
            # If no world is specified, assume that it means that the intended world is free space.
            wbound = 3
            self.world = World.empty((-wbound, wbound, -wbound,
                                      wbound, -wbound, wbound))
        else:
            self.world = world

        return

    def run(self, t_final=10,  # The maximum duration of the environment in seconds
            terminate=False,
            verbose=False,  # Boolean: will print statistics regarding the simulation.
            fname=None
            # Filename is specified if you want to save the animation. Default location is the home directory.
            ):

        """
        Run the simulator
        """

        self.t_step = 1 / self.sim_rate
        self.t_final = t_final
        self.t_final = t_final
        self.terminate = terminate

        start_time = clk.time()
        (time, states, controls, flats, exit) = simulate(self.world,
                                                         [vehicle.initial_state for vehicle in self.vehicles],
                                                         self.vehicles,
                                                         self.controllers,
                                                         self.trajectories,
                                                         self.t_final,
                                                         self.t_step,
                                                         self.safety_margin,
                                                         terminate=self.terminate,
                                                         )
        if verbose:
            # Print relevant statistics or simulator status indicators here
            print('-------------------RESULTS-----------------------')
            print('SIM TIME -- %3.2f seconds | WALL TIME -- %3.2f seconds' % (
                min(self.t_final, time[-1]), (clk.time() - start_time)))
            print('EXIT STATUS -- ' + exit.value)

        self.result = dict(time=time, states=states, controls=controls, flats=flats, exit=exit)

        x = np.concatenate([state['x'][:, None, :] for state in states], axis=1)
        q = np.concatenate([state['q'][:, None, :] for state in states], axis=1)
        R = matrices_from_quaternions(q)
        wind = np.zeros_like(x)
        print('x shape:', x.shape)  # (T, n, 3)
        print('R shape:', R.shape)  # (T, n, 3, 3)
        print('wind shape:', wind.shape)  # (T, n, 3)

        ani = animate(time, x, R, wind, False, self.world, filename=None, blit=False, show_axes=True,
                      close_on_finish=False)
        plt.show()

        return self.result


if __name__ == "__main__":
    from rotorpy.vehicles.crazyflie_params import quad_params
    from rotorpy.trajectories.hover_traj import HoverTraj
    from rotorpy.trajectories.circular_traj import CircularTraj

    from rotorpy.vehicles.multirotor import Multirotor
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.controllers.controller_template import MultirotorControlTemplate

    n_drones = 5
    trajectories = [HoverTraj(np.array([i, i, i])) for i in range(n_drones)]
    sim = Environment(vehicles=[Multirotor(quad_params, control_abstraction='cmd_vel')] * n_drones,
                      controllers=[MultirotorControlTemplate(quad_params)] * n_drones,
                      trajectories=trajectories,
                      sim_rate=100
                      )
    run_sim = True
    if run_sim:
        result = sim.run(t_final=10)
    else:
        X0 = jnp.load("../jax_traj_gen/solution_X_center_y_0.0.npy")
        U0 = jnp.load("../jax_traj_gen/solution_U_center_y_0.0.npy")

        T = 40
        Nx = 13  # state dimension
        n = 3  # number of agents
        Nu = 4  # control dimension
        dt = 0.05  # time step
        time = np.arange(0, T * dt + dt, dt)
        # print(X0.shape, U0.shape)
        state = np.array(X0.reshape((T + 1, n, Nx)))
        control = np.array(U0.reshape((T, Nu, n)))

        # x = np.concatenate([state['x'][:, None, :] for state in states], axis=1)
        # q = np.concatenate([state['q'][:, None, :] for state in states], axis=1)
        x = state[:, :, :3]
        q = state[:, :, 3:7]
        R = matrices_from_quaternions(q)
        wind = np.zeros_like(x)
        print('time shape:', time.shape)
        print('x shape:', x.shape)  # (T, n, 3)
        print('R shape:', R.shape)  # (T, n, 3, 3)
        print('wind shape:', wind.shape)  # (T, n, 3)

        wbound = 4
        world = World.empty((-wbound, wbound, -wbound,
                             wbound, -wbound, wbound))
        # print(type(world))

        ani = animate(time, x, R, wind, False, world, filename=None, blit=False, show_axes=True,
                      close_on_finish=False)

        import matplotlib.animation as animation

        # saving to m4 using ffmpeg writer
        writer = animation.PillowWriter(fps=30,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save('increasingStraightLine.gif', writer=writer)
        plt.show()
