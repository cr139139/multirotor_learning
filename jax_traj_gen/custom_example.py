from jax.config import config
# from trajax import integrators
from trajax.experimental.sqp import shootsqp, util
config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

key = jax.random.key(42)
mode = 2 # 0: no control, 1: animation, 2: optimization
n = 1  # number of drones
dt = 0.05  # time step
T_total = 30
r_nominal = jnp.zeros((n, 3))
use_warm_start = False


def matrices_from_quaternions(Q):
    """
    Q : (..., w, x, y, z)
    Rs : (..., 3, 3)
    """
    Q = jnp.asarray(Q)

    w = Q[..., 0]
    x = Q[..., 1]
    y = Q[..., 2]
    z = Q[..., 3]

    x2 = 2.0 * x * x
    y2 = 2.0 * y * y
    z2 = 2.0 * z * z
    xy = 2.0 * x * y
    xz = 2.0 * x * z
    yz = 2.0 * y * z
    xw = 2.0 * x * w
    yw = 2.0 * y * w
    zw = 2.0 * z * w

    out = jnp.empty(w.shape + (3, 3))
    out = out.at[..., 0, 0].set(1.0 - y2 - z2)
    out = out.at[..., 0, 1].set(xy - zw)
    out = out.at[..., 0, 2].set(xz + yw)
    out = out.at[..., 1, 0].set(xy + zw)
    out = out.at[..., 1, 1].set(1.0 - x2 - z2)
    out = out.at[..., 1, 2].set(yz - xw)
    out = out.at[..., 2, 0].set(xz - yw)
    out = out.at[..., 2, 1].set(yz + xw)
    out = out.at[..., 2, 2].set(1.0 - x2 - y2)

    return out


def quat_dot(quat, omega):
    """
    Parameters:
        quat, [w,i,j,k]
        omega, angular velocity of body in body axes

    Returns
        duat_dot, [w,i,j,k]

    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    G = jnp.array([[w, z, -y, -x],
                   [-z, w, x, -y],
                   [y, -x, w, -z]])
    quat_dot = 0.5 * G.T @ omega
    # Augment to maintain unit quaternion.
    quat_err = jnp.sum(quat ** 2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    (x, y, z, w) = quat_dot
    return jnp.array([w, x, y, z])


quat_dot_batched = jax.vmap(quat_dot, (0, 0), 0)

def get_attachment(x, R):
    global r_nominal
    return x[None, :] + (R @ r_nominal[..., None])[..., 0]

if __name__ == "__main__":
    # this is a few quadrotors trying to lift a load together, last mass is the load, the rest are drones
    # environment parameters
    Nx = 14  # state dimension
    Nu = 4  # control dimension
    g = 9.81
    K = 100  # tension spring constant

    # drone and load parameters
    mass = jnp.ones(n + 1)
    mass = mass.at[-1].set(3.0)  # load mass
    inertia = jnp.eye(3)[None, ...].repeat(n + 1, axis=0)
    inertia_inv = jnp.linalg.inv(inertia)
    # r_nominal = jax.random.normal(key, shape=(n, 3)) * 0.01
    # print(mass.shape, inertia.shape, inertia_inv.shape, r_nominal.shape)

    # state: (n+1) x (3+4+1+3+3 = 14)
    # Tension value in the load state is not used
    # x (x, y, z), q (qw, qx, qy, qz), T, dxdt (vx, vy, vz), Omega (wx, wy, wz)
    state = jnp.zeros((n + 1, 14))
    # set the quad to be in a triangle symmetic across the origin
    a = 0

    # quadrotors 120 deg apart!!!
    # drone 1 (a, 0)
    # drone 2 a * (cos(120), sin(120))
    # drone 3 a * (cos(-120), sin(-120))
    # state = state.at[0, 0].set(a)
    # state = state.at[0, 1].set(0.0)
    # state = state.at[1, 0].set(a * jnp.cos(2 * jnp.pi / 3))
    # state = state.at[1, 1].set(a * jnp.sin(2 * jnp.pi / 3))
    # state = state.at[2, 0].set(a * jnp.cos(-2 * jnp.pi / 3))
    # state = state.at[2, 1].set(a * jnp.sin(-2 * jnp.pi / 3))

    state = state.at[n, 2].set(-1.0)  # z
    state = state.at[:, 3].set(0.998)
    state = state.at[:, 4].set(0.044)
    state = state.at[:, 5].set(0.044)
    state = state.at[:, 6].set(0.002)
    # state = state.at[:, 7].set(1.0)  # T

    # f, M (x, y, z), deleted dTdt n x 4
    control = jnp.zeros((n, 4)) 
    control = control.at[:,0].set(jnp.sum(mass)/n*g ) # 4 properller,  n drone, sharing load weight
    R = matrices_from_quaternions(state[-1, 3:7])
    r_ini = get_attachment(state[-1,0:3], R)
    L = jnp.linalg.norm(r_ini - state[0:n,0:3], axis=1) - jnp.sum(mass[-1])*g/K/n


    # # batched quaternion to matrices test
    # Q = jnp.zeros((n, 4))
    # Q = Q.at[:, 0].set(1.)
    # print(matrices_from_quaternions(Q).shape)

    # # batched quaternion dot test
    # Q = jnp.zeros((n, 4))
    # Q = Q.at[:, 0].set(1.)
    # Omega = jnp.ones((n, 3))
    # Omega = Omega.at[-1, :].set(-1.)
    # print(quat_dot(Q[0], -Omega[0]).shape)
    # print(quat_dot_batched(Q, Omega).shape)


    # # get_attachment func test
    # x = jnp.zeros(3)
    # R = jnp.eye(3)
    # print(get_attachment(x, R).shape)

    # @jax.jit
    def ode(state, control, t):
        del t
        state = state.reshape((n + 1, Nx))
        x = state[:, 0:3]
        q = state[:, 3:7]
        # T = state[:, 7:8]
        dxdt = state[:, 8:11]
        Omega = state[:, 11:14]

        control = control.reshape((n, Nu))
        f = control[:, 0:1]
        M = control[:, 1:4]
        # dTdt = control[:, 4:5]

        R = matrices_from_quaternions(q)
        r = get_attachment(x[-1], R[-1])
        e3 = jnp.array([0., 0., 1.])
        q_vector_full = r - x[0:n]
        q_vec_norm = jnp.linalg.norm(q_vector_full, axis=1, keepdims=True)+1e-9
        # control dTdt is a one sided spring
        dTdt = jnp.zeros((n,1)) #(K * (q_vec_norm.flatten() - L)).reshape((n,1)) # n x 1 
        T = (K * (q_vec_norm.flatten() - L))#.reshape((n,1))
        q_vector = q_vector_full / q_vec_norm

        # ddxdt_drone: (nx3)
        ddxdt_drone = (f * (R[0:n] @ e3)  # (nx1) * (nx3x3 @ 3) = (nx3)
                       - mass[0:n, None] * g * e3[None, :]  # (nx1) * 1 * (1x3) = (nx3)
                       )#+ (R[n] @ q_vector[:, :, None])[..., 0] * T[0:n])  # (3x3 @ nx3x1)[...,0] * (nx1) = (nx3)

        # Omegadt_drone: (nx3)
        # ((nx3x3) @ ((nx3) - cross((nx3), (nx3x3 @ nx3x1)[...,0]))[...,None])[...,0] = (nx3)
        Omegadt_drone = \
            (inertia_inv[0:n] @ (M - jnp.cross(Omega[0:n], (inertia_inv[0:n] @ Omega[0:n, :, None])[..., 0]))[
                ..., None])[
                ..., 0]

        # ddxdt_load: (1x3)
        ddxdt_load = jnp.zeros((1,3))#(-((R[n] @ q_vector[:, :, None])[..., 0] * T[:n]).sum(axis=0,
                                      #                                     keepdims=True)  # (3x3 @ nx3x1)[...,0] * nx1 = nx3 -> 1x3
                     # - mass[n] * g * e3[None, :])  # (1 x 1 * 1x3) = (1x3)

        # Omegadt_load: (1X3)
        # (3x3 @ (cross(nx3, (nx1)*(nx3)).sum() - cross(3, 3x3 @ 3)))[None, :] = 1x3
        Omegadt_load = jnp.zeros((1,3))# (inertia_inv[n] @ (
               # jnp.cross(r, -T[:n] * q_vector).sum(axis=0) - jnp.cross(Omega[n], (inertia_inv[n] @ Omega[n, :]))))[
                 #      None, :]

        ddxdt = jnp.concatenate([ddxdt_drone, ddxdt_load], axis=0)
        Omegadt = jnp.concatenate([Omegadt_drone, Omegadt_load], axis=0)
        dqdt = quat_dot_batched(q, Omega)
        dTdt_dummy = jnp.concatenate([dTdt, jnp.array([[0.]])], axis=0)

        statedt = jnp.concatenate([dxdt, dqdt, dTdt_dummy, ddxdt, Omegadt], axis=1).flatten()
        return statedt


    def euler(dynamics, dt=0.01):
        @jax.jit
        def integrator(x, u, t):
            x_new = x + dt * dynamics(x, u, t)
            x_new = x_new.reshape((n + 1, 14))
            x_new = x_new.at[:, 3:7].set(x_new[:, 3:7] / (jnp.linalg.norm(x_new[:, 3:7]) + 1e-9))
            # x_new = x_new.at[:, 7:8].set(jnp.clip(x_new[:, 7:8], 0.))
            x_new = x_new.flatten()
            return x_new

        return integrator


    # # ode func test
    print(ode(state.flatten(), control.flatten(), 0.0).reshape((n + 1, 14))[0])

    # # euler func test (failed)
    dynamics = euler(ode, dt)
    
    
    def loss_fn(x_target, x, u):
        x_new = dynamics(x.flatten(), u.flatten(), 0.1)
        return jnp.sum((x_target - x_new.reshape(n + 1, 14))[:, :3] ** 2)
    
    

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    R = matrices_from_quaternions(state[:, 3:7]) #* 0.01
    if mode == 0:
        quiver = ax.quiver(state[:, 0], state[:, 1], state[:, 2], R[:, 0, 2], R[:, 1, 2], R[:, 2, 2])
    state_temp = jnp.copy(state)
    control_temp = jnp.copy(control)
    
    def autonomous_system(frame):
        global state
        global state_temp
        global control_temp


    def update(frame):
        global state
        global state_temp
        global control_temp

        # last control input is tension derivative, It acts as a spring but only one sided 
        
        for _ in range(10):
            grad = jax.grad(loss_fn, argnums=2)(state, state_temp, control_temp)
            control_temp -= grad
            print(grad)
        # x (x, y, z), q (qw, qx, qy, qz), T, dxdt (vx, vy, vz), Omega (wx, wy, wz)
        state_temp = dynamics(state_temp.flatten(), control_temp.flatten(), t=1.0).reshape((n + 1, 14))


        # controls = jnp.concatenate([jnp.zeros((n, 4)), state_temp[: ]
                                    

        R = matrices_from_quaternions(state_temp[:, 3:7]) * 0.1
        segs = jnp.concatenate([state_temp[:, :3], state_temp[:, :3] + R[:, :, 2]], axis=1).T
        new_segs = [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*segs.tolist())]
        quiver.set_segments(new_segs)
        print(loss_fn(state, state_temp, control_temp))
    if mode == 0:
        state_history = []
        
        for _ in range(60):
            state_temp = dynamics(state_temp.flatten(), control_temp.flatten(), t=1.0).reshape((n + 1, 14))
            state_history.append(state_temp)
        # draw 3d
        state_history = jnp.array(state_history)
        print(state_history.shape)
        for i in range(n + 1):
            ax.plot(state_history[:, i, 0], state_history[:, i, 1], state_history[:, i, 2])
        plt.axis('equal')
        plt.show()
    elif mode == 1:
        # ax.plot(state_history[:, ,0], state_history[:, 1], state_history[:, 2])
# 
        ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
        plt.show()

    elif mode == 2:

        wQ = jnp.zeros((Nx * (n+1), Nx * (n+1)))
        # wQ = wQ.at[Nx * n , Nx * n ].set(1.0) # load x
        # wQ = wQ.at[Nx * n + 1, Nx * n + 1].set(1.0) # load y
        # wQ = wQ.at[Nx * n + 2, Nx * n + 2].set(1.0) # load z
        wQ = wQ.at[0,0].set(1.0) # drone x
        wQ = wQ.at[1,1].set(1.0) # drone y
        wQ = wQ.at[2,2].set(1.0) # drone z
        wQ = wQ.at[3,3].set(.5) # drone q
        wQ = wQ.at[4,4].set(.5) # drone q
        wQ = wQ.at[5,5].set(.5) # drone q
        wQ = wQ.at[6,6].set(.5) # drone q
        wQ = wQ.at[7,7].set(0.0) # drone Tension
        wQ = wQ.at[8,8].set(1.0) # drone vx
        wQ = wQ.at[9,9].set(1.0) # drone vy
        wQ = wQ.at[10,10].set(1.0) # drone vz
        wQ = wQ.at[11,11].set(1.0) # drone wx
        wQ = wQ.at[12,12].set(1.0) # drone wy
        wQ = wQ.at[13,13].set(1.0) # drone wz


        wR = jnp.eye(Nu * n) * 1.0
        goal_default = jnp.zeros( (n+1, Nx), dtype=jnp.float64)
        goal_default = goal_default.at[0, 0].set(2.0)
        goal_default = goal_default.at[0, 1].set(2.0)
        goal_default = goal_default.at[0, 2].set(0.0)
        goal_default = goal_default.at[0, 3].set(1.0)
        goal_default = goal_default.at[0, 4].set(0.0)
        goal_default = goal_default.at[0, 5].set(0.0)
        goal_default = goal_default.at[0, 6].set(0.0)
        # goal_default = goal_default.at[0, 7].set(0.0) # tension doesnt matter
        goal_default = goal_default.at[0, 8].set(0.0)
        goal_default = goal_default.at[0, 9].set(0.0)
        goal_default = goal_default.at[0, 10].set(0.0)
        goal_default = goal_default.at[0, 11].set(0.0)
        goal_default = goal_default.at[0, 12].set(0.0)
        goal_default = goal_default.at[0, 13].set(0.0)
        
        # goal_default = goal_default.at[0, 3].set(1.0)

    
        # goal_default = goal_default.at[Nx * n].set(1.0)
        # goal_default = goal_default.at[Nx * n + 1].set(1.0)
        # goal_default = goal_default.at[Nx * n + 2].set(-1.0) # don't change
        #set quaternion w to 1, but doesnt matter now since weight is 0
        # set goal of load xyz to 0 # already so
        # goal_default = goal_default.at

        @jax.jit
        def cost(x, u, t, goal=goal_default.flatten(), u_ref = control.flatten()):
            stage_cost = dt * jnp.vdot((u-u_ref), wR @ (u-u_ref))
            delta = x - goal
            term_cost = jnp.vdot(delta, wQ @ delta)
            return jnp.where(t == T_total, term_cost, stage_cost)

    # Control box bounds
    # n x 4 control inputs
        # control_bounds = (jnp.array([0, -10, -10, -10, 0, -10, -10, -10, 0, -10, -10, -10], dtype=jnp.float64),
        #                 jnp.array([100, 10, 10, 10, 100, 10, 10, 10, 100, 10, 10, 10], dtype=jnp.float64))
        control_bounds = (jnp.array([0, -10, -10, -10], dtype=jnp.float64),
                        jnp.array([100, 10, 10, 10], dtype=jnp.float64))

        print(control_bounds)

    # Obstacle avoidance constraint function
    # def obs_constraint(pos):
    #   def avoid_obs(pos_c, ob):
    #     delta_body = pos_c - ob[0]
    #     delta_dist_sq = jnp.vdot(delta_body, delta_body) - (ob[1]**2)
    #     return delta_dist_sq
    #   return jnp.array([avoid_obs(pos, ob) for ob in obs])

    # State constraint function
        @jax.jit
        def state_constraint(x, t):
            del t
            del x
            return jnp.ones(Nx * (n+1)) # no constraint always positive
        solver_options = dict(method=shootsqp.SQP_METHOD.SENS,
                            ddp_options={'ddp_gamma': 1e-4},
                            hess="full", verbose=True,
                            max_iter=100, ls_eta=0.49, ls_beta=0.8,
                            primal_tol=1e-3, dual_tol=1e-3, stall_check="abs",
                            debug=False)
        solver = shootsqp.ShootSQP(Nx*(n+1), Nu*n, T_total, dynamics, cost, control_bounds,
                                state_constraint, s1_ind=None,  **solver_options)
        x0 = state.flatten()
        # copy flattened control T_total times
        U0 = jnp.tile(control.flatten(), T_total).reshape(T_total, Nu*n)
        X0 = jnp.load("solution_X.npy") if use_warm_start else None
        solver.opt.proj_init = False

        # solver.opt.proj_init = True

        solver.opt.max_iter = 1
        _ = solver.solve(x0, U0, X0)
     

# Run to completion
        solver.opt.max_iter = 100
        soln = solver.solve(x0, U0, X0)

        # get trajectory from solver
        # x_traj, u_traj = soln.get_trajectories()

        plt.rcParams.update({'font.size': 20})
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42



# fig, ax = render_scene()
        U, X = soln.primals
#         print(U.shape, X.shape, "U, X shapes")
#         ax.plot(X[:, 0], X[:, 1], 'r-', linewidth=2)

#         for t in jnp.arange(0, solver._T+1, 5):
#             ax.arrow(X[t, 0], X[t, 1],
#                 0.2 * jnp.sin(X[t, 2]), 0.2 * jnp.cos(X[t, 2]),
#                 width=0.05, color='c')

#         # Start
#         ax.add_patch(plt.Circle([x0[0], x0[1]], 0.1, color='g', alpha=0.3))
#         # End
#         ax.add_patch(plt.Circle([goal_default[0], goal_default[1]], 0.1, color='r', alpha=0.3))

#         ax.set_aspect('equal')
     

# # @title {vertical-output: true}

#         fig = plt.figure(figsize=(6, 6))
#         ax = fig.add_subplot(111)
#         ax.grid(True)
#         plt.plot(solver._timesteps[:-1]*dt, U, markersize=5)
#         ax.set_ylabel('U')
#         ax.set_xlabel('Time [s]')

            

#         import seaborn as sns
#         colors = sns.color_palette("tab10")
            

#         history = soln.history
#         history.keys()
            

#         #@title {vertical-output: true}

#         plt.rcParams.update({'font.size': 24})
#         matplotlib.rcParams['pdf.fonttype'] = 42
#         matplotlib.rcParams['ps.fonttype'] = 42

#         fig, axs = plt.subplots(2, 2, figsize=(15, 15))

#         axs[0][0].plot(history['steplength'], color=colors[0], linewidth=2)
#         axs[0][0].set_title('Step size')
#         axs[0][0].grid(True)

#         axs[0][1].plot(history['obj'], color=colors[0], linewidth=2)
#         axs[0][1].set_title('Objective')
#         axs[0][1].set_yscale('log')
#         axs[0][1].grid(True)

#         axs[1][0].plot(history['min_viol'], color=colors[0], linewidth=2)
#         axs[1][0].set_title('Min constraint viol.')
#         axs[1][0].set_xlabel('Iteration')
#         axs[1][0].grid(True)

#         if 'ddp_err' in history:
#             axs[1][1].plot(history['ddp_err'], color=colors[0], linewidth=2)
#             axs2 = axs[1][1].twinx()
#             axs2.plot(history['ddp_err_grad'], color=colors[1], linewidth=2)
#             axs2.set_yscale("log")
#             axs[1][1].set_title('DDP errors')
#             axs[1][1].set_xlabel('Iteration')
#             axs[1][1].grid(True)
     
        # save X,U to file so we can warm start the next run
        jnp.save("solution_X.npy", X)
        jnp.save("solution_U.npy", U)

#         pt.max_iter = 100
# soln = solver.solve(x0, U0, X0)