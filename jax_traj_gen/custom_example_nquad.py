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
T_total = 40
r_nominal = jnp.zeros((n, 3))
use_warm_start = False
Nx = 13  # state dimension
Nu = 4  # control dimension
g = 9.81

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



@jax.jit
def ode(state, control, t):
    del t
    state = state.reshape((n, Nx))
    x = state[:, 0:3]
    q = state[:, 3:7]
    # T = state[:, 7:8]
    dxdt = state[:, 7:10]
    Omega = state[:, 10:13]

    control = control.reshape((n, Nu))
    f = control[:, 0:1]
    M = control[:, 1:4]
    # dTdt = control[:, 4:5]

    R = matrices_from_quaternions(q)
    # r = get_attachment(x[-1], R[-1])
    e3 = jnp.array([0., 0., 1.])
    # q_vector_full = r - x[0:n]
    # q_vec_norm = jnp.linalg.norm(q_vector_full, axis=1, keepdims=True)+1e-9
    # control dTdt is a one sided spring
    # dTdt = jnp.zeros((n,1)) #(K * (q_vec_norm.flatten() - L)).reshape((n,1)) # n x 1 
    # T = (K * (q_vec_norm.flatten() - L))#.reshape((n,1))
    # q_vector = q_vector_full / q_vec_norm

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
    # ddxdt_load = jnp.zeros((1,3))#(-((R[n] @ q_vector[:, :, None])[..., 0] * T[:n]).sum(axis=0,
    #                               #                                     keepdims=True)  # (3x3 @ nx3x1)[...,0] * nx1 = nx3 -> 1x3
    #              # - mass[n] * g * e3[None, :])  # (1 x 1 * 1x3) = (1x3)

    # # Omegadt_load: (1X3)
    # # (3x3 @ (cross(nx3, (nx1)*(nx3)).sum() - cross(3, 3x3 @ 3)))[None, :] = 1x3
    # Omegadt_load = jnp.zeros((1,3))# (inertia_inv[n] @ (
            # jnp.cross(r, -T[:n] * q_vector).sum(axis=0) - jnp.cross(Omega[n], (inertia_inv[n] @ Omega[n, :]))))[
                #      None, :]

    # ddxdt = jnp.concatenate([ddxdt_drone, ddxdt_load], axis=0)
    # Omegadt = jnp.concatenate([Omegadt_drone, Omegadt_load], axis=0)
    dqdt = quat_dot_batched(q, Omega)
    # dTdt_dummy = jnp.concatenate([dTdt, jnp.array([[0.]])], axis=0)

    statedt = jnp.concatenate([dxdt, dqdt, ddxdt_drone, Omegadt_drone], axis=1).flatten()
    return statedt


def euler(dynamics, dt=0.01):
    @jax.jit
    def integrator(x, u, t):
        x_new = x + dt * dynamics(x, u, t)
        x_new = x_new.reshape((n, 13))
        x_new = x_new.at[:, 3:7].set(x_new[:, 3:7] / (jnp.linalg.norm(x_new[:, 3:7]) + 1e-9))
        # x_new = x_new.at[:, 7:8].set(jnp.clip(x_new[:, 7:8], 0.))
        x_new = x_new.flatten()
        return x_new

    return integrator
dynamics = euler(ode, dt)


def loss_fn(x_target, x, u):
    x_new = dynamics(x.flatten(), u.flatten(), 0.1)
    return jnp.sum((x_target - x_new.reshape(n , 13))[:, :3] ** 2)
    
@jax.jit
def cost(x, u, t, goal, u_ref):
    stage_cost = dt * jnp.vdot((u-u_ref), wR @ (u-u_ref))
    delta = x - goal
    term_cost = jnp.vdot(delta, wQ @ delta)
    return jnp.where(t == T_total, term_cost, stage_cost)
@jax.jit
def state_constraint(x, t):
    del t
    # del x
    # each quad needs to stay at least 0.2 m away from each other
    # reshape x to (n, Nx)
    x = x.reshape((n, Nx))
    if n == 1:
        return jnp.array([1.0])
    else:
        return jnp.array([jnp.linalg.norm(x[i, 0:3] - x[j, 0:3]) - 0.7 for i in range(n) for j in range(i+1, n)])
if __name__ == "__main__":
    # this is a few quadrotors trying to lift a load together, last mass is the load, the rest are drones
    # environment parameters

    # K = 100  # tension spring constant

    # drone and load parameters
    mass = jnp.ones(n)
    # mass = mass.at[-1].set(3.0)  # load mass
    inertia = 0.1*jnp.eye(3)[None, ...].repeat(n, axis=0)
    inertia_inv = jnp.linalg.inv(inertia)
    # r_nominal = jax.random.normal(key, shape=(n, 3)) * 0.01
    # print(mass.shape, inertia.shape, inertia_inv.shape, r_nominal.shape)

    # state: (n) x (3+4+1+3+3 = 14)
    # Tension value in the load state is not used
    # x (x, y, z), q (qw, qx, qy, qz), T, dxdt (vx, vy, vz), Omega (wx, wy, wz)
    
    
    # state = state.at[:, 7].set(1.0)  # T

    # f, M (x, y, z), deleted dTdt n x 4
    control = jnp.zeros((n, 4)) 
    control = control.at[:,0].set(jnp.sum(mass)/n*g ) # 4 properller,  n drone, sharing load weight

    state = jnp.zeros((n, 13))
    # set the quad to be in a triangle symmetic across the origin
    a = 0.5
    # List the names of the parameters to vary and save the solution with the parameter name and value in the filename
    center_x = 3.0
    params_names = ["center_y"]
    # center_y = 3.0
    for center_y in jnp.linspace(-2, 2, 5):
        for i in range(n): # quadrotors in a circle
            key1 = jax.random.key(i*10)
            key2 = jax.random.key(i*10+1)
            key3 = jax.random.key(i*10+2)
            state = state.at[i, 0].set(a * jnp.cos(2 * jnp.pi / n * i) + center_x + 0.1 * jax.random.normal(key1))
            state = state.at[i, 1].set(a * jnp.sin(2 * jnp.pi / n * i) + center_y + 0.1 * jax.random.normal(key2))
            # state = state.at[i, 2].set(jax.random.uniform(key3) * 0.3)
            # set z = 0
            
        state = state.at[:, 3].set(1.0)# all quads somewhat facing up
        state = state.at[:, 4].set(0.0)
        state = state.at[:, 5].set(0.0)
        state = state.at[:, 6].set(0.0)

        print(ode(state.flatten(), control.flatten(), 0.0).reshape((n, 13))[0])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        R = matrices_from_quaternions(state[:, 3:7]) #* 0.01
        if mode == 0:
            quiver = ax.quiver(state[:, 0], state[:, 1], state[:, 2], R[:, 0, 2], R[:, 1, 2], R[:, 2, 2])
        state_temp = jnp.copy(state)
        control_temp = jnp.copy(control)
        
        if mode == 0:
            state_history = []
            
            for _ in range(60):
                state_temp = dynamics(state_temp.flatten(), control_temp.flatten(), t=1.0).reshape((n, 13))
                state_history.append(state_temp)
            # draw 3d
            state_history = jnp.array(state_history)
            print(state_history.shape)
            for i in range(n):
                ax.plot(state_history[:, i, 0], state_history[:, i, 1], state_history[:, i, 2])
            plt.axis('equal')
            plt.show()

        elif mode == 2:

            wQ = jnp.zeros((Nx * (n), Nx * (n)))
            # wQ = wQ.at[Nx * n , Nx * n ].set(1.0) # load x
            # wQ = wQ.at[Nx * n + 1, Nx * n + 1].set(1.0) # load y
            # wQ = wQ.at[Nx * n + 2, Nx * n + 2].set(1.0) # load z
            for i in range(n):
                wQ = wQ.at[Nx * i, Nx * i].set(3.0)
                wQ = wQ.at[Nx * i + 1, Nx * i + 1].set(3.0)
                wQ = wQ.at[Nx * i + 2, Nx * i + 2].set(3.0)
                wQ = wQ.at[Nx * i + 3, Nx * i + 3].set(0.5)
                wQ = wQ.at[Nx * i + 4, Nx * i + 4].set(0.5)
                wQ = wQ.at[Nx * i + 5, Nx * i + 5].set(0.5)
                wQ = wQ.at[Nx * i + 6, Nx * i + 6].set(0.5)
                wQ = wQ.at[Nx * i + 7, Nx * i + 7].set(1.0)
                wQ = wQ.at[Nx * i + 8, Nx * i + 8].set(1.0)
                wQ = wQ.at[Nx * i + 9, Nx * i + 9].set(1.0)
                wQ = wQ.at[Nx * i + 10, Nx * i + 10].set(1.0)
                wQ = wQ.at[Nx * i + 11, Nx * i + 11].set(1.0)
                wQ = wQ.at[Nx * i + 12, Nx * i + 12].set(1.0)
                
            # wQ = wQ.at[0,0].set(1.0) # drone x
            # wQ = wQ.at[1,1].set(1.0) # drone y
            # wQ = wQ.at[2,2].set(1.0) # drone z
            # wQ = wQ.at[3,3].set(.5) # drone q
            # wQ = wQ.at[4,4].set(.5) # drone q
            # wQ = wQ.at[5,5].set(.5) # drone q
            # wQ = wQ.at[6,6].set(.5) # drone q
            # wQ = wQ.at[7,7].set(0.0) # drone Tension
            # wQ = wQ.at[8,8].set(1.0) # drone vx
            # wQ = wQ.at[9,9].set(1.0) # drone vy
            # wQ = wQ.at[10,10].set(1.0) # drone vz
            # wQ = wQ.at[11,11].set(1.0) # drone wx
            # wQ = wQ.at[12,12].set(1.0) # drone wy
            # wQ = wQ.at[13,13].set(1.0) # drone wz


            wR = jnp.eye(Nu * n) * 0.1


            goal_default = jnp.zeros( (n, Nx), dtype=jnp.float64)
            for i in range(n):
                goal_default = goal_default.at[i, 0].set(0.0)
                goal_default = goal_default.at[i, 1].set(0.0)
                goal_default = goal_default.at[i, 2].set(0.0)
                goal_default = goal_default.at[i, 3].set(1.0)
                goal_default = goal_default.at[i, 4].set(0.0)
                goal_default = goal_default.at[i, 5].set(0.0)
                goal_default = goal_default.at[i, 6].set(0.0)
                goal_default = goal_default.at[i, 7].set(0.0)
                goal_default = goal_default.at[i, 8].set(0.0)
                goal_default = goal_default.at[i, 9].set(0.0)
                goal_default = goal_default.at[i, 10].set(0.0)
                goal_default = goal_default.at[i, 11].set(0.0)
                goal_default = goal_default.at[i, 12].set(0.0)
                goal_default = goal_default.at[i, 13].set(0.0)

            cost_fun = partial(cost, goal=goal_default.flatten(), u_ref=control.flatten())

    # Control box bounds
            max_torque = 5.0
        # n x 4 control inputs
            # control_bounds = (jnp.array([0, -10, -10, -10, 0, -10, -10, -10, 0, -10, -10, -10], dtype=jnp.float64),
            #                 jnp.array([100, 10, 10, 10, 100, 10, 10, 10, 100, 10, 10, 10], dtype=jnp.float64))
            control_bounds = (jnp.array([0, -max_torque, -max_torque, -max_torque], dtype=jnp.float64),
                            jnp.array([100, max_torque, max_torque, max_torque], dtype=jnp.float64))
            #copy it n times
            control_bounds = (jnp.tile(control_bounds[0], n), jnp.tile(control_bounds[1], n))
            print(control_bounds)

                        
            solver_options = dict(method=shootsqp.SQP_METHOD.SENS,
                                ddp_options={'ddp_gamma': 1e-4},
                                hess="full", verbose=True,
                                max_iter=100, ls_eta=0.49, ls_beta=0.8,
                                primal_tol=1e-3, dual_tol=1e-3, stall_check="abs",
                                debug=False)
            # print()
            solver = shootsqp.ShootSQP(Nx*n, Nu*n, T_total, dynamics, cost_fun, control_bounds,
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



    # fig, ax = render_scene()
            U, X = soln.primals

            jnp.save("solution_X_center_y_{}.npy".format(center_y), X)
            jnp.save("solution_U_center_y_{}.npy".format(center_y), U)

#         pt.max_iter = 100
# soln = solver.solve(x0, U0, X0)