from enum import Enum
import copy
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


class ExitStatus(Enum):
    """ Exit status values indicate the reason for simulation termination. """
    COMPLETE = 'Success: End reached.'
    TIMEOUT = 'Timeout: Simulation end time reached.'
    INF_VALUE = 'Failure: Your controller returned inf motor speeds.'
    NAN_VALUE = 'Failure: Your controller returned nan motor speeds.'
    OVER_SPEED = 'Failure: Your quadrotor is out of control; it is going faster than 100 m/s. The Guinness World Speed Record is 73 m/s.'
    OVER_SPIN = 'Failure: Your quadrotor is out of control; it is spinning faster than 100 rad/s. The onboard IMU can only measure up to 52 rad/s (3000 deg/s).'
    FLY_AWAY = 'Failure: Your quadrotor is out of control; it flew away with a position error greater than 20 meters.'
    COLLISION = 'Failure: Your quadrotor collided with an object.'


def simulate(world, initial_states, vehicles, controllers, trajectories, t_final, t_step, safety_margin,
             terminate=None):
    initial_states = [{k: np.array(v) for k, v in initial_state.items()} for initial_state in initial_states]

    if terminate is None:  # Default exit. Terminate at final position of trajectory.
        for initial_state, trajectory in zip(initial_states, trajectories):
            normal_exit = traj_end_exit(initial_state, trajectory, using_vio=False)
            if normal_exit == True:
                break
    elif terminate is False:  # Never exit before timeout.
        normal_exit = lambda t, s: None
    else:  # Custom exit.
        normal_exit = terminate

    time = [0]
    states = [[copy.deepcopy(initial_state)] for initial_state in initial_states]

    flats = [[sanitize_trajectory_dic(trajectory.update(time[-1]))] for trajectory in trajectories]
    controls = [[sanitize_control_dic(controller.update(time[-1], state[-1], flat[-1]))] for controller, state, flat in zip(controllers, states, flats)]
    exit_status = None

    while True:
        for state, flat, control in zip(states, flats, controls):
            exit_status = exit_status or safety_exit(world, safety_margin, state[-1], flat[-1], control[-1])
            exit_status = exit_status or normal_exit(time[-1], state[-1])
            exit_status = exit_status or time_exit(time[-1], t_final)
        if exit_status:
            break
        time.append(time[-1] + t_step)
        for state, control, flat, trajectory, controller, vehicle in zip(states, controls, flats, trajectories, controllers, vehicles):
            state.append(vehicle.step(state[-1], control[-1], t_step))
            flat.append(sanitize_trajectory_dic(trajectory.update(time[-1])))
            control.append(sanitize_control_dic(controller.update(time[-1], state[-1], flat[-1])))

    time = np.array(time, dtype=float)
    states = [merge_dicts(state) for state in states]
    controls = [merge_dicts(control) for control in controls]
    flats = [merge_dicts(flat) for flat in flats]

    return (time, states, controls, flats, exit_status)


def merge_dicts(dicts_in):
    """
    Concatenates contents of a list of N state dicts into a single dict by
    prepending a new dimension of size N. This is more convenient for plotting
    and analysis. Requires dicts to have consistent keys and have values that
    are numpy arrays.
    """
    dict_out = {}
    for k in dicts_in[0].keys():
        dict_out[k] = []
        for d in dicts_in:
            dict_out[k].append(d[k])
        dict_out[k] = np.array(dict_out[k])
    return dict_out


def traj_end_exit(initial_state, trajectory, using_vio=False):
    """
    Returns a exit function. The exit function returns an exit status message if
    the quadrotor is near hover at the end of the provided trajectory. If the
    initial state is already at the end of the trajectory, the simulation will
    run for at least one second before testing again.
    """

    xf = trajectory.update(np.inf)['x']
    yawf = trajectory.update(np.inf)['yaw']
    rotf = Rotation.from_rotvec(yawf * np.array([0, 0, 1]))  # create rotation object that describes yaw
    if np.array_equal(initial_state['x'], xf):
        min_time = 1.0
    else:
        min_time = 0

    def exit_fn(time, state):
        cur_attitude = Rotation.from_quat(state['q'])
        err_attitude = rotf * cur_attitude.inv()  # Rotation between current and final
        angle = norm(err_attitude.as_rotvec())  # angle in radians from vertical
        # Success is reaching near-zero speed with near-zero position error.
        if using_vio:
            # set larger threshold for VIO due to noisy measurements
            if time >= min_time and norm(state['x'] - xf) < 1 and norm(state['v']) <= 1 and angle <= 1:
                return ExitStatus.COMPLETE
        else:
            if time >= min_time and norm(state['x'] - xf) < 0.02 and norm(state['v']) <= 0.03 and angle <= 0.02:
                return ExitStatus.COMPLETE
        return None

    return exit_fn


def time_exit(time, t_final):
    """
    Return exit status if the time exceeds t_final, otherwise None.
    """
    if time >= t_final:
        return ExitStatus.TIMEOUT
    return None


def safety_exit(world, margin, state, flat, control):
    """
    Return exit status if any safety condition is violated, otherwise None.
    """
    if np.any(np.isinf(control['cmd_motor_speeds'])):
        return ExitStatus.INF_VALUE
    if np.any(np.isnan(control['cmd_motor_speeds'])):
        return ExitStatus.NAN_VALUE
    if np.any(np.abs(state['v']) > 100):
        return ExitStatus.OVER_SPEED
    if np.any(np.abs(state['w']) > 100):
        return ExitStatus.OVER_SPIN
    if np.any(np.abs(state['x'] - flat['x']) > 20):
        return ExitStatus.FLY_AWAY

    if len(world.world.get('blocks', [])) > 0:
        # If a world has objects in it we need to check for collisions.  
        collision_pts = world.path_collisions(state['x'], margin)
        no_collision = collision_pts.size == 0
        if not no_collision:
            return ExitStatus.COLLISION
    return None


def sanitize_control_dic(control_dic):
    """
    Return a sanitized version of the control dictionary where all of the elements are np arrays
    """
    control_dic['cmd_motor_speeds'] = np.asarray(control_dic['cmd_motor_speeds'], np.float64).ravel()
    control_dic['cmd_moment'] = np.asarray(control_dic['cmd_moment'], np.float64).ravel()
    control_dic['cmd_q'] = np.asarray(control_dic['cmd_q'], np.float64).ravel()
    return control_dic


def sanitize_trajectory_dic(trajectory_dic):
    """
    Return a sanitized version of the trajectory dictionary where all of the elements are np arrays
    """
    trajectory_dic['x'] = np.asarray(trajectory_dic['x'], np.float64).ravel()
    trajectory_dic['x_dot'] = np.asarray(trajectory_dic['x_dot'], np.float64).ravel()
    trajectory_dic['x_ddot'] = np.asarray(trajectory_dic['x_ddot'], np.float64).ravel()
    trajectory_dic['x_dddot'] = np.asarray(trajectory_dic['x_dddot'], np.float64).ravel()
    trajectory_dic['x_ddddot'] = np.asarray(trajectory_dic['x_ddddot'], np.float64).ravel()

    return trajectory_dic
