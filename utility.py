import numpy as np
import matplotlib as mpl
import scipy.interpolate as si
import TrajectoryGenerator

def gen_arrow_head_marker(angle):
    """generate a marker to plot with matplotlib scatter, plot, ...

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    angle=0: positive x direction
    Parameters
    ----------
    angle : float
        rotation in rad
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """
    arr = np.array([[.1, .3], [.1, -.3], [1, 0], [.1, .3]])  # arrow shape
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
        ])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO,mpl.path.Path.LINETO, mpl.path.Path.CLOSEPOLY]
    arrow_head_marker = mpl.path.Path(arr, codes)
    return arrow_head_marker, scale

def get_middle_point(state1: np.array, state2: np.array) -> np.array:
    """
    Get the middle point between two points.

    Args:
    ----
        state1 (np.array): Point 1.
        state2 (np.array): Point 2.

    Returns:
        np.array: Middle point.

    """
    return np.array([(state1[0] + state2[0]) / 2, (state1[1] + state2[1]) / 2])

def calculate_perpendicular_angle(point1: np.array, point2: np.array) -> float:
    """
    Calculate the angle that is perpendicular to the line connected two points.

    Args:
    ----
        point1 (np.array): The point 1.
        point2 (np.array): The point 2.

    Returns:
    -------
        float: the value of the perpendicular angle (rad).

    """
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]

    # Calculate the perpendicular vector (90° clockwise rotation)
    perpendicular_vector = (dy, -dx)

    # Calculate the angle of the perpendicular vector
    angle = np.arctan2(perpendicular_vector[1], perpendicular_vector[0])
    return angle

def get_trajectory_part(trajectory: TrajectoryGenerator, parameter: float,
                        steps: int, target_velocity: float, delta_t: float) -> list:
    """
    Get part of the trajectory with the constant velocity.

    Args:
    ----
        trajectory (TrajectoryGenerator): trajectory spline.
        parameter (float): start parameter of the trajectory.
        steps (int): number of steps.

    Returns:
    -------
        list: list of target states and controls.

    """
    vector = []
    
    for st in range(0, steps):
        # check is this is the end of trajectory
        terminal_step = parameter > 1.0

        # Limit the trajectory parameter for predictions
        if terminal_step:
            parameter = 1.0

        # Get target coordinate
        target_point = trajectory.get_trajectory_point(parameter)

        # Calculate tangent vector at the point
        trajectory_derivative = trajectory.get_trajectory_derivative(parameter)
        target_yaw = np.arctan2(trajectory_derivative[1], trajectory_derivative[0])

        # Configure target state
        target_state = np.append(target_point, target_yaw)

        # Calculate a new parameter delta to preserve velocity constant
        parameter_delta_delta = target_velocity * delta_t / np.linalg.norm(trajectory_derivative)
        parameter = parameter + parameter_delta_delta

        # Add state vector within the horizon
        vector.append(target_state)
        # Add control vector within the horizon (zero linear velocity if this is the terminal state)
        vector.append(np.array([target_velocity if not terminal_step else 0.0, 0.0]))
    return vector

def get_circular_trajectory(trajectory_states:np.array, angular_velocity: float,
                            phase_shift: float, time: float, rotation_radius: float, delta_t: float) -> list:
    """
    Get part of the circular trajectory around the target trajectory.

    Args:
    ----
        trajectory_states (np.array): trajectory to rotate around.
        angular_velocity (float): angular velocity of circular rotation.
        phase_shift (float): phase shift of circular rotation.
        time (float): current time.

    Returns:
    -------
        list: circular trajectory and init vector (states and controls).

    """
    circular_trajectory = []
    circular_init_vector = []

    # -1 because predictions N+1, but we need only N points for initial states
    for i in range(trajectory_states.shape[1] - 1):
        target_point = trajectory_states[:2, i]

        circular_point_x = target_point[0] + rotation_radius * np.sin(time * angular_velocity + phase_shift)
        circular_point_y = target_point[1] + rotation_radius * np.cos(time * angular_velocity + phase_shift)

        # Configure target state
        target_state = np.array([])
        target_state = np.append(target_state, circular_point_x)
        target_state = np.append(target_state, circular_point_y)
        target_state = np.append(target_state, calculate_perpendicular_angle(target_state, target_point))

        time = time + delta_t

        # Add state vector within the horizon
        circular_init_vector.append(target_state)
        circular_trajectory.append(np.array(target_state))
        # Add control vector within the horizon (zero linear velocity if this is the terminal state)
        circular_init_vector.append(np.array([angular_velocity * rotation_radius, -angular_velocity]))

    circular_trajectory = np.vstack(circular_trajectory)
    return (circular_trajectory, circular_init_vector)