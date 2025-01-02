import numpy as np
import scipy.interpolate as si
from TrajectoryGenerator import TrajectoryGenerator

class TargetModel:
    """Target model to facilitate modelling."""
    def __init__(self, control_points: np.array, target_velocity: float):
        # Control points of the trajectory
        self.control_points = control_points
        
        # Target velocity
        self.target_velocity = target_velocity
        
        # Current parameter of the trajectory
        self.current_parameter = 0

        # Create trajectory instance for the target
        self.trajectory = TrajectoryGenerator(self.control_points)

    def get_trajectory(self) -> np.array:
        """
        Get full trajectory of the target.

        Returns:
            np.array: Trajectory of the target.

        """
        # Get all list of normalized parameters to build the target trajectory
        parameter_list = np.linspace(0, 1, 100)
        trajectory_points = self.trajectory.get_trajectory_point(parameter_list)
        return trajectory_points

    def get_tracker_telemetry(self) -> list[np.array, float]:
        """
        Get current telemetry of the target (state and velocity).

        Returns:
        -------
            list[np.array, float]: state (3x1) and linear velocity

        """
        # Get tracker position
        target_point = self.trajectory.get_trajectory_point(self.current_parameter)

        # Get trajectory derivative and heading of the tracker
        trajectory_derivative = self.trajectory.get_trajectory_derivative(self.current_parameter)
        target_yaw = np.arctan2(trajectory_derivative[1], trajectory_derivative[0])

        # Form the state of the tracker
        state = np.append(target_point, target_yaw).reshape((-1, 1))

        # We assume that linear velocity is constant
        return (state, self.target_velocity if self.current_parameter < 1.0 else 0.0)

    def update_tracker_position(self, delta_t: float):
        """
        Update tracker position.

        Args:
        ----
            delta_t (float): delta time of simulation.

        """
        if self.current_parameter > 1.0:
            self.current_parameter = 1.0

        trajectory_derivative = self.trajectory.get_trajectory_derivative(self.current_parameter)

        # Calculate a new parameter delta for target motion to preserve velocity constant
        parameter_delta = self.target_velocity * delta_t / np.linalg.norm(trajectory_derivative)
        self.current_parameter = self.current_parameter + parameter_delta