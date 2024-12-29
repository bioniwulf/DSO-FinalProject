import numpy as np
import scipy.interpolate as si

class TrajectoryGenerator:
    """Generate the trajectory based on control points by B-splines."""

    def __init__(self, control_points: np.array):
        # Build 3d-order B-spline (and derivative) for target trajectory
        (self.trajectory_spline, self.trajectory_spline_derivative) =\
            self.generate_bspline(control_points, degree=3)

    def generate_bspline(self, cv: np.array, degree=3) -> list:
        """
        Generate a 2D B-spline by the set of control point.

        Args:
        ----
            cv (np.array): Set of control points.
            degree (int, optional): B-spline degree. Defaults to 3.

        Returns:
        -------
            list: (BSpline, BSpline) spline and derivative spline, accordingly.

        """
        control_vertices = np.asarray(cv)
        count = control_vertices.shape[0]

        # Prevent degree from exceeding count-1, otherwise splev will crash
        degree = np.clip(degree, 1, count - 1)

        # Calculate knot vector
        knot_vector = np.array([0] * degree + list(range(count - degree + 1)) + [count - degree] * degree,
                            dtype='int')

        # Normalize the knot vector to 1
        knot_vector = knot_vector / (count - degree)

        # Calculate splines
        spline = si.BSpline(knot_vector, control_vertices, degree)
        spline_derivative = spline.derivative(1)

        return (spline, spline_derivative)
    
    def get_trajectory_point(self, parameter: float) -> np.array:
        """
        Get trajectory point.

        Args:
        ----
            parameter (float): Trajectory normalized parameter [0, 1]

        Returns:
        -------
            np.array: Trajectory point.

        """
        return self.trajectory_spline(parameter)

    def get_trajectory_derivative(self, parameter: float) -> np.array:
        """
        Get trajectory derivative (tangent).

        Args:
        ----
            parameter (float): Trajectory normalized parameter [0, 1]

        Returns:
        -------
            np.array: Trajectory derivative.

        """
        return self.trajectory_spline_derivative(parameter)