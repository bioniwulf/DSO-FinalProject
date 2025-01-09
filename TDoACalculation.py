import numpy as np

class TDoACalculation():
    """Class for TDoA (time-difference of arrival) calculation."""
    
    def __init__(self, parameter_max: float, discretization_step: int):
        """
        Class initialization.

        Args:
        ----
            parameter_max (float): max value (plus/minus) of hyperbolic parameter for discretization.
            discretization_step (int): steps for the hyperbolic parameter discretization.
        """
        self.hyperbolic_parameters = self.hyperbolic_discretization((-parameter_max, parameter_max), discretization_step)

    def convert2inertial(self, T: np.array, x_series: np.array, y_series: np.array) -> list:
        """
        Convert Hyperbolic frame to the Inertial frame.

        Args:
        ----
            T (np.array): transformation matrix: rotation and translation (3x3)
            x_series (np.array): series of X coordinates for conversion.
            y_series (np.array): series of Y coordinates for conversion.

        Returns:
        -------
            list: list of (X, Y) coordinates (np.array, np.array)

        """
        # Step 1: Organize into a matrix (2 x N)
        points = np.vstack((x_series, y_series))
        
        # Step 2: Convert to homogeneous coordinates (3 x N)
        points_homogeneous = np.vstack((points, np.ones(points.shape[1])))
        
        # Step 3: Apply the transformation
        transformed_points_homogeneous = T @ points_homogeneous

        # Step 4: Extract transformed coordinates (2 x N)
        transformed_points = transformed_points_homogeneous[:2, :]  # Only x' and y'
        
        x_transformed, y_transformed = transformed_points
        return (x_transformed, y_transformed)

    def find_transformation_matrix(self, point_1:np.array, point_2:np.array) -> np.array:
        """
        Find transformation matrix for transition from Hyperbolic frame to Inertial frame.

        Args:
        -----
            point_1 (np.array): the coordinate (x,y) of the first acoustic receiver.
            point_2 (np.array): the coordinate (x,y) of the second acoustic receiver.

        Returns:
        -------
            np.array: transformation matrix: rotation and translation (3x3)
            
        """
        # Step 1: New origin (midpoint)
        origin = (point_1 + point_2) / 2
        
        # Step 2: x-axis direction
        v_x = (point_1 - origin)
        v_x = v_x / np.linalg.norm(v_x)
        
        v_y = np.array([v_x[1], -v_x[0]])
        
        # Step 4: Rotation matrix
        R = np.column_stack((v_x, v_y))

        # Step 5: Translation vector
        T = -np.dot(R, origin)  # Apply rotation to the origin shift

        # Final result: Transformation matrix
        T_final = np.eye(3)
        T_final[:2, :2] = R
        T_final[:2, 2] = T

        T_inverse = np.eye(3)
        T_inverse[:2, :2] = R.T
        T_inverse[:2, 2] = -np.matmul(R.T, T)
        return T_inverse

    def find_range_diff(self,
                        pos_target: np.array,
                        pos_tracker_1: np.array,
                        pos_tracker_2: np.array) -> float:
        """
        Find range difference between acoustic receivers.

        Args:
        ----
            pos_target (np.array): Tte position of the acoustic source (x, y).
            pos_tracker_1 (np.array): the position of the first acoustic receiver (x, y).
            pos_tracker_2 (np.array): the position of the second acoustic receiver (x, y).

        Returns:
        -------
            float: signed range difference from the first receiver to the second one (it's not the distance between them).

        """
        return np.linalg.norm(pos_target - pos_tracker_1) - \
               np.linalg.norm(pos_target - pos_tracker_2)

    def calculate_semiaxis(self, range_difference: float,
                           tracker_distance: float) -> list:
        """
        Calculate hyperbolic's semiaxis.

        Args:
        ----
            range_difference (float): range difference between receivers.
            tracker_distance (float): distance between receivers.

        Returns:
        -------
            list: semiaxes (a^2, b^2).

        """
        a_square = (0.5 * range_difference)**2
        b_square = (0.5 * tracker_distance)**2 - a_square
        return (a_square, b_square)

    def hyperbolic_discretization(self, parameter_range:list, samples: int) -> list:
        """
        Prepare list of parameters for discretization that gives equal delta difference for sinh function.
        
        Args:
        ----
            parameter_range (list): range for parameter discretization (min, max).
            samples (int): steps for the hyperbolic parameter discretization.

        Returns:
        -------
            list: list of parameters.

        """
        y_pos_min = np.sinh(min(parameter_range))
        y_pos_max = np.sinh(max(parameter_range))
        delta = (y_pos_max - y_pos_min) / (samples - 1)
        parameter_list = list()
        for i in range(0, samples):
            parameter_list.append(np.arcsinh(y_pos_min + i * delta))
        return parameter_list

    def hyperbolic_solution(self,
                            a_square: float,
                            b_square: float,
                            fist_tracker_nearest: bool) -> list:
        """
        Find discretized set of hyperbolic solutions in hyperbolic frame.

        Args:
        ----
            a_square (float): hyperbolci semiaxes A.
            b_square (float): hyperbolci semiaxes B.
            fist_tracker_nearest (bool): is the first acoustic receiver nearer to the source.

        Returns:
        -------
            list: list of coordinates with discretized solutions, (X, Y) points (hyperbolic frame).

        """
        solution_x = np.sqrt(a_square) * np.cosh(self.hyperbolic_parameters)
        solution_y = np.sqrt(b_square) * np.sinh(self.hyperbolic_parameters)
        
        if not fist_tracker_nearest:
            solution_x = -solution_x

        return (solution_x, solution_y)

    def find_hyperbolic_solution(self, pos_target: np.array,
                                 pos_tracker_1: np.array,
                                 pos_tracker_2: np.array) -> list:
        """
        Find set of hyperbolic solutions.

        Args:
        ----
            pos_target (np.array): position of the acoustic source (x,y).
            pos_tracker_1 (np.array): position of the first acoustic receiver (x,y).
            pos_tracker_2 (np.array): position of the second acoustic receiver (x,y).

        Returns:
        -------
            list: list of coordinates with discretized solutions, (X, Y) points (inertial frame).

        """
        range_diff = self.find_range_diff(pos_target, pos_tracker_1, pos_tracker_2)
        
        (a_square, b_square) = self.calculate_semiaxis(range_diff, np.linalg.norm(pos_tracker_1 - pos_tracker_2))

        # Tracker 1 is the nearest if range_diff < 0
        (solution_x, solution_y) = self.hyperbolic_solution(a_square, b_square, range_diff < 0)

        transform_matrix = self.find_transformation_matrix(pos_tracker_1, pos_tracker_2)
        return self.convert2inertial(transform_matrix, solution_x, solution_y)