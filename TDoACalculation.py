
import numpy as np

class TDoACalculation():
    
    def __init__(self, parameter_max: float):
        self.hyperbolic_parameters = self.hyperbolic_discretization((-parameter_max, parameter_max), 1000)

    def convert2inertial(self, T: np.array, x_series: np.array, y_series: np.array) -> list:
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

    def find_transformation_matrix(self, point_1:np.array,
                                   point_2:np.array):
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
        return np.linalg.norm(pos_target - pos_tracker_1) - \
               np.linalg.norm(pos_target - pos_tracker_2)

    def calculate_semiaxis(self, range_difference: float,
                           tracker_distance: float) -> list:
        a_square = (0.5 * range_difference)**2
        b_square = (0.5 * tracker_distance)**2 - a_square
        return (a_square, b_square)

    def hyperbolic_discretization(self, parameter_range:list, samples: int):
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
        solution_x = np.sqrt(a_square) * np.cosh(self.hyperbolic_parameters)
        solution_y = np.sqrt(b_square) * np.sinh(self.hyperbolic_parameters)
        
        if not fist_tracker_nearest:
            solution_x = -solution_x

        return (solution_x, solution_y)

    def find_hyperbolic_solution(self, pos_target: np.array,
                                 pos_tracker_1: np.array,
                                 pos_tracker_2: np.array) -> list:
        range_diff = self.find_range_diff(pos_target, pos_tracker_1, pos_tracker_2)
        
        (a_square, b_square) = self.calculate_semiaxis(range_diff, np.linalg.norm(pos_tracker_1 - pos_tracker_2))

        # Tracker 1 is the nearest if range_diff < 0
        (solution_x, solution_y) = self.hyperbolic_solution(a_square, b_square, range_diff < 0)

        transform_matrix = self.find_transformation_matrix(pos_tracker_1, pos_tracker_2)
        return self.convert2inertial(transform_matrix, solution_x, solution_y)