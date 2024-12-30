from casadi import *
import numpy as np

class TrackerModel:
    def __init__(self, label: str, prediction_horizon: int):
        self._label = label
        self._horizon = prediction_horizon

        # Define the states and state vector
        sym_x = SX.sym(self._label + '_x')
        sym_y = SX.sym(self._label + '_y')
        sym_theta = SX.sym(self._label + '_theta')
        self._sym_state_vector = vertcat(sym_x, sym_y, sym_theta)

        # Define the controls and control vector
        sym_v = SX.sym(self._label + 'v')
        sym_omega = SX.sym(self._label + 'omega')
        self._sym_control_vector = vertcat(sym_v, sym_omega)

        st_size   = self._sym_state_vector.size()[0]
        ctrl_size = self._sym_control_vector.size()[0]

        # Define the Right-Hand Side (RHS) of kinematic model
        rhs = np.array([[sym_v * np.cos(sym_theta)],
                        [sym_v * np.sin(sym_theta)],
                        [sym_omega]])

        # Define the symbolic kinematic model mapping of RHS for Casadi
        # This mapping gives the prediction of the next state knowing current state and control
        self._sym_kin_function = Function(self._label + 'KinFunction',
                                          [self._sym_state_vector, self._sym_control_vector], [rhs])

        # Define initial state
        self._sym_state_init = SX.sym(self._label + 'Init', st_size)

        # Define reference state and control for all prediction steps
        self._sym_reference = SX.sym(self._label + 'Reference', self._horizon * (st_size + ctrl_size))

        # Define the symbolic control prediction for all prediction steps
        self._sym_prediction_control = SX.sym(self._label + 'Control', ctrl_size, self._horizon)

        # Define the symbolic state predictions for all prediction steps + 1
        self._sym_prediction_state = SX.sym(self._label + 'Prediction', st_size, (self._horizon + 1))

    def create_objective(self, weight_state: np.array, weight_control: np.array):
        # Define objective function
        objective_fn = 0
        
        st_size   = self._sym_state_vector.size()[0]
        ctrl_size = self._sym_control_vector.size()[0]

        for step in range(0, self._horizon):
            # Get state and control on the current step
            state = self._sym_prediction_state[:, step]
            control = self._sym_prediction_control[:, step]

            # Accumulate objective function for all steps in the prediction horizon
            period = st_size + ctrl_size

            # Target state to control on the current step of prediction
            target_state = self._sym_reference[step * period : step * period + st_size]
            # Change offset for the target control (this is initial state + first control state)
            offset = st_size + st_size

            # Target control
            target_control = self._sym_reference[step * period + st_size : step * period + st_size + ctrl_size]
            objective_fn = objective_fn + \
                (state - target_state).T @ weight_state @ (state - target_state) + \
                (control - target_control).T @ weight_control @ (control - target_control)
        return objective_fn

    def create_constraints(self, time_discrete: float):
        # Temporary define equality constraints as a python list
        constraints_equality = []
        constraints_equality.append(self._sym_prediction_state[:, 0] - self._sym_state_init)

        # Add constraints connected with kinematic model of the tracker 
        for step in range(0, self._horizon):
            # Get the current and the next state
            state = self._sym_prediction_state[:, step]
            state_next = self._sym_prediction_state[:, step + 1]
            # Get the current control
            control = self._sym_prediction_control[:, step]
            # Compute rhs function value for the current state and control
            rhs_value = self._sym_kin_function(state, control)
            # Make prediction for the next step
            state_next_euler = state + (time_discrete * rhs_value)
            constraints_equality.append(state_next - state_next_euler)
        return constraints_equality
    
    def get_global_limits(self):
        st_size = self._sym_state_vector.size()[0]
        
        lower_bounds = np.zeros((st_size * (self._horizon + 1), 1))
        upper_bounds = np.zeros((st_size * (self._horizon + 1), 1))
         
        return (lower_bounds, upper_bounds)

    def get_state_limits(self, linear_velocity: list, angular_velocity: list):
        st_size   = self._sym_state_vector.size()[0]
        ctrl_size = self._sym_control_vector.size()[0]

        # Set the lower bound for the optimized vector (both state and control vector)
        lower_bounds = np.zeros((st_size * (self._horizon + 1) + ctrl_size * self._horizon, 1))
        # Fist let's fill the limit for state vector. First st_size * (self._horizon + 1) part
        # (x, y parts)
        lower_bounds[0:(3 * self._horizon + 1) + 0:3] = -np.inf
        lower_bounds[1:(3 * self._horizon + 1) + 1:3] = -np.inf
        # (\phi part)
        lower_bounds[2:(3 * self._horizon + 1) + 2:3] = -np.inf
        # Second let's fill the limit for control vector. Second ctrl_size * self._horizon part
        lower_bounds[st_size * (self._horizon + 1) + 0:st_size * (self._horizon + 1) + ctrl_size * self._horizon:2] = np.min(linear_velocity)
        lower_bounds[st_size * (self._horizon + 1) + 1:st_size * (self._horizon + 1) + ctrl_size * self._horizon:2] = np.min(angular_velocity)

        # Set the upper bound for the optimized vector (both state and control vector) symmetricaly
        upper_bounds = np.zeros((ctrl_size * self._horizon + st_size * (self._horizon + 1), 1))
        upper_bounds[0:(3 * self._horizon + 1) + 0:3] = np.inf
        upper_bounds[1:(3 * self._horizon + 1) + 1:3] = np.inf
        upper_bounds[2:(3 * self._horizon + 1) + 2:3] = np.inf
        upper_bounds[st_size * (self._horizon + 1) + 0:st_size * (self._horizon + 1) + ctrl_size * self._horizon:2] = np.max(linear_velocity)
        upper_bounds[st_size * (self._horizon + 1) + 1:st_size * (self._horizon + 1) + ctrl_size * self._horizon:2] = np.max(angular_velocity)
        
        return (lower_bounds, upper_bounds)

    def get_state_prediction(self, step=None):
        if step is None:
            return self._sym_prediction_state
        else:
            return self._sym_prediction_state[:, step]

    def init_storages(self):
        # Storage for the states throughout the simulation (python list of vectors)
        self._storage_states = []
        # Storage for predictions (python list of 2D matrix) throughout the simulation
        self._storage_predictions = []
        # Storage for control actions (2D matrix: (cumulative) x Control Vector Size)
        self._storage_control = []

    def convert_storages(self):
        # Convert list of matrix to 3D np matrix and put list into the 3rd dimension
        self._storage_predictions = np.stack(self._storage_predictions, axis=2)

        # Concatinate all data along the second axis (because state is the column-vector)
        self._storage_states = np.concatenate(self._storage_states, axis=1)

        # Concatinate all data along the first axis (because control is the row-vector)
        self._storage_control = np.concatenate(self._storage_control, axis=0)
    
    @property
    def storage_states(self):
        return self._storage_states
    
    @property
    def storage_predictions(self):
        return self._storage_predictions
    
    @property
    def storage_target_states(self):
        return self._storage_target_states
    
    @property
    def storage_control(self):
        return self._storage_control
    

    @property
    def state_init(self):
        return self._sym_state_init

    @property
    def state_reference(self):
        return self._sym_reference

    @property
    def state_vector(self):
        return self._sym_state_vector

    @property
    def control_vector(self):
        return self._sym_control_vector

    @property
    def control_prediction(self):
        return self._sym_prediction_control

    @property
    def kin_function(self):
        return self._sym_kin_function