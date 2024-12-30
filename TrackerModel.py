from casadi import *
import numpy as np

class TrackerModel:
    """MPC Tracker class to facilitate modelling."""
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
        
        # Storage for the states throughout the simulation (python list of vectors)
        self._storage_states = []
        
        # Storage for predictions (python list of 2D matrix) throughout the simulation
        self._storage_predictions = []
        
        # Storage for control actions (2D matrix: (cumulative) x Control Vector Size)
        self._storage_control = []

    def create_objective(self, weight_state: np.array, weight_control: np.array) -> SX:
        """
        Create objective function for MPC (only for the tracker itself).

        Args:
        ----
            weight_state (np.array): weights for state (Diag matrix 3x3).
            weight_control (np.array): weights for control (Diag matrix 2x2).

        Returns:
        -------
            SX: Objective function (Casadi).

        """
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

    def create_constraints(self, time_discrete: float) -> list[SX]:
        """
        Create list of equality symbolic constraints for each state within prediction horizon.

        Args:
        ----
            time_discrete (float): time discrete, [s].

        Returns:
        -------
            list[SX]: list of symbolic constraints.

        """
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
    
    def get_global_limits(self) -> list:
        """
        Create list list of global limits for equality constraints.

        Returns:
        -------
            list: list of global limits.

        """
        st_size = self._sym_state_vector.size()[0]

        lower_bounds = np.zeros((st_size * (self._horizon + 1), 1))
        upper_bounds = np.zeros((st_size * (self._horizon + 1), 1))

        return (lower_bounds, upper_bounds)

    def get_state_limits(self, linear_velocity: list, angular_velocity: list) -> list[np.array]:
        """
        Create list of constraints on state and control vectors (lower and upper bounds).

        Args:
        ----
            linear_velocity (list): linear velocity constraints, (min, max), [m/s].
            angular_velocity (list): angular velocity constraints, (min, max), [rad/s].

        Returns:
        -------
            list[np.array]: list of lower and upper constraints.

        """
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

        # Set the upper bound for the optimized vector (both state and control vector) symmetrically
        upper_bounds = np.zeros((ctrl_size * self._horizon + st_size * (self._horizon + 1), 1))
        upper_bounds[0:(3 * self._horizon + 1) + 0:3] = np.inf
        upper_bounds[1:(3 * self._horizon + 1) + 1:3] = np.inf
        upper_bounds[2:(3 * self._horizon + 1) + 2:3] = np.inf
        upper_bounds[st_size * (self._horizon + 1) + 0:st_size * (self._horizon + 1) + ctrl_size * self._horizon:2] = np.max(linear_velocity)
        upper_bounds[st_size * (self._horizon + 1) + 1:st_size * (self._horizon + 1) + ctrl_size * self._horizon:2] = np.max(angular_velocity)

        return (lower_bounds, upper_bounds)

    def get_state_prediction(self, step: int = None) -> np.array:
        """
        Get symbolic state predictions within the prediction horizon.

        Args:
        ----
            step (int, optional): particular step of the prediction horizon. Defaults to None.

        Returns:
        -------
            np.array: symbolic state predictions.
        """
        if step is None:
            return self._sym_prediction_state
        else:
            return self._sym_prediction_state[:, step]

    @property
    def state_vector(self) -> DM:
        """
        Get access to the symbolic state vector of the tracker.

        Returns:
        -------
            DM: symbolic state vector of the tracker.

        """
        return self._sym_state_vector

    @property
    def control_vector(self) -> DM:
        """
        Get access to the symbolic control vector of the tracker.

        Returns:
        -------
            DM: symbolic control vector of the tracker.

        """
        return self._sym_control_vector

    @property
    def control_prediction(self) -> SX:
        """
        Get access to the symbolic control predictions within the prediction horizon.

        Returns:
        -------
            SX: symbolic control predictions.

        """
        return self._sym_prediction_control

    @property
    def kin_function(self) -> Function:
        """
        Get access to the symbolic kinematic function of the tracker.

        Returns:
        -------
            Function: symbolic kinematic function.

        """
        return self._sym_kin_function

    @property
    def state_init(self) -> SX:
        """
        Get access to the symbolic initial state of the tracker.

        Returns:
        -------
            SX: Symbolic initial state of the tracker.

        """
        return self._sym_state_init

    @property
    def state_reference(self) -> SX:
        """
        Get access to the symbolic reference states of the tracker within the prediction horizon.

        Returns:
        -------
            SX: symbolic reference states.

        """
        return self._sym_reference

    def storage_state(self, state: np.array) -> None:
        """
        Add state to the internal storage.

        Args:
        ----
            state (np.array): tracker state.

        """
        self._storage_states.append(state)

    def get_state_storage(self) -> np.array:
        """
        Return accumulated states as np matrix.

        Returns:
        -------
            np.array: storage of states.

        """
        return np.concatenate(self._storage_states, axis=1)

    def storage_control(self, control: np.array) -> None:
        """
        Add control to the internal storage.

        Args:
        ----
            control (np.array): tracker control.

        """
        self._storage_control.append(control)
    
    def get_control_storage(self) -> np.array:
        """
        Return accumulated controls as np matrix.

        Returns:
        -------
            np.array: storage of controls

        """
        return np.concatenate(self._storage_control, axis=0)

    def storage_predictions(self, predictions: np.array) -> None:
        """
        Add predictions to the internal storage.

        Args:
        ----
            predictions (np.array): predictions.

        """
        self._storage_predictions.append(predictions)
    
    def get_predictions_storage(self) -> np.array:
        """
        Return accumulated predictions as np matrix.

        Returns:
        -------
            np.array: storage of predictions.

        """
        return np.stack(self._storage_predictions, axis=2)
