
import numpy as np
import do_mpc
from .util import FixedKeysDict, SetDict


class Simulator(object):
    def __init__(self, f, h, dt=0.01, n=None, m=None,
                 state_names=None, input_names=None, measurement_names=None,
                 params_simulator=None):

        """ Simulator.

        :param callable f: dynamics function f(X, U, t)
        :param callable h: measurement function h(X, U, t)
        :param float dt: sampling time in seconds
        :param int n: number of states, optional but cannot be set if state_names is set
        :param int m: number of inputs, optional but cannot be set if input_names is set
        :param list state_names: names of states, optional but cannot be set if n is set
        :param list input_names: names of inputs, optional but cannot be set if m is set
        :param list measurement_names: names of measurements, optional
        :param dict params_simulator: simulation parameters, optional
        """

        self.f = f
        self.h = h
        self.dt = dt

        # Set state names
        if state_names is None:  # default state names
            if n is None:
                raise ValueError('must set state_names or n')
            else:
                self.n = int(n)

            self.state_names = ['x_' + str(n) for n in range(self.n)]
        else:  # state names given
            if n is not None:
                raise ValueError('cannot set state_names and n')

            self.state_names = list(state_names)
            self.n = len(self.state_names)
            # if len(self.state_names) != self.n:
            #     raise ValueError('state_names must have length equal to x0')

        # Set input names
        if input_names is None:  # default input names
            if m is None:
                raise ValueError('must set input_names or m')
            else:
                self.m = int(m)

            self.input_names = ['u_' + str(m) for m in range(self.m)]
        else:  # input names given
            if m is not None:
                raise ValueError('cannot set in and n')

            self.input_names = list(input_names)
            self.m = len(self.input_names)
            # if len(self.input_names) != self.m:
            #     raise ValueError('input_names must have length equal to u0')

        # Run measurement function to get measurement size
        x0 = np.ones(self.n)
        u0 = np.ones(self.m)
        y = self.h(x0, u0, 0)
        self.p = len(y)  # number of measurements

        # Set measurement names
        if measurement_names is None:  # default measurement names
            self.measurement_names = ['y_' + str(p) for p in range(self.p)]
        else:
            self.measurement_names = measurement_names
            if len(self.measurement_names) != self.p:
                raise ValueError('measurement_names must have length equal to y')

        # Initialize time vector
        w = 10  # initialize for w time-steps, but this can change later
        self.time = np.arange(0, w * self.dt + self.dt / 2, step=self.dt)  # time vector

        # Define initial states & initialize state time-series
        self.x0 = {}
        self.x = {}
        for n, state_name in enumerate(self.state_names):
            self.x0[state_name] = x0[n]
            self.x[state_name] = x0[n] * np.ones(w)

        self.x0 = FixedKeysDict(self.x0)

        # Initialize input time-series
        self.u = {}
        for m, input_name in enumerate(self.input_names):
            self.u[input_name] = u0[m] * np.ones(w)

        self.u = FixedKeysDict(self.u)

        # Initialize measurement time-series
        self.y = {}
        for p, measurement_name in enumerate(self.measurement_names):
            self.y[measurement_name] = 0.0 * np.ones(w)

        self.y = FixedKeysDict(self.y)

        # Define continuous-time MPC model
        self.model = do_mpc.model.Model('continuous')

        # Define state variables
        X = []
        for n, state_name in enumerate(self.state_names):
            x = self.model.set_variable(var_type='_x', var_name=state_name, shape=(1, 1))
            X.append(x)

        # Define input variables
        U = []
        for m, input_name in enumerate(self.input_names):
            u = self.model.set_variable(var_type='_u', var_name=input_name, shape=(1, 1))
            U.append(u)

        # Define dynamics
        Xdot = self.f(X, U, 0)
        for n, state_name in enumerate(self.state_names):
            self.model.set_rhs(state_name, Xdot[n])

        # Build model
        self.model.setup()

        # Define simulator & simulator parameters
        self.simulator = do_mpc.simulator.Simulator(self.model)

        # Set simulation parameters
        if params_simulator is None:
            self.params_simulator = {
                'integration_tool': 'idas',  # cvodes, idas
                'abstol': 1e-8,
                'reltol': 1e-8,
                't_step': self.dt
            }
        else:
            self.params_simulator = params_simulator

        self.simulator.set_param(**self.params_simulator)
        self.simulator.setup()

    def set_initial_state(self, x0):
        """ Update the initial state.
        """

        if x0 is not None:  # initial state given
            if isinstance(x0, dict):  # in dict format
                SetDict().set_dict_with_overwrite(self.x0, x0)  # update only the states in the dict given
            elif isinstance(x0, list) or isinstance(x0, tuple) or (
            x0, np.ndarray):  # list, tuple,  or numpy array format
                x0 = np.array(x0).squeeze()
                for n, key in enumerate(self.x0.keys()):  # each state
                    self.x0[key] = x0[n]
            else:
                raise Exception('x0 must be either a dict, tuple, list, or numpy array')

    def set_inputs(self, u):
        """ Update the inputs.
        """

        if u is not None:  # inputs given
            if isinstance(u, dict):  # in dict format
                SetDict().set_dict_with_overwrite(self.u, u)  # update only the inputs in the dict given
            elif isinstance(u, list) or isinstance(u, tuple):  # list or tuple format, each input vector in each element
                for n, k in enumerate(self.u.keys()):  # each input
                    self.u[k] = u[n]
            elif isinstance(u, np.ndarray):  # numpy array format given as matrix where columns are the different inputs
                if len(u.shape) <= 1:  # given as 1d array, so convert to column vector
                    u = np.atleast_2d(u).T

                for m, key in enumerate(self.u.keys()):  # each input
                    self.u[key] = u[:, m]

            else:
                raise Exception('u must be either a dict, tuple, list, or numpy array')

        # Make sure inputs are the same size
        points = np.array([self.u[key].shape[0] for key in self.u.keys()])
        points_check = points == points[0]
        if not np.all(points_check):
            raise Exception('inputs are not the same size')

    def simulate(self, x0=None, u=None, return_full_output=False):
        """
        Simulate the system.

        :params x0: initial state dict or array
        :params u: input dict or array
        :params return_full_output: boolean to run (time, x, u, y) instead of y
        """

        # Update the initial state
        self.set_initial_state(x0=x0.copy())

        # Update the inputs
        self.set_inputs(u=u)

        # Concatenate the inputs, where rows are individual inputs and columns are time-steps
        u_sim = np.vstack(list(self.u.values())).T
        n_point = u_sim.shape[0]

        # Update time vector
        T = (n_point - 1) * self.dt
        self.time = np.linspace(0, T, num=n_point)

        # Set array to store simulated states, where rows are individual states and columns are time-steps
        x_step = np.array(list(self.x0.values()))  # initialize state
        x_sim = np.nan * np.zeros((n_point, self.n))
        x_sim[0, :] = x_step.copy()

        # Initialize the simulator
        self.simulator.t0 = self.time[0]
        self.simulator.x0 = x_step.copy()
        self.simulator.set_initial_guess()

        # Run simulation
        for k in range(1, n_point):
            # Set input
            u_step = u_sim[k - 1:k, :].T

            # Store inputs
            u_sim[k - 1, :] = u_step.squeeze()

            # Simulate one time step given current inputs
            x_step = self.simulator.make_step(u_step)

            # Store new states
            x_sim[k, :] = x_step.squeeze()

        # Update the inputs
        self.set_inputs(u=u_sim)

        # Update state trajectory
        for n, key in enumerate(self.x.keys()):
            self.x[key] = x_sim[:, n]

        # Calculate measurements
        x_list = list(self.x.values())
        u_list = list(self.u.values())
        y = self.h(x_list, u_list, 0)

        # Set outputs
        self.y = {}
        for p, measurement_name in enumerate(self.measurement_names):
            self.y[measurement_name] = y[p]

        # Return the outputs in array format
        y_array = np.vstack(list(self.y.values())).T

        if return_full_output:
            return self.time.copy(), self.x.copy(), self.u.copy(), self.u.copy()
        else:
            return y_array

    def get_time_states_input_measurements(self):
        return self.time.copy(), self.x.copy(), self.u.copy(), self.u.copy()


