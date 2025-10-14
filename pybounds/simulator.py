
import warnings
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
import casadi
import do_mpc
from .util import FixedKeysDict, SetDict


class Simulator(object):
    def __init__(self, f, h, dt=0.01,
                 discrete=False,
                 n=None, m=None,
                 state_names=None, input_names=None, measurement_names=None,
                 params_simulator=None, mpc_horizon=10):

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
        self.h = ensure_float_output(h)
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
            if len(self.state_names) != self.n:
                raise ValueError('state_names must have length equal to x0')

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
            if len(self.input_names) != self.m:
                raise ValueError('input_names must have length equal to u0')

        # Run measurement function to get measurement size
        x0 = np.ones(self.n)
        u0 = np.ones(self.m)
        y = self.h(np.ravel(x0), np.ravel(u0))
        self.p = len(y)  # number of measurements

        # Set measurement names
        if measurement_names is None:  # default measurement names
            self.measurement_names = ['y_' + str(p) for p in range(self.p)]
        else:
            self.measurement_names = measurement_names
            if len(self.measurement_names) != self.p:
                raise ValueError('measurement_names must have length equal to y')

        # Initialize time vector
        self.w = 11  # initialize for w time-steps, but this can change later
        self.time = np.arange(0, self.w * self.dt, step=self.dt)  # time vector

        # Define initial states & initialize state time-series
        self.x0 = {}
        self.x = {}
        for n, state_name in enumerate(self.state_names):
            self.x0[state_name] = x0[n]
            self.x[state_name] = x0[n] * np.ones(self.w)

        self.x0 = FixedKeysDict(self.x0)
        self.x = FixedKeysDict(self.x)

        # Initialize input time-series
        self.u = {}
        for m, input_name in enumerate(self.input_names):
            self.u[input_name] = u0[m] * np.ones(self.w)

        self.u = FixedKeysDict(self.u)

        # Initialize measurement time-series
        self.y = {}
        for p, measurement_name in enumerate(self.measurement_names):
            self.y[measurement_name] = 0.0 * np.ones(self.w)

        self.y = FixedKeysDict(self.y)

        # Initialize state set-points
        self.setpoint = {}
        for n, state_name in enumerate(self.state_names):
            self.setpoint[state_name] = 0.0 * np.ones(self.w)

        self.setpoint = FixedKeysDict(self.setpoint)

        # Define MPC model
        if discrete:
            model_type = 'discrete'
        else:
            model_type = 'continuous'
        self.model = do_mpc.model.Model(model_type)

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
        Xdot = self.f(X, U)
        for n, state_name in enumerate(self.state_names):
            self.model.set_rhs(state_name, casadi.SX(Xdot[n]))

        # Add time-varying set-point variables for later use with MPC
        for n, state_name in enumerate(self.state_names):
            x = self.model.set_variable(var_type='_tvp', var_name=state_name + str('_set'), shape=(1, 1))

        # Build model
        self.model.setup()

        # Define simulator & simulator parameters
        self.simulator = do_mpc.simulator.Simulator(self.model)

        # Set simulation parameters
        if params_simulator is None:
            if self.model.model_type == 'continuous':
                self.params_simulator = {
                    'integration_tool': 'idas',  # cvodes, idas
                    'abstol': 1e-8,
                    'reltol': 1e-8,
                    't_step': self.dt
                }
            else:
                self.params_simulator = {
                    't_step': self.dt
                }
        else:
            self.params_simulator = params_simulator

        self.simulator.set_param(**self.params_simulator)

        # Setup MPC
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc_horizon = mpc_horizon
        setup_mpc = {
            'n_horizon': self.mpc_horizon,
            'n_robust': 0,
            'open_loop': 0,
            't_step': self.dt,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 2,
            'collocation_ni': 1,
            'store_full_solution': False,

            # Use MA27 linear solver in ipopt for faster calculations:
            'nlpsol_opts': {'ipopt.linear_solver': 'mumps',  # mumps, MA27
                            'ipopt.print_level': 0,
                            'ipopt.sb': 'yes',
                            'print_time': 0,
                            }
        }

        self.mpc.set_param(**setup_mpc)

        # Get template's for MPC time-varying parameters
        self.mpc_tvp_template = self.mpc.get_tvp_template()
        self.simulator_tvp_template = self.simulator.get_tvp_template()

        # Set time-varying set-point functions
        self.mpc.set_tvp_fun(self.mpc_tvp_function)
        self.simulator.set_tvp_fun(self.simulator_tvp_function)

        # Setup simulator
        self.simulator.setup()

    def simulator_tvp_function(self, t):
        """ Set the set-point function for MPC simulator.
        :param t: current time
        """

        mpc_horizon = self.mpc._settings.n_horizon

        # Set current step index
        k_step = int(np.round(t / self.dt))
        if k_step >= mpc_horizon:  # point is beyond end of input data
            k_step = mpc_horizon - 1  # set point beyond input data to last point

        # Update current set-point
        for n, state_name in enumerate(self.state_names):
            self.simulator_tvp_template[state_name + '_set'] = self.setpoint[state_name][k_step]

        return self.simulator_tvp_template

    def mpc_tvp_function(self, t):
        """ Set the set-point function for MPC optimizer.
        """

        mpc_horizon = self.mpc._settings.n_horizon

        # Set current step index
        k_step = int(np.round(t / self.dt))

        # Update set-point time horizon
        for k in range(mpc_horizon + 1):
            k_set = k_step + k
            if k_set >= self.w:  # horizon is beyond end of input data
                k_set = self.w - 1  # set part of horizon beyond input data to last point

            # Update each set-point over time horizon
            for n, state_name in enumerate(self.state_names):
                self.mpc_tvp_template['_tvp', k, state_name + '_set'] = self.setpoint[state_name][k_set]

        return self.mpc_tvp_template

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

    def update_dict(self, data=None, name=None):
        """ Update.
        """

        update = getattr(self, name)

        if data is not None:  # data given
            if isinstance(data, dict):  # in dict format
                SetDict().set_dict_with_overwrite(update, data)  # update only the inputs in the dict given

                # Normalize unset keys to be the length of the set keys be repeating the 1st element
                unset_key = set(update.keys()) - set(data.keys())  # find keys that were not set
                set_key = set(data.keys())  # find keys that were set
                if unset_key != set_key:
                    w = data[list(set_key)[0]].squeeze().shape[0]  # size of 1st set key
                    for k in unset_key:  # update each unset key
                        update[k] = update[k][0] * np.ones(w)

            elif isinstance(data, list) or isinstance(data, tuple):  # list or tuple format, each input vector in each element
                for n, k in enumerate(update.keys()):  # each state
                    update[k] = data[n]
            elif isinstance(data, np.ndarray):  # numpy array format given as matrix where columns are the different inputs
                if len(data.shape) <= 1:  # given as 1d array, so convert to column vector
                    data = np.atleast_2d(data).T

                for n, key in enumerate(update.keys()):  # each input
                    update[key] = data[:, n]

            else:
                raise Exception(name + ' must be either a dict, tuple, list, or numpy array')

        # Make sure inputs are the same size
        points = np.array([update[key].shape[0] for key in update.keys()])
        points_check = points == points[0]
        if not np.all(points_check):
            raise Exception(name + ' not the same size')

    def simulate(self, x0=None, u=None, aux=None, mpc=False, return_full_output=False):
        """
        Simulate the system.

        :params x0: initial state dict or array
        :params u: input dict or array, if True then mpc must be None
        :params aux: auxiliary input
        :params mpc: boolean to run MPC, if True then u must be None
        :params return_full_output: boolean to run (time, x, u, y) instead of y
        """

        if (mpc is True) and (u is not None):
            raise Exception('u must be None if running MPC')

        if (mpc is False) and (u is None):
            warnings.warn('not running MPC or setting u directly')

        # Update the initial state
        if x0 is None:
            if mpc:  # set the initial state to start at set-point if running MPC
                x0 = {}
                for state_name in self.state_names:
                    x0[state_name] = self.setpoint[state_name][0]

                self.set_initial_state(x0=x0)
        else:
            self.set_initial_state(x0=x0)

        # Update the inputs
        self.update_dict(u, name='u')

        # Concatenate the inputs, where rows are individual inputs and columns are time-steps
        if mpc:
            self.w = np.vstack(list(self.setpoint.values())).shape[1]
            u_sim = np.zeros((self.w, self.m))  # preallocate input array
        else:
            self.w = np.vstack(list(self.u.values())).shape[1]
            u_sim = np.vstack(list(self.u.values())).T

        # Update time vector
        T = (self.w - 1) * self.dt
        self.time = np.linspace(0, T, num=self.w)

        # Set array to store simulated states, where rows are individual states and columns are time-steps
        x_step = np.array(list(self.x0.values()))  # initialize state
        x = np.nan * np.zeros((self.w, self.n))
        x[0, :] = x_step.copy()

        # Set array to store simulated measurements
        y = np.nan * np.zeros((self.w, self.p))

        # Initialize the simulator
        # self.simulator = do_mpc.simulator.Simulator(self.model)
        # self.simulator.set_param(**self.params_simulator)
        # self.simulator.set_tvp_fun(self.simulator_tvp_function)
        # self.simulator.setup()
        self.simulator.reset_history()  # reset simulator history (super important for speed)
        self.simulator.t0 = self.time[0]
        self.simulator.x0 = x_step.copy()
        self.simulator.set_initial_guess()

        # Initialize MPC
        if mpc:
            self.mpc.setup()
            self.mpc.t0 = self.time[0]
            self.mpc.x0 = x_step.copy()
            self.mpc.u0 = np.zeros((self.m, 1))
            self.mpc.set_initial_guess()

        # Run simulation
        for k in range(1, self.w):
            # Set input
            if mpc:  # run MPC step
                u_step = self.mpc.make_step(x_step)
            else:  # use inputs directly
                u_step = u_sim[k - 1:k, :].T

            # Calculate current measurements
            y_step = self.h(np.ravel(x_step), np.ravel(u_step))

            # Simulate one time step given current inputs
            x_step = self.simulator.make_step(u_step)

            # Store inputs
            u_sim[k - 1, :] = u_step.squeeze()

            # Store state
            x[k, :] = x_step.squeeze()

            # Store measurements
            y[k - 1, :] = y_step.squeeze()

        # Last input has no effect, so keep it the same as previous time-step
        if mpc:
            u_sim[-1, :] = u_sim[-2, :]

        # Last measurement
        y[-1, :] = self.h(np.ravel(x[-1, :]), np.ravel(u_sim[-1, :]))

        # Update the inputs
        self.update_dict(u_sim, name='u')

        # Update state trajectory
        self.update_dict(x, name='x')

        # Update measurements
        self.update_dict(y, name='y')

        # Return the measurements in array format
        y_array = np.vstack(list(self.y.values())).T

        if return_full_output:
            return self.time.copy(), self.x.copy(), self.u.copy(), self.y.copy()
        else:
            return y_array

    def get_time_states_inputs_measurements(self):
        return self.time.copy(), self.x.copy(), self.u.copy(), self.y.copy()

    def plot(self, name='x', dpi=150, plot_kwargs=None):
        """ Plot states, inputs.
        """

        if plot_kwargs is None:
            plot_kwargs = {
                'color': 'black',
                'linewidth': 2.0,
                'linestyle': '-',
                'marker': '.',
                'markersize': 0
            }

            if name == 'x':
                plot_kwargs['color'] = 'firebrick'
            elif name == 'u':
                plot_kwargs['color'] = 'royalblue'
            elif name == 'y':
                plot_kwargs['color'] = 'seagreen'
            elif name == 'setpoint':
                plot_kwargs['color'] = 'gray'

        plot_dict = getattr(self, name)
        plot_data = np.array(list(plot_dict.values()))
        n = plot_data.shape[0]

        fig, ax = plt.subplots(n, 1, figsize=(4, n * 1.5), dpi=dpi, sharex=True)
        ax = np.atleast_1d(ax)

        for n, key in enumerate(plot_dict.keys()):
            ax[n].plot(self.time, plot_dict[key], label=name, **plot_kwargs)
            ax[n].set_ylabel(key, fontsize=7)

            # Also plot the states if plotting setpoint
            if name == 'setpoint':
                ax[n].plot(self.time, self.x[key], label=key, color='firebrick', linestyle='-', linewidth=0.5)
                ax[n].legend(fontsize=6)

                y = self.x[key]
            else:
                y = plot_dict[key]

            # Set y-axis limits
            y_min = np.min(y)
            y_max = np.max(y)
            delta = y_max - y_min
            if np.abs(delta) < 0.01:
                margin = 0.1
                ax[n].set_ylim(y_min - margin, y_max + margin)

        ax[-1].set_xlabel('time', fontsize=7)
        ax[0].set_title(name, fontsize=8, fontweight='bold')

        for a in ax.flat:
            a.tick_params(axis='both', labelsize=6)


def ensure_float_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        return np.array([float(e) for e in output])
    return wrapper
