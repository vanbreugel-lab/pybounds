import numpy as np
import do_mpc
import casadi
from util import FixedKeysDict, SetDict


class MonoCamera:
    def __init__(self, dt=0.01, output_mode=None, params_simulator=None, n_horizon=10, u_weight=1e-3,
                 setpoint_x=None):
        """
        Simulator.
        :param float dt: is the sampling time in seconds
        :param (list, tuple, or set) output_mode:  array of strings containing variables names of outputs
        the strings must correspond to a keys in self.sim_data
        ex: output_mode = ('x', 'v' 'r') # to output states x & v & the optic flow r
        :param dict params_simulator: dictionary containing simulator parameters
        """

        # Set time-step
        self.dt = dt  # [seconds]

        # Define initial states
        self.x0 = FixedKeysDict(x=1.0,  # position
                                v=0.0,  # velocity
                                z=1.0,  # altitude
                                m=1.0,  # mass
                                b=1.0,  # damping
                                k=1.0   # spring constant
                                )

        self.state_names = tuple(self.x0.keys())
        self.n = len(self.state_names)  # number of states

        # Initialize the state trajectory
        j = 100  # initialize for j time-steps, but this can change later
        self.x = self.x0.copy()
        for key in self.x.keys():
            self.x[key] = self.x0[key] * np.ones(j)

        self.t = np.arange(0, j * self.dt + self.dt / 2, step=self.dt)  # time vector

        # Initialize the inputs as 0
        self.u = FixedKeysDict(u_1=np.zeros(j),  # main force
                               u_2=np.zeros(j)  # auxiliary force
                               )

        self.input_names = tuple(self.u.keys())
        self.m = len(self.input_names)  # number of inputs

        # Initialize simulation data
        self.sim_data = {}
        self.calculate_simulation_data()

        # Set output mode
        self.p = 2
        self.output_mode = ['x', 'v']  # default output mode is [position, velocity]
        self.set_output_mode(output_mode)

        # Initialize outputs
        self.y = {}
        for output_name in self.output_mode:
            self.y[output_name] = self.sim_data[output_name]

        # Define continuous-time MPC model
        self.model = do_mpc.model.Model('continuous')

        # Define state variables
        x = self.model.set_variable(var_type='_x', var_name='x', shape=(1, 1))  # position
        v = self.model.set_variable(var_type='_x', var_name='v', shape=(1, 1))  # velocity
        z = self.model.set_variable(var_type='_x', var_name='z', shape=(1, 1))  # altitude
        m = self.model.set_variable(var_type='_x', var_name='m', shape=(1, 1))  # mass
        b = self.model.set_variable(var_type='_x', var_name='b', shape=(1, 1))  # damping
        k = self.model.set_variable(var_type='_x', var_name='k', shape=(1, 1))  # spring

        # Define input variables
        u_1 = self.model.set_variable(var_type='_u', var_name='u_1', shape=(1, 1))  # main input
        u_2 = self.model.set_variable(var_type='_u', var_name='u_2', shape=(1, 1))  # altitude input

        # Define dynamics equations
        self.model.set_rhs('x', v)
        self.model.set_rhs('v', -(k / m) * x - (b / m) * v + (1 / m) * u_1)
        self.model.set_rhs('z', u_2)
        self.model.set_rhs('m', x * 0)
        self.model.set_rhs('b', x * 0)
        self.model.set_rhs('k', x * 0)

        # Add set-point variables for later use with MPC
        setpoint_x_var = self.model.set_variable(var_type='_tvp', var_name='setpoint_x')

        # Build model
        self.model.setup()

        # Define simulator & simulator parameters
        self.simulator = do_mpc.simulator.Simulator(self.model)

        # Setup MPC
        self.mpc = do_mpc.controller.MPC(self.model)

        # Set MPC parameters
        self.n_horizon = n_horizon
        setup_mpc = {
            'n_horizon': self.n_horizon,
            'n_robust': 0,
            'open_loop': 0,
            't_step': self.dt,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 3,
            'collocation_ni': 1,
            'store_full_solution': True,

            # Use MA27 linear solver in ipopt for faster calculations:
            'nlpsol_opts': {'ipopt.linear_solver': 'mumps',  # mumps, MA27
                            'ipopt.print_level': 0,
                            'ipopt.sb': 'yes',
                            'print_time': 0,
                            }
        }

        self.mpc.set_param(**setup_mpc)

        # Run MPC if set-point specified
        if setpoint_x is not None:
            self.setpoint_x = np.atleast_1d(setpoint_x.copy())
            self.mpc_points = self.setpoint_x.shape[0]
        else:
            self.mpc_points = 10
            self.setpoint_x = np.zeros(self.mpc_points)

        # Get template's for MPC time-varying parameters
        self.mpc_tvp_template = self.mpc.get_tvp_template()
        self.simulator_tvp_template = self.simulator.get_tvp_template()

        # Set time-varying set-point functions
        self.mpc.set_tvp_fun(self.mpc_tvp_function)
        self.simulator.set_tvp_fun(self.simulator_tvp_function)

        # Set MPC objective function
        self.set_mpc_objective(u_weight=u_weight)

        # Setup MPC
        self.mpc.setup()

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

    def set_output_mode(self, output_mode):
        """ Set output mode.
        """

        if output_mode is not None:
            # Check if variables in output_mode were calculated
            var_list = set(self.sim_data.keys())
            for y in output_mode:
                if y not in var_list:
                    raise Exception('variable name not available')

            # Set output mode
            self.output_mode = tuple(output_mode)
            self.p = len(self.output_mode)  # number of outputs

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

    def calculate_simulation_data(self):
        """ Calculate non-state variables.
        """

        # Initialize a dict with the time, states, & inputs
        self.sim_data = dict(t=self.t.copy())

        for key in self.x.keys():
            self.sim_data[key] = self.x[key]

        for key in self.u.keys():
            self.sim_data[key] = self.u[key]

        # Add other variables
        # Optic flow: velocity divided by position
        self.sim_data['r'] = self.sim_data['v'] / self.sim_data['z']

        # Position multiplied by velocity...whatever this is
        self.sim_data['xv'] = self.sim_data['x'] * self.sim_data['v']

    def simulate(self, x0=None, u=None, output_mode=None, run_mpc=False):
        """
        Simulate the system.

        :params x0: initial state dict or array
        :params u: input dict or array
        :params output_mode: array of strings containing variable names of outputs
        :params run_mpc: boolean to run MPC controller
        """

        # Update the initial state.
        self.set_initial_state(x0=x0.copy())

        # Set the output mode
        self.set_output_mode(output_mode)

        if run_mpc:  # run MPC to get inputs
            if u is not None:
                raise Exception('u must be None to run MPC')

            n_point = self.mpc_points
            u_sim = np.zeros((n_point, self.m))
            self.x0['x'] = self.setpoint_x[0]
            print(self.x0)

        else:  # use open loop inputs
            # Update the inputs
            self.set_inputs(u=u)

            # Concatenate the inputs, where rows are individual inputs and columns are time-steps
            u_sim = np.vstack(list(self.u.values())).T
            n_point = u_sim.shape[0]

        # Update time vector
        T = (n_point - 1) * self.dt
        self.t = np.linspace(0, T, num=n_point)

        # Set array to store simulated states, where rows are individual states and columns are time-steps
        x_step = np.array(list(self.x0.values()))  # initialize state
        x_sim = np.nan * np.zeros((n_point, self.n))
        x_sim[0, :] = x_step.copy()

        # Initialize the simulator
        self.simulator.t0 = self.t[0]
        self.simulator.x0 = x_step.copy()
        self.simulator.set_initial_guess()

        if run_mpc:
            self.mpc.t0 = self.t[0]
            self.mpc.x0 = x_step.copy()
            self.mpc.u0 = np.zeros((self.m, 1))
            self.mpc.set_initial_guess()

        # Run simulation
        for k in range(1, n_point):
            # Set input
            if run_mpc:
                u_step = self.mpc.make_step(x_step)
            else:
                u_step = u_sim[k - 1:k, :].T

            # Store controls
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

        # Calculate non-state variables
        self.calculate_simulation_data()

        # Set outputs
        self.y = {}
        for output_name in self.output_mode:
            self.y[output_name] = self.sim_data[output_name]

        # Return the outputs in array format
        y_array = np.vstack(list(self.y.values())).T
        return y_array

    def set_mpc_objective(self, u_weight=1e-3):
        """ Set MCP objective function.


            Inputs:
                case: type of objective function
                r_weight: weight for control penalty
        """

        # Set stage cost
        lterm = (self.model.x['x'] - self.model.tvp['setpoint_x']) ** 2
        # lterm = (self.model.x['x'] - self.model.tvp['setpoint_x']) ** 2 + casadi.fabs(self.model.x['x'] - self.model.tvp['setpoint_x'])

        # Set terminal cost same as state cost
        mterm = lterm

        # Set objective
        self.mpc.set_objective(mterm=mterm, lterm=lterm)  # objective function
        self.mpc.set_rterm(u_1=u_weight, u_2=u_weight)  # input penalty

    def mpc_tvp_function(self, t):
        """ Set the set-point function for MPC optimizer.
        """

        # Set current step index
        k_step = int(np.round(t / self.dt))

        # Update set-point time horizon
        for n in range(self.n_horizon + 1):
            k_set = k_step + n
            if k_set >= self.mpc_points:  # horizon is beyond end of input data
                k_set = self.mpc_points - 1  # set part of horizon beyond input data to last point

            # Update each set-point over time horizon
            self.mpc_tvp_template['_tvp', n, 'setpoint_x'] = self.setpoint_x[k_set]

        return self.mpc_tvp_template

    def simulator_tvp_function(self, t):
        """ Set the set-point function for MPC simulator.
        :param t: current time
        """

        # Set current step index
        k_step = int(np.round(t / self.dt))
        if k_step >= self.mpc_points:  # point is beyond end of input data
            k_step = self.mpc_points - 1  # set point beyond input data to last point

        # Update current set-point
        self.simulator_tvp_template['setpoint_x'] = self.setpoint_x[k_step]

        return self.simulator_tvp_template
