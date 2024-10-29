import numpy as np
import do_mpc
import casadi
from util import FixedKeysDict, SetDict


class FlyWind:
    def __init__(self, dt=0.01, output_mode=None, params_simulator=None, n_horizon=10, u_weight=1e-6, setpoint=None):
        """
        Simulator.
        :param float dt: is the sampling time in seconds
        :param (list, tuple, or set) output_mode:  array of strings containing variables names of outputs
                the strings must correspond to a keys in self.sim_data
        :param dict params_simulator: dictionary containing simulator parameters
        """

        # Set time-step
        self.dt = dt  # [seconds]

        # Parameters in SI units
        m = 0.25e-6  # [kg]
        I = 5.2e-13  # [N*m*s^2] yaw mass moment of inertia: 10.1242/jeb.02369
        # I = 4.971e-12  # [N*m*s^2] yaw mass moment of inertia: 10.1242/jeb.038778
        C_phi = 27.36e-12  # [N*m*s] yaw damping: 10.1242/jeb.038778
        C_para = m / 0.170  # [N*s/m] calculate using the mass and time constant reported in 10.1242/jeb.098665
        C_perp = C_para  # assume same as C_para

        # Scale Paramaters
        m = m * 1e6  # [mg]
        I = I * 1e6 * (1e3) ** 2  # [mg*mm/s^2 * mm*s^2]
        C_phi = C_phi * 1e6 * (1e3) ** 2  # [mg*mm/s^2 *m*s]
        C_para = C_para * 1e6  # [mg/s]
        C_perp = C_perp * 1e6  # [mg/s]

        # Define initial states
        self.x0 = FixedKeysDict(x=0.0,  # x position [m]
                                y=0.0,  # y position [m]
                                v_para=0.1,  # parallel velocity [m/s]
                                v_perp=0.0,  # perpendicular velocity [m/s]
                                phi=0.0,  # heading [rad]
                                phidot=0.0,  # angular velocity [rad/s]
                                w=1.0,  # ambient wind speed [m/s]
                                zeta=0.0,  # ambient wind angle [rad]
                                z=1.0,  # altitude [m]
                                m=m,  # mass [kg]
                                I=I,  # inertia [kg*m^2]
                                C_para=C_para,  # parallel damping [N*s/m]
                                C_perp=C_perp,  # perpendicular damping [N*s/m]
                                C_phi=C_phi,  # rotational damping [N·m/rad/]
                                km1=1.0,  # parallel motor calibration coefficient
                                km2=0.0,  # offset motor calibration coefficient
                                km3=1.0,  # perpendicular motor calibration coefficient
                                km4=1.0,  # rotational motor calibration coefficient
                                ks1=1.0,  # sensor calibration coefficient
                                ks2=1.0,  # sensor calibration coefficient
                                ks3=1.0,  # sensor calibration coefficient
                                ks4=1.0,  # sensor calibration coefficient
                                ks5=1.0,  # sensor calibration coefficient
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
        self.u = FixedKeysDict(u_para=np.zeros(j),  # parallel force [N]
                               u_perp=np.zeros(j),  # perpendicular force [N]
                               u_phi=np.zeros(j),  # torque [N*m]
                               u_w=np.zeros(j),  # wind magnitude input
                               u_zeta=np.zeros(j),  # wind direction input
                               u_z=np.zeros(j)  # altitude input
                               )

        self.input_names = tuple(self.u.keys())
        self.m = len(self.input_names)  # number of inputs

        # Initialize set-point data
        self.mpc_points = j
        self.closed_loop = False
        if setpoint is not None:
            self.set_setpoint(setpoint)

        # Initialize simulation data
        self.sim_data = {}
        self.calculate_simulation_data()

        # Set output mode
        self.output_mode = ('phi', 'psi', 'gamma')  # default output mode
        self.p = len(self.output_mode)
        self.set_output_mode(output_mode)

        # Initialize outputs
        self.y = {}
        for output_name in self.output_mode:
            self.y[output_name] = self.sim_data[output_name]

        # Define continuous-time MPC model
        self.model = do_mpc.model.Model('continuous')

        # Define state variables
        x = self.model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
        y = self.model.set_variable(var_type='_x', var_name='y', shape=(1, 1))
        v_para = self.model.set_variable(var_type='_x', var_name='v_para', shape=(1, 1))
        v_perp = self.model.set_variable(var_type='_x', var_name='v_perp', shape=(1, 1))
        phi = self.model.set_variable(var_type='_x', var_name='phi', shape=(1, 1))
        phidot = self.model.set_variable(var_type='_x', var_name='phidot', shape=(1, 1))
        w = self.model.set_variable(var_type='_x', var_name='w', shape=(1, 1))
        zeta = self.model.set_variable(var_type='_x', var_name='zeta', shape=(1, 1))
        z = self.model.set_variable(var_type='_x', var_name='z', shape=(1, 1))
        m = self.model.set_variable(var_type='_x', var_name='m', shape=(1, 1))
        I = self.model.set_variable(var_type='_x', var_name='I', shape=(1, 1))
        C_para = self.model.set_variable(var_type='_x', var_name='C_para', shape=(1, 1))
        C_perp = self.model.set_variable(var_type='_x', var_name='C_perp', shape=(1, 1))
        C_phi = self.model.set_variable(var_type='_x', var_name='C_phi', shape=(1, 1))
        km1 = self.model.set_variable(var_type='_x', var_name='km1', shape=(1, 1))
        km2 = self.model.set_variable(var_type='_x', var_name='km2', shape=(1, 1))
        km3 = self.model.set_variable(var_type='_x', var_name='km3', shape=(1, 1))
        km4 = self.model.set_variable(var_type='_x', var_name='km4', shape=(1, 1))
        ks1 = self.model.set_variable(var_type='_x', var_name='ks1', shape=(1, 1))
        ks2 = self.model.set_variable(var_type='_x', var_name='ks2', shape=(1, 1))
        ks3 = self.model.set_variable(var_type='_x', var_name='ks3', shape=(1, 1))
        ks4 = self.model.set_variable(var_type='_x', var_name='ks4', shape=(1, 1))
        ks5 = self.model.set_variable(var_type='_x', var_name='ks5', shape=(1, 1))

        # Define input variables
        u_para = self.model.set_variable(var_type='_u', var_name='u_para', shape=(1, 1))
        u_perp = self.model.set_variable(var_type='_u', var_name='u_perp', shape=(1, 1))
        u_phi = self.model.set_variable(var_type='_u', var_name='u_phi', shape=(1, 1))
        u_w = self.model.set_variable(var_type='_u', var_name='u_w', shape=(1, 1))
        u_zeta = self.model.set_variable(var_type='_u', var_name='u_zeta', shape=(1, 1))
        u_z = self.model.set_variable(var_type='_u', var_name='u_z', shape=(1, 1))

        # Calculate dynamics
        g, psi, a_para, a_perp, a, gamma, v_para_dot, v_perp_dot, q, alpha, phiddot = (
            calculate_dynamics(v_para, v_perp, phi, phidot, w, zeta,
                               m, I, C_para, C_perp, C_phi,
                               km1, km2, km3, km4,
                               u_para, u_perp, u_phi))

        # Define dynamics equations
        self.model.set_rhs('x', v_para * np.cos(phi) - v_perp * np.sin(phi))
        self.model.set_rhs('y', v_para * np.sin(phi) + v_perp * np.cos(phi))
        self.model.set_rhs('v_para', v_para_dot)
        self.model.set_rhs('v_perp', v_perp_dot)
        self.model.set_rhs('phi', phidot)
        self.model.set_rhs('phidot', phiddot)
        self.model.set_rhs('w', u_w)
        self.model.set_rhs('zeta', u_zeta)
        self.model.set_rhs('z', u_z)
        self.model.set_rhs('m', phi * 0)
        self.model.set_rhs('I', phi * 0)
        self.model.set_rhs('C_para', phi * 0)
        self.model.set_rhs('C_perp', phi * 0)
        self.model.set_rhs('C_phi', phi * 0)
        self.model.set_rhs('km1', phi * 0)
        self.model.set_rhs('km2', phi * 0)
        self.model.set_rhs('km3', phi * 0)
        self.model.set_rhs('km4', phi * 0)
        self.model.set_rhs('ks1', phi * 0)
        self.model.set_rhs('ks2', phi * 0)
        self.model.set_rhs('ks3', phi * 0)
        self.model.set_rhs('ks4', phi * 0)
        self.model.set_rhs('ks5', phi * 0)

        # Add set-point variables for later use with MPC
        setpoint_v_para = self.model.set_variable(var_type='_tvp', var_name='setpoint_v_para')
        setpoint_v_perp = self.model.set_variable(var_type='_tvp', var_name='setpoint_v_perp')
        setpoint_phi = self.model.set_variable(var_type='_tvp', var_name='setpoint_phi')
        setpoint_w = self.model.set_variable(var_type='_tvp', var_name='setpoint_w')
        setpoint_zeta = self.model.set_variable(var_type='_tvp', var_name='setpoint_zeta')
        setpoint_z = self.model.set_variable(var_type='_tvp', var_name='setpoint_z')
        setpoint_psi_global = self.model.set_variable(var_type='_tvp', var_name='setpoint_psi_global')

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

        # Initialize the MPC set-point
        self.setpoint = FixedKeysDict(v_para=np.zeros(j),  # parallel velocity set-point
                                      v_perp=np.zeros(j),  # perpendicular velocity set-point
                                      phi=np.zeros(j),  # heading set-point
                                      w=np.zeros(j),  # wind magnitude set-point
                                      zeta=np.zeros(j),  # wind direction set-point
                                      z=np.zeros(j),  # altitude set-point
                                      psi_global=np.zeros(j)  # global course direction set-point
                                      )

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

    def set_setpoint(self, setpoint):
        """ Update the inputs.
        """

        if setpoint is not None:  # inputs given
            self.closed_loop = True
            if isinstance(setpoint, dict):  # in dict format
                SetDict().set_dict_with_overwrite(self.setpoint, setpoint)  # update only the inputs in the dict given
            elif isinstance(setpoint, list) or isinstance(setpoint, tuple):  # list or tuple format, each input vector in each element
                for n, k in enumerate(self.setpoint.keys()):  # each input
                    self.setpoint[k] = setpoint[n]
            elif isinstance(setpoint, np.ndarray):  # numpy array format given as matrix where columns are the different inputs
                if len(setpoint.shape) <= 1:  # given as 1d array, so convert to column vector
                    setpoint = np.atleast_2d(setpoint).T

                for m, key in enumerate(self.setpoint.keys()):  # each input
                    self.setpoint[key] = setpoint[:, m]

            else:
                raise Exception('setpoint must be either a dict, tuple, list, or numpy array')
        else:
            self.closed_loop = False

        # Make sure inputs are the same size
        points = np.array([self.setpoint[key].shape[0] for key in self.setpoint.keys()])
        points_check = points == points[0]
        if not np.all(points_check):
            raise Exception('inputs are not the same size')

        first_key = next(iter(self.setpoint))
        self.mpc_points = self.setpoint[first_key].shape[0]

    def calculate_simulation_data(self):
        """ Calculate all variables to store.
        """

        # Initialize a dict with the time, states, & inputs
        self.sim_data = dict(time=self.t.copy())

        for key in self.x.keys():
            self.sim_data[key] = self.x[key]

        for key in self.u.keys():
            self.sim_data[key] = self.u[key]

        # Inputs in polar form
        u_g, u_psi = cart2polar(self.sim_data['u_para'], self.sim_data['u_perp'])
        self.sim_data['u_g'] = u_g
        self.sim_data['u_psi'] = u_psi

        # Calculate dynamics
        g, psi, a_para, a_perp, a, gamma, v_para_dot, v_perp_dot, q, alpha, phiddot = (
            calculate_dynamics(self.sim_data['v_para'],
                               self.sim_data['v_perp'],
                               self.sim_data['phi'],
                               self.sim_data['phidot'],
                               self.sim_data['w'],
                               self.sim_data['zeta'],
                               self.sim_data['m'],
                               self.sim_data['I'],
                               self.sim_data['C_para'],
                               self.sim_data['C_perp'],
                               self.sim_data['C_phi'],
                               self.sim_data['km1'],
                               self.sim_data['km2'],
                               self.sim_data['km3'],
                               self.sim_data['km4'],
                               self.sim_data['u_para'],
                               self.sim_data['u_perp'],
                               self.sim_data['u_phi']))

        # Ground velocity
        self.sim_data['g'] = g  # ground velocity magnitude
        self.sim_data['psi'] = psi  # ground velocity angle
        self.sim_data['psi_global'] = psi + self.sim_data['phi']  # global course direction

        # Air velocity
        self.sim_data['a'] = a  # air velocity magnitude
        self.sim_data['gamma'] = gamma  # air velocity angle
        self.sim_data['a_para'] = a_para  # parallel air velocity
        self.sim_data['a_perp'] = a_perp  # perpendicular air velocity

        # Translational acceleration
        self.sim_data['q'] = q  # acceleration flow magnitude
        self.sim_data['alpha'] = alpha  # acceleration angle
        self.sim_data['a_para'] = v_para_dot  # parallel acceleration
        self.sim_data['a_perp'] = v_perp_dot  # perpendicular acceleration

        # Angular acceleration
        self.sim_data['phiddot'] = phiddot

        # Optic flow: ground speed divided by altitude
        self.sim_data['r'] = self.sim_data['g'] / self.sim_data['z']  # optic flow magnitude
        self.sim_data['r_para'] = self.sim_data['v_para'] / self.sim_data['z']  # optic in parallel direction
        self.sim_data['r_perp'] = self.sim_data['v_perp'] / self.sim_data['z']  # optic in perpendicular direction

        # Unwrap angles
        self.sim_data['phi'] = np.unwrap(self.sim_data['phi'])
        self.sim_data['psi'] = np.unwrap(self.sim_data['psi'])
        self.sim_data['gamma'] = np.unwrap(self.sim_data['gamma'])
        self.sim_data['alpha'] = np.unwrap(self.sim_data['alpha'])

        # Set-points
        if self.closed_loop:
            for key in self.setpoint.keys():
                new_key = 'setpoint_' + str(key)
                self.sim_data[new_key] = self.setpoint[key]

    def simulate(self, x0=None, u=None, output_mode=None, setpoint=None, run_mpc=False):
        """
        Simulate the system.

        :params x0: initial state dict or array
        :params u: input dict or array
        :params output_mode: array of strings containing variable names of outputs
        :params run_mpc: boolean to run MPC controller
        """

        # Update initial state.
        self.set_initial_state(x0=x0)

        # Update output mode
        self.set_output_mode(output_mode)

        # Update set-point
        self.set_setpoint(setpoint)

        if run_mpc:  # run MPC to get inputs
            if u is not None:
                raise Exception('u must be None to run MPC')

            # Start initial conditions at set-point
            self.x0['v_para'] = self.setpoint['v_para'][0]
            self.x0['v_perp'] = self.setpoint['v_perp'][0]
            self.x0['phi'] = self.setpoint['phi'][0]
            self.x0['w'] = self.setpoint['w'][0]
            self.x0['zeta'] = self.setpoint['zeta'][0]
            self.x0['z'] = self.setpoint['z'][0]

            # Preallocate the control input array
            n_point = self.mpc_points
            u_sim = np.zeros((n_point, self.m))

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

    def set_mpc_objective(self, u_weight=1e-6):
        """ Set MCP objective function.


            Inputs:
                case: type of objective function
                u_weight: weight for control penalty
        """

        # Set stage cost
        lterm = ((self.model.x['v_para'] - self.model.tvp['setpoint_v_para']) ** 2 +
                 (self.model.x['v_perp'] - self.model.tvp['setpoint_v_perp']) ** 2 +
                 (self.model.x['phi'] - self.model.tvp['setpoint_phi']) ** 2 +
                 (self.model.x['w'] - self.model.tvp['setpoint_w']) ** 2 +
                 (self.model.x['zeta'] - self.model.tvp['setpoint_zeta']) ** 2 +
                 (self.model.x['z'] - self.model.tvp['setpoint_z']) ** 2)


        # Set terminal cost same as state cost
        mterm = lterm

        # Set objective
        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        # Input penalty
        self.mpc.set_rterm(u_para=u_weight, u_perp=u_weight, u_phi=u_weight, u_w=0.0, u_zeta=0.0, u_z=0.0)

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
            self.mpc_tvp_template['_tvp', n, 'setpoint_v_para'] = self.setpoint['v_para'][k_set]
            self.mpc_tvp_template['_tvp', n, 'setpoint_v_perp'] = self.setpoint['v_perp'][k_set]
            self.mpc_tvp_template['_tvp', n, 'setpoint_phi'] = self.setpoint['phi'][k_set]
            self.mpc_tvp_template['_tvp', n, 'setpoint_phi'] = self.setpoint['phi'][k_set]
            self.mpc_tvp_template['_tvp', n, 'setpoint_w'] = self.setpoint['w'][k_set]
            self.mpc_tvp_template['_tvp', n, 'setpoint_zeta'] = self.setpoint['zeta'][k_set]
            self.mpc_tvp_template['_tvp', n, 'setpoint_z'] = self.setpoint['z'][k_set]
            self.mpc_tvp_template['_tvp', n, 'setpoint_psi_global'] = self.setpoint['psi_global'][k_set]

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
        self.simulator_tvp_template['setpoint_v_para'] = self.setpoint['v_para'][k_step]
        self.simulator_tvp_template['setpoint_v_perp'] = self.setpoint['v_perp'][k_step]
        self.simulator_tvp_template['setpoint_phi'] = self.setpoint['phi'][k_step]
        self.simulator_tvp_template['setpoint_w'] = self.setpoint['w'][k_step]
        self.simulator_tvp_template['setpoint_zeta'] = self.setpoint['zeta'][k_step]
        self.simulator_tvp_template['setpoint_z'] = self.setpoint['z'][k_step]
        self.simulator_tvp_template['setpoint_psi_global'] = self.setpoint['psi_global'][k_step]

        return self.simulator_tvp_template


def calculate_air_velocity(v_para, v_perp, phi, w, zeta):
    """ Calculate air velocity.
    """

    # Air speed in parallel & perpendicular directions
    a_para = v_para - w * np.cos(phi - zeta)
    a_perp = v_perp + w * np.sin(phi - zeta)

    # Air velocity angle & magnitude
    # a = np.linalg.norm((a_perp, a_para), ord=2, axis=0)  # air velocity magnitude
    a = np.sqrt(a_para ** 2 + a_perp ** 2)
    gamma = np.arctan2(a_perp, a_para)  # air velocity angle

    return a_para, a_perp, a, gamma


def calculate_dynamics(v_para, v_perp, phi, phidot, w, zeta, m, I, C_para, C_perp, C_phi, km1, km2, km3, km4,
                       u_para, u_perp, u_phi):
    """ Calculate air velocity, translational acceleration, & angular acceleration.
    """

    # Velocity angle & magnitude
    # g = np.linalg.norm((v_perp, v_para), ord=2, axis=0)
    # psi = np.arctan2(v_perp, v_para)
    g, psi = cart2polar(v_para, v_perp)

    # Air velocity
    a_para, a_perp, a, gamma = calculate_air_velocity(v_para, v_perp, phi, w, zeta)

    # Acceleration
    v_para_dot = ((km1 * u_para - C_para * a_para) / m) + (v_perp * phidot)
    v_perp_dot = ((km3 * u_perp - C_perp * a_perp) / m) - (v_para * phidot)

    # Acceleration angle & magnitude
    # q = np.linalg.norm((v_perp_dot, v_para_dot), ord=2, axis=0)  # acceleration magnitude
    q = np.sqrt(v_para_dot ** 2 + v_perp_dot ** 2)
    alpha = np.arctan2(v_perp_dot, v_para_dot)  # acceleration angle

    # Angular acceleration
    phiddot = (km4 * u_phi / I) - (C_phi * phidot / I) + (km2 * u_para / I)

    return g, psi, a_para, a_perp, a, gamma, v_para_dot, v_perp_dot, q, alpha, phiddot


def cart2polar(x, y):
    # Transform cartesian to polar
    r = np.sqrt((x ** 2) + (y ** 2))
    theta = np.arctan2(y, x)

    return r, theta
