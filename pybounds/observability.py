
import numpy as np
import pandas as pd
from multiprocessing import Pool
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sympy as sp
from util import LatexStates


class EmpiricalObservabilityMatrix:
    def __init__(self, simulator, x0, time, u, eps=1e-5, parallel=False):
        """ Construct an empirical observability matrix O.

        :param callable Simulator: Simulator object : y = simulator(x0, u, **kwargs)
            y is (w x p) array. w is the number of time-steps and p is the number of measurements
        :param dict/list/np.array x0: initial state for Simulator
        :param dict/np.array u: inputs array
        :param float eps: epsilon value for perturbations to construct O, should be small number
        :param bool parallel: if True, run the perturbations in parallel
        """

        # Store inputs
        self.simulator = simulator
        self.time = time.copy()
        self.eps = eps
        self.parallel = parallel

        if isinstance(x0, dict):
            self.x0 = np.array(list(x0.values()))
        else:
            self.x0 = np.array(x0).squeeze()

        if isinstance(u, dict):
            self.u = np.vstack(list(u.values())).T
        else:
            self.u = np.array(u)

        # Number of states
        self.n = self.x0.shape[0]

        # Simulate once for nominal trajectory
        self.y_nominal = self.simulator.simulate(self.x0, self.u)

        # Number of outputs
        self.p = self.y_nominal.shape[1]

        # Check for state/measurement names
        if hasattr(self.simulator, 'state_names'):
            self.state_names = self.simulator.state_names
        else:
            self.state_names = ['x_' + str(n) for n in range(self.n)]

        if hasattr(self.simulator, 'measurement_names'):
            self.measurement_names = self.simulator.measurement_names
        else:
            self.measurement_names = ['y_' + str(p) for p in range(self.p)]

        # Perturbation amounts
        self.w = len(self.time)  # of points in time window
        self.delta_x = eps * np.eye(self.n)  # perturbation amount for each state
        self.delta_y = np.zeros((self.p, self.n, self.w))  # preallocate delta_y
        self.y_plus = np.zeros((self.w, self.n, self.p))
        self.y_minus = np.zeros((self.w, self.n, self.p))

        # Observability matrix
        self.O = np.nan * np.zeros((self.p * self.w, self.n))
        self.O_df = pd.DataFrame(self.O)

        # Set measurement names
        self.measurement_labels = []
        self.time_labels = []
        for w in range(self.w):
            tl = (w * np.ones(self.p)).astype(int)
            self.time_labels.append(tl)
            self.measurement_labels = self.measurement_labels + list(self.measurement_names)

        self.time_labels = np.hstack(self.time_labels)

        # Run
        self.run()

    def run(self, parallel=None):
        """ Construct empirical observability matrix.
        """

        if parallel is not None:
            self.parallel = parallel

        # Run simulations for perturbed initial conditions
        state_index = np.arange(0, self.n).tolist()
        if self.parallel:  # multiprocessing
            with Pool(4) as pool:
                results = pool.map(self.simulate, state_index)

            for n, r in enumerate(results):
                delta_y, y_plus, y_minus = r
                self.delta_y[:, n, :] = delta_y
                self.y_plus[:, n, :] = y_plus
                self.y_minus[:, n, :] = y_minus

        else:  # sequential
            for n in state_index:
                delta_y, y_plus, y_minus = self.simulate(n)
                self.delta_y[:, n, :] = delta_y
                self.y_plus[:, n, :] = y_plus
                self.y_minus[:, n, :] = y_minus

        # Construct O by stacking the 3rd dimension of delta_y along the 1st dimension, O is a (p*w x n) matrix
        self.O = np.zeros((self.p * self.w, self.n))
        for w in range(self.w):
            if w == 0:
                start_index = 0
            else:
                start_index = int(w * self.p)

            end_index = start_index + self.p
            self.O[start_index:end_index] = self.delta_y[:, :, w]

        # Make O into a data-frame for interpretability
        self.O_df = pd.DataFrame(self.O, columns=self.state_names, index=self.measurement_labels)
        self.O_df['time_step'] = self.time_labels
        self.O_df = self.O_df.set_index('time_step', append=True)
        self.O_df.index.names = ['sensor', 'time_step']

    def simulate(self, n):
        """ Run the simulator for specified state index (n).
        """

        # Perturb initial condition in both directions
        x0_plus = self.x0 + self.delta_x[:, n]
        x0_minus = self.x0 - self.delta_x[:, n]

        # Simulate measurements from perturbed initial conditions
        y_plus = self.simulator.simulate(x0=x0_plus, u=self.u)
        y_minus = self.simulator.simulate(x0=x0_minus, u=self.u)

        # Calculate the numerical Jacobian & normalize by 2x the perturbation amount
        delta_y = np.array(y_plus - y_minus).T / (2 * self.eps)

        return delta_y, y_plus, y_minus


class SlidingEmpiricalObservabilityMatrix:
    def __init__(self, simulator, t_sim, x_sim, u_sim, w=None, eps=1e-5,
                 parallel_sliding=False, parallel_perturbation=False):
        """ Construct empirical observability matrix O in sliding windows along a trajectory.
        
        :param callable simulator: Simulator object : y = simulator(x0, u, **kwargs)
            y is (w x p) array. w is the number of time-steps and p is the number of measurements
        :param np.array t_sim: time vector size N
        :param np.array x_sim: state trajectory array (N, n), can also be dict
        :param np.array u_sim: input array (N, m), can also be dict
        :param np.array w: window size for O calculations, will automatically set how many windows to compute
        :params float eps: tolerance for sliding windows
        :param dict/np.array u: inputs array
        :param float eps: epsilon value for perturbations to construct O's, should be small number
        :param bool parallel_sliding: if True, run the sliding windows in parallel
        :param bool parallel_perturbation: if True, run the perturbations in parallel
        """

        self.simulator = simulator
        self.eps = eps
        self.parallel_sliding = parallel_sliding
        self.parallel_perturbation = parallel_perturbation

        # Set time vector
        self.t_sim = np.array(t_sim)

        # Number of points
        self.N = self.t_sim.shape[0]

        # Make x_sim & u_sim arrays
        if isinstance(x_sim, dict):
            self.x_sim = np.vstack((list(x_sim.values()))).T
        else:
            self.x_sim = np.array(x_sim).squeeze()

        if isinstance(u_sim, dict):
            self.u_sim = np.vstack(list(u_sim.values())).T
        else:
            self.u_sim = np.array(u_sim).squeeze()

        # Check sizes
        if self.N != self.x_sim.shape[0]:
            raise ValueError('t_sim & x_sim must have same number of rows')
        elif self.N != self.u_sim.shape[0]:
            raise ValueError('t_sim & u_sim must have same number of rows')
        elif self.x_sim.shape[0] != self.u_sim.shape[0]:
            raise ValueError('x_sim & u_sim must have same number of rows')

        # Set time-window to calculate O's
        if w is None:  # set window size to full time-series size
            self.w = self.N
        else:
            self.w = w

        if self.w > self.N:
            raise ValueError('window size must be smaller than trajectory length')

        # All the indices to calculate O
        self.O_index = np.arange(0, self.N - self.w + 1,  step=1)  # indices to compute O
        self.O_time = self.t_sim[self.O_index]  # times to compute O
        self.n_point = len(self.O_index)  # # of times to calculate O

        # Where to store sliding window trajectory data & O's
        self.window_data = {}
        self.O_sliding = []
        self.O_df_sliding = []

        # Run
        self.EOM = None
        self.run()

    def run(self, parallel_sliding=None):
        """ Run.
        """

        if parallel_sliding is not None:
            self.parallel_sliding = parallel_sliding

        # Where to store sliding window trajectory data & O's
        self.window_data = {'t': [], 'u': [], 'y': [], 'y_plus': [], 'y_minus': []}
        self.O_sliding = []
        self.O_df_sliding = []

        # Construct O's
        n_point_range = np.arange(0, self.n_point).astype(int)
        if self.parallel_sliding:  # multiprocessing
            with Pool(4) as pool:
                results = pool.map(self.construct, n_point_range)
                for r in results:
                    self.O_sliding.append(r[0])
                    self.O_df_sliding.append(r[1])
                    for k in self.window_data.keys():
                        self.window_data[k].append(r[2][k])

        else:
            for n in n_point_range:  # each point on trajectory
                O_sliding, O_df_sliding, window_data = self.construct(n)
                self.O_sliding.append(O_sliding)
                self.O_df_sliding.append(O_df_sliding)
                for k in self.window_data.keys():
                    self.window_data[k].append(window_data[k])

    def construct(self, n):
        # Start simulation at point along nominal trajectory
        x0 = np.squeeze(self.x_sim[self.O_index[n], :])  # get state on trajectory & set it as the initial condition

        # Get the range to pull out time & input data for simulation
        win = np.arange(self.O_index[n], self.O_index[n] + self.w, step=1)  # index range

        # Remove part of window if it is past the end of the nominal trajectory
        within_win = win < self.N
        win = win[within_win]

        # Pull out time & control inputs in window
        t_win = self.t_sim[win]  # time in window
        t_win0 = t_win - t_win[0]  # start at 0
        u_win = self.u_sim[win, :]  # inputs in window

        # Calculate O for window
        EOM = EmpiricalObservabilityMatrix(self.simulator, x0, t_win0, u_win, eps=self.eps, parallel=self.parallel_perturbation)
        self.EOM = EOM

        # Store data
        O_sliding = EOM.O.copy()
        O_df_sliding = EOM.O_df.copy()

        window_data = {'t': t_win.copy(),
                       'u': u_win.copy(),
                       'y': EOM.y_nominal.copy(),
                       'y_plus': EOM.y_plus.copy(),
                       'y_minus': EOM.y_minus.copy()}

        return O_sliding, O_df_sliding, window_data

    def get_observability_matrix(self):
        return self.O_df_sliding.copy()


class FisherObservability:
    def __init__(self, O, R=None, sensor_noise_dict=None, lam=None):
        """ Evaluate the observability of a state variable(s) using the Fisher Information Matrix.

        :param np.array O: observability matrix (w*p, n)
            w is the number of time-steps, p is the number of measurements, and n in the number of states
            can also be set as pd.DataFrame where columns set the state names & a multilevel index sets the
            measurement names: O.index names must be ('sensor', 'time_step')
        :param np.array R: measurement noise covariance matrix (w*p x w*p)
            can also be set as pd.DataFrame where R.index = R.columns = O.index
            can also be a scaler where R = R * I
        :param dict sensor_noise_dict: constructs R by setting the noise levels for each sensor across time-steps
            keys must correspond to the 'sensor' index in O data-frame, can only be set if R is None
        :param float lam: lamda parameter, if lam='limit' compute F^-1 symbolically, otherwise use Chernoff inverse
        """

        # Make O a data-frame
        self.pw = O.shape[0]  # number of sensors * time-steps
        self.n = O.shape[1]  # number of states
        if isinstance(O, pd.DataFrame):  # data-frame given
            self.O = O.copy()
            self.sensor_names = tuple(O.index.get_level_values('sensor'))
            self.state_names = tuple(O.columns)
        elif isinstance(O, np.ndarray):  # array given
            self.sensor_names = tuple(['y' for n in range(self.pw)])
            self.state_names = tuple(['x_' + str(n) for n in range(self.n)])
            self.O = pd.DataFrame(O, index=self.sensor_names, columns=self.state_names)
        else:
            raise TypeError('O is not a pandas data-frame or numpy array')

        # Set measurement noise covariance matrix
        self.R = pd.DataFrame(np.eye(self.pw), index=self.O.index, columns=self.O.index)
        self.R_inv = pd.DataFrame(np.eye(self.pw), index=self.O.index, columns=self.O.index)
        self.set_noise_covariance(R=R, sensor_noise_dict=sensor_noise_dict)

        # Calculate Fisher Information Matrix
        self.F = self.O.values.T @ self.R_inv.values @ self.O.values
        self.F = pd.DataFrame(self.F, index=O.columns, columns=O.columns)

        # Set sigma
        if lam is None:
            # np.linalg.eig(self.F)
            self.lam = 0.0
        else:
            self.lam = lam

        # Invert F
        if self.lam == 'limit':  # calculate limit with symbolic sigma
            sigma_sym = sp.symbols('sigma')
            F_hat = self.F.values + sp.Matrix(sigma_sym * np.eye(self.n))
            F_hat_inv = F_hat.inv()
            F_hat_inv_limit = F_hat_inv.applyfunc(lambda elem: sp.limit(elem, sigma_sym, 0))
            self.F_inv = np.array(F_hat_inv_limit, dtype=np.float64)
        else:  # numeric sigma
            F_epsilon = self.F.values + (self.lam * np.eye(self.n))
            self.F_inv = np.linalg.inv(F_epsilon)

        self.F_inv = pd.DataFrame(self.F_inv, index=O.columns, columns=O.columns)

        # Pull out diagonal elements
        self.error_variance = pd.DataFrame(np.diag(self.F_inv), index=self.O.columns).T

    def set_noise_covariance(self, R=None, sensor_noise_dict=None):
        """ Set the measurement noise covariance matrix.
        """

        # Preallocate the noise covariance matrix R
        self.R = pd.DataFrame(np.eye(self.pw), index=self.O.index, columns=self.O.index)

        # Set R based on values in dict
        if sensor_noise_dict is not None:  # set each distinct sensor's noise level
            if R is not None:
                raise Exception('R can not be set directly if sensor_noise_dict is set')
            else:
                # for s in self.R.index.levels[0]:
                for s in pd.unique(self.R.index.get_level_values('sensor')):
                    R_sensor = self.R.loc[[s], [s]]
                    for r in range(R_sensor.shape[0]):
                        R_sensor.iloc[r, r] = sensor_noise_dict[s]

                    self.R.loc[[s], [s]] = R_sensor.values
        else:
            if R is None:
                warnings.warn('R not set, defaulting to identity matrix')
            else:  # set R directly
                if np.atleast_1d(R).shape[0] == 1:  # given scalar
                    self.R = R * self.R
                elif isinstance(R, pd.DataFrame):  # matrix R in data-frame
                    self.R = R.copy()
                elif isinstance(R, np.ndarray):  # matrix in array
                    self.R = pd.DataFrame(R, index=self.R.index, columns=self.R.columns)
                else:
                    raise Exception('R must be a numpy array, pandas data-frame, or scalar value')

        # Inverse of R
        R_diagonal = np.diag(self.R.values)
        is_diagonal = np.all(self.R.values == np.diag(R_diagonal))
        if is_diagonal:
            self.R_inv = np.diag(1 / R_diagonal)
        else:
            self.R_inv = np.linalg.inv(self.R.values)

        self.R_inv = pd.DataFrame(self.R_inv, index=self.R.index, columns=self.R.index)

    def get_fisher_information(self):
        return self.F.copy(), self.F_inv.copy(), self.R.copy()


class SlidingFisherObservability:
    def __init__(self, O_list, time=None, lam=1e6, R=None, sensor_noise_dict=None,
                 states=None, sensors=None, time_steps=None, w=None):

        """ Compute the Fisher information matrix & inverse in sliding windows and pull put the minimum error variance.
        :param list O_list: list of observability matrices O (stored as pd.DataFrame)
        :param np.array time: time vector the same size as O_list
        :param np.array lam: lamda parameter, if lam='limit' compute F^-1 symbolically, otherwise use Chernoff inverse
        :param np.array R: measurement noise covariance matrix (w*p x w*p)
            can also be set as pd.DataFrame where R.index = R.columns = O.index
            can also be a scaler where R = R * I
        :param dict sensor_noise_dict: constructs R by setting the noise levels for each sensor across time-steps
            keys must correspond to the 'sensor' index in O data-frame, can only be set if R is None
        :param list states: list of states to use from O's. ex: ['g', 'd']
        :param list sensors: list of sensors to use from O's, ex: ['r']
        :param np.array time_steps: array of time steps to use from O's, ex: np.array([0, 1, 2])
        :param np.array w: window size to use from O's, if None then just grab it from O
        """

        self.O_list = O_list
        self.n_window = len(O_list)

        # Set time & time-step
        if time is None:
            self.time = np.arange(0, self.n_window, step=1)
        else:
            self.time = np.array(time)

        self.dt = np.mean(np.diff(self.time))

        # Get single O
        O = O_list[0]

        # Set window size
        if w is None:  # set automatically
            self.w = np.max(np.array(O.index.get_level_values('time_step'))) + 1
        else:
            self.w = w

        # Set the states to use
        if states is None:
            self.states = O.columns
        else:
            self.states = states

        # Set the sensors to use
        if sensors is None:
            self.sensors = O.index.get_level_values('sensor')
        else:
            self.sensors = sensors

        # Set the time-steps to use
        if time_steps is None:
            self.time_steps = O.index.get_level_values('time_step')
        else:
            self.time_steps = np.array(time_steps)

        # Compute Fisher information matrix & inverse for each sliding window
        self.EV = []  # collect error variance data for each state over time
        self.FO = []
        self.shift_index = int(np.round((1 / 2) * self.w))
        self.shift_time = self.shift_index * self.dt  # shift the time forward by half the window size
        for k in range(self.n_window):  # each window
            # Get full O
            O = self.O_list[k]

            # Get subset of O
            O_subset = O.loc[(self.sensors, self.time_steps), self.states].sort_values(['time_step', 'sensor'])

            # Compute Fisher information & inverse
            FO = FisherObservability(O_subset, sensor_noise_dict=sensor_noise_dict, R=R, lam=lam)
            self.FO.append(FO)

            # Collect error variance data
            ev = FO.error_variance.copy()
            ev.insert(0, 'time_initial', self.time[k])
            self.EV.append(ev)

        # Concatenate error variance & make same size as simulation data
        self.EV = pd.concat(self.EV, axis=0, ignore_index=True)
        self.EV.index = np.arange(self.shift_index, self.EV.shape[0] + self.shift_index, step=1, dtype=int)
        time_df = pd.DataFrame(np.atleast_2d(self.time).T, columns=['time'])
        self.EV_aligned = pd.concat((time_df, self.EV), axis=1)

    def get_minimum_error_variance(self):
        return self.EV_aligned.copy()


class ObservabilityMatrixImage:
    def __init__(self, O, state_names=None, sensor_names=None, vmax_percentile=100, vmin_ratio=1.0, cmap='bwr'):
        """ Display an image of an observability matrix.
        """

        # Plotting parameters
        self.vmax_percentile = vmax_percentile
        self.vmin_ratio = vmin_ratio
        self.cmap = cmap
        self.crange = None
        self.fig = None
        self.ax = None
        self.cbar = None

        # Get O
        self.pw, self.n = O.shape
        if isinstance(O, pd.DataFrame):  # data-frame
            self.O = O.copy()  # O in matrix form

            # Default state names based on data-frame columns
            self.state_names_default = list(O.columns)

            # Default sensor names based on data-frame 'sensor' index
            sensor_names_all = list(np.unique(O.index.get_level_values('sensor')))
            self.sensor_names_default = list(O.index.get_level_values('sensor')[0:len(sensor_names_all)])

        else:  # numpy matrix
            raise TypeError('n-sensor must be an integer value when O is given as a numpy matrix')

        self.n_sensor = len(self.sensor_names_default)  # number of sensors
        self.n_time_step = int(self.pw / self.n_sensor)  # number of time-steps

        # Set state names
        if state_names is not None:
            if len(state_names) == self.n:
                self.state_names = state_names.copy()
            elif len(state_names) == 1:
                self.state_names = ['$' + state_names[0] + '_{' + str(n) + '}$' for n in range(1, self.n + 1)]
            else:
                raise TypeError('state_names must be of length n or length 1')
        else:
            self.state_names = self.state_names_default.copy()

        # Convert to Latex
        LatexConverter = LatexStates()
        self.state_names = LatexConverter.convert_to_latex(self.state_names)

        # Set sensor & measurement names
        if sensor_names is not None:
            if len(sensor_names) == self.n_sensor:
                self.sensor_names = sensor_names.copy()
                self.sensor_names = LatexConverter.convert_to_latex(self.sensor_names, remove_dollar_signs=True)
                self.measurement_names = []
                for w in range(self.n_time_step):
                    for p in range(self.n_sensor):
                        m = '$' + self.sensor_names[p] + ',_{' + 'k=' + str(w) + '}$'
                        self.measurement_names.append(m)

            elif len(sensor_names) == 1:
                self.sensor_names = [sensor_names[0] + '_{' + str(n) + '}$' for n in range(1, self.n_sensor + 1)]
                self.sensor_names = LatexConverter.convert_to_latex(self.sensor_names, remove_dollar_signs=True)
                self.measurement_names = []
                for w in range(self.n_time_step):
                    for p in range(self.n_sensor):
                        m = '$' + sensor_names[0] + '_{' + str(p) + ',k=' + str(w) + '}$'
                        self.measurement_names.append(m)
            else:
                raise TypeError('sensor_names must be of length p or length 1')

        else:
            self.sensor_names = self.sensor_names_default.copy()
            self.sensor_names = LatexConverter.convert_to_latex(self.sensor_names, remove_dollar_signs=True)
            self.measurement_names = []
            for w in range(self.n_time_step):
                for p in range(self.n_sensor):
                    m = '$' + self.sensor_names[p] + '_{' + ',k=' + str(w) + '}$'
                    self.measurement_names.append(m)

    def plot(self, vmax_percentile=100, vmin_ratio=0.0, vmax_override=None, cmap='bwr', grid=True, scale=1.0, dpi=150,
             ax=None):
        """ Plot the observability matrix.
        """

        # Plot properties
        self.vmax_percentile = vmax_percentile
        self.vmin_ratio = vmin_ratio
        self.cmap = cmap

        if vmax_override is None:
            self.crange = np.percentile(np.abs(self.O), self.vmax_percentile)
        else:
            self.crange = vmax_override

        # Display O
        O_disp = self.O.values
        # O_disp = np.nan_to_num(np.sign(O_disp) * np.log(np.abs(O_disp)), nan=0.0)
        for n in range(self.n):
            for m in range(self.pw):
                oval = O_disp[m, n]
                if (np.abs(oval) < (self.vmin_ratio * self.crange)) and (np.abs(oval) > 1e-6):
                    O_disp[m, n] = self.vmin_ratio * self.crange * np.sign(oval)

        # Plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(0.3 * self.n * scale, 0.3 * self.pw * scale),
                                   dpi=dpi)
        else:
            fig = None

        O_data = ax.imshow(O_disp, vmin=-self.crange, vmax=self.crange, cmap=self.cmap)
        ax.grid(visible=False)

        ax.set_xlim(-0.5, self.n - 0.5)
        ax.set_ylim(self.pw - 0.5, -0.5)

        ax.set_xticks(np.arange(0, self.n))
        ax.set_yticks(np.arange(0, self.pw))

        ax.set_xlabel('States', fontsize=10, fontweight='bold')
        ax.set_ylabel('Measurements', fontsize=10, fontweight='bold')

        ax.set_xticklabels(self.state_names)
        ax.set_yticklabels(self.measurement_names)

        ax.tick_params(axis='x', which='major', labelsize=7, pad=-1.0)
        ax.tick_params(axis='y', which='major', labelsize=7, pad=-0.0, left=False)
        ax.tick_params(axis='x', which='both', top=False, labeltop=True, bottom=False, labelbottom=False)
        ax.xaxis.set_label_position('top')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # Draw grid
        if grid:
            grid_color = [0.8, 0.8, 0.8, 1.0]
            grid_lw = 1.0
            for n in np.arange(-0.5, self.pw + 1.5):
                ax.axhline(y=n, color=grid_color, linewidth=grid_lw)
            for n in np.arange(-0.5, self.n + 1.5):
                ax.axvline(x=n, color=grid_color, linewidth=grid_lw)

        # Make colorbar
        axins = inset_axes(ax, width='100%', height=0.1, loc='lower left',
                           bbox_to_anchor=(0.0, -1.0 * (1.0 / self.pw), 1, 1), bbox_transform=ax.transAxes,
                           borderpad=0)

        cbar = plt.colorbar(O_data, cax=axins, orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('matrix values', fontsize=9, fontweight='bold', rotation=0)

        # Store figure & axis
        self.fig = fig
        self.ax = ax
        self.cbar = cbar