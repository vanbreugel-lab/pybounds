
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
    def __init__(self, simulator, x0, t_sim, u_sim, eps=1e-5, parallel=False):
        """ Construct an empirical observability matrix O.

        Inputs
            simulator:          simulator object: y = simulator(x0, t_sim, u_sim)
            x0:                 initial state
            t_sim:              simulation time
            u_sim:              simulation inputs
            eps:                amount to perturb initial states
        """

        # Store inputs
        self.simulator = simulator
        self.t_sim = t_sim.copy()
        self.eps = eps
        self.parallel = parallel

        if isinstance(x0, dict):
            self.x0 = np.array(list(x0.values()))
        else:
            self.x0 = np.array(x0).squeeze()

        if isinstance(u_sim, dict):
            self.u_sim = np.vstack(list(u_sim.values())).T
        else:
            self.u_sim = np.array(u_sim)

        # Simulate once for nominal trajectory
        self.y_nominal = self.simulator.simulate(self.x0, self.u_sim)
        self.x_nominal = self.simulator.x.copy()
        self.u_nominal = self.simulator.u.copy()
        # self.sim_data_nominal = self.simulator.sim_data.copy()

        # Perturbation amounts
        self.w = len(t_sim)  # of points in time window
        self.delta_x = eps * np.eye(self.simulator.n)  # perturbation amount for each state
        self.delta_y = np.zeros((self.simulator.p, self.simulator.n, self.w))  # preallocate delta_y
        self.y_plus = np.zeros((self.w, self.simulator.n, self.simulator.p))
        self.y_minus = np.zeros((self.w, self.simulator.n, self.simulator.p))

        # Observability matrix
        self.O = np.nan * np.zeros((self.simulator.p * self.w, self.simulator.n))
        self.O_df = pd.DataFrame(self.O)

        # Set measurement names
        self.measurement_labels = []
        self.time_labels = []
        for w in range(self.w):
            tl = (w * np.ones(self.simulator.p)).astype(int)
            self.time_labels.append(tl)
            self.measurement_labels = self.measurement_labels + list(self.simulator.output_mode)

        self.time_labels = np.hstack(self.time_labels)

        # Run
        self.run()

    def run(self, parallel=None):
        """ Construct empirical observability matrix.
        """

        if parallel is not None:
            self.parallel = parallel

        # Run simulations for perturbed initial conditions
        state_index = np.arange(0, self.simulator.n).tolist()
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
        self.O = np.zeros((self.simulator.p * self.w, self.simulator.n))
        for w in range(self.w):
            if w == 0:
                start_index = 0
            else:
                start_index = int(w * self.simulator.p)

            end_index = start_index + self.simulator.p
            self.O[start_index:end_index] = self.delta_y[:, :, w]

        # Make O into a data-frame for interpretability
        self.O_df = pd.DataFrame(self.O, columns=self.simulator.state_names, index=self.measurement_labels)
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
        y_plus = self.simulator.simulate(x0=x0_plus, u=self.u_sim)
        y_minus = self.simulator.simulate(x0=x0_minus, u=self.u_sim)

        # Calculate the numerical Jacobian & normalize by 2x the perturbation amount
        delta_y = np.array(y_plus - y_minus).T / (2 * self.eps)

        return delta_y, y_plus, y_minus


class SlidingEmpiricalObservabilityMatrix:
    def __init__(self, simulator, t_sim, x_sim, u_sim, w=None, eps=1e-5, parallel=False):
        """ Construct an empirical observability matrix O in sliding windows along a trajectory.

            Inputs
                simulator:          simulator object
                t_sim:              simulation time along trajectory
                u_sim:              simulation inputs along trajectory
                x_sim:              state trajectory
                w:                  simulation window size for each calculation of O
                eps:                amount to perturb initial state
        """

        self.simulator = simulator
        self.t_sim = t_sim.copy()
        self.eps = eps
        self.parallel = parallel

        if isinstance(x_sim, dict):
            self.x_sim = np.vstack((list(x_sim.values()))).T
        else:
            self.x_sim = np.array(x_sim).squeeze()

        if isinstance(u_sim, dict):
            self.u_sim = np.vstack(list(u_sim.values())).T
        else:
            self.u_sim = np.array(u_sim).squeeze()

        # self.dt = np.round(np.mean(np.diff(self.t_sim)), 6)
        self.N = self.t_sim.shape[0]

        if w is None:  # set window size to full time-series size
            self.w = self.N
        else:
            self.w = w

        if self.w > self.N:
            raise ValueError('Window size must be smaller than trajectory length')

        # All the indices to calculate O
        self.O_index = np.arange(0, self.N - self.w + 1,  step=1)  # indices to compute O
        self.O_time = self.t_sim[self.O_index]  # times to compute O
        self.n_point = len(self.O_index)  # # of times to calculate O

        # Where to store sliding window trajectory data & O's
        self.window_data = {'t': [], 'u': [], 'x': [], 'y': [], 'y_plus': [], 'y_minus': []}
        self.O_sliding = []
        self.O_df_sliding = []

        # Run
        self.EOM = None
        self.run()

    def run(self, parallel=None):
        """ Run.
        """

        if parallel is not None:
            self.parallel = parallel

        # Where to store sliding window trajectory data & O's
        self.window_data = {'t': [], 'u': [], 'x': [], 'y': [], 'y_plus': [], 'y_minus': []}
        self.O_sliding = []
        self.O_df_sliding = []

        # Construct O's
        n_point_range = np.arange(0, self.n_point).astype(int)
        if self.parallel:  # multiprocessing
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
        EOM = EmpiricalObservabilityMatrix(self.simulator, x0, t_win0, u_win, eps=self.eps, parallel=False)
        self.EOM = EOM

        # Store data
        O_sliding = EOM.O.copy()
        O_df_sliding = EOM.O_df.copy()

        window_data = {'t': t_win.copy(),
                       'u': u_win.copy(),
                       'x': EOM.x_nominal.copy(),
                       'y': EOM.y_nominal.copy(),
                       'y_plus': EOM.y_plus.copy(),
                       'y_minus': EOM.y_minus.copy()}

        return O_sliding, O_df_sliding, window_data


class FisherObservability:
    def __init__(self, O, R=None, sensor_noise_dict=None, sigma=None):
        """ Evaluate the observability of a state variable(s) using the Fisher Information Matrix.

            Inputs
                O: observability matrix. Can be numpy array or pandas data-frame (pxw x n)
                R: measurement noise covariance matrix (p x p)
                beta: reconstruction error bound for binary observability
                epsilon: F = F + epsilon*I if not None
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
        if sigma is None:
            # np.linalg.eig(self.F)
            self.sigma = 0.0
        else:
            self.sigma = sigma

        # Invert F
        if self.sigma == 'limit':  # calculate limit with symbolic sigma
            sigma_sym = sp.symbols('sigma')
            F_hat = self.F.values + sp.Matrix(sigma_sym * np.eye(self.n))
            F_hat_inv = F_hat.inv()
            F_hat_inv_limit = F_hat_inv.applyfunc(lambda elem: sp.limit(elem, sigma_sym, 0))
            self.F_inv = np.array(F_hat_inv_limit, dtype=np.float64)
        else:  # numeric sigma
            F_epsilon = self.F.values + (self.sigma * np.eye(self.n))
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


class SlidingFisherObservability:
    def __init__(self, O_list, time=None, sigma=1e6, R=None, sensor_noise_dict=None,
                 states=None, sensors=None, time_steps=None, w=None):

        """ Compute the Fisher information matrix & inverse in sliding windows and pull put the minimum error variance.
        :param O_list: list of observability matrices O
        :param time: time vector the same size as O_list
        :param states: list of states to use from O's
        :param sensors: list of sensors to use from O's
        :param time_steps: list of time steps to use from O's
        """

        self.O_list = O_list
        self.n_window = len(O_list)

        # Set time & time-step
        if time is None:
            self.time = np.arange(0, self.n_window, step=1)
        else:
            self.time = time

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
            self.time_steps = time_steps

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
            FO = FisherObservability(O_subset, sensor_noise_dict=sensor_noise_dict, R=R, sigma=sigma)
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