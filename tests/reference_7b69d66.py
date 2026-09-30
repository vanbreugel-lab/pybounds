"""
Reference copy of FisherObservability and SlidingFisherObservability from pybounds commit 7b69d66
(the version before the memory changes), copied verbatim. Tests compare the current implementation
against it bit for bit. Do not edit.
"""

import warnings

import numpy as np
import pandas as pd
import sympy as sp

DEFAULT_LAM = 1e-8


class FisherObservability:
    def __init__(self, O, R=None, lam=DEFAULT_LAM, force_R_scalar=False,
                 states=None, sensors=None, time_steps=None, w=None):
        """ Evaluate the observability of a state variable(s) using the Fisher Information Matrix.

        :param np.array O: observability matrix (w*p, n)
            w is the number of time-steps, p is the number of measurements, and n in the number of states
            can also be set as pd.DataFrame where columns set the state names & a multilevel index sets the
            measurement names: O.index names must be ('sensor', 'time_step')
        :param None | np.array | float | dict  R: measurement noise covariance matrix (w*p x w*p)
            as an array, rows/columns follow the row order of the O passed in (it is subset and reordered with O)
            can also be set as pd.DataFrame where R.index = R.columns = O.index (aligned by label)
            can also be a scaler where R = R * I_(nxn)
            can also be dict where keys must correspond to the 'sensor' index in O data-frame
            if None, then R = I_(nxn)
        :param float | str lam: regularization for inverting F, computed as (F + lam*I)^-1 (Chernoff inverse).
            1/lam is the ceiling on the minimum error variance: no state's error variance can exceed 1/lam,
            so a value near 1/lam means the state is unobservable (or nearly so), not that it has that variance.
            lam is absolute, so it should be small relative to the eigenvalues of F, which scale with 1/R
            and with the units of each state. Default 1e-8 (ceiling of 1e8).
            If lam='limit', compute the limit lam -> 0 symbolically.
        :param bool force_R_scalar: force R to be a scalar, useful when the resulting R matrix is too big to fit in memory
        :param None | tuple | list states: list of states to use from O's. ex: ['g', 'd']
        :param None | tuple | list sensors: list of sensors to use from O's, ex: ['r']
        :param None | tuple | list | np.array time_steps: array of time steps to use from O's, ex: np.array([0, 1, 2])
        :param None | tuple | list | np.array w: window size to use from O's,
            if None then just grab it from O as the maximum window size        """

        # Make O a data-frame
        self.pw = O.shape[0]  # number of sensors * time-steps
        self.n = O.shape[1]  # number of states
        if isinstance(O, pd.DataFrame):  # data-frame given
            self.O = O.copy()
            self.sensor_names = tuple(O.index.get_level_values('sensor'))
            self.state_names = tuple(O.columns)
        elif isinstance(O, np.ndarray):  # array given, treat each row as one time-step of a single sensor 'y'
            self.sensor_names = tuple(['y' for _ in range(self.pw)])
            self.state_names = tuple(['x_' + str(n) for n in range(self.n)])
            index = pd.MultiIndex.from_arrays([self.sensor_names, np.arange(self.pw)], names=['sensor', 'time_step'])
            self.O = pd.DataFrame(O, index=index, columns=self.state_names)
        else:
            raise TypeError('O is not a pandas data-frame or numpy array')

        # Set window size
        if w is None:  # set automatically
            self.w = np.max(np.array(self.O.index.get_level_values('time_step'))) + 1
        else:
            self.w = w

        # Set the states to use
        if states is None:
            self.states = self.O.columns
        else:
            self.states = states

        # Set the sensors to use
        if sensors is None:
            self.sensors = self.O.index.get_level_values('sensor')
        else:
            self.sensors = sensors

        # Set the time-steps to use
        if time_steps is None:
            self.time_steps = self.O.index.get_level_values('time_step')
        else:
            self.time_steps = np.array(time_steps)

        # Get subset of O, keeping the full index so a matrix R can be aligned with it
        self._O_index_full = self.O.index
        self.O = self.O.loc[(self.sensors, self.time_steps), self.states].sort_values(['time_step', 'sensor'])

        # Reset the size of O
        self.pw = self.O.shape[0]  # number of sensors * time-steps
        self.n = self.O.shape[1]  # number of states

        # Set measurement noise covariance matrix & calculate Fisher Information Matrix
        if force_R_scalar and np.isscalar(R):  # scalar R
            self.R = pd.DataFrame({'R': {'index': float(R)}})
            self.R_inv = pd.DataFrame({'R_inv': {'index': 1 / self.R.values.squeeze()}})

            # Calculate Fisher Information Matrix for scalar R
            self.F = self.R_inv.values.squeeze() * (self.O.values.T @ self.O.values)

        elif force_R_scalar and not np.isscalar(R):
            raise Exception('R must be a scalar')

        else:  # non-scalar R
            self.R = pd.DataFrame(np.eye(self.pw), index=self.O.index, columns=self.O.index)
            self.R_inv = pd.DataFrame(np.eye(self.pw), index=self.O.index, columns=self.O.index)
            self.set_noise_covariance(R=R)

            # Calculate Fisher Information Matrix for non-scalar R
            self.F = self.O.values.T @ self.R_inv.values @ self.O.values

        self.F = pd.DataFrame(self.F, index=self.O.columns, columns=self.O.columns)

        # Set sigma
        if lam is None:
            self.lam = DEFAULT_LAM
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

        self.F_inv = pd.DataFrame(self.F_inv, index=self.O.columns, columns=self.O.columns)

        # Pull out diagonal elements
        self.error_variance = pd.DataFrame(np.diag(self.F_inv), index=self.O.columns).T

    def set_noise_covariance(self, R=None):
        """ Set the measurement noise covariance matrix.
        """

        # Preallocate the noise covariance matrix R
        self.R = pd.DataFrame(np.eye(self.pw), index=self.O.index, columns=self.O.index)

        # Set R based on values in dict
        if isinstance(R, dict):  # set each distinct sensor's noise level
            for s in pd.unique(self.R.index.get_level_values('sensor')):
                R_sensor = self.R.loc[[s], [s]]
                for r in range(R_sensor.shape[0]):
                    R_sensor.iloc[r, r] = R[s]

                self.R.loc[[s], [s]] = R_sensor.values
        else:
            if R is None:  # set R as identity matrix
                warnings.warn('R not set, defaulting to identity matrix')
            else:  # set R directly
                if isinstance(R, pd.DataFrame):  # matrix R in data-frame, aligned with O by index labels
                    self.R = R.loc[self.O.index, self.O.index].copy()
                elif isinstance(R, np.ndarray) and R.ndim == 2:  # matrix in array
                    n_full = len(self._O_index_full)
                    if R.shape == (n_full, n_full):  # rows/columns in the order of the O passed in
                        R_full = pd.DataFrame(R, index=self._O_index_full, columns=self._O_index_full)
                        self.R = R_full.loc[self.O.index, self.O.index].copy()
                    elif R.shape == (self.pw, self.pw):  # already matches the subset & sorted O
                        self.R = pd.DataFrame(R, index=self.R.index, columns=self.R.columns)
                    else:
                        raise ValueError(f'R array must be ({n_full}, {n_full}) to match O, '
                                         f'or ({self.pw}, {self.pw}) to match the selected subset of O')
                elif np.size(R) == 1:  # scalar multiplied by identity matrix
                    self.R = float(np.squeeze(R)) * self.R
                else:
                    raise Exception('R must be a dict, numpy array, pandas data-frame, or scalar value')

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
    def __init__(self, O_list, R=None, lam=DEFAULT_LAM, time=None,
                 states=None, sensors=None, time_steps=None, w=None, force_R_scalar=False):

        """ Compute the Fisher information matrix & inverse in sliding windows and pull put the minimum error variance.

        :param list O_list: list of observability matrices O (stored as pd.DataFrame)
        :param None | np.array | float| dict  R: measurement noise covariance matrix (w*p x w*p)
            can also be set as pd.DataFrame where R.index = R.columns = O.index
            can also be a scaler where R = R * I_(nxn)
            can also be dict where keys must correspond to the 'sensor' index in O data-frame
            if None, then R = I_(nxn)
        :param float | str lam: regularization for inverting F in each window, computed as (F + lam*I)^-1.
            1/lam is the ceiling on the minimum error variance (see FisherObservability). Default 1e-8.
            If lam='limit', compute the limit lam -> 0 symbolically.
        :param None | np.array time: time vector the same size as O_list
        :param None | tuple | list states: list of states to use from O's. ex: ['g', 'd']
        :param None | tuple | list sensors: list of sensors to use from O's, ex: ['r']
        :param None | tuple | list | np.array time_steps: array of time steps to use from O's, ex: np.array([0, 1, 2])
        :param None | tuple | list | np.array w: window size to use from O's,
            if None then just grab it from O as the maximum window size
        :param bool force_R_scalar: force R to be a scalar in each window (see FisherObservability)
        """

        self.O_list = O_list
        self.n_window = len(O_list)

        # Set time & time-step
        if time is None:
            self.time = np.arange(0, self.n_window, step=1)
        else:
            self.time = np.array(time)

        # Set time-step
        if time is not None:
            if len(self.time) > 1:  # compute time-step from vector
                self.dt = np.mean(np.diff(self.time))
            else:
                self.dt = 0.0
        else:  # default is time-step of 1
            self.dt = 1

        # Compute Fisher information matrix & inverse for each sliding window
        self.EV = []  # collect error variance data for each state over windows
        self.FO = []  # collect FisherObservability objects over windows
        for k in range(self.n_window):  # each window
            # Get full O
            O = self.O_list[k]

            # Compute Fisher information & inverse
            FO = FisherObservability(O, R=R, lam=lam, force_R_scalar=force_R_scalar,
                                     states=states, sensors=sensors, time_steps=time_steps, w=w)
            self.FO.append(FO)

            # Collect error variance data
            ev = FO.error_variance.copy()
            ev.insert(0, 'time_initial', self.time[k])
            self.EV.append(ev)

        # Concatenate error variance & make same size as simulation data
        # Shift the time forward by half the window size. Floor division puts odd windows at their center
        # time-step (w-1)/2; np.round's banker's rounding gave 2, 2, 4, 4 for w = 3, 5, 7, 9.
        self.shift_index = int(FO.w) // 2
        self.shift_time = self.shift_index * self.dt
        self.EV = pd.concat(self.EV, axis=0, ignore_index=True)
        if self.n_window > 1 or time is not None:  # align windows with the time vector
            self.EV.index = np.arange(self.shift_index, self.EV.shape[0] + self.shift_index, step=1, dtype=int)
            time_df = pd.DataFrame(np.atleast_2d(self.time).T, columns=['time'])
            self.EV_aligned = pd.concat((time_df, self.EV), axis=1)
        else:  # single window without a time vector: time in units of time-steps
            self.EV_aligned = self.EV.copy()
            self.EV_aligned.insert(0, 'time', self.EV['time_initial'] + self.shift_time)

    def get_minimum_error_variance(self):
        return self.EV_aligned.copy()
