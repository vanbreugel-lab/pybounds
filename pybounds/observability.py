import numpy as np
import pandas as pd
import sympy as sp
import warnings
import pickle
import pickletools
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from .simulator import Simulator
from .util import LatexStates
from .jacobian import SymbolicJacobian

# Default regularization for the Fisher information inverse (F + lam*I)^-1
DEFAULT_LAM = 1e-8


# ---------------------------------------------------------------------------
# Module-level helpers for process-based parallel sliding window computation.
# Must live at module scope so they are picklable by multiprocessing.
# ---------------------------------------------------------------------------

# Per-process Simulator instance (set in pool initialiser).
_process_simulator = None


def _pool_initializer(factory):
    """Create one Simulator per worker process."""
    global _process_simulator
    _process_simulator = factory()


def _check_spawn_picklable(obj, name):
    """Raise a clear error if spawned worker processes could not load obj.

    Spawned workers import functions by module and name. Functions defined in an interactive
    __main__ (e.g. a Jupyter notebook) cannot be imported, which makes the pool hang, and
    lambdas / nested functions cannot be pickled at all.
    """
    if obj is None:
        return

    try:
        data = pickle.dumps(obj, protocol=2)  # protocol 2 names every global as 'module name'
    except Exception as e:
        raise ValueError(f'{name} must be picklable for process-based parallelism '
                         f'(lambdas and nested functions are not); define it at module level') from e

    main_has_file = getattr(sys.modules.get('__main__'), '__file__', None) is not None
    if not main_has_file:
        for opcode, arg, _ in pickletools.genops(data):
            if opcode.name == 'GLOBAL' and arg.split(' ')[0] == '__main__':
                raise ValueError(
                    f'{name} uses {arg.split(" ", 1)[1]!r}, which is defined in an interactive session '
                    f'(e.g. a Jupyter notebook) that worker processes cannot import. Move it into a .py '
                    f'file and import it from there, or use parallel_sliding=False.')


def _compute_window(args):
    """Compute EmpiricalObservabilityMatrix for a single window (called in worker)."""
    global _process_simulator
    n, O_index, x_sim, u_sim, t_sim, N, w, eps, aux_list, z_function, z_state_names, with_data = args

    x0 = np.squeeze(x_sim[O_index[n], :])
    win = np.arange(O_index[n], O_index[n] + w, step=1)
    win = win[win < N]
    t_win = t_sim[win]
    u_win = u_sim[win, :]

    EOM = EmpiricalObservabilityMatrix(_process_simulator, x0, u_win,
                                       aux=aux_list[n], eps=eps,
                                       parallel=False,
                                       z_function=z_function,
                                       z_state_names=z_state_names)
    window_data = None
    if with_data:
        window_data = {
            't': t_win.copy(), 'u': u_win.copy(),
            'y': EOM.y_nominal.copy(),
            'y_plus': EOM.y_plus.copy(),
            'y_minus': EOM.y_minus.copy(),
        }
    return EOM.O.copy(), EOM.O_df.copy(), window_data


def _ordered_thread_map(fn, items, max_workers, max_in_flight):
    """Like ThreadPoolExecutor.map, in order, but with at most max_in_flight unconsumed results."""
    from collections import deque
    items = iter(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = deque()
        for item in items:
            pending.append(executor.submit(fn, item))
            if len(pending) >= max_in_flight:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _reject_jax_simulator(simulator, cls_name):
    """Raise a clear error when a JaxSimulator is passed to a CasADi-backend class."""
    jax_module = sys.modules.get(f'{__package__}.jax_simulator')
    if jax_module is not None and isinstance(simulator, jax_module.JaxSimulator):
        raise TypeError(f'{cls_name} does not accept a JaxSimulator; use Jax{cls_name} instead '
                        '(or compute_observability(..., use_jax=True)).')


def _ordered_values(d, names, label):
    """Values of dict d ordered by the simulator's names (insertion order if it has none)."""
    if names is None:
        return list(d.values())
    names = list(names)
    if set(d.keys()) != set(names):
        raise ValueError(f'{label} keys {list(d.keys())} must match the simulator names {names}')
    return [d[k] for k in names]


class _TransformJacobianAliases:
    """Deprecated attribute names from before the transform Jacobians were correctly labeled."""

    @property
    def dzdx(self):
        warnings.warn('dzdx is deprecated: it has always held dx/dz. Use dxdz instead.',
                      DeprecationWarning, stacklevel=2)
        return self.dxdz

    @property
    def dxdz_sym(self):
        warnings.warn('dxdz_sym is deprecated: it has always held the symbolic dz/dx. Use dzdx_sym instead.',
                      DeprecationWarning, stacklevel=2)
        return self.dzdx_sym


class EmpiricalObservabilityMatrix(_TransformJacobianAliases):
    def __init__(self, simulator, x0, u, aux=None, eps=1e-5, parallel=False,
                 z_function=None, z_state_names=None):
        """ Construct an empirical observability matrix O.

        :param callable simulator: simulator object  that has a method y = simulator.simulate(x0, u, **kwargs)
            y is (w x p) array. w is the number of time-steps and p is the number of measurements
        :param dict/list/np.array x0: initial state for Simulator
        :param dict/np.array u: inputs array
        :param aux: auxiliary input that can be passed to Simulator class
        :param float eps: epsilon value for perturbations to construct O, should be small number
        :param bool parallel: if True, run the perturbations in parallel using threads.
            Only safe for thread-safe custom simulators; ignored (with a warning) for pybounds.Simulator
        :param callable z_function: function that transforms coordinates from original to new states
            must be of the form z = z_function(x), where x & z are the same size
            should use sympy functions wherever possible
            leave as None to maintain original coordinates
        :param list | tuple | None z_state_names: (optional) names of states in new coordinates.
            will only have an effect if O is a data-frame & z_function is not None
        """

        # Store inputs
        _reject_jax_simulator(simulator, 'EmpiricalObservabilityMatrix')
        self.simulator = simulator
        self.aux = aux
        self.eps = eps
        self.parallel = parallel

        if isinstance(x0, dict):
            self.x0 = np.array(_ordered_values(x0, getattr(simulator, 'state_names', None), 'x0'))
        else:
            self.x0 = np.ravel(np.array(x0))  # 1-D, also for a single state

        if isinstance(u, dict):
            self.u = np.vstack(_ordered_values(u, getattr(simulator, 'input_names', None), 'u')).T
        else:
            self.u = np.array(u)

        # Number of states
        self.n = self.x0.shape[0]

        # Simulate once for nominal trajectory
        self.y_nominal = self.simulator.simulate(x0=self.x0, u=self.u, aux=self.aux)

        # Number of outputs
        self.p = self.y_nominal.shape[1]

        # Number of time-steps
        self.w = self.y_nominal.shape[0]  # of points in time window

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

        # Run simulations to construct O
        self.run()

        # Perform coordinate transformation on O, if specified
        if z_function is not None:
            self.O_df, self.dxdz, self.dzdx_sym = transform_states(O=self.O_df,
                                                                   square_flag=False,
                                                                   z_function=z_function,
                                                                   x0=self.x0,
                                                                   z_state_names=z_state_names)
            self.state_names = tuple(self.O_df.columns)
            self.O = self.O_df.values
        else:
            self.dxdz = None
            self.dzdx_sym = None

    def run(self, parallel=None):
        """ Construct empirical observability matrix.
        """

        if parallel is not None:
            self.parallel = parallel

        # The CasADi/IDAS integrator inside Simulator is stateful, so threads sharing one instance
        # corrupt each other's runs. Only custom thread-safe simulators can run in parallel here.
        if self.parallel and isinstance(self.simulator, Simulator):
            warnings.warn(
                'parallel=True is not thread-safe with pybounds.Simulator (CasADi/IDAS); '
                'running perturbations sequentially instead. '
                'Use SlidingEmpiricalObservabilityMatrix(parallel_sliding=True, simulator_factory=...) '
                'for process-based parallelism.',
                RuntimeWarning, stacklevel=2,
            )
            self.parallel = False

        # Run simulations for perturbed initial conditions
        state_index = np.arange(0, self.n).tolist()
        if self.parallel:  # multiprocessing
            # with Pool(4) as pool:
            #     results = pool.map(self.simulate, state_index)

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self.simulate, state_index))

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
        y_plus = self.simulator.simulate(x0=x0_plus, u=self.u, aux=self.aux)
        y_minus = self.simulator.simulate(x0=x0_minus, u=self.u, aux=self.aux)

        # Calculate the numerical Jacobian & normalize by 2x the perturbation amount
        delta_y = np.array(y_plus - y_minus).T / (2 * self.eps)

        return delta_y, y_plus, y_minus


class SlidingEmpiricalObservabilityMatrix:
    def __init__(self, simulator, t_sim, x_sim, u_sim, aux_list=None, w=None, eps=1e-5,
                 parallel_sliding=False, parallel_perturbation=False,
                 simulator_factory=None, n_workers=None,
                 z_function=None, z_state_names=None):
        """ Construct empirical observability matrix O in sliding windows along a trajectory.

        :param callable simulator: Simulator object : y = simulator(x0, u, **kwargs)
            y is (w x p) array. w is the number of time-steps and p is the number of measurements
        :param np.array t_sim: time values along state trajectory array (N, 1)
        :param np.array x_sim: state trajectory array (N, n), can also be dict
        :param np.array u_sim: input array (N, m), can also be dict
        :param aux_list: auxiliary input that can be passed to Simulator class
        :param np.array w: window size for O calculations, will automatically set how many windows to compute
        :param float eps: tolerance for sliding windows
        :param float eps: epsilon value for perturbations to construct O's, should be small number
        :param bool parallel_sliding: if True, run the sliding windows in parallel using processes.
            Requires ``simulator_factory`` when parallel_sliding=True (see below).
            Without a factory, custom simulators run in threads (they must be thread-safe), and
            pybounds.Simulator runs sequentially with a warning.
        :param bool parallel_perturbation: if True, run the perturbations in parallel (thread-based,
            only safe when the simulator's simulate() is thread-safe; ignored for pybounds.Simulator).
        :param callable simulator_factory: zero-argument callable that returns a fresh Simulator.
            Required for correct process-based parallelism (``parallel_sliding=True``).
            It (and z_function) must be importable by worker processes: define it in a .py file,
            not in a Jupyter notebook or interactive session, or a ValueError is raised.
            Each worker process will call factory() once to create its own Simulator instance,
            avoiding the thread-safety issues of CasADi/IDAS.  Example::

                def make_sim():
                    return pybounds.Simulator(dynamics_f, h, dt=0.01,
                                             state_names=['g', 'd'], ...)

                SEOM = SlidingEmpiricalObservabilityMatrix(
                    simulator, ..., parallel_sliding=True, simulator_factory=make_sim)

        :param int n_workers: number of worker processes for process-based parallelism.
            Defaults to min(n_windows, os.cpu_count()).
        """
        self._prepare(simulator, t_sim, x_sim, u_sim, aux_list=aux_list, w=w, eps=eps,
                      parallel_sliding=parallel_sliding, parallel_perturbation=parallel_perturbation,
                      simulator_factory=simulator_factory, n_workers=n_workers,
                      z_function=z_function, z_state_names=z_state_names)
        self.run()

    def _prepare(self, simulator, t_sim, x_sim, u_sim, aux_list=None, w=None, eps=1e-5,
                 parallel_sliding=False, parallel_perturbation=False,
                 simulator_factory=None, n_workers=None,
                 z_function=None, z_state_names=None):

        _reject_jax_simulator(simulator, 'SlidingEmpiricalObservabilityMatrix')
        self.simulator = simulator
        self.eps = eps
        self.parallel_sliding = parallel_sliding
        self.parallel_perturbation = parallel_perturbation
        self.simulator_factory = simulator_factory
        self.n_workers = n_workers
        self.z_function = z_function
        self.z_state_names = z_state_names

        # Set time vector
        self.t_sim = np.array(t_sim)

        # Number of points
        self.N = self.t_sim.shape[0]

        # Make x_sim & u_sim arrays
        if isinstance(x_sim, dict):
            self.x_sim = np.vstack(_ordered_values(x_sim, getattr(simulator, 'state_names', None), 'x_sim')).T
        else:
            x_sim = np.array(x_sim)
            self.x_sim = x_sim.reshape(x_sim.shape[0], -1)  # (N, n), also for a single state

        if isinstance(u_sim, dict):
            self.u_sim = np.vstack(_ordered_values(u_sim, getattr(simulator, 'input_names', None), 'u_sim')).T
        else:
            self.u_sim = np.array(u_sim)

        # Check sizes
        if self.N != self.x_sim.shape[0]:
            raise ValueError('t_sim & x_sim must have same number of rows')
        elif self.N != self.u_sim.shape[0]:
            raise ValueError('t_sim & u_sim must have same number of rows')
        elif self.x_sim.shape[0] != self.u_sim.shape[0]:
            raise ValueError('x_sim & u_sim must have same number of rows')

        # Set aux inputs
        if aux_list is None:
            self.aux_list = [None for k in range(self.N)]
        else:
            self.aux_list = aux_list

        if len(self.aux_list) != self.N:
            raise ValueError('aux_list must have same number of elements as t_sim')

        # Set time-window to calculate O's
        if w is None:  # set window size to full time-series size
            self.w = self.N
        else:
            self.w = w

        if self.w < 1:
            raise ValueError(f'window size ({self.w}) must be at least 1')
        if self.w > self.N:
            raise ValueError(f'window size ({self.w}) must be smaller than trajectory length ({self.N})')

        # All the indices to calculate O
        self.O_index = np.arange(0, self.N - self.w + 1, step=1)  # indices to compute O
        self.O_time = self.t_sim[self.O_index]  # times to compute O
        self.n_point = len(self.O_index)  # # of times to calculate O

        # Where to store sliding window trajectory data & O's
        self.window_data = {}
        self.O_sliding = []
        self.O_df_sliding = []
        self.EOM = None

    @classmethod
    def _prepared(cls, *args, **kwargs):
        """Validated inputs, ready to compute windows, without computing any (used to stream windows)."""
        self = cls.__new__(cls)
        self._prepare(*args, **kwargs)
        return self

    def run(self, parallel_sliding=None):
        """ Run.
        """

        if parallel_sliding is not None:
            self.parallel_sliding = parallel_sliding

        # Where to store sliding window trajectory data & O's
        self.window_data = {'t': [], 'u': [], 'y': [], 'y_plus': [], 'y_minus': []}
        self.O_sliding = []
        self.O_df_sliding = []

        for O_sliding, O_df_sliding, window_data in self._iter_windows(with_data=True, copy=True):
            self.O_sliding.append(O_sliding)
            self.O_df_sliding.append(O_df_sliding)
            for k in self.window_data.keys():
                self.window_data[k].append(window_data[k])

    def _iter_windows(self, with_data=True, copy=True):
        """Yield (O, O_df, window_data) for each window in order, computing one window at a time.

        :param bool with_data: also build each window's trajectory data (window_data is None otherwise)
        :param bool copy: return copies of O and O_df (not needed when each window is consumed and dropped)
        """
        # Threads sharing one pybounds Simulator corrupt each other's CasADi/IDAS runs
        if self.parallel_sliding and self.simulator_factory is None and isinstance(self.simulator, Simulator):
            warnings.warn(
                'parallel_sliding=True without simulator_factory is not thread-safe with pybounds.Simulator '
                '(CasADi/IDAS); running windows sequentially instead. '
                'Pass simulator_factory=<callable> to use process-based parallelism.',
                RuntimeWarning, stacklevel=3)
            self.parallel_sliding = False

        # Construct O's
        n_point_range = np.arange(0, self.n_point).astype(int)
        if self.parallel_sliding:
            if self.simulator_factory is not None:
                # ---- Process-based parallelism (safe with CasADi/IDAS) ----
                # Each worker process gets its own Simulator via the factory.
                import os
                _check_spawn_picklable(self.simulator_factory, 'simulator_factory')
                _check_spawn_picklable(self.z_function, 'z_function')
                n_workers = self.n_workers or min(self.n_point, os.cpu_count() or 1)
                args_iter = ((n, self.O_index, self.x_sim, self.u_sim, self.t_sim,
                              self.N, self.w, self.eps, self.aux_list,
                              self.z_function, self.z_state_names, with_data)
                             for n in n_point_range)
                ctx = multiprocessing.get_context('spawn')
                with ctx.Pool(processes=n_workers,
                              initializer=_pool_initializer,
                              initargs=(self.simulator_factory,)) as pool:
                    yield from pool.imap(_compute_window, args_iter)   # in order, as windows finish

            else:
                # ---- Thread-based parallelism, only reached for custom (thread-safe) simulators ----
                yield from _ordered_thread_map(lambda n: self._window(n, with_data, copy), n_point_range,
                                               max_workers=12, max_in_flight=24)

        else:
            for n in n_point_range:  # each point on trajectory
                yield self._window(n, with_data, copy)

    def construct(self, n):
        return self._window(n, with_data=True, copy=True)

    def _window(self, n, with_data=True, copy=True):
        # Start simulation at point along nominal trajectory
        x0 = np.squeeze(self.x_sim[self.O_index[n], :])  # get state on trajectory & set it as the initial condition

        # Get the range to pull out time & input data for simulation
        win = np.arange(self.O_index[n], self.O_index[n] + self.w, step=1)  # index range

        # Remove part of window if it is past the end of the nominal trajectory
        within_win = win < self.N
        win = win[within_win]

        # Pull out time & control inputs in window
        t_win = self.t_sim[win]  # time in window
        # t_win0 = t_win - t_win[0]  # start at 0
        u_win = self.u_sim[win, :]  # inputs in window

        # Calculate O for window
        EOM = EmpiricalObservabilityMatrix(self.simulator, x0, u_win,
                                           aux=self.aux_list[n],
                                           eps=self.eps,
                                           parallel=self.parallel_perturbation,
                                           z_function=self.z_function,
                                           z_state_names=self.z_state_names)
        self.EOM = EOM

        # Store data
        O_sliding = EOM.O.copy() if copy else EOM.O
        O_df_sliding = EOM.O_df.copy() if copy else EOM.O_df

        window_data = None
        if with_data:
            window_data = {'t': t_win.copy(),
                           'u': u_win.copy(),
                           'y': EOM.y_nominal.copy(),
                           'y_plus': EOM.y_plus.copy(),
                           'y_minus': EOM.y_minus.copy()}

        return O_sliding, O_df_sliding, window_data

    def get_observability_matrix(self):
        return self.O_df_sliding.copy()


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
        self._R = None       # R and R_inv are built on first access when R is diagonal (scalar or dict)
        self._R_inv = None
        self._R_diag = None
        if isinstance(O, pd.DataFrame):  # data-frame given (not modified: the subset below is a new frame)
            self.O = O
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
            self.set_noise_covariance(R=R)

            # Calculate Fisher Information Matrix for non-scalar R
            if self._R_diag is not None:  # diagonal R: O^T R^-1 O without a (w*p x w*p) matrix
                # (O^T * r_inv) equals O^T @ diag(r_inv) exactly (one nonzero term per element), so F is
                # bit-identical to the dense product below
                O_values = self.O.values
                self.F = np.ascontiguousarray(O_values.T * (1 / self._R_diag)) @ O_values
            else:
                self.F = self.O.values.T @ self.R_inv.values @ self.O.values

        self.F = pd.DataFrame(self.F, index=self.O.columns, columns=self.O.columns)

        # Set sigma
        if lam is None:
            self.lam = DEFAULT_LAM
        else:
            self.lam = lam

        # Invert F
        self.F_inv = _fisher_inverse(self.F.values, self.lam)

        self.F_inv = pd.DataFrame(self.F_inv, index=self.O.columns, columns=self.O.columns)

        # Pull out diagonal elements
        self.error_variance = pd.DataFrame(np.diag(self.F_inv), index=self.O.columns).T

    def set_noise_covariance(self, R=None):
        """ Set the measurement noise covariance matrix.

        A scalar, dict or None R is diagonal: it is kept as one variance per row of O (self._R_diag), and
        the R / R_inv data-frames are only built if accessed. A matrix R is stored as a data-frame.
        """
        self._R = self._R_inv = self._R_diag = None

        # Diagonal R: one variance per row of O
        if isinstance(R, dict):  # set each distinct sensor's noise level
            self._R_diag = np.array([float(R[s]) for s in self.O.index.get_level_values('sensor')])
            return
        if R is None:  # set R as identity matrix
            warnings.warn('R not set, defaulting to identity matrix')
            self._R_diag = np.ones(self.pw)
            return
        if not isinstance(R, (pd.DataFrame, np.ndarray)) or (isinstance(R, np.ndarray) and R.ndim != 2):
            if np.size(R) == 1:  # scalar multiplied by identity matrix
                self._R_diag = np.full(self.pw, float(np.squeeze(R)))
                return
            raise Exception('R must be a dict, numpy array, pandas data-frame, or scalar value')

        # Matrix R
        if isinstance(R, pd.DataFrame):  # matrix R in data-frame, aligned with O by index labels
            self.R = R.loc[self.O.index, self.O.index].copy()
        else:  # matrix in array
            n_full = len(self._O_index_full)
            if R.shape == (n_full, n_full):  # rows/columns in the order of the O passed in
                R_full = pd.DataFrame(R, index=self._O_index_full, columns=self._O_index_full)
                self.R = R_full.loc[self.O.index, self.O.index].copy()
            elif R.shape == (self.pw, self.pw):  # already matches the subset & sorted O
                self.R = pd.DataFrame(R, index=self.O.index, columns=self.O.index)
            else:
                raise ValueError(f'R array must be ({n_full}, {n_full}) to match O, '
                                 f'or ({self.pw}, {self.pw}) to match the selected subset of O')

        # Inverse of R
        R_diagonal = np.diag(self.R.values)
        is_diagonal = np.all(self.R.values == np.diag(R_diagonal))
        if is_diagonal:
            self.R_inv = np.diag(1 / R_diagonal)
        else:
            self.R_inv = np.linalg.inv(self.R.values)

        self.R_inv = pd.DataFrame(self.R_inv, index=self.R.index, columns=self.R.index)

    @property
    def R(self):
        """Measurement noise covariance as a (w*p x w*p) data-frame (built on first access for diagonal R)."""
        if self._R is None and self._R_diag is not None:
            self._R = pd.DataFrame(np.diag(self._R_diag), index=self.O.index, columns=self.O.index)
        return self._R

    @R.setter
    def R(self, value):
        self._R = value

    @property
    def R_inv(self):
        """Inverse of R as a data-frame (built on first access for diagonal R)."""
        if self._R_inv is None and self._R_diag is not None:
            self._R_inv = pd.DataFrame(np.diag(1 / self._R_diag), index=self.O.index, columns=self.O.index)
        return self._R_inv

    @R_inv.setter
    def R_inv(self, value):
        self._R_inv = value

    def get_fisher_information(self):
        return self.F.copy(), self.F_inv.copy(), self.R.copy()


class SlidingFisherObservability:
    def __init__(self, O_list, R=None, lam=DEFAULT_LAM, time=None,
                 states=None, sensors=None, time_steps=None, w=None, force_R_scalar=False, keep_windows=True):

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
        :param bool keep_windows: keep each window's FisherObservability object in self.FO. With False, only
            the error variance is kept (self.FO stays empty), so memory does not grow with the number of windows
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
            if keep_windows:
                self.FO.append(FO)

            # Collect error variance data
            ev = FO.error_variance.copy()
            ev.insert(0, 'time_initial', self.time[k])
            self.EV.append(ev)

        # Concatenate error variance & make same size as simulation data
        self.shift_index = int(FO.w) // 2
        self.shift_time = self.shift_index * self.dt
        self.EV, self.EV_aligned = _align_error_variance(pd.concat(self.EV, axis=0, ignore_index=True),
                                                         self.time, self.shift_index, self.shift_time,
                                                         aligned=self.n_window > 1 or time is not None)

    def get_minimum_error_variance(self):
        return self.EV_aligned.copy()


def _fisher_inverse(F, lam):
    """(F + lam*I)^-1 for an (n, n) array F; lam='limit' takes lam -> 0 symbolically."""
    n = F.shape[0]
    if lam == 'limit':  # calculate limit with symbolic sigma
        sigma_sym = sp.symbols('sigma')
        F_hat = F + sp.Matrix(sigma_sym * np.eye(n))
        F_hat_inv = F_hat.inv()
        F_hat_inv_limit = F_hat_inv.applyfunc(lambda elem: sp.limit(elem, sigma_sym, 0))
        return np.array(F_hat_inv_limit, dtype=np.float64)
    F_epsilon = F + (lam * np.eye(n))  # numeric sigma
    return np.linalg.inv(F_epsilon)


def _align_error_variance(EV, time, shift_index, shift_time, aligned):
    """Place one row per window (columns 'time_initial' + states) on the trajectory's time axis.

    Each window is shifted forward by half its size (floor division puts odd windows at their center
    time-step (w-1)/2). Returns (EV with the shifted index, EV_aligned with a 'time' column).
    """
    if aligned:  # align windows with the time vector
        EV.index = np.arange(shift_index, EV.shape[0] + shift_index, step=1, dtype=int)
        time_df = pd.DataFrame(np.atleast_2d(time).T, columns=['time'])
        return EV, pd.concat((time_df, EV), axis=1)
    # single window without a time vector: time in units of time-steps
    EV_aligned = EV.copy()
    EV_aligned.insert(0, 'time', EV['time_initial'] + shift_time)
    return EV, EV_aligned


def transform_states(O=None, square_flag=False, z_function=None, x0=None, z_state_names=None):
    """ Transform the coordinates of an observability matrix (O) or Fisher information matrix (F)
        from the original coordinates (x) to new user defined coordinates (z).

        :param O: observability matrix or Fisher information matrix
        :param boolean square_flag: whether to square the transform Jacobian or not
            should be set to False if passing squared an observability matrix as O
            should be set to True if passing squared a Fisher information matrix as O
        :param callable z_function: function that transforms coordinates from original to new states
            must be of the form z = z_function(x), where x & z are the same size
            should use sympy functions wherever possible
        :param np.array x0: initial state in original coordinates
        :param list | tuple z_state_names: (optional) names of states in new coordinates.
            will only have an effect if O or F is a data-frame

        :return:
            Z: observability matrix or Fisher information matrix in transformed coordinates
            dxdz: numerical Jacobian dx/dz (inverse of dz/dx) evaluated at x0, so that O_z = O @ dxdz
            dzdx_sym: symbolic Jacobian dz/dx of z_function
    """

    # Symbolic vector of original states
    x_sym = sp.symbols('x_0:%d' % O.shape[1])

    # Initialize the Jacobian calculator with a Python function
    jacobian_calculator_func = SymbolicJacobian(func=z_function, state_vars=x_sym)

    # Get the symbolic Jacobian dz/dx
    dzdx_sym = jacobian_calculator_func.jacobian_symbolic

    # Get the Jacobian calculator function
    dzdx_function = jacobian_calculator_func.get_jacobian_function()

    # Evaluate the Jacobian at x0
    dzdx = dzdx_function(np.array(x0))

    # Take the inverse
    dxdz = np.linalg.inv(dzdx)

    # Compute the new O or F (chain rule: dy/dz = dy/dx @ dx/dz)
    if square_flag:  # F
        O_z = dxdz.T @ O @ dxdz
    else:  # O
        O_z = O @ dxdz

    # Set column/index names if data-frame was passed
    if isinstance(O_z, pd.DataFrame):
        if z_state_names is not None:
            O_z.columns = z_state_names
            if square_flag:
                O_z.index = z_state_names

    return O_z, dxdz, dzdx_sym


def _z_jacobian_function(z_function, n):
    """Numerical dz/dx function of a coordinate transform over n states (symbolic Jacobian built once)."""
    x_sym = sp.symbols('x_0:%d' % n)
    return SymbolicJacobian(func=z_function, state_vars=x_sym).get_jacobian_function()


def _transform_O_df(O_df, x0, dzdx_function, z_state_names):
    """One window of transform_states: returns (O_z, dx/dz) with O_z = O_df @ dx/dz at x0."""
    dxdz = np.linalg.inv(dzdx_function(np.array(x0)))
    O_z = O_df @ dxdz
    if z_state_names is not None:
        O_z.columns = z_state_names
    return O_z, dxdz


def _transform_O_df_list(O_df_list, x0_list, z_function, z_state_names, return_dxdz=False):
    """Apply ``transform_states`` to each O data-frame at its own x0.

    Gives the same result as calling ``transform_states`` per window, but
    builds (and simplifies) the symbolic Jacobian only once. With return_dxdz=True,
    also returns the list of numerical dx/dz Jacobians (one per window).
    """
    dzdx_function = _z_jacobian_function(z_function, O_df_list[0].shape[1])

    O_df_z = []
    dxdz_list = []
    for O_df, x0 in zip(O_df_list, x0_list):
        O_z, dxdz = _transform_O_df(O_df, x0, dzdx_function, z_state_names)
        O_df_z.append(O_z)
        dxdz_list.append(dxdz)

    if return_dxdz:
        return O_df_z, dxdz_list
    return O_df_z


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

            # Default sensor names based on data-frame 'sensor' index, in order of first appearance
            self.sensors = list(O.index.get_level_values('sensor'))
            self.time_steps = np.array(O.index.get_level_values('time_step'))
            self.sensor_names_default = list(pd.unique(np.array(self.sensors, dtype=object)))
            self.time_steps_default = np.unique(self.time_steps)
        else:  # numpy matrix
            raise TypeError('n-sensor must be an integer value when O is given as a numpy matrix')

        self.n_sensor = len(self.sensor_names_default)  # number of sensors
        self.n_time_step = int(self.pw / self.n_sensor)  # number of time-steps

        # Set state names
        if state_names is not None:
            if len(state_names) == self.n:
                self.state_names = list(state_names)
            elif len(state_names) == 1:
                self.state_names = ['${' + state_names[0] + '}_{' + str(n) + '}$' for n in range(1, self.n + 1)]
            else:
                raise TypeError('state_names must be of length n or length 1')
        else:
            self.state_names = self.state_names_default.copy()

        # Convert to Latex
        LatexConverter = LatexStates()
        self.state_names = LatexConverter.convert_to_latex(self.state_names)

        # Set sensor & measurement names. Each row is labeled from its own (sensor, time_step) index,
        # so the labels are right whatever order the rows of O are in.
        if sensor_names is not None:
            if len(sensor_names) == self.n_sensor:
                self.sensor_names = list(sensor_names)
                self.sensor_names = LatexConverter.convert_to_latex(self.sensor_names, remove_dollar_signs=True)

                def label(p, k):
                    return '$' + self.sensor_names[p] + ',_{' + 'k=' + str(k) + '}$'

            elif len(sensor_names) == 1:
                self.sensor_names = [sensor_names[0] + '_{' + str(n) + '}$' for n in range(1, self.n_sensor + 1)]
                self.sensor_names = LatexConverter.convert_to_latex(self.sensor_names, remove_dollar_signs=True)

                def label(p, k):
                    return '${' + sensor_names[0] + '}_{' + str(p) + ',k=' + str(k) + '}$'
            else:
                raise TypeError('sensor_names must be of length p or length 1')

        else:
            self.sensor_names = self.sensor_names_default.copy()
            self.sensor_names = LatexConverter.convert_to_latex(self.sensor_names, remove_dollar_signs=True)

            def label(p, k):
                # braces keep a '_' in the sensor name (e.g. the default 'y_0') from making a double subscript
                return '${' + self.sensor_names[p] + '}_{' + ',k=' + str(k) + '}$'

        self.measurement_names = [label(self.sensor_names_default.index(s), k)
                                  for s, k in zip(self.sensors, self.time_steps)]

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

        # Display O (a copy: clipping must not modify self.O, and .values is read-only under pandas copy-on-write)
        O_disp = self.O.to_numpy(dtype=float, copy=True)
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


def compute_observability(simulator, t_sim, x_sim, u_sim, R,
                          w=6, eps=1e-4, lam=DEFAULT_LAM, use_jax=False):
    """Compute sliding-window Fisher observability in one call.

    Parameters
    ----------
    simulator : Simulator or JaxSimulator
        A configured simulator instance.  Pass a ``JaxSimulator`` when
        ``use_jax=True``.
    t_sim, x_sim, u_sim : trajectory returned by simulator.simulate(..., return_full_output=True)
    R : dict  — sensor noise covariance, e.g. {'r': 0.1}
    w : int   — sliding window length (time steps)
    eps : float — finite-difference perturbation size (ignored when use_jax=True)
    lam : float — Chernoff regularization for Fisher inversion, (F + lam*I)^-1.
        1/lam is the ceiling on the minimum error variance, so values near 1/lam
        (1e8 for the default) indicate unobservable states.
    use_jax : bool — if True, use JAX autodiff (exact Jacobians, faster for many windows).
        Requires JAX to be installed and ``simulator`` to be a ``JaxSimulator`` whose
        ``f`` and ``h`` functions are written with ``jax.numpy`` (``jnp``) instead of
        ``numpy``.  See ``JaxSimulator`` for details.

    Returns
    -------
    DataFrame with columns 'time', 'time_initial', and one column per state
    containing the minimum error variance for each sliding window.

    See ``ObservabilityAnalysis`` to keep the observability matrices and query
    other selections of states, sensors and time-steps without recomputing them.
    """
    from .analysis import ObservabilityAnalysis   # imported here: analysis imports this module

    method_options = {} if use_jax else {'eps': eps}   # eps does not apply to the JAX backend
    analysis = ObservabilityAnalysis(simulator, t_sim, x_sim, u_sim, method='jax' if use_jax else 'empirical',
                                     w=w, R=R, lam=lam, **method_options)
    return analysis.run().min_error_variance(states=simulator.state_names, sensors=simulator.measurement_names,
                                             time_steps=None if w is None else np.arange(w))
