"""
JAX-based simulator and observability matrix for pybounds.

Provides drop-in alternatives to ``Simulator`` and ``EmpiricalObservabilityMatrix``
that use JAX's forward-mode autodiff (``jax.jacfwd``) instead of numerical
perturbations.  All classes produce the same output formats as their legacy
counterparts so downstream ``FisherObservability`` / ``SlidingFisherObservability``
require no changes.

Requirements
------------
- JAX must be installed: ``pip install "jax[cpu]"``
- ``f_jax`` and ``h_jax`` must use ``jax.numpy`` (not ``numpy``) for math
  operations so that JAX can trace through them.  Plain Python arithmetic
  operators (``+``, ``-``, ``*``, ``/``) work with both backends as-is.

Precision
---------
Computations run in float64 to match do_mpc, but only inside pybounds calls:
importing pybounds does not change JAX's global ``jax_enable_x64`` setting.

Pipeline hand-off
-----------------
do_mpc ``Simulator`` remains the entry point for MPC trajectory reconstruction
(finding control inputs from a measured trajectory).  Once ``(t_sim, x_sim,
u_sim)`` are known, ``JaxSimulator`` takes over for the observability analysis:

    do_mpc Simulator  →  MPC  →  (t_sim, x_sim, u_sim)
                                        ↓
                            JaxSimulator / JaxEmpiricalObservabilityMatrix
                                        ↓
                              O_df  (same format as legacy)
                                        ↓
                      FisherObservability / SlidingFisherObservability  (unchanged)
"""

import functools
import warnings
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from .observability import transform_states, _TransformJacobianAliases, _transform_O_df_list



def _x64():
    """Context manager that enables float64 (to match do_mpc precision) without changing JAX's global setting."""
    if hasattr(jax, 'enable_x64'):  # newer JAX: config state usable as a context manager
        return jax.enable_x64(True)
    from jax.experimental import enable_x64  # older JAX
    return enable_x64()


def _with_x64(method):
    """Run a method with float64 enabled only for its own JAX computations."""
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with _x64():
            return method(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# JaxSimulator
# ---------------------------------------------------------------------------

class JaxSimulator:
    """Open-loop forward simulator implemented in pure JAX.

    Uses ``jax.lax.scan`` over RK4 (or Euler) time steps so the simulation
    function is fully JAX-traceable.  This makes the measurement trajectory
    Y differentiable with respect to the initial state x₀ via ``jax.jacfwd``.

    Parameters
    ----------
    f_jax : callable
        Dynamics function ``f_jax(x, u) -> x_dot``.
        ``x`` and ``u`` are 1-D ``jnp`` arrays; return value must also be a
        ``jnp`` array of shape ``(n,)``.  When auxiliary data is passed
        (``aux`` / ``aux_list``), it is called as ``f_jax(x, u, aux)``.
    h_jax : callable
        Measurement function ``h_jax(x, u) -> y``, or ``h_jax(x, u, aux)``
        when auxiliary data is passed.
        Returns a ``jnp`` array of shape ``(p,)``.
    dt : float
        Sample time step (seconds): one input and one measurement per ``dt``.
    state_names : list of str
        Names of state variables (length n).
    input_names : list of str
        Names of input variables (length m).
    measurement_names : list of str
        Names of measurement variables (length p).
    integrator : {'rk4', 'euler'}
        Numerical integration scheme.  ``'rk4'`` is more accurate and
        recommended; ``'euler'`` is faster but first-order.
    substeps : int
        Number of integration steps per sample, each of size ``dt / substeps``,
        with the input held constant over the sample.  The default of 1 takes a
        single step per ``dt``.  Increase it when ``dt`` is coarse relative to
        the system's fastest dynamics (including moderately stiff systems);
        measurements are still returned once per ``dt``.
    """

    def __init__(self, f_jax, h_jax, dt,
                 state_names, input_names, measurement_names,
                 integrator='rk4', substeps=1):
        if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
            raise ValueError(f'substeps must be a positive integer, got {substeps!r}')
        if integrator not in ('rk4', 'euler'):
            raise ValueError(f"integrator must be 'rk4' or 'euler', got {integrator!r}")

        self.f_jax = f_jax
        self.h_jax = h_jax
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.state_names = list(state_names)
        self.input_names = list(input_names)
        self.measurement_names = list(measurement_names)
        self.n = len(state_names)
        self.m = len(input_names)
        self.p = len(measurement_names)
        self.integrator = integrator

        # Build the pure JAX simulation function once and store it.
        self._simulate_jax = self._build_simulate()

    def _build_simulate(self):
        """Return a pure JAX function  simulate(x0, u_seq, aux=None) -> y_traj.

        x0    : shape (n,)
        u_seq : shape (w, m)  — one input vector per time step
        aux   : None, or any JAX pytree passed to f and h at every time step
        y_traj: shape (w, p)  — measurement at every time step
        """
        substeps = self.substeps
        dt = self.dt / substeps  # integration step
        f = self.f_jax
        h = self.h_jax
        integrator = self.integrator

        def euler_step(f_xu, x, u):
            return x + dt * jnp.asarray(f_xu(x, u))

        def rk4_step(f_xu, x, u):
            k1 = jnp.asarray(f_xu(x,              u))
            k2 = jnp.asarray(f_xu(x + dt / 2 * k1, u))
            k3 = jnp.asarray(f_xu(x + dt / 2 * k2, u))
            k4 = jnp.asarray(f_xu(x + dt * k3,      u))
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        step_fn = rk4_step if integrator == 'rk4' else euler_step

        def simulate(x0, u_seq, aux=None):
            """Integrate forward and return measurement trajectory.

            Measurement at step k is evaluated *before* integrating step k,
            matching the convention used by ``Simulator.simulate()``.
            """
            # With aux, f and h are called as f(x, u, aux) and h(x, u, aux)
            if aux is None:
                f_xu, h_xu = f, h
            else:
                def f_xu(x, u):
                    return f(x, u, aux)

                def h_xu(x, u):
                    return h(x, u, aux)

            def scan_fn(x, u):
                y = jnp.asarray(h_xu(x, u))
                if substeps == 1:
                    x_next = step_fn(f_xu, x, u)
                else:
                    x_next = jax.lax.fori_loop(0, substeps, lambda _, x_k: step_fn(f_xu, x_k, u), x)
                return x_next, y

            x0_arr = jnp.asarray(x0, dtype=jnp.float64)
            u_arr = jnp.asarray(u_seq, dtype=jnp.float64)
            _, y_traj = jax.lax.scan(scan_fn, x0_arr, u_arr)
            return y_traj   # shape (w, p)

        return simulate

    @_with_x64
    def simulate(self, x0, u_seq, aux=None):
        """Run the open-loop simulation.

        Parameters
        ----------
        x0 : array-like or dict
            Initial state, shape ``(n,)`` or dict mapping state name → value.
        u_seq : array-like or dict
            Input sequence, shape ``(w, m)`` or dict mapping input name →
            array of length ``w``.
        aux : optional
            Auxiliary data (scalar, array, or dict/tuple of arrays) passed
            unchanged to ``f_jax(x, u, aux)`` and ``h_jax(x, u, aux)`` at every
            time step.  If None, ``f_jax(x, u)`` and ``h_jax(x, u)`` are called.

        Returns
        -------
        y_traj : np.ndarray, shape (w, p)
            Measurement trajectory.
        """
        x0_arr = _to_array(x0, self.state_names)
        u_arr = _to_u_array(u_seq, self.input_names)
        y = np.array(self._simulate_jax(x0_arr, u_arr, _to_aux(aux)))
        if not np.isfinite(y).all():
            _warn_nonfinite('JaxSimulator.simulate output')
        return y


# ---------------------------------------------------------------------------
# JaxEmpiricalObservabilityMatrix
# ---------------------------------------------------------------------------

class JaxEmpiricalObservabilityMatrix(_TransformJacobianAliases):
    """Observability matrix via JAX forward-mode autodiff.

    Replaces the 2n numerical perturbation simulations used by
    ``EmpiricalObservabilityMatrix`` with a single ``jax.jacfwd`` call,
    giving the *exact* Jacobian dY/dx₀ in one forward pass.

    Output attributes match those of ``EmpiricalObservabilityMatrix`` so this
    class can be used as a drop-in replacement.

    Parameters
    ----------
    jax_simulator : JaxSimulator
        A configured ``JaxSimulator`` instance.
    x0 : array-like or dict
        Initial state, shape ``(n,)`` or dict.
    u_seq : array-like or dict
        Input sequence, shape ``(w, m)`` or dict.
    eps : float, optional
        Accepted for API compatibility but not used (the Jacobian is exact).
    aux : optional
        Auxiliary data passed to ``f_jax(x, u, aux)`` and ``h_jax(x, u, aux)``
        (see ``JaxSimulator.simulate``).  The Jacobian is taken with respect
        to x₀ only.
    z_function : callable, optional
        Transforms coordinates from original to new states, ``z = z_function(x)``,
        using sympy functions.  Same as ``EmpiricalObservabilityMatrix``;
        leave as None to keep the original coordinates.
    z_state_names : list of str, optional
        Names of the states in the new coordinates.
    """

    @_with_x64
    def __init__(self, jax_simulator, x0, u_seq, eps=None, aux=None,
                 z_function=None, z_state_names=None):
        _require_jax_simulator(jax_simulator, 'JaxEmpiricalObservabilityMatrix')
        self.jax_simulator = jax_simulator
        self.eps = eps  # kept for API compat; not used

        x0_arr = _to_array(x0, jax_simulator.state_names)
        u_arr = _to_u_array(u_seq, jax_simulator.input_names)
        aux_arr = _to_aux(aux)

        self.x0 = x0_arr
        self.u = u_arr
        self.aux = aux
        self.n = jax_simulator.n
        self.p = jax_simulator.p
        self.w = u_arr.shape[0]
        self.state_names = jax_simulator.state_names
        self.measurement_names = jax_simulator.measurement_names

        # Nominal trajectory
        self.y_nominal = np.array(jax_simulator._simulate_jax(x0_arr, u_arr, aux_arr))  # (w, p)

        # Jacobian: dY/dx0, shape (w, p, n)
        jac_fn = jax.jit(jax.jacfwd(jax_simulator._simulate_jax, argnums=0))
        jac = np.array(jac_fn(x0_arr, u_arr, aux_arr))   # (w, p, n)

        # Reshape to (w*p, n) matching EmpiricalObservabilityMatrix.O
        # Row order: [sensor_0 t=0, sensor_1 t=0, ..., sensor_p t=0,
        #             sensor_0 t=1, ...]
        self.O = jac.reshape(self.w * self.p, self.n)
        if not (np.isfinite(self.y_nominal).all() and np.isfinite(self.O).all()):
            _warn_nonfinite('JaxEmpiricalObservabilityMatrix')

        # Build MultiIndex DataFrame matching EmpiricalObservabilityMatrix.O_df
        measurement_labels = self.measurement_names * self.w
        time_labels = np.repeat(np.arange(self.w), self.p).astype(int)

        self.O_df = pd.DataFrame(
            self.O,
            columns=self.state_names,
            index=measurement_labels,
        )
        self.O_df['time_step'] = time_labels
        self.O_df = self.O_df.set_index('time_step', append=True)
        self.O_df.index.names = ['sensor', 'time_step']

        # Perform coordinate transformation on O, if specified
        if z_function is not None:
            self.O_df, self.dxdz, self.dzdx_sym = transform_states(O=self.O_df,
                                                                   square_flag=False,
                                                                   z_function=z_function,
                                                                   x0=np.array(x0_arr),
                                                                   z_state_names=z_state_names)
            self.state_names = list(self.O_df.columns)
            self.O = self.O_df.values
        else:
            self.dxdz = None
            self.dzdx_sym = None


# ---------------------------------------------------------------------------
# JaxSlidingEmpiricalObservabilityMatrix
# ---------------------------------------------------------------------------

class JaxSlidingEmpiricalObservabilityMatrix:
    """Sliding observability matrix computed via JAX vmap + jacfwd.

    Batches all sliding windows into a single vmapped ``jax.jacfwd`` call,
    giving exact Jacobians for every window in one XLA kernel launch.

    Output attributes match those of ``SlidingEmpiricalObservabilityMatrix``
    (``O_sliding``, ``O_df_sliding``, ``O_time``, ``O_index``, ``t_sim``,
    ``window_data``).  ``window_data`` holds per-window lists under ``'t'``,
    ``'u'`` and ``'y'`` (nominal measurements); it has no ``'y_plus'`` /
    ``'y_minus'`` keys because the Jacobian is exact and no perturbed
    simulations are run.

    Parameters
    ----------
    jax_simulator : JaxSimulator
        A configured ``JaxSimulator`` instance.
    t_sim : array-like, shape (T,)
        Time vector for the trajectory.
    x_sim : array-like or dict, shape (T, n)
        State trajectory.
    u_sim : array-like or dict, shape (T, m)
        Input trajectory.
    w : int, optional
        Window size in time steps.  If None, use the full trajectory (one window).
    aux_list : list, optional
        Auxiliary data, one entry per time step of the trajectory (length T),
        as in ``SlidingEmpiricalObservabilityMatrix``.  Window i passes
        ``aux_list[i]`` to ``f_jax(x, u, aux)`` and ``h_jax(x, u, aux)``.
        Entries may be scalars, arrays, or dicts/tuples of arrays, but they must
        all have the same structure and shapes so they can be batched with vmap.
    z_function : callable, optional
        Transforms coordinates from original to new states, ``z = z_function(x)``,
        using sympy functions.  Each window's O is transformed at that window's
        initial state, as in ``SlidingEmpiricalObservabilityMatrix``.
    z_state_names : list of str, optional
        Names of the states in the new coordinates.
    batch_size : int, optional
        Compute at most this many windows per batched (vmap) call. None (default) computes all windows in
        one call. batch_size trades memory against speed:

        - Memory: JAX's working memory for the Jacobians scales with batch_size, so the peak falls as
          batch_size gets small relative to the number of windows. A batch close to the number of windows
          can use more memory than no batching, because the chunks are then copied into separate storage.
        - Speed: smaller batches mean more calls and a slower run. For example, with 790 windows
          (40 states, 66 sensors, nonlinear dynamics): 1.5 s unbatched, and 1.7 / 2.0 / 3.0 s at
          batch_size 128 / 32 / 8, whose peaks were 1.5x / 1.14x / 1.05x one copy of O instead of 2.2x.
        - Results: repeatable for a given batch_size, but XLA vectorizes across the batch, so a few windows
          can differ from the unbatched result in the last bit (~1e-16 relative). To replay a run exactly,
          use the same batch_size (or None).

        A batch_size of a few percent of the number of windows keeps most of the memory saving at a moderate
        time cost; in general, use the largest batch that fits in memory.
    """

    @_with_x64
    def __init__(self, jax_simulator, t_sim, x_sim, u_sim, w=None, aux_list=None,
                 z_function=None, z_state_names=None, batch_size=None):
        self._prepare(jax_simulator, t_sim, x_sim, u_sim, w=w, aux_list=aux_list)
        jac_batch, y_batch = self._compute(batch_size=batch_size)
        self._build_windows(jac_batch, y_batch, z_function, z_state_names)

    @classmethod
    def _prepared(cls, *args, **kwargs):
        """Validated, batched inputs without computing anything (used to get O as one array)."""
        self = cls.__new__(cls)
        self._prepare(*args, **kwargs)
        return self

    @_with_x64
    def _prepare(self, jax_simulator, t_sim, x_sim, u_sim, w=None, aux_list=None):
        _require_jax_simulator(jax_simulator, 'JaxSlidingEmpiricalObservabilityMatrix')
        self.jax_simulator = jax_simulator
        self.n = jax_simulator.n
        self.p = jax_simulator.p
        self.state_names = jax_simulator.state_names
        self.measurement_names = jax_simulator.measurement_names

        self.t_sim = np.asarray(t_sim).ravel()
        N = len(self.t_sim)

        # Convert x_sim to (T, n) array
        if isinstance(x_sim, dict):
            x_arr = np.column_stack([np.asarray(x_sim[k]).ravel()
                                     for k in jax_simulator.state_names])
        else:
            x_arr = np.asarray(x_sim)
            if x_arr.ndim == 1:
                x_arr = x_arr[:, None]

        # Convert u_sim to (T, m) array
        if isinstance(u_sim, dict):
            u_arr = np.column_stack([np.asarray(u_sim[k]).ravel()
                                     for k in jax_simulator.input_names])
        else:
            u_arr = np.asarray(u_sim)
            if u_arr.ndim == 1:
                u_arr = u_arr[:, None]

        if N != x_arr.shape[0]:
            raise ValueError('t_sim & x_sim must have same number of rows')
        if N != u_arr.shape[0]:
            raise ValueError('t_sim & u_sim must have same number of rows')
        if w is None:  # set window size to full time-series size, as in SlidingEmpiricalObservabilityMatrix
            w = N
        if w < 1:
            raise ValueError(f'window size ({w}) must be at least 1')
        if w > N:
            raise ValueError(f'window size ({w}) must be smaller than trajectory length ({N})')
        self.w = w

        self.O_index = np.arange(0, N - w + 1, step=1)
        self.O_time = self.t_sim[self.O_index]
        n_windows = len(self.O_index)

        # Build batched arrays: x0_batch (n_windows, n), u_batch (n_windows, w, m)
        x0_batch = jnp.array(
            np.stack([x_arr[i] for i in self.O_index]), dtype=jnp.float64)
        u_batch = jnp.array(
            np.stack([u_arr[i:i + w] for i in self.O_index]), dtype=jnp.float64)
        self._u_arr = u_arr

        # Batch aux over windows: window i uses aux_list[i], as in SlidingEmpiricalObservabilityMatrix
        self.aux_list = aux_list
        if aux_list is None:
            aux_batch, aux_axis = None, None
        else:
            if len(aux_list) != N:
                raise ValueError('aux_list must have same number of elements as t_sim')
            try:
                aux_batch = jax.tree_util.tree_map(
                    lambda *leaves: jnp.stack(leaves),
                    *[_to_aux(aux_list[i]) for i in range(n_windows)])
            except (ValueError, TypeError) as e:
                raise ValueError('aux_list entries must all have the same structure and array shapes '
                                 'so they can be batched across windows') from e
            aux_axis = 0
        self._batch = (x0_batch, u_batch, aux_batch, aux_axis)

    def _batched_functions(self):
        """jit(vmap(jacfwd(simulate))) and jit(vmap(simulate)) over windows."""
        aux_axis = self._batch[3]
        sim = self.jax_simulator._simulate_jax
        return (jax.jit(jax.vmap(jax.jacfwd(sim, argnums=0), in_axes=(0, 0, aux_axis))),
                jax.jit(jax.vmap(sim, in_axes=(0, 0, aux_axis))))

    @_with_x64
    def _compute(self, copy=True, batch_size=None):
        """Jacobian (n_windows, w, p, n) and nominal trajectory (n_windows, w, p) of every window.

        With copy=False the Jacobian may share JAX's result buffer (read-only), avoiding a second full copy.
        With batch_size, windows are computed in chunks (see _iter_chunks) into preallocated arrays.
        """
        x0_batch, u_batch, aux_batch, aux_axis = self._batch
        n_windows = len(self.O_index)
        if batch_size is not None:
            jac_batch = np.empty((n_windows, self.w, self.p, self.n))
            y_batch = np.empty((n_windows, self.w, self.p))
            for start, jac, y in self._iter_chunks(batch_size, warn=False):
                jac_batch[start:start + len(jac)] = jac
                y_batch[start:start + len(y)] = y
        else:
            # Single vmapped jacfwd call — one XLA kernel for all windows
            vmapped_jac, vmapped_sim = self._batched_functions()
            jac_result = vmapped_jac(x0_batch, u_batch, aux_batch)
            jac_batch = np.array(jac_result) if copy else np.asarray(jac_result)  # (n_windows, w, p, n)
            del jac_result
            y_batch = np.array(vmapped_sim(x0_batch, u_batch, aux_batch))    # (n_windows, w, p)

        bad = _nonfinite_windows(jac_batch, y_batch)
        if bad:
            _warn_nonfinite(f'JaxSlidingEmpiricalObservabilityMatrix ({bad} of {n_windows} windows)')
        return jac_batch, y_batch

    def _iter_chunks(self, batch_size, warn=True):
        """Yield (start, jac, y) for consecutive chunks of at most batch_size windows.

        Every call uses exactly batch_size windows (the last chunk is padded by repeating its last window and
        the padding is dropped), so the batched functions are compiled once. jac may share JAX's buffer
        (read-only). With warn, NaN/inf values are reported after the last chunk.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError(f'batch_size must be a positive integer, got {batch_size!r}')
        x0_batch, u_batch, aux_batch, aux_axis = self._batch
        n_windows = len(self.O_index)
        # XLA compiles a batch of one window differently (the batch axis disappears); computing it as a padded
        # batch of two keeps it on the same code path as larger batches (unless there is only one window)
        batch_size = min(max(int(batch_size), 2), n_windows)
        vmapped_jac, vmapped_sim = self._batched_functions()
        bad = 0
        for start in range(0, n_windows, batch_size):
            count = min(batch_size, n_windows - start)
            idx = np.minimum(np.arange(start, start + batch_size), n_windows - 1)
            with _x64():
                aux = None if aux_batch is None else jax.tree_util.tree_map(lambda leaf: leaf[idx], aux_batch)
                jac = np.asarray(vmapped_jac(x0_batch[idx], u_batch[idx], aux))[:count]
                y = np.asarray(vmapped_sim(x0_batch[idx], u_batch[idx], aux))[:count]
            bad += _nonfinite_windows(jac, y)
            yield start, jac, y
        if warn and bad:
            _warn_nonfinite(f'JaxSlidingEmpiricalObservabilityMatrix ({bad} of {n_windows} windows)')

    def _window_frame(self, O_i):
        """One window's O as a DataFrame (same format as SlidingEmpiricalObservabilityMatrix)."""
        O_df_i = pd.DataFrame(O_i, columns=self.state_names, index=self.measurement_names * self.w)
        O_df_i['time_step'] = np.repeat(np.arange(self.w), self.p).astype(int)
        O_df_i = O_df_i.set_index('time_step', append=True)
        O_df_i.index.names = ['sensor', 'time_step']
        return O_df_i

    def _build_windows(self, jac_batch, y_batch, z_function, z_state_names):
        """Per-window O arrays and DataFrames, window_data, and the optional coordinate transform."""
        w, u_arr, x0_batch = self.w, self._u_arr, self._batch[0]
        n_windows = len(self.O_index)

        self.O_sliding = []
        self.O_df_sliding = []
        self.y_nominal_sliding = []

        # Per-window trajectory data, matching SlidingEmpiricalObservabilityMatrix.window_data.
        # There are no 'y_plus' / 'y_minus' keys because no perturbed simulations are run.
        self.window_data = {'t': [], 'u': [], 'y': []}

        for i in range(n_windows):
            O_i = jac_batch[i].reshape(w * self.p, self.n)
            self.O_sliding.append(O_i)
            self.y_nominal_sliding.append(y_batch[i])

            win = np.arange(self.O_index[i], self.O_index[i] + w)
            self.window_data['t'].append(self.t_sim[win].copy())
            self.window_data['u'].append(u_arr[win].copy())
            self.window_data['y'].append(y_batch[i].copy())

            self.O_df_sliding.append(self._window_frame(O_i))

        # Perform coordinate transformation on each window's O, if specified
        if z_function is not None:
            self.O_df_sliding = _transform_O_df_list(self.O_df_sliding, np.array(x0_batch),
                                                     z_function, z_state_names)
            self.O_sliding = [O_df_i.values for O_df_i in self.O_df_sliding]
            self.state_names = list(self.O_df_sliding[0].columns)

    def get_observability_matrix(self):
        """Return a copy of the sliding O_df list."""
        return self.O_df_sliding.copy()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_jax_simulator(simulator, cls_name):
    """Raise a clear error when a non-JAX simulator is passed to a JAX-backend class."""
    if not isinstance(simulator, JaxSimulator):
        raise TypeError(f'{cls_name} requires a JaxSimulator, got {type(simulator).__name__}; '
                        f'use {cls_name[3:]} instead (compute_observability: use_jax=False, or leave use_jax unset '
                        f'to pick the backend from the simulator).')


def _nonfinite_windows(jac, y):
    """Number of windows whose Jacobian or nominal trajectory has NaN or inf values."""
    bad = ~(np.isfinite(jac).all(axis=(1, 2, 3)) & np.isfinite(y).all(axis=(1, 2)))
    return int(bad.sum())


def _warn_nonfinite(context):
    """Warn about NaN or inf values, which usually mean the integration diverged."""
    warnings.warn(
        f'{context} contains NaN or inf values; the integration may have diverged or f/h '
        'produced invalid values. For coarse dt or stiff dynamics, increase JaxSimulator(substeps=...), '
        'reduce dt, or use the CasADi backend (IDAS).',
        RuntimeWarning, stacklevel=3)


def _to_aux(aux):
    """Convert aux (None, scalar, array, or pytree of those) to jnp leaves."""
    if aux is None:
        return None
    return jax.tree_util.tree_map(jnp.asarray, aux)


def _to_array(x, names):
    """Convert x0 (dict or array-like) to a 1-D float64 jnp array."""
    if isinstance(x, dict):
        return jnp.array([x[k] for k in names], dtype=jnp.float64)
    return jnp.array(x, dtype=jnp.float64).ravel()


def _to_u_array(u, names):
    """Convert u (dict or array-like) to a 2-D float64 jnp array (w, m)."""
    if isinstance(u, dict):
        cols = [np.asarray(u[k]).ravel() for k in names]
        return jnp.array(np.column_stack(cols), dtype=jnp.float64)
    arr = np.asarray(u)
    if arr.ndim == 1:
        arr = arr[:, None]
    return jnp.array(arr, dtype=jnp.float64)
