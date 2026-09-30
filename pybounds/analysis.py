"""
One object for a sliding-window observability analysis.

``ObservabilityAnalysis`` holds the settings, the observability matrices (O) of every
window once ``run()`` has been called, and methods to query the minimum error
variance for any selection of states, sensors and time-steps without rebuilding O.

How O is built is pluggable: each method name maps to a builder in ``_BUILDERS``
that returns a ``SlidingO``. Adding a new way of building O (e.g. analytical or
data-driven) means writing one builder function and registering it; the rest of
the class does not change.
"""

import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Iterator, NamedTuple

import numpy as np
import pandas as pd
import yaml

from .observability import (DEFAULT_LAM, SlidingEmpiricalObservabilityMatrix, SlidingFisherObservability,
                            FisherObservability, ObservabilityMatrixImage, _ordered_values, _transform_O_df,
                            _z_jacobian_function, _fisher_inverse, _align_error_variance)


class _Unset:
    """Sentinel for 'use the analysis setting' (R=None is a valid value meaning identity)."""

    def __repr__(self):
        return '<unset>'


_UNSET = _Unset()
_NOCACHE = object()


# ---------------------------------------------------------------------------
# O builders
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlidingO:
    """Observability matrices for every sliding window, as returned by an O builder.

    Give the matrices either as ``O_df_sliding`` (one DataFrame per window) or as ``O`` (one array)
    with ``index`` and ``state_names``. All windows must share the same rows and states.

    :param list O_df_sliding: one pd.DataFrame per window with index names ('sensor', 'time_step')
        and one column per state, in the original (untransformed) coordinates
    :param np.ndarray | None t_sim: time of every point of the trajectory, shape (N,)
    :param np.ndarray O_index: index into t_sim at which each window starts (default 0, 1, 2, ...)
    :param int w: window size in time-steps (default: from the 'time_step' level)
    :param dict | None window_data: optional per-window trajectory data
    :param source: the builder's native object, for advanced use
    :param np.ndarray O: alternative to O_df_sliding: array of shape (n_windows, w*p, n). It is used as is,
        without a copy (the analysis never modifies it)
    :param pd.MultiIndex index: row index of every window, names ('sensor', 'time_step') (with O)
    :param list state_names: column names (with O)
    """
    O_df_sliding: list = None
    t_sim: np.ndarray = None
    O_index: np.ndarray = None
    w: int = None
    window_data: dict = None
    source: object = None
    O: np.ndarray = None
    index: pd.MultiIndex = None
    state_names: list = None


@dataclass
class _WindowStream:
    """Windows produced one at a time by a builder: iterating `windows` yields (O_k, index, state_names)."""
    windows: Iterator
    n_windows: int
    t_sim: np.ndarray = None
    O_index: np.ndarray = None
    w: int = None
    window_data: dict = None
    source: object = None


class _Builder(NamedTuple):
    # func(simulator, t_sim, x_sim, u_sim, *, w, stream, **options) -> SlidingO, or a _WindowStream when
    # stream=True (a builder may also return a SlidingO then, e.g. when it computes all windows at once)
    func: Callable
    options: frozenset      # option names the builder accepts


def _from_sliding_object(obj):
    """SlidingO from any object exposing O_df_sliding, t_sim, O_index (and optionally w, window_data)."""
    if isinstance(obj, SlidingO):
        return obj
    return SlidingO(O_df_sliding=list(obj.O_df_sliding), t_sim=getattr(obj, 't_sim', None),
                    O_index=getattr(obj, 'O_index', None), w=getattr(obj, 'w', None),
                    window_data=getattr(obj, 'window_data', None), source=obj)


def _from_native(native):
    """SlidingO holding one stacked array from a sliding object with O_sliding (arrays) and O_df_sliding."""
    return SlidingO(O=np.stack(native.O_sliding), index=native.O_df_sliding[0].index,
                    state_names=list(native.O_df_sliding[0].columns), t_sim=native.t_sim,
                    O_index=native.O_index, w=native.w, window_data=native.window_data, source=native)


def _window_array(sliding):
    """(O, index, state_names) of a SlidingO, with O as one (n_windows, rows, n) float array."""
    if sliding.O is not None:
        O = np.asarray(sliding.O, dtype=float)
        if O.ndim != 3:
            raise ValueError(f'SlidingO.O must have shape (n_windows, w*p, n), got {O.shape}')
        if sliding.index is None or sliding.state_names is None:
            raise ValueError('SlidingO.O needs index (rows) and state_names (columns)')
        index, state_names = sliding.index, list(sliding.state_names)
        if len(index) != O.shape[1] or len(state_names) != O.shape[2]:
            raise ValueError(f'SlidingO.index has {len(index)} rows and state_names {len(state_names)} names, '
                             f'but O has shape {O.shape}')
    else:
        frames = list(sliding.O_df_sliding or [])
        if not frames:
            raise ValueError('the observability matrix builder returned no windows')
        index, state_names = frames[0].index, list(frames[0].columns)
        O = np.empty((len(frames), len(index), len(state_names)))
        for k, frame in enumerate(frames):
            if not frame.index.equals(index) or list(frame.columns) != state_names:
                raise ValueError(f'window {k} has different rows or states than window 0; '
                                 'all windows must share the same rows and states')
            O[k] = frame.to_numpy(dtype=float)
    if list(index.names) != ['sensor', 'time_step']:
        raise ValueError(f"the row index names must be ['sensor', 'time_step'], got {list(index.names)}")
    return O, index, state_names


class _FastWindows:
    """Per-window Fisher information and error variance straight from the stored O array.

    Reproduces FisherObservability for a scalar, dict or None R: the same selected rows and columns, laid out
    in Fortran order as FisherObservability's DataFrame gives them, and the same 2-D matrix products and
    inversion, so results are bit-identical, without per-window pandas indexing.
    """

    def __init__(self, analysis, rows, cols, reference, lam):
        self._analysis, self._rows, self._cols, self._lam = analysis, rows, cols, lam
        self._columns = reference.O.columns
        if reference._R_diag is None:   # force_R_scalar: F = R_inv * (O^T O)
            self._scale, self._r_inv = reference.R_inv.values.squeeze(), None
        else:                            # diagonal R: F = (O^T * r_inv) @ O
            self._scale, self._r_inv = None, 1 / reference._R_diag

    def _O(self, k):
        return np.asfortranarray(self._analysis._O[k][np.ix_(self._rows, self._cols)])

    def fisher(self, k):
        O_values = self._O(k)
        if self._r_inv is None:
            return self._scale * (O_values.T @ O_values)
        return np.ascontiguousarray(O_values.T * self._r_inv) @ O_values

    def error_variance_row(self, k):
        F = pd.DataFrame(self.fisher(k), index=self._columns, columns=self._columns)
        return np.diag(_fisher_inverse(F.values, self._lam))

    def fisher_information(self):
        n_windows = self._analysis._n_windows
        return np.stack([pd.DataFrame(self.fisher(k), index=self._columns, columns=self._columns).to_numpy()
                         for k in range(n_windows)])

    def error_variance(self):
        """Aligned error variance, as SlidingFisherObservability(...).get_minimum_error_variance() returns it."""
        a = self._analysis
        n_window = a._n_windows
        if a._t_sim is None:
            time, dt = np.arange(0, n_window, step=1), 1
        else:
            time = np.array(a._t_sim)
            dt = np.mean(np.diff(time)) if len(time) > 1 else 0.0
        rows = []
        for k in range(n_window):
            ev = pd.DataFrame(self.error_variance_row(k), index=self._columns).T
            ev.insert(0, 'time_initial', time[k])
            rows.append(ev)
        shift_index = a._w // 2
        return _align_error_variance(pd.concat(rows, axis=0, ignore_index=True), time, shift_index,
                                     shift_index * dt, aligned=n_window > 1 or a._t_sim is not None)[1]


class _WindowFrames(Sequence):
    """Read-only sequence of per-window DataFrames, built on access from one stored array."""

    def __init__(self, O, index, state_names):
        self._O, self._index, self._state_names = O, index, state_names

    def __len__(self):
        return self._O.shape[0]

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self[i] for i in range(*k.indices(len(self)))]
        return pd.DataFrame(self._O[k], index=self._index, columns=self._state_names, copy=True)


def _build_empirical(simulator, t_sim, x_sim, u_sim, *, w, stream=False, **options):
    """Finite-difference O from a CasADi/do_mpc (or custom) simulator.

    With stream=True the windows are computed and handed over one at a time, so all windows' O,
    perturbed trajectories and DataFrames never exist at once.
    """
    if not stream:
        return _from_native(SlidingEmpiricalObservabilityMatrix(simulator, t_sim, x_sim, u_sim, w=w, **options))
    native = SlidingEmpiricalObservabilityMatrix._prepared(simulator, t_sim, x_sim, u_sim, w=w, **options)

    def windows():
        for O, O_df, _ in native._iter_windows(with_data=False, copy=False):
            yield O, O_df.index, list(O_df.columns)

    return _WindowStream(windows=windows(), n_windows=native.n_point, t_sim=native.t_sim,
                         O_index=native.O_index, w=native.w)


def _build_jax(simulator, t_sim, x_sim, u_sim, *, w, stream=False, batch_size=None, **options):
    """Exact (autodiff) O from a JaxSimulator.

    Without batch_size, all windows are computed in one batched call and, with stream=True, the resulting
    array is used directly, without per-window arrays or DataFrames. With batch_size, chunks of that many
    windows are computed and streamed one window at a time.
    """
    try:
        from .jax_simulator import JaxSlidingEmpiricalObservabilityMatrix
    except ImportError:
        raise ImportError("JAX is not installed. Install it with: pip install jax[cpu]") from None
    if not stream:
        return _from_native(JaxSlidingEmpiricalObservabilityMatrix(simulator, t_sim, x_sim, u_sim, w=w,
                                                                   batch_size=batch_size, **options))
    native = JaxSlidingEmpiricalObservabilityMatrix._prepared(simulator, t_sim, x_sim, u_sim, w=w, **options)
    if batch_size is not None:
        rows, n = native.w * native.p, native.n
        index = native._window_frame(np.zeros((rows, n))).index
        state_names = list(native.state_names)

        def windows():
            for _, jac, _ in native._iter_chunks(batch_size):
                for jac_k in jac:
                    yield jac_k.reshape(rows, n), index, state_names

        return _WindowStream(windows=windows(), n_windows=len(native.O_index), t_sim=native.t_sim,
                             O_index=native.O_index, w=native.w)
    jac_batch, _ = native._compute(copy=False)   # read-only, possibly sharing JAX's buffer: stored as is
    O = jac_batch.reshape(jac_batch.shape[0], native.w * native.p, native.n)
    return SlidingO(O=O, index=native._window_frame(O[0]).index, state_names=list(native.state_names),
                    t_sim=native.t_sim, O_index=native.O_index, w=native.w)


_BUILDERS = {
    'empirical': _Builder(_build_empirical, frozenset({'aux_list', 'eps', 'parallel_sliding', 'parallel_perturbation',
                                                       'simulator_factory', 'n_workers'})),
    'jax': _Builder(_build_jax, frozenset({'aux_list', 'batch_size'})),
}


STORAGE_MODES = ('observability', 'fisher_per_sensor', 'fisher')

# What each storage mode can answer after run(), for error messages
_NEEDS_O = "storage='observability'"


def _group_rows(index, groups):
    """Row numbers of each sensor group in a window's O."""
    sensor_level = np.asarray(index.get_level_values('sensor'), dtype=object)
    return [np.flatnonzero(np.isin(sensor_level, list(group))) for group in groups]


def _pack_window(O_k, group_rows, iu):
    """Unit-noise Fisher information O_g^T O_g of each sensor group in one window, packed symmetric:
    an array of shape (len(groups), n(n+1)/2) holding the upper triangles."""
    packed = np.empty((len(group_rows), len(iu[0])))
    for j, rows in enumerate(group_rows):
        O_g = O_k[rows]
        packed[j] = (O_g.T @ O_g)[iu]
    return packed


def _unpack_fisher(F_packed, n):
    """(n_windows, n(n+1)/2) packed upper triangles -> (n_windows, n, n) symmetric matrices."""
    iu = np.triu_indices(n)
    F = np.zeros((F_packed.shape[0], n, n))
    F[:, iu[0], iu[1]] = F_packed
    F[:, iu[1], iu[0]] = F_packed
    return F


def _is_matrix(R):
    return isinstance(R, pd.DataFrame) or (isinstance(R, np.ndarray) and R.ndim == 2)


def _is_jax_simulator(simulator):
    jax_module = sys.modules.get(f'{__package__}.jax_simulator')
    return jax_module is not None and isinstance(simulator, jax_module.JaxSimulator)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freeze(x):
    """Hashable version of a query argument for the result cache, or _NOCACHE if not practical."""
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        items = tuple(sorted((str(k), _freeze(v)) for k, v in x.items()))
        return _NOCACHE if any(v is _NOCACHE for _, v in items) else ('dict', items)
    if isinstance(x, (list, tuple)) or (isinstance(x, np.ndarray) and x.ndim <= 1):
        items = tuple(_freeze(v) for v in np.ravel(x).tolist()) if isinstance(x, np.ndarray) \
            else tuple(_freeze(v) for v in x)
        return _NOCACHE if any(v is _NOCACHE for v in items) else ('seq', items)
    return _NOCACHE


def _as_list(x, name):
    """Normalize a selection (None, a single name/int, or a sequence) to None or a list."""
    if x is None:
        return None
    if isinstance(x, (str, int, np.integer)):
        x = [x]
    x = [v.item() if isinstance(v, np.generic) else v for v in x]
    if len(x) == 0:
        raise ValueError(f'{name} must not be empty; use None to select all')
    duplicates = sorted({v for v in x if x.count(v) > 1}, key=str)
    if duplicates:
        raise ValueError(f'{name} has duplicate entries: {duplicates}')
    return x


def _to_plain(x):
    """Recursively convert numpy / tuple values to plain Python for yaml.safe_dump."""
    if isinstance(x, dict):
        return {str(k): _to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _to_plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def _is_plain(x):
    """True if x can be written with yaml.safe_dump after _to_plain."""
    try:
        yaml.safe_dump(_to_plain(x))
        return True
    except yaml.YAMLError:
        return False


def _reference(obj):
    """Record of a non-serializable setting: 'module:qualname' for callables, '<set>' otherwise."""
    if obj is None:
        return None
    qualname = getattr(obj, '__qualname__', None)
    if callable(obj) and qualname is not None:
        return f'{getattr(obj, "__module__", "?")}:{qualname}'
    return '<set>'


def _R_to_yaml(R):
    if R is None or isinstance(R, (int, float, np.number)):
        return None if R is None else float(R)
    if isinstance(R, dict):
        return {str(k): float(v) for k, v in R.items()}
    if isinstance(R, pd.DataFrame):
        return {'_matrix': _to_plain(R.values), '_index': [_to_plain(list(i)) for i in R.index]}
    R = np.asarray(R, dtype=float)
    if R.ndim == 2:
        return {'_matrix': _to_plain(R), '_index': None}
    return float(R.squeeze())


def _R_from_yaml(R):
    if isinstance(R, dict) and '_matrix' in R:
        matrix = np.asarray(R['_matrix'], dtype=float)
        if R.get('_index') is None:
            return matrix
        index = pd.MultiIndex.from_tuples([tuple(i) for i in R['_index']], names=['sensor', 'time_step'])
        return pd.DataFrame(matrix, index=index, columns=index)
    return R


def _number(x):
    """Numbers written by hand like 1e-8 are read by YAML as strings; convert them back."""
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    return x


def _coerce_numbers(settings):
    for key in ('lam', 'eps'):
        if key in settings:
            settings[key] = _number(settings[key])
    if isinstance(settings.get('w'), (str, float)):
        settings['w'] = int(float(settings['w']))
    R = settings.get('R')
    if isinstance(R, dict) and '_matrix' not in R:
        settings['R'] = {k: _number(v) for k, v in R.items()}
    elif isinstance(R, str):
        settings['R'] = _number(R)
    return settings


def _pybounds_version():
    try:
        from importlib.metadata import version
        return version('pybounds')
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ObservabilityAnalysis
# ---------------------------------------------------------------------------

class ObservabilityAnalysis:
    """Sliding-window observability analysis: configure, ``run()``, then query.

    Nothing is computed until ``run()`` is called, so settings can be changed first
    (``update_settings``). After ``run()``, the minimum error variance can be queried for
    any selection of states, sensors and time-steps without rebuilding O. Selecting states
    is conditional: the other states are treated as known (as in FisherObservability).

    :param simulator: a pybounds Simulator, a JaxSimulator, or a custom simulator object
    :param t_sim: time of every point of the trajectory, shape (N,)
    :param x_sim: state trajectory, (N, n) array or dict of state name -> (N,) array
    :param u_sim: input trajectory, (N, m) array or dict of input name -> (N,) array
    :param str method: how to build O: 'empirical' (finite differences) or 'jax' (autodiff).
        None picks 'jax' for a JaxSimulator and 'empirical' otherwise
    :param int w: window size in time-steps; None uses the full trajectory (one window)
    :param list aux_list: auxiliary data, one entry per time-step, passed to the simulator per window
    :param callable z_function: coordinate transform z = z_function(x) using sympy functions; each window's
        O is transformed at that window's initial state. States are then selected by the new names
    :param list z_state_names: names of the transformed states
    :param R: default measurement noise covariance for queries (scalar, dict per sensor, or matrix);
        None means identity
    :param float | str lam: default regularization for inverting F; 1/lam is the ceiling on the
        minimum error variance. 'limit' computes lam -> 0 symbolically
    :param str storage: what run() keeps (default 'observability'):
        'observability' keeps every window's O and can answer every query (selections of states, sensors
        and time_steps; any R including matrices; the observability matrices themselves).
        'fisher_per_sensor' keeps one unit-noise Fisher information matrix per sensor per window (packed
        symmetric, n(n+1)/2 numbers each) and discards O: it answers selections of states and sensors with a
        scalar or per-sensor (dict) R, but not time_steps selections or matrix R. It is smaller than O when
        (n+1)/2 < w, i.e. for long windows.
        'fisher' keeps one Fisher information matrix per window, summed over fisher_sensors with unit noise:
        the smallest, but only states can be selected and R must be a scalar.
        In the Fisher modes, z_function is applied to O before F is formed, and observability_matrix(k)
        recomputes that one window's O on demand. Results agree with 'observability' to rounding error
        (different summation order), which inversion amplifies by up to the condition number of F + lam*I
    :param list fisher_sensors: sensors summed into F with storage='fisher' (default: all)
    :param bool keep_source: keep the builder's native object (``source``) and its per-window trajectory
        data (``window_data``) after run(). Off by default because they hold extra copies of O (and, for
        'empirical', the perturbed simulations), several times the memory of O itself
    :param method_options: options for the chosen method, forwarded to its builder. Only options
        that are given are forwarded, so the builder's own defaults apply otherwise.
        'empirical': eps, parallel_sliding, parallel_perturbation, simulator_factory, n_workers.
        'jax': batch_size, to compute at most that many windows per batched call and cap JAX's working
        memory (set integrator/substeps on the JaxSimulator)

    Memory: after run() the observability matrices are held once, as one (n_windows, w*p, n) float
    array (8 * n_windows * w * p * n bytes). Queries build one window's data at a time.
    """

    _O_SETTINGS = ('method', 'w', 'aux_list', 'z_function', 'z_state_names', 'storage', 'fisher_sensors',
                   'keep_source')
    _QUERY_SETTINGS = ('R', 'lam')

    def __init__(self, simulator, t_sim, x_sim, u_sim, *, method=None, w=None, aux_list=None,
                 z_function=None, z_state_names=None, R=None, lam=DEFAULT_LAM, storage='observability',
                 fisher_sensors=None, keep_source=False, **method_options):
        self.simulator = simulator
        self._t_sim_in = t_sim
        self._x_sim_in = x_sim
        self._u_sim_in = u_sim
        self._external = False

        if method is None:
            method = 'jax' if _is_jax_simulator(simulator) else 'empirical'

        self._settings = {'method': method, 'w': w, 'aux_list': aux_list, 'z_function': z_function,
                          'z_state_names': z_state_names, 'storage': storage, 'fisher_sensors': fisher_sensors,
                          'keep_source': bool(keep_source), 'R': R, 'lam': lam}
        self._method_options = {}
        self._validate_storage(storage, fisher_sensors)
        self._validate_method(method, method_options, aux_list)
        self._method_options = dict(method_options)

        self._discard_results()

    # ------------------------------------------------------------------ construction from existing O

    @classmethod
    def from_sliding(cls, obj, *, R=None, lam=DEFAULT_LAM, storage='observability', fisher_sensors=None,
                     keep_source=False):
        """Wrap observability matrices that were already computed.

        :param obj: a SlidingO, or an object with O_df_sliding, t_sim and O_index attributes
            (e.g. SlidingEmpiricalObservabilityMatrix, JaxSlidingEmpiricalObservabilityMatrix).
            A list of DataFrames is copied into one array (the caller's list is not kept); a
            SlidingO(O=array, index=..., state_names=...) is used without a copy.
        :param str storage: what to keep, see the class docstring; with a Fisher mode, O is converted and
            not kept
        :param list fisher_sensors: sensors summed into F with storage='fisher'
        :param bool keep_source: keep obj (as ``source``) and its window_data
        """
        cls._validate_storage(storage, fisher_sensors)
        self = cls.__new__(cls)
        self.simulator = None
        self._t_sim_in = self._x_sim_in = self._u_sim_in = None
        self._external = True
        self._settings = {'method': 'external', 'w': None, 'aux_list': None, 'z_function': None,
                          'z_state_names': None, 'storage': storage, 'fisher_sensors': fisher_sensors,
                          'keep_source': bool(keep_source), 'R': R, 'lam': lam}
        self._method_options = {}
        self._discard_results()
        self._store(_from_sliding_object(obj))
        return self

    # ------------------------------------------------------------------ settings

    @property
    def method(self):
        return self._settings['method']

    @property
    def settings(self):
        """All current settings (O-building, query defaults and method options) as a new dict."""
        return {**self._settings, **self._method_options}

    def update_settings(self, **settings):
        """Change settings. Changing any O-building setting (method, w, aux_list, z_function,
        z_state_names or a method option) discards computed results, so call run() again.
        Setting a method option to None removes it, so the builder's default applies."""
        if not settings:
            return self
        new_settings = dict(self._settings)
        new_options = dict(self._method_options)
        for key, value in settings.items():
            if key in self._settings:
                new_settings[key] = value
            elif value is None:
                new_options.pop(key, None)
            else:
                new_options[key] = value

        o_changed = any(k not in self._QUERY_SETTINGS for k in settings)
        if o_changed and self._external:
            raise ValueError('this analysis wraps existing observability matrices (from_sliding); '
                             'only R and lam can be changed')
        if new_settings['method'] is None:
            new_settings['method'] = 'jax' if _is_jax_simulator(self.simulator) else 'empirical'
        if o_changed:
            self._validate_storage(new_settings['storage'], new_settings['fisher_sensors'])
            self._validate_method(new_settings['method'], new_options, new_settings['aux_list'])

        self._settings = new_settings
        self._method_options = new_options
        if o_changed:
            self._discard_results()
        else:
            self.clear_cache()
        return self

    # ------------------------------------------------------------------ settings files (YAML)

    _REFERENCE_SETTINGS = ('aux_list', 'z_function')

    def _settings_document(self):
        """All settings as plain data: hyperparameters, references to non-serializable settings,
        a record of the simulator, and metadata."""
        hyperparameters = {'method': self.method, 'w': self._settings['w'],
                           'z_state_names': self._settings['z_state_names'],
                           'storage': self._settings['storage'],
                           'fisher_sensors': self._settings['fisher_sensors'],
                           'keep_source': self._settings['keep_source'],
                           'R': _R_to_yaml(self._settings['R']), 'lam': self._settings['lam']}
        references = {k: _reference(self._settings[k]) for k in self._REFERENCE_SETTINGS}
        for key, value in self._method_options.items():
            if callable(value) or not _is_plain(value):
                references[key] = _reference(value)
            else:
                hyperparameters[key] = value
        return {'pybounds_version': _pybounds_version(),
                'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'settings': _to_plain(hyperparameters),
                'references': references,
                'simulator': self._simulator_record()}

    def _simulator_record(self):
        sim = self.simulator
        if sim is None:
            return None
        record = {'type': f'{type(sim).__module__}.{type(sim).__qualname__}'}
        for attr in ('dt', 'state_names', 'input_names', 'measurement_names', 'integrator', 'substeps',
                     'mpc_horizon', 'params_simulator'):
            if hasattr(sim, attr):
                value = _to_plain(getattr(sim, attr))
                if _is_plain(value):
                    record[attr] = value
        return record

    def save_settings(self, path):
        """Write all settings to a YAML file.

        Callables and other non-serializable settings (z_function, simulator_factory, aux_list) are
        recorded by reference only; pass them again when loading. The simulator is recorded for checking.
        """
        with open(path, 'w') as f:
            yaml.safe_dump(self._settings_document(), f, sort_keys=False)
        return path

    def load_settings(self, path):
        """Apply the settings in a YAML file written by save_settings (or a save_results sidecar).

        Method options not in the file are removed. Non-serializable settings are never loaded from
        the file: if the file references one that isn't set on this analysis, a warning says to pass
        it with update_settings. Differences from the recorded simulator are warned about.
        """
        with open(path) as f:
            document = yaml.safe_load(f)
        if 'settings' not in document and isinstance(document.get('analysis'), dict):
            document = document['analysis']   # a save_results sidecar
        settings = _coerce_numbers(dict(document.get('settings') or {}))
        references = document.get('references') or {}
        if 'R' in settings:
            settings['R'] = _R_from_yaml(settings['R'])

        if self._external:
            ignored = sorted(set(settings) - set(self._QUERY_SETTINGS))
            if ignored:
                warnings.warn(f'this analysis wraps existing observability matrices; ignoring {ignored}',
                              UserWarning, stacklevel=2)
            self.update_settings(**{k: v for k, v in settings.items() if k in self._QUERY_SETTINGS})
            return self

        # Replace the method options with the file's, keeping currently set non-serializable ones
        for key in self._method_options:
            if key not in settings and key not in references:
                settings[key] = None
        missing = [k for k, ref in references.items() if ref is not None and self.settings.get(k) is None]
        if missing:
            warnings.warn(f'{path} references {missing}, which cannot be loaded from a file; '
                          f'set them with update_settings(...)', UserWarning, stacklevel=2)

        recorded = document.get('simulator')
        current = self._simulator_record()
        if recorded and current:
            differs = sorted(k for k in set(recorded) | set(current)
                             if k != 'type' and recorded.get(k) != current.get(k))
            if recorded.get('type') != current.get('type'):
                differs.insert(0, 'type')
            if differs:
                warnings.warn(f'the simulator differs from the one recorded in {path}: {differs}',
                              UserWarning, stacklevel=2)

        self.update_settings(**settings)
        return self

    @staticmethod
    def _validate_storage(storage, fisher_sensors):
        if storage not in STORAGE_MODES:
            raise ValueError(f'unknown storage {storage!r}; valid storage modes: {list(STORAGE_MODES)}')
        if fisher_sensors is not None and storage != 'fisher':
            raise ValueError("fisher_sensors only applies to storage='fisher'")

    @staticmethod
    def _validate_method(method, method_options, aux_list):
        if method not in _BUILDERS:
            raise ValueError(f'unknown method {method!r}; valid methods: {sorted(_BUILDERS)}')
        given = set(method_options) | ({'aux_list'} if aux_list is not None else set())
        unsupported = given - _BUILDERS[method].options
        if unsupported:
            raise TypeError(f'method {method!r} does not accept: {sorted(unsupported)}; '
                            f'valid options: {sorted(_BUILDERS[method].options)}')

    # ------------------------------------------------------------------ computation

    @property
    def is_computed(self):
        return self._n_windows is not None

    @property
    def storage(self):
        return self._settings['storage']

    def run(self):
        """Build the observability matrix of every window with the current settings. Returns self."""
        if self._external:
            return self
        builder = _BUILDERS[self.method]
        options = dict(self._method_options)
        if self._settings['aux_list'] is not None:
            options['aux_list'] = self._settings['aux_list']
        result = builder.func(self.simulator, self._t_sim_in, self._x_sim_in, self._u_sim_in,
                              w=self._settings['w'], stream=not self._settings['keep_source'], **options)
        self._store(result)
        return self

    def _store(self, result):
        """Keep a builder result: the observability matrices as one array, or their Fisher information.

        Windows are consumed one at a time (applying the coordinate transform if one is set), so a
        streamed result never holds more than one window besides what is stored.
        """
        if isinstance(result, _WindowStream):
            n_windows, windows, materialized = result.n_windows, result.windows, None
        else:
            materialized = _window_array(result)
            O_all, index_all, names_all = materialized
            n_windows = O_all.shape[0]
            windows = ((O_all[k], index_all, names_all) for k in range(n_windows))
        if n_windows == 0:
            raise ValueError('the observability matrix builder returned no windows')
        O_index = np.arange(n_windows) if result.O_index is None else np.asarray(result.O_index)
        if not np.array_equal(O_index, np.arange(n_windows)):
            raise NotImplementedError('only windows starting at every time-step (O_index = 0, 1, 2, ...) are '
                                      'supported for time alignment')

        storage = self._settings['storage']
        z_function = self._settings['z_function']
        if z_function is not None and self._settings['z_state_names'] is None:
            warnings.warn('z_function is set without z_state_names, so the transformed states are named '
                          '0, 1, 2, ...', UserWarning, stacklevel=3)

        if storage == 'observability' and z_function is None and materialized is not None:
            O, index, state_names = materialized   # already one array: keep it without a copy
            dxdz_sliding = None
        else:
            O, index, state_names, dxdz_sliding = self._assemble(windows, n_windows, O_index, storage)

        self._index = index
        self._n_windows = n_windows
        if storage == 'observability':
            O = O.view()
            O.flags.writeable = False   # never modified; also protects an array passed in via SlidingO(O=...)
            self._O = O
        else:
            self._F = O
            self._O = None
        self._state_names = state_names
        self._dxdz_sliding = dxdz_sliding
        self._sensor_names = list(pd.unique(np.asarray(index.get_level_values('sensor'), dtype=object)))
        self._time_steps = sorted(int(k) for k in pd.unique(index.get_level_values('time_step')))
        self._t_sim = None if result.t_sim is None else np.ravel(np.asarray(result.t_sim))
        self._O_index = O_index
        self._w = int(result.w) if result.w is not None else self._time_steps[-1] + 1
        keep = self._settings['keep_source']
        self._source = result.source if keep else None
        self._window_data = result.window_data if keep else None
        self.clear_cache()

    def _assemble(self, windows, n_windows, O_index, storage):
        """Consume (O_k, index, state_names) windows into the stored array (O, or packed Fisher information)."""
        z_function = self._settings['z_function']
        out = index0 = names0 = state_names = None
        dxdz_sliding = [] if z_function is not None else None
        count = 0
        for k, (O_k, index, names) in enumerate(windows):
            if k == 0:
                index0, names0 = index, list(names)
                if list(index0.names) != ['sensor', 'time_step']:
                    raise ValueError(f"the row index names must be ['sensor', 'time_step'], got {list(index0.names)}")
                n = O_k.shape[1]
                if z_function is not None:
                    x0_list = self._trajectory_states(n)[O_index]
                if storage == 'observability':
                    out = np.empty((n_windows, O_k.shape[0], n))
                else:
                    sensor_names = list(pd.unique(np.asarray(index0.get_level_values('sensor'), dtype=object)))
                    if storage == 'fisher_per_sensor':
                        groups = [[s] for s in sensor_names]
                    else:
                        groups = [list(self._settings['fisher_sensors'] or sensor_names)]
                        unknown = [s for s in groups[0] if s not in sensor_names]
                        if unknown:
                            raise ValueError(f'unknown fisher_sensors {unknown}; available sensors: {sensor_names}')
                    self._F_groups = groups
                    group_rows = _group_rows(index0, groups)
                    iu = np.triu_indices(n)
                    out = np.empty((n_windows, len(groups), len(iu[0])))
            elif not index.equals(index0) or list(names) != names0:
                raise ValueError(f'window {k} has different rows or states than window 0; '
                                 'all windows must share the same rows and states')

            if z_function is not None:
                O_k, state_names, dxdz = self._transform_window(O_k, index0, names0, x0_list[k])
                dxdz_sliding.append(dxdz)
            if storage == 'observability':
                out[k] = O_k
            else:
                out[k] = _pack_window(O_k, group_rows, iu)
            count += 1
        if count != n_windows:
            raise ValueError(f'the builder produced {count} windows, expected {n_windows}')
        return out, index0, state_names if z_function is not None else names0, dxdz_sliding

    def _transform_window(self, O_k, index, state_names, x0):
        """One window in z coordinates at its initial state: (O_z, z state names, dx/dz)."""
        if self._dzdx_function is None:
            self._dzdx_function = _z_jacobian_function(self._settings['z_function'], O_k.shape[1])
        frame = pd.DataFrame(O_k, index=index, columns=state_names, copy=True)
        frame_z, dxdz = _transform_O_df(frame, x0, self._dzdx_function, self._settings['z_state_names'])
        return frame_z.to_numpy(dtype=float), list(frame_z.columns), dxdz

    def _trajectory_states(self, n):
        """x_sim as an (N, n) array, in the simulator's state order."""
        x_sim = self._x_sim_in
        if isinstance(x_sim, dict):
            x_sim = np.vstack(_ordered_values(x_sim, getattr(self.simulator, 'state_names', None), 'x_sim')).T
        x_sim = np.asarray(x_sim, dtype=float)
        return x_sim.reshape(x_sim.shape[0], n)

    def _discard_results(self):
        self._O = self._index = self._n_windows = None
        self._F = self._F_groups = None
        self._dzdx_function = None
        self._dxdz_sliding = None
        self._state_names = self._sensor_names = self._time_steps = None
        self._t_sim = self._O_index = self._w = None
        self._source = self._window_data = None
        self._cache = {}

    def clear_cache(self):
        """Forget cached minimum error variance results (computed O is kept)."""
        self._cache = {}

    def _require_computed(self):
        if self._n_windows is None:
            raise RuntimeError('observability matrices have not been computed with the current settings; '
                               'call run() first')

    def _frames(self, what='this query'):
        """Per-window DataFrames, built on access (storage='observability' only)."""
        if self._O is None:
            raise ValueError(f"{what} needs the observability matrices, which storage={self.storage!r} does not "
                             f"keep; use {_NEEDS_O}")
        return _WindowFrames(self._O, self._index, self._state_names)

    # ------------------------------------------------------------------ results (after run)

    def _computed(self, value):
        self._require_computed()
        return value

    @property
    def w(self):
        return self._computed(self._w)

    @property
    def n_windows(self):
        return self._computed(self._n_windows)

    @property
    def state_names(self):
        """State names of O (the transformed names when z_function is set)."""
        return list(self._computed(self._state_names))

    @property
    def sensor_names(self):
        return list(self._computed(self._sensor_names))

    @property
    def time_steps(self):
        return list(self._computed(self._time_steps))

    @property
    def t_sim(self):
        t_sim = self._computed(self._t_sim)
        return None if t_sim is None else t_sim.copy()

    @property
    def O_index(self):
        return np.array(self._computed(self._O_index))

    @property
    def O_time(self):
        t_sim = self.t_sim
        return None if t_sim is None else t_sim[self.O_index]

    @property
    def O_df_sliding(self):
        """New DataFrames of every window's observability matrix (transformed if z_function is set).

        This builds all windows at once, a full copy of O; use observability_matrix(k) for one window.
        """
        self._require_computed()
        return list(self._frames('O_df_sliding'))

    @property
    def window_data(self):
        """The builder's per-window trajectory data (only with keep_source=True)."""
        self._require_computed()
        if not self._settings['keep_source']:
            raise RuntimeError('window_data is not kept by default; set keep_source=True and call run()')
        return self._window_data

    @property
    def dxdz_sliding(self):
        """dx/dz of the coordinate transform at each window's initial state (None without z_function)."""
        self._require_computed()
        return None if self._dxdz_sliding is None else [d.copy() for d in self._dxdz_sliding]

    @property
    def source(self):
        """The builder's native object, e.g. the SlidingEmpiricalObservabilityMatrix (only with
        keep_source=True). O is untransformed there."""
        self._require_computed()
        if not self._settings['keep_source']:
            raise RuntimeError('source is not kept by default; set keep_source=True and call run()')
        return self._source

    # ------------------------------------------------------------------ queries

    def _select(self, states, sensors, time_steps):
        self._require_computed()
        states = _as_list(states, 'states')
        sensors = _as_list(sensors, 'sensors')
        time_steps = _as_list(time_steps, 'time_steps')

        if states is not None:
            unknown = [s for s in states if s not in self._state_names]
            if unknown:
                note = ' (z_function is set, so states use the transformed names)' \
                    if self._settings['z_function'] is not None else ''
                raise ValueError(f'unknown states {unknown}; available states: {self._state_names}{note}')
        if sensors is not None:
            unknown = [s for s in sensors if s not in self._sensor_names]
            if unknown:
                raise ValueError(f'unknown sensors {unknown}; available sensors: {self._sensor_names}')
        if time_steps is not None:
            unknown = [k for k in time_steps if k not in self._time_steps]
            if unknown:
                raise ValueError(f'unknown time_steps {unknown}; available time_steps: '
                                 f'{self._time_steps[0]}..{self._time_steps[-1]}')
            time_steps = [int(k) for k in time_steps]
        return states, sensors, time_steps

    def _resolve(self, R, lam, sensors):
        R = self._settings['R'] if R is _UNSET else R
        lam = self._settings['lam'] if lam is _UNSET else lam
        if isinstance(R, dict):
            missing = [s for s in (sensors or self._sensor_names) if s not in R]
            if missing:
                raise ValueError(f'R has no noise level for sensors {missing}')
        return R, lam

    def fisher(self, states=None, sensors=None, time_steps=None, *, R=_UNSET, lam=_UNSET, force_R_scalar=False):
        """SlidingFisherObservability for the selection, with F, F_inv and R of every window (not cached).

        Selections default to all states, sensors and time-steps; R and lam default to the settings.
        """
        states, sensors, time_steps = self._select(states, sensors, time_steps)
        R, lam = self._resolve(R, lam, sensors)
        if self._O is None:
            raise ValueError(f"fisher() builds per-window FisherObservability objects from the observability "
                             f"matrices, which storage={self.storage!r} does not keep; use fisher_information() "
                             f"for each window's F, or {_NEEDS_O}")
        return self._sliding_fisher(states, sensors, time_steps, R, lam, force_R_scalar, keep_windows=True)

    def _sliding_fisher(self, states, sensors, time_steps, R, lam, force_R_scalar, keep_windows):
        return SlidingFisherObservability(self._frames(), R=R, lam=lam, time=self._t_sim,
                                          states=states, sensors=sensors,
                                          time_steps=None if time_steps is None else np.array(time_steps),
                                          w=None, force_R_scalar=force_R_scalar, keep_windows=keep_windows)

    def min_error_variance(self, states=None, sensors=None, time_steps=None, *,
                           R=_UNSET, lam=_UNSET, force_R_scalar=False):
        """Minimum error variance of the selected states over time.

        Returns a DataFrame with 'time', 'time_initial' and one column per selected state, aligned with
        the trajectory: each window's result is placed at its center time-step (w // 2). The shift uses
        the full window size even when a subset of time_steps is selected. Results are cached.
        """
        states_l, sensors_l, time_steps_l = self._select(states, sensors, time_steps)
        R_r, lam_r = self._resolve(R, lam, sensors_l)
        key = tuple(_freeze(v) for v in (states_l, sensors_l, time_steps_l, R_r, lam_r, bool(force_R_scalar)))
        cacheable = not any(k is _NOCACHE for k in key)
        if cacheable and key in self._cache:
            return self._cache[key].copy()

        if self._O is not None:
            fast = self._fast_windows(states_l, sensors_l, time_steps_l, R_r, lam_r, force_R_scalar)
            if fast is not None:
                ev = fast.error_variance()
            else:   # matrix R, or a selection the fast path doesn't reproduce exactly
                ev = self._sliding_fisher(states_l, sensors_l, time_steps_l, R_r, lam_r, force_R_scalar,
                                          keep_windows=False).get_minimum_error_variance()
        else:
            ev = self._stored_fisher_error_variance(states_l, sensors_l, time_steps_l, R_r, lam_r, force_R_scalar)
        if cacheable:
            self._cache[key] = ev.copy()
        return ev

    def fisher_information(self, states=None, sensors=None, *, R=_UNSET, force_R_scalar=False):
        """Fisher information F = O^T R^-1 O of every window for a selection, as an array of shape
        (n_windows, n_states, n_states) with states in the order given (default: state_names).

        No regularization is added, so it can be combined with, e.g., a per-state lam. Works with every
        storage mode. With storage='observability' each window equals fisher(...).FO[k].F exactly.
        """
        states, sensors, _ = self._select(states, sensors, None)
        R, _ = self._resolve(R, _UNSET, sensors)
        if self._O is None:
            return self._stored_fisher(states, sensors, None, R, force_R_scalar)
        fast = self._fast_windows(states, sensors, None, R, DEFAULT_LAM, force_R_scalar)
        if fast is not None:
            return fast.fisher_information()
        frames = self._frames()
        return np.stack([FisherObservability(frames[k], R=R, force_R_scalar=force_R_scalar, states=states,
                                             sensors=sensors).F.to_numpy() for k in range(self._n_windows)])

    def _fast_windows(self, states, sensors, time_steps, R, lam, force_R_scalar):
        """_FastWindows for a query with storage='observability', or None to use FisherObservability per window.

        All windows share one row index, so the selected rows (in FisherObservability's order), columns and
        noise weights are computed once. Window 0 is computed both ways; any difference falls back.
        """
        if _is_matrix(R):
            return None
        frames = self._frames()
        reference = FisherObservability(frames[0], R=R, lam=lam, force_R_scalar=force_R_scalar, states=states,
                                        sensors=sensors, time_steps=None if time_steps is None else np.array(time_steps))
        sensor_level = np.asarray(self._index.get_level_values('sensor'), dtype=object)
        step_level = np.asarray(self._index.get_level_values('time_step'))
        mask = np.ones(len(self._index), dtype=bool)
        if sensors is not None:
            mask &= np.isin(sensor_level, list(sensors))
        if time_steps is not None:
            mask &= np.isin(step_level, list(time_steps))
        rows = np.flatnonzero(mask)
        rows = rows[np.lexsort((sensor_level[rows], step_level[rows]))]   # sort_values(['time_step', 'sensor'])
        cols = np.arange(len(self._state_names)) if states is None else \
            np.array([self._state_names.index(x) for x in states])
        if not (self._index[rows].equals(reference.O.index)
                and list(reference.O.columns) == [self._state_names[c] for c in cols]):
            return None
        fast = _FastWindows(self, rows, cols, reference, lam)
        if not (np.array_equal(fast.fisher(0), reference.F.to_numpy())
                and np.array_equal(fast.error_variance_row(0), reference.error_variance.to_numpy()[0])):
            return None
        return fast



    def _stored_fisher(self, states, sensors, time_steps, R, force_R_scalar):
        """Selected F of every window from the stored per-sensor / summed Fisher information."""
        if time_steps is not None:
            raise ValueError(f"selecting time_steps needs the observability matrices, which storage={self.storage!r} "
                             f"does not keep; use {_NEEDS_O}")
        if _is_matrix(R):
            raise ValueError(f"a matrix R needs the observability matrices, which storage={self.storage!r} does not "
                             f"keep; use a scalar or per-sensor dict R, or {_NEEDS_O}")
        if force_R_scalar and not np.isscalar(R):
            raise Exception('R must be a scalar')
        if R is None:
            warnings.warn('R not set, defaulting to identity matrix', stacklevel=3)

        if self.storage == 'fisher':
            summed = self._F_groups[0]
            if sensors is not None and set(sensors) != set(summed):
                raise ValueError(f"storage='fisher' keeps F summed over the sensors {summed}, so sensors cannot be "
                                 f"selected; use storage='fisher_per_sensor' or {_NEEDS_O}")
            if isinstance(R, dict):
                raise ValueError("storage='fisher' keeps F for unit noise summed over sensors, so R must be a scalar; "
                                 f"use storage='fisher_per_sensor' or {_NEEDS_O} for a per-sensor R")
            group_index, weights = [0], np.array([1.0 if R is None else 1 / float(R)])
        else:
            selected = sensors or self._sensor_names
            group_index = [self._sensor_names.index(s) for s in selected]
            weights = np.array([1.0 if R is None else 1 / float(R[s] if isinstance(R, dict) else R)
                                for s in selected])

        packed = np.zeros((self._F.shape[0], self._F.shape[2]))
        for j, weight in zip(group_index, weights):   # one sensor at a time: no copy of the stored F
            packed += weight * self._F[:, j, :]
        F = _unpack_fisher(packed, len(self._state_names))
        if states is not None:
            idx = [self._state_names.index(x) for x in states]
            F = F[:, idx][:, :, idx]
        return F

    def _stored_fisher_error_variance(self, states, sensors, time_steps, R, lam, force_R_scalar):
        """min_error_variance from stored Fisher information, aligned like SlidingFisherObservability."""
        F = self._stored_fisher(states, sensors, time_steps, R, force_R_scalar)
        names = states or self._state_names
        EV = pd.DataFrame(np.array([np.diag(_fisher_inverse(F_k, lam)) for F_k in F]), columns=names)
        n_window = F.shape[0]
        if self._t_sim is None:
            time, dt = np.arange(0, n_window, step=1), 1
        else:
            time = np.array(self._t_sim)
            dt = np.mean(np.diff(time)) if len(time) > 1 else 0.0
        EV.insert(0, 'time_initial', time[:n_window])
        shift_index = self._w // 2
        return _align_error_variance(EV, time, shift_index, shift_index * dt,
                                     aligned=n_window > 1 or self._t_sim is not None)[1]

    def observability_matrix(self, window=0, states=None, sensors=None, time_steps=None):
        """Copy of one window's observability matrix, optionally restricted to a selection
        (rows then ordered by time_step, sensor, as used for the Fisher information).

        With a Fisher storage mode, O is not stored and this window is recomputed from the simulator.
        """
        states, sensors, time_steps = self._select(states, sensors, time_steps)
        O = self._frames()[window] if self._O is not None else self._recompute_window(window)
        if states is None and sensors is None and time_steps is None:
            return O
        return O.loc[(sensors or self._sensor_names, time_steps or self._time_steps),
                     states or self._state_names].sort_values(['time_step', 'sensor'])

    def _recompute_window(self, window):
        """Rebuild one window's observability matrix (for Fisher storage modes, where O is not kept)."""
        if self._external:
            raise ValueError(f"storage={self.storage!r} does not keep the observability matrices, and this analysis "
                             f"wraps precomputed data (from_sliding), so they cannot be recomputed; use {_NEEDS_O}")
        k = range(self._n_windows)[window]   # IndexError when out of range; supports negative indices
        rows = slice(k, k + self._w)

        def cut(data):
            if isinstance(data, dict):
                return {name: np.asarray(v)[rows] for name, v in data.items()}
            return np.asarray(data)[rows]

        options = {key: v for key, v in self._method_options.items()
                   if key not in ('parallel_sliding', 'simulator_factory', 'n_workers')}
        if self._settings['aux_list'] is not None:
            options['aux_list'] = list(self._settings['aux_list'])[rows]
        result = _BUILDERS[self.method].func(self.simulator, np.ravel(np.asarray(self._t_sim_in))[rows],
                                             cut(self._x_sim_in), cut(self._u_sim_in), w=self._w, **options)
        O, index, state_names = _window_array(result)
        O_k = O[0]
        if self._settings['z_function'] is not None:
            x0 = self._trajectory_states(O_k.shape[1])[k]
            O_k, state_names, _ = self._transform_window(O_k, index, state_names, x0)
        return pd.DataFrame(O_k, index=index, columns=state_names, copy=True)

    def plot_observability_matrix(self, window=0, states=None, sensors=None, time_steps=None, *,
                                  state_names=None, sensor_names=None, **plot_kwargs):
        """Plot one window's observability matrix. Returns the ObservabilityMatrixImage (with fig, ax, cbar).

        state_names / sensor_names set the axis labels; other keyword arguments go to
        ObservabilityMatrixImage.plot (vmax_percentile, vmin_ratio, vmax_override, cmap, grid, scale, dpi, ax).
        """
        O = self.observability_matrix(window, states=states, sensors=sensors, time_steps=time_steps)
        image = ObservabilityMatrixImage(O, state_names=state_names, sensor_names=sensor_names)
        image.plot(**plot_kwargs)
        return image

    # ------------------------------------------------------------------ saving results

    def save_results(self, directory, states=None, sensors=None, time_steps=None, *, R=_UNSET, lam=_UNSET,
                     force_R_scalar=False, include_observability_matrices=False, overwrite=False):
        """Save the minimum error variance for a selection, with a YAML sidecar, into a directory.

        Files written (the directory is created if needed):
          - min_error_variance.csv: the time series returned by min_error_variance(...)
          - min_error_variance.yaml: the selection (states, sensors, time_steps, R, lam, force_R_scalar),
            the full lists of states, sensors and time-steps, and all analysis settings (loadable with
            load_settings)
          - observability_matrices.npz (if include_observability_matrices): 'O' with shape
            (n_windows, w*p, n) plus 'state_names', 'sensor' and 'time_step' labels, 'O_index', and
            't_sim' / 'window_time_initial' when the trajectory time is known. O is in the transformed
            coordinates when z_function is set.

        :param bool overwrite: replace existing files instead of raising FileExistsError
        :return dict: path of each file written, keyed by 'min_error_variance', 'sidecar' and
            'observability_matrices'
        """
        states_l, sensors_l, time_steps_l = self._select(states, sensors, time_steps)
        R_r, lam_r = self._resolve(R, lam, sensors_l)
        if include_observability_matrices and self._O is None:
            raise ValueError(f"include_observability_matrices needs the observability matrices, which "
                             f"storage={self.storage!r} does not keep; use {_NEEDS_O}")
        ev = self.min_error_variance(states_l, sensors_l, time_steps_l, R=R_r, lam=lam_r,
                                     force_R_scalar=force_R_scalar)

        directory = Path(directory)
        files = {'min_error_variance': directory / 'min_error_variance.csv',
                 'sidecar': directory / 'min_error_variance.yaml'}
        if include_observability_matrices:
            files['observability_matrices'] = directory / 'observability_matrices.npz'
        existing = [str(f) for f in files.values() if f.exists()]
        if existing and not overwrite:
            raise FileExistsError(f'would overwrite {existing}; pass overwrite=True to replace them')
        directory.mkdir(parents=True, exist_ok=True)

        ev.to_csv(files['min_error_variance'], index=False)
        if include_observability_matrices:
            self._save_observability_matrices(files['observability_matrices'])

        sidecar = {
            'pybounds_version': _pybounds_version(),
            'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'files': {k: f.name for k, f in files.items() if k != 'sidecar'},
            'columns': list(ev.columns),
            'selection': {'states': states_l or self._state_names,
                          'sensors': sensors_l or self._sensor_names,
                          'time_steps': time_steps_l or self._time_steps,
                          'R': _R_to_yaml(R_r), 'lam': lam_r, 'force_R_scalar': bool(force_R_scalar)},
            'all_states': self._state_names,
            'all_sensors': self._sensor_names,
            'all_time_steps': self._time_steps,
            'transformed_coordinates': self._settings['z_function'] is not None,
            'w': self._w,
            'n_windows': self._n_windows,
            'storage': self.storage,
            'time_alignment': 'each window is stamped at its center time-step, time_initial + (w // 2) * dt',
            'analysis': self._settings_document(),
        }
        with open(files['sidecar'], 'w') as f:
            yaml.safe_dump(_to_plain(sidecar), f, sort_keys=False)
        return {k: str(f) for k, f in files.items()}

    def _save_observability_matrices(self, path):
        arrays = {'O': self._O,
                  'state_names': np.array(self._state_names, dtype=str),
                  'sensor': np.array(self._index.get_level_values('sensor'), dtype=str),
                  'time_step': np.array(self._index.get_level_values('time_step'), dtype=int),
                  'O_index': np.asarray(self._O_index, dtype=int)}
        if self._t_sim is not None:
            arrays['t_sim'] = np.asarray(self._t_sim, dtype=float)
            arrays['window_time_initial'] = arrays['t_sim'][arrays['O_index']]
        np.savez(path, **arrays)

    def __repr__(self):
        status = f'computed, {self._n_windows} windows' if self.is_computed else 'not computed'
        if self.storage != 'observability':
            status += f', storage={self.storage!r}'
        return f'ObservabilityAnalysis(method={self.method!r}, w={self._settings["w"]!r}, {status})'
