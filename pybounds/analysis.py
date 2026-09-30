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
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import yaml

from .observability import (DEFAULT_LAM, SlidingEmpiricalObservabilityMatrix, SlidingFisherObservability,
                            ObservabilityMatrixImage, _ordered_values, _transform_O_df, _z_jacobian_function)


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


class _Builder(NamedTuple):
    func: Callable          # func(simulator, t_sim, x_sim, u_sim, *, w, **options) -> SlidingO
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


def _build_empirical(simulator, t_sim, x_sim, u_sim, *, w, **options):
    """Finite-difference O from a CasADi/do_mpc (or custom) simulator."""
    return _from_native(SlidingEmpiricalObservabilityMatrix(simulator, t_sim, x_sim, u_sim, w=w, **options))


def _build_jax(simulator, t_sim, x_sim, u_sim, *, w, **options):
    """Exact (autodiff) O from a JaxSimulator."""
    try:
        from .jax_simulator import JaxSlidingEmpiricalObservabilityMatrix
    except ImportError:
        raise ImportError("JAX is not installed. Install it with: pip install jax[cpu]") from None
    return _from_native(JaxSlidingEmpiricalObservabilityMatrix(simulator, t_sim, x_sim, u_sim, w=w, **options))


_BUILDERS = {
    'empirical': _Builder(_build_empirical, frozenset({'aux_list', 'eps', 'parallel_sliding', 'parallel_perturbation',
                                                       'simulator_factory', 'n_workers'})),
    'jax': _Builder(_build_jax, frozenset({'aux_list'})),
}


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
    :param bool keep_source: keep the builder's native object (``source``) and its per-window trajectory
        data (``window_data``) after run(). Off by default because they hold extra copies of O (and, for
        'empirical', the perturbed simulations), several times the memory of O itself
    :param method_options: options for the chosen method, forwarded to its builder. Only options
        that are given are forwarded, so the builder's own defaults apply otherwise.
        'empirical': eps, parallel_sliding, parallel_perturbation, simulator_factory, n_workers.
        'jax': none (set integrator/substeps on the JaxSimulator)

    Memory: after run() the observability matrices are held once, as one (n_windows, w*p, n) float
    array (8 * n_windows * w * p * n bytes). Queries build one window's data at a time.
    """

    _O_SETTINGS = ('method', 'w', 'aux_list', 'z_function', 'z_state_names', 'keep_source')
    _QUERY_SETTINGS = ('R', 'lam')

    def __init__(self, simulator, t_sim, x_sim, u_sim, *, method=None, w=None, aux_list=None,
                 z_function=None, z_state_names=None, R=None, lam=DEFAULT_LAM, keep_source=False,
                 **method_options):
        self.simulator = simulator
        self._t_sim_in = t_sim
        self._x_sim_in = x_sim
        self._u_sim_in = u_sim
        self._external = False

        if method is None:
            method = 'jax' if _is_jax_simulator(simulator) else 'empirical'

        self._settings = {'method': method, 'w': w, 'aux_list': aux_list, 'z_function': z_function,
                          'z_state_names': z_state_names, 'keep_source': bool(keep_source), 'R': R, 'lam': lam}
        self._method_options = {}
        self._validate_method(method, method_options, aux_list)
        self._method_options = dict(method_options)

        self._discard_results()

    # ------------------------------------------------------------------ construction from existing O

    @classmethod
    def from_sliding(cls, obj, *, R=None, lam=DEFAULT_LAM, keep_source=False):
        """Wrap observability matrices that were already computed.

        :param obj: a SlidingO, or an object with O_df_sliding, t_sim and O_index attributes
            (e.g. SlidingEmpiricalObservabilityMatrix, JaxSlidingEmpiricalObservabilityMatrix).
            A list of DataFrames is copied into one array (the caller's list is not kept); a
            SlidingO(O=array, index=..., state_names=...) is used without a copy.
        :param bool keep_source: keep obj (as ``source``) and its window_data
        """
        self = cls.__new__(cls)
        self.simulator = None
        self._t_sim_in = self._x_sim_in = self._u_sim_in = None
        self._external = True
        self._settings = {'method': 'external', 'w': None, 'aux_list': None, 'z_function': None,
                          'z_state_names': None, 'keep_source': bool(keep_source), 'R': R, 'lam': lam}
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
        return self._O is not None

    def run(self):
        """Build the observability matrix of every window with the current settings. Returns self."""
        if self._external:
            return self
        builder = _BUILDERS[self.method]
        options = dict(self._method_options)
        if self._settings['aux_list'] is not None:
            options['aux_list'] = self._settings['aux_list']
        result = builder.func(self.simulator, self._t_sim_in, self._x_sim_in, self._u_sim_in,
                              w=self._settings['w'], **options)
        self._store(result)
        return self

    def _store(self, result):
        """Keep a builder result as one array, applying the coordinate transform if one is set."""
        O, index, state_names = _window_array(result)
        n_windows = O.shape[0]
        O_index = np.arange(n_windows) if result.O_index is None else np.asarray(result.O_index)
        if n_windows == 0:
            raise ValueError('the observability matrix builder returned no windows')
        if not np.array_equal(O_index, np.arange(n_windows)):
            raise NotImplementedError('only windows starting at every time-step (O_index = 0, 1, 2, ...) are '
                                      'supported for time alignment')

        dxdz_sliding = None
        z_function = self._settings['z_function']
        if z_function is not None:
            if self._settings['z_state_names'] is None:
                warnings.warn('z_function is set without z_state_names, so the transformed states are named '
                              '0, 1, 2, ...', UserWarning, stacklevel=3)
            O, state_names, dxdz_sliding = self._transform(O, index, state_names, O_index)

        O = O.view()
        O.flags.writeable = False   # never modified; also protects an array passed in via SlidingO(O=...)
        self._O = O
        self._index = index
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

    def _transform(self, O, index, state_names, O_index):
        """Transform every window to z coordinates at its initial state, one window at a time."""
        x0_list = self._trajectory_states(O.shape[2])[O_index]
        dzdx_function = _z_jacobian_function(self._settings['z_function'], O.shape[2])
        O_z = np.empty_like(O)
        dxdz_sliding = []
        z_names = None
        for k in range(O.shape[0]):
            frame = pd.DataFrame(O[k], index=index, columns=state_names, copy=True)
            frame_z, dxdz = _transform_O_df(frame, x0_list[k], dzdx_function, self._settings['z_state_names'])
            O_z[k] = frame_z.to_numpy(dtype=float)
            dxdz_sliding.append(dxdz)
            z_names = list(frame_z.columns)
        return O_z, z_names, dxdz_sliding

    def _trajectory_states(self, n):
        """x_sim as an (N, n) array, in the simulator's state order."""
        x_sim = self._x_sim_in
        if isinstance(x_sim, dict):
            x_sim = np.vstack(_ordered_values(x_sim, getattr(self.simulator, 'state_names', None), 'x_sim')).T
        x_sim = np.asarray(x_sim, dtype=float)
        return x_sim.reshape(x_sim.shape[0], n)

    def _discard_results(self):
        self._O = self._index = None
        self._dxdz_sliding = None
        self._state_names = self._sensor_names = self._time_steps = None
        self._t_sim = self._O_index = self._w = None
        self._source = self._window_data = None
        self._cache = {}

    def clear_cache(self):
        """Forget cached minimum error variance results (computed O is kept)."""
        self._cache = {}

    def _require_computed(self):
        if self._O is None:
            raise RuntimeError('observability matrices have not been computed with the current settings; '
                               'call run() first')

    def _frames(self):
        """Per-window DataFrames, built on access."""
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
        return self._computed(self._O).shape[0]

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
        return list(self._frames())

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

        ev = self._sliding_fisher(states_l, sensors_l, time_steps_l, R_r, lam_r, force_R_scalar,
                                  keep_windows=False).get_minimum_error_variance()
        if cacheable:
            self._cache[key] = ev.copy()
        return ev

    def observability_matrix(self, window=0, states=None, sensors=None, time_steps=None):
        """Copy of one window's observability matrix, optionally restricted to a selection
        (rows then ordered by time_step, sensor, as used for the Fisher information)."""
        states, sensors, time_steps = self._select(states, sensors, time_steps)
        O = self._frames()[window]
        if states is None and sensors is None and time_steps is None:
            return O
        return O.loc[(sensors or self._sensor_names, time_steps or self._time_steps),
                     states or self._state_names].sort_values(['time_step', 'sensor'])

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
            'n_windows': self._O.shape[0],
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
        status = f'computed, {self._O.shape[0]} windows' if self.is_computed else 'not computed'
        return f'ObservabilityAnalysis(method={self.method!r}, w={self._settings["w"]!r}, {status})'
