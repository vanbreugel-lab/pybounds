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
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import yaml

from .observability import (DEFAULT_LAM, SlidingEmpiricalObservabilityMatrix, SlidingFisherObservability,
                            ObservabilityMatrixImage, _ordered_values, _transform_O_df_list)


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

    :param list O_df_sliding: one pd.DataFrame per window with index names ('sensor', 'time_step')
        and one column per state, in the original (untransformed) coordinates
    :param np.ndarray | None t_sim: time of every point of the trajectory, shape (N,)
    :param np.ndarray O_index: index into t_sim at which each window starts
    :param int w: window size in time-steps
    :param dict | None window_data: optional per-window trajectory data
    :param source: the builder's native object, for advanced use
    """
    O_df_sliding: list
    t_sim: np.ndarray
    O_index: np.ndarray
    w: int
    window_data: dict = None
    source: object = None


class _Builder(NamedTuple):
    func: Callable          # func(simulator, t_sim, x_sim, u_sim, *, w, **options) -> SlidingO
    options: frozenset      # option names the builder accepts


def _from_sliding_object(obj):
    """SlidingO from any object exposing O_df_sliding, t_sim, O_index (and optionally w, window_data)."""
    if isinstance(obj, SlidingO):
        return obj
    O_df_sliding = list(obj.O_df_sliding)
    w = getattr(obj, 'w', None)
    if w is None:
        w = int(np.max(O_df_sliding[0].index.get_level_values('time_step'))) + 1
    t_sim = getattr(obj, 't_sim', None)
    O_index = getattr(obj, 'O_index', None)
    return SlidingO(O_df_sliding=O_df_sliding,
                    t_sim=None if t_sim is None else np.ravel(np.asarray(t_sim)),
                    O_index=np.arange(len(O_df_sliding)) if O_index is None else np.asarray(O_index),
                    w=int(w), window_data=getattr(obj, 'window_data', None), source=obj)


def _build_empirical(simulator, t_sim, x_sim, u_sim, *, w, **options):
    """Finite-difference O from a CasADi/do_mpc (or custom) simulator."""
    return _from_sliding_object(SlidingEmpiricalObservabilityMatrix(simulator, t_sim, x_sim, u_sim, w=w, **options))


def _build_jax(simulator, t_sim, x_sim, u_sim, *, w, **options):
    """Exact (autodiff) O from a JaxSimulator."""
    try:
        from .jax_simulator import JaxSlidingEmpiricalObservabilityMatrix
    except ImportError:
        raise ImportError("JAX is not installed. Install it with: pip install jax[cpu]") from None
    return _from_sliding_object(JaxSlidingEmpiricalObservabilityMatrix(simulator, t_sim, x_sim, u_sim, w=w, **options))


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
    :param method_options: options for the chosen method, forwarded to its builder. Only options
        that are given are forwarded, so the builder's own defaults apply otherwise.
        'empirical': eps, parallel_sliding, parallel_perturbation, simulator_factory, n_workers.
        'jax': none (set integrator/substeps on the JaxSimulator)
    """

    _O_SETTINGS = ('method', 'w', 'aux_list', 'z_function', 'z_state_names')
    _QUERY_SETTINGS = ('R', 'lam')

    def __init__(self, simulator, t_sim, x_sim, u_sim, *, method=None, w=None, aux_list=None,
                 z_function=None, z_state_names=None, R=None, lam=DEFAULT_LAM, **method_options):
        self.simulator = simulator
        self._t_sim_in = t_sim
        self._x_sim_in = x_sim
        self._u_sim_in = u_sim
        self._external = False

        if method is None:
            method = 'jax' if _is_jax_simulator(simulator) else 'empirical'

        self._settings = {'method': method, 'w': w, 'aux_list': aux_list, 'z_function': z_function,
                          'z_state_names': z_state_names, 'R': R, 'lam': lam}
        self._method_options = {}
        self._validate_method(method, method_options, aux_list)
        self._method_options = dict(method_options)

        self._discard_results()

    # ------------------------------------------------------------------ construction from existing O

    @classmethod
    def from_sliding(cls, obj, *, R=None, lam=DEFAULT_LAM):
        """Wrap observability matrices that were already computed.

        :param obj: a SlidingO, or an object with O_df_sliding, t_sim and O_index attributes
            (e.g. SlidingEmpiricalObservabilityMatrix, JaxSlidingEmpiricalObservabilityMatrix)
        """
        self = cls.__new__(cls)
        self.simulator = None
        self._t_sim_in = self._x_sim_in = self._u_sim_in = None
        self._external = True
        self._settings = {'method': 'external', 'w': None, 'aux_list': None, 'z_function': None,
                          'z_state_names': None, 'R': R, 'lam': lam}
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
        return self._result is not None

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
        """Keep a builder result, applying the coordinate transform if one is set."""
        n_windows = len(result.O_df_sliding)
        if n_windows == 0:
            raise ValueError('the observability matrix builder returned no windows')
        if not np.array_equal(np.asarray(result.O_index), np.arange(n_windows)):
            raise NotImplementedError('only windows starting at every time-step (O_index = 0, 1, 2, ...) are '
                                      'supported for time alignment')

        O_df_sliding = [O.copy() for O in result.O_df_sliding]
        dxdz_sliding = None
        z_function = self._settings['z_function']
        if z_function is not None:
            if self._settings['z_state_names'] is None:
                warnings.warn('z_function is set without z_state_names, so the transformed states keep the '
                              'original state names', UserWarning, stacklevel=3)
            x0_list = self._trajectory_states(O_df_sliding[0].shape[1])[np.asarray(result.O_index)]
            O_df_sliding, dxdz_sliding = _transform_O_df_list(O_df_sliding, x0_list, z_function,
                                                              self._settings['z_state_names'], return_dxdz=True)

        self._result = result
        self._O_df_sliding = O_df_sliding
        self._dxdz_sliding = dxdz_sliding
        self._state_names = list(O_df_sliding[0].columns)
        index = O_df_sliding[0].index
        self._sensor_names = list(pd.unique(np.asarray(index.get_level_values('sensor'), dtype=object)))
        self._time_steps = sorted(int(k) for k in pd.unique(index.get_level_values('time_step')))
        self.clear_cache()

    def _trajectory_states(self, n):
        """x_sim as an (N, n) array, in the simulator's state order."""
        x_sim = self._x_sim_in
        if isinstance(x_sim, dict):
            x_sim = np.vstack(_ordered_values(x_sim, getattr(self.simulator, 'state_names', None), 'x_sim')).T
        x_sim = np.asarray(x_sim, dtype=float)
        return x_sim.reshape(x_sim.shape[0], n)

    def _discard_results(self):
        self._result = None
        self._O_df_sliding = None
        self._dxdz_sliding = None
        self._state_names = self._sensor_names = self._time_steps = None
        self._cache = {}

    def clear_cache(self):
        """Forget cached minimum error variance results (computed O is kept)."""
        self._cache = {}

    def _require_computed(self):
        if self._result is None:
            raise RuntimeError('observability matrices have not been computed with the current settings; '
                               'call run() first')

    # ------------------------------------------------------------------ results (after run)

    def _computed(self, value):
        self._require_computed()
        return value

    @property
    def w(self):
        return self._computed(self._result).w

    @property
    def n_windows(self):
        return len(self._computed(self._O_df_sliding))

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
        t_sim = self._computed(self._result).t_sim
        return None if t_sim is None else t_sim.copy()

    @property
    def O_index(self):
        return np.array(self._computed(self._result).O_index)

    @property
    def O_time(self):
        t_sim = self.t_sim
        return None if t_sim is None else t_sim[self.O_index]

    @property
    def O_df_sliding(self):
        """Copies of the observability matrix of every window (transformed if z_function is set)."""
        return [O.copy() for O in self._computed(self._O_df_sliding)]

    @property
    def window_data(self):
        return self._computed(self._result).window_data

    @property
    def dxdz_sliding(self):
        """dx/dz of the coordinate transform at each window's initial state (None without z_function)."""
        self._require_computed()
        return None if self._dxdz_sliding is None else [d.copy() for d in self._dxdz_sliding]

    @property
    def source(self):
        """The builder's native object (e.g. the SlidingEmpiricalObservabilityMatrix); O is untransformed there."""
        return self._computed(self._result).source

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
        return SlidingFisherObservability(self._O_df_sliding, R=R, lam=lam, time=self._result.t_sim,
                                          states=states, sensors=sensors,
                                          time_steps=None if time_steps is None else np.array(time_steps),
                                          w=None, force_R_scalar=force_R_scalar)

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

        ev = self.fisher(states_l, sensors_l, time_steps_l, R=R_r, lam=lam_r,
                         force_R_scalar=force_R_scalar).get_minimum_error_variance()
        if cacheable:
            self._cache[key] = ev.copy()
        return ev

    def observability_matrix(self, window=0, states=None, sensors=None, time_steps=None):
        """Copy of one window's observability matrix, optionally restricted to a selection
        (rows then ordered by time_step, sensor, as used for the Fisher information)."""
        states, sensors, time_steps = self._select(states, sensors, time_steps)
        O = self._O_df_sliding[window].copy()
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
            'w': self._result.w,
            'n_windows': len(self._O_df_sliding),
            'time_alignment': 'each window is stamped at its center time-step, time_initial + (w // 2) * dt',
            'analysis': self._settings_document(),
        }
        with open(files['sidecar'], 'w') as f:
            yaml.safe_dump(_to_plain(sidecar), f, sort_keys=False)
        return {k: str(f) for k, f in files.items()}

    def _save_observability_matrices(self, path):
        index = self._O_df_sliding[0].index
        if any(not O.index.equals(index) or list(O.columns) != self._state_names for O in self._O_df_sliding):
            raise ValueError('windows have different rows or columns; cannot stack them into one array')
        arrays = {'O': np.stack([O.to_numpy(dtype=float) for O in self._O_df_sliding]),
                  'state_names': np.array(self._state_names, dtype=str),
                  'sensor': np.array(index.get_level_values('sensor'), dtype=str),
                  'time_step': np.array(index.get_level_values('time_step'), dtype=int),
                  'O_index': np.asarray(self._result.O_index, dtype=int)}
        if self._result.t_sim is not None:
            arrays['t_sim'] = np.asarray(self._result.t_sim, dtype=float)
            arrays['window_time_initial'] = arrays['t_sim'][arrays['O_index']]
        np.savez(path, **arrays)

    def __repr__(self):
        status = f'computed, {len(self._O_df_sliding)} windows' if self.is_computed else 'not computed'
        return f'ObservabilityAnalysis(method={self.method!r}, w={self._settings["w"]!r}, {status})'
