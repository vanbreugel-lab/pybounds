"""The fast per-query path (shared row index) is bit-identical to FisherObservability per window."""
import time

import numpy as np
import pandas as pd
import pytest

import pybounds
from pybounds import ObservabilityAnalysis, analysis
from pybounds.analysis import SlidingO

SENSORS = ['s2', 's10', 's1', 'a']     # string order differs from list order ('s10' < 's2')
STATES = [f'x{i}' for i in range(6)]
W = 7


def _sliding(sensor_major=False, n_windows=9, seed=0, sensors=SENSORS, states=STATES):
    rng = np.random.default_rng(seed)
    if sensor_major:
        rows = [(s, k) for s in sensors for k in range(W)]
    else:
        rows = [(s, k) for k in range(W) for s in sensors]
    index = pd.MultiIndex.from_tuples(rows, names=['sensor', 'time_step'])
    O = rng.normal(size=(n_windows, len(rows), len(states)))
    return SlidingO(O=O, index=index, state_names=states, t_sim=0.1 * np.arange(n_windows + W - 1), w=W)


def _slow(oa, **query):
    """The pre-fast-path implementation: SlidingFisherObservability over per-window DataFrames."""
    states, sensors, time_steps = oa._select(query.get('states'), query.get('sensors'), query.get('time_steps'))
    R, lam = oa._resolve(query.get('R', analysis._UNSET), query.get('lam', analysis._UNSET), sensors)
    return oa._sliding_fisher(states, sensors, time_steps, R, lam, query.get('force_R_scalar', False),
                              keep_windows=True)


R_DICT = {'s2': 0.3, 's10': 0.05, 's1': 1.7, 'a': 0.9}
QUERIES = [dict(), dict(states=['x3']), dict(states=['x4', 'x0', 'x2']), dict(sensors=['s10', 's2']),
           dict(sensors=['a']), dict(time_steps=[5, 0, 3]), dict(sensors=['s1', 'a'], time_steps=[6]),
           dict(states=['x5', 'x1'], sensors=['s2', 's1'], time_steps=[1, 2, 4]), dict(R=0.37), dict(R=3),
           dict(R=np.float32(0.25)), dict(R=R_DICT), dict(R=R_DICT, sensors=['s10', 'a'], states=['x2']),
           dict(R=None), dict(lam=1e-3), dict(R=0.2, force_R_scalar=True, states=['x1', 'x0'])]


@pytest.mark.parametrize('sensor_major', [False, True])
@pytest.mark.parametrize('query', QUERIES)
def test_bit_identical_to_per_window_path(query, sensor_major):
    oa = ObservabilityAnalysis.from_sliding(_sliding(sensor_major), R=0.1)
    q = {k: v for k, v in query.items()}
    fast = oa._fast_windows(*oa._select(q.get('states'), q.get('sensors'), q.get('time_steps')),
                            q.get('R', 0.1), q.get('lam', 1e-8), q.get('force_R_scalar', False))
    assert fast is not None                      # the fast path is used for all of these
    with pytest.warns(UserWarning) if 'R' in q and q['R'] is None else _nullcontext():
        slow = _slow(oa, **q)
        ev = oa.min_error_variance(**q)
    pd.testing.assert_frame_equal(ev, slow.get_minimum_error_variance(), check_exact=True)
    if 'time_steps' not in q:
        with pytest.warns(UserWarning) if 'R' in q and q['R'] is None else _nullcontext():
            F = oa.fisher_information(states=q.get('states'), sensors=q.get('sensors'), R=q.get('R', 0.1),
                                      force_R_scalar=q.get('force_R_scalar', False))
        for k in range(oa.n_windows):
            np.testing.assert_array_equal(F[k], slow.FO[k].F.to_numpy())


def test_lam_limit():
    oa = ObservabilityAnalysis.from_sliding(_sliding(n_windows=2, states=STATES[:3]), R=R_DICT, lam='limit')
    pd.testing.assert_frame_equal(oa.min_error_variance(), _slow(oa).get_minimum_error_variance(), check_exact=True)


def test_z_function_names():
    """Transformed states without z_state_names are named 0..n-1 (integers)."""
    sim = pybounds.Simulator(lambda X, U: [U[0], 0 * U[0]], lambda X, U: [X[0] / X[1]], dt=0.01,
                             state_names=['g', 'd'], input_names=['u'], measurement_names=['r'])
    t, x, u, _ = sim.simulate(x0={'g': 2., 'd': 3.}, u={'u': 0.1 * np.ones(20)}, return_full_output=True)
    import sympy as sp
    with pytest.warns(UserWarning, match='without z_state_names'):
        oa = ObservabilityAnalysis(sim, t, x, u, w=5, R=0.1, z_function=lambda v: sp.Matrix([v[0] / v[1], v[1]])).run()
    for q in (dict(), dict(states=[1])):
        pd.testing.assert_frame_equal(oa.min_error_variance(**q), _slow(oa, **q).get_minimum_error_variance(),
                                      check_exact=True)


def test_matrix_R_uses_per_window_path():
    oa = ObservabilityAnalysis.from_sliding(_sliding(), R=0.1)
    R = 0.2 * np.eye(W * len(SENSORS))
    assert oa._fast_windows(None, None, None, R, 1e-8, False) is None
    pd.testing.assert_frame_equal(oa.min_error_variance(R=R), _slow(oa, R=R).get_minimum_error_variance(),
                                  check_exact=True)


def test_guard_falls_back(monkeypatch):
    """If window 0 computed both ways ever differs, the per-window path is used instead."""
    oa = ObservabilityAnalysis.from_sliding(_sliding(), R=0.1)
    monkeypatch.setattr(analysis._FastWindows, 'fisher', lambda self, k: np.zeros((6, 6)))
    assert oa._fast_windows(None, None, None, 0.1, 1e-8, False) is None
    pd.testing.assert_frame_equal(oa.min_error_variance(), _slow(oa).get_minimum_error_variance(), check_exact=True)


def test_fast_path_is_faster():
    rng = np.random.default_rng(0)
    sensors = [f's{j}' for j in range(20)]
    index = pd.MultiIndex.from_tuples([(s, k) for k in range(30) for s in sensors], names=['sensor', 'time_step'])
    oa = ObservabilityAnalysis.from_sliding(SlidingO(O=rng.normal(size=(120, 600, 12)), index=index,
                                                     state_names=[f'x{i}' for i in range(12)]), R=0.1)
    q = dict(states=['x1', 'x4', 'x7'], sensors=sensors[:5])
    t0 = time.perf_counter()
    oa.min_error_variance(**q)
    fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    _slow(oa, **q).get_minimum_error_variance()
    slow = time.perf_counter() - t0
    assert fast < slow / 5


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
