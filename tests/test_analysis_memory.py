"""ObservabilityAnalysis stores O once and gives bit-identical results to 7b69d66."""
import gc
import tracemalloc
import weakref

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import sympy as sp

import pybounds
from pybounds import ObservabilityAnalysis
from pybounds.analysis import SlidingO
from pybounds.observability import _transform_O_df_list
import reference_7b69d66 as ref


class LinearSim:
    """Fast deterministic simulator: x_{k+1} = Ad x_k + Bd u_k, y_k = C x_k."""

    def __init__(self, n, p, dt=0.01, seed=0):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(n, n)) / np.sqrt(n) - 1.5 * np.eye(n)
        self.Ad = scipy.linalg.expm(A * dt)
        self.Bd = dt * rng.normal(size=(n, 1))
        self.C = rng.normal(size=(p, n))
        self.state_names = [f'x{i}' for i in range(n)]
        self.input_names = ['u']
        self.measurement_names = [f's{j}' for j in range(p)]

    def simulate(self, x0, u, aux=None):
        x = np.array(x0, dtype=float)
        u = np.asarray(u, dtype=float).reshape(len(u), -1)
        y = np.empty((len(u), self.C.shape[0]))
        for k in range(len(u)):
            y[k] = self.C @ x
            x = self.Ad @ x + self.Bd @ u[k]
        return y

    def trajectory(self, N, seed=1):
        rng = np.random.default_rng(seed)
        u = rng.normal(size=(N, 1))
        x = np.empty((N, len(self.state_names)))
        x[0] = rng.normal(size=len(self.state_names))
        for k in range(N - 1):
            x[k + 1] = self.Ad @ x[k] + self.Bd @ u[k]
        return 0.01 * np.arange(N), x, u


def z_fn(x):
    """Nonlinear coordinate change on 6 states."""
    return sp.Matrix([x[0] / (1 + x[1] ** 2)] + [x[i] + 0.1 * x[0] * x[i] for i in range(1, 6)])


SIM = LinearSim(6, 5)
T, X, U = SIM.trajectory(40)
R_DICT = {s: 0.1 * (j + 1) for j, s in enumerate(SIM.measurement_names)}
QUERIES = [dict(), dict(states=['x3']), dict(states=['x4', 'x0', 'x2']), dict(sensors=['s3', 's0']),
           dict(states=['x1', 'x5'], sensors=['s1', 's2', 's4']), dict(R=0.37), dict(R=R_DICT), dict(lam=1e-4),
           dict(states=['x2'], R=R_DICT, lam=1e-6), dict(time_steps=[0, 3])]


def _reference(w, z=False, z_state_names=None):
    """The 7b69d66 pipeline: builder O copied per window, optional z transform, old SlidingFisher."""
    seom = pybounds.SlidingEmpiricalObservabilityMatrix(SIM, T, X, U, w=w, eps=1e-4)
    O_list = [O.copy() for O in seom.O_df_sliding]
    if z:
        O_list = _transform_O_df_list(O_list, X[seom.O_index], z_fn, z_state_names)
    return O_list


def _rename(query, oa):
    """Map x-state selections onto the analysis' state names (for z coordinates)."""
    if 'states' not in query:
        return query
    return dict(query, states=[oa.state_names[int(s[1:])] for s in query['states']])


@pytest.mark.parametrize('config', [dict(w=4), dict(w=15), dict(w=4, z=True, z_state_names=[f'z{i}' for i in range(6)]),
                                    dict(w=4, z=True)])
def test_bit_identical_to_7b69d66(config):
    config = dict(config)
    z = config.pop('z', False)
    O_list = _reference(config['w'], z=z, z_state_names=config.get('z_state_names'))
    kwargs = dict(w=config['w'], eps=1e-4, R=0.1)
    if z:
        kwargs.update(z_function=z_fn, z_state_names=config.get('z_state_names'))
    with pytest.warns(UserWarning) if z and config.get('z_state_names') is None else _nullcontext():
        oa = ObservabilityAnalysis(SIM, T, X, U, **kwargs).run()
    for k in (0, len(O_list) // 2, len(O_list) - 1):
        pd.testing.assert_frame_equal(oa.observability_matrix(k), O_list[k], check_exact=True)
    for query in QUERIES:
        q = _rename(query, oa)
        old = ref.SlidingFisherObservability(O_list, time=T, R=q.get('R', 0.1), lam=q.get('lam', 1e-8),
                                             states=q.get('states'), sensors=q.get('sensors'),
                                             time_steps=q.get('time_steps'))
        pd.testing.assert_frame_equal(oa.min_error_variance(**q), old.get_minimum_error_variance(), check_exact=True)
        new = oa.fisher(**q)
        for k in (0, len(O_list) - 1):
            pd.testing.assert_frame_equal(new.FO[k].F, old.FO[k].F, check_exact=True)
            pd.testing.assert_frame_equal(new.FO[k].F_inv, old.FO[k].F_inv, check_exact=True)


@pytest.mark.parametrize('form', ['frames', 'array'])
def test_from_sliding_bit_identical(form):
    O_list = _reference(6)
    if form == 'frames':
        sliding = SlidingO(O_list, T, np.arange(len(O_list)), 6)
    else:
        sliding = SlidingO(O=np.stack([O.values for O in O_list]), index=O_list[0].index,
                           state_names=list(O_list[0].columns), t_sim=T, w=6)
    oa = ObservabilityAnalysis.from_sliding(sliding, R=R_DICT)
    for query in QUERIES:
        old = ref.SlidingFisherObservability(O_list, time=T, R=query.get('R', R_DICT), lam=query.get('lam', 1e-8),
                                             states=query.get('states'), sensors=query.get('sensors'),
                                             time_steps=query.get('time_steps'))
        pd.testing.assert_frame_equal(oa.min_error_variance(**query), old.get_minimum_error_variance(),
                                      check_exact=True)
        pd.testing.assert_frame_equal(oa.fisher(**query).FO[3].F, old.FO[3].F, check_exact=True)


class TestStorage:

    def test_single_array(self):
        oa = ObservabilityAnalysis(SIM, T, X, U, w=5, eps=1e-4).run()
        assert oa._O.shape == (36, 5 * 5, 6)
        assert not oa._O.flags.writeable
        frames = oa.O_df_sliding
        assert len(frames) == 36 and all(isinstance(f, pd.DataFrame) for f in frames)
        frames[0].iloc[:, :] = 0.0   # the public copies are writable and don't touch the stored array
        assert np.abs(oa.observability_matrix(0).values).sum() > 0

    def test_builder_streams_by_default(self, monkeypatch):
        """By default windows are streamed: no full builder object (with y_plus/y_minus for every window) exists."""
        from pybounds import analysis
        original = analysis._BUILDERS['bounds-empirical']
        results = []

        def func(*args, **kwargs):
            result = original.func(*args, **kwargs)
            results.append(result)
            return result

        monkeypatch.setitem(analysis._BUILDERS, 'bounds-empirical', analysis._Builder(func, original.options))
        oa = ObservabilityAnalysis(SIM, T, X, U, w=5, eps=1e-4).run()
        assert isinstance(results[0], analysis._WindowStream) and results[0].source is None
        assert oa._source is None and oa._window_data is None
        kept = ObservabilityAnalysis(SIM, T, X, U, w=5, eps=1e-4, keep_source=True).run()
        assert isinstance(results[1].source, pybounds.SlidingEmpiricalObservabilityMatrix)
        assert kept.source is results[1].source
        np.testing.assert_array_equal(kept._O, oa._O)   # same O either way

    def test_array_input_is_not_copied(self):
        O_list = _reference(6)
        O = np.stack([f.values for f in O_list])
        oa = ObservabilityAnalysis.from_sliding(SlidingO(O=O, index=O_list[0].index,
                                                         state_names=list(O_list[0].columns), t_sim=T))
        assert np.shares_memory(oa._O, O)
        assert O.flags.writeable            # the caller's array is untouched ...
        assert not oa._O.flags.writeable    # ... but the analysis' view of it is read-only

    def test_caller_frames_are_not_kept(self):
        O_list = _reference(6)
        refs = [weakref.ref(f) for f in O_list]
        oa = ObservabilityAnalysis.from_sliding(SlidingO(O_list, T))
        del O_list
        gc.collect()
        assert all(r() is None for r in refs)
        assert oa.n_windows == 35

    def test_windows_with_different_rows_raise(self):
        O_list = _reference(6)
        O_list[2] = O_list[2].iloc[1:]
        with pytest.raises(ValueError, match='window 2 has different rows or states'):
            ObservabilityAnalysis.from_sliding(SlidingO(O_list, T))

    def test_array_form_needs_labels(self):
        with pytest.raises(ValueError, match='needs index'):
            ObservabilityAnalysis.from_sliding(SlidingO(O=np.zeros((2, 3, 4))))

    def test_held_memory_is_one_copy_of_O(self):
        sim = LinearSim(20, 30)
        t, x, u = sim.trajectory(120)
        oa = ObservabilityAnalysis(sim, t, x, u, w=40, eps=1e-4, R=0.1)
        gc.collect()
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        oa.run()
        gc.collect()
        held = tracemalloc.get_traced_memory()[0] - before
        tracemalloc.stop()
        O_bytes = oa._O.nbytes
        assert O_bytes == 81 * 40 * 30 * 20 * 8
        assert held < 1.15 * O_bytes


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _linear_factory():
    """Module-level (importable) factory for process-based parallel windows."""
    return LinearSim(6, 5)


class TestStreaming:

    @pytest.mark.parametrize('options', [dict(parallel_sliding=True, simulator_factory=_linear_factory, n_workers=2),
                                         dict(parallel_sliding=True)])   # processes; threads (custom simulator)
    def test_parallel_windows_are_identical(self, options):
        sequential = ObservabilityAnalysis(SIM, T, X, U, w=5, eps=1e-4, R=R_DICT).run()
        parallel = ObservabilityAnalysis(SIM, T, X, U, w=5, eps=1e-4, R=R_DICT, **options).run()
        np.testing.assert_array_equal(parallel._O, sequential._O)
        pd.testing.assert_frame_equal(parallel.min_error_variance(states=['x1']),
                                      sequential.min_error_variance(states=['x1']), check_exact=True)

    def test_seom_run_is_unchanged(self):
        """The public class still collects every window, with its window_data."""
        seom = pybounds.SlidingEmpiricalObservabilityMatrix(SIM, T, X, U, w=5, eps=1e-4)
        assert len(seom.O_sliding) == len(seom.O_df_sliding) == 36
        assert set(seom.window_data) == {'t', 'u', 'y', 'y_plus', 'y_minus'}
        assert len(seom.window_data['y_plus']) == 36
        oa = ObservabilityAnalysis(SIM, T, X, U, w=5, eps=1e-4).run()
        for k in (0, 35):
            np.testing.assert_array_equal(oa._O[k], seom.O_sliding[k])

    @pytest.mark.parametrize('storage', ['observability', 'fisher_per_sensor', 'fisher'])
    def test_run_peak_memory(self, storage):
        """run() never holds all windows' builder data: the peak is the stored size plus a few windows."""
        sim = LinearSim(20, 30)
        t, x, u = sim.trajectory(120)
        w = 40
        oa = ObservabilityAnalysis(sim, t, x, u, w=w, eps=1e-4, R=0.1, storage=storage)
        gc.collect()
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        oa.run()
        peak = tracemalloc.get_traced_memory()[1] - before
        tracemalloc.stop()
        n_windows, n, p = 81, 20, 30
        stored = {'observability': 8 * n_windows * w * p * n,
                  'fisher_per_sensor': 8 * n_windows * p * n * (n + 1) // 2,
                  'fisher': 8 * n_windows * n * (n + 1) // 2}[storage]
        one_window = 8 * w * p * n
        assert peak < stored + 12 * one_window + 1_000_000   # the old builder peaked at ~5x O
