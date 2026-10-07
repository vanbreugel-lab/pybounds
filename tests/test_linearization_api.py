"""ObservabilityAnalysis.from_linearization, .linearization, .model_state_names, .deterministic_states, and keeping
a stochastic method's linearization when only w or the coordinate transform change."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sympy as sp

import pybounds
from pybounds import analysis
from pybounds.analysis import ObservabilityAnalysis, Linearization
from conftest import N_STEPS_SLIDING, WINDOW_SIZE, EPS

N_WINDOWS = N_STEPS_SLIDING - WINDOW_SIZE + 1
KINDS = ['observability', 'constructability']


def z_optic_flow(x):
    return sp.Matrix([x[0] / x[1], x[1]])


@pytest.fixture(scope='module')
def trajectory(simulation_output):
    t_sim, x_sim, u_sim, _ = simulation_output
    return (t_sim[:N_STEPS_SLIDING],
            {k: v[:N_STEPS_SLIDING] for k, v in x_sim.items()},
            {k: v[:N_STEPS_SLIDING] for k, v in u_sim.items()})


def _run(simulator, trajectory, kind, **settings):
    settings = {'w': WINDOW_SIZE, 'R': 0.1, 'Q': 1e-4, **settings}
    return ObservabilityAnalysis(simulator, *trajectory, method=f'stochastic-{kind}-classic', **settings).run()


def _fields(lin):
    """The fields of a Linearization without copying the arrays (dataclasses.asdict deep-copies them)."""
    return {f.name: getattr(lin, f.name) for f in dataclasses.fields(lin)}


QUERIES = [{}, {'states': ['d']}, {'states': ['d', 'g']}, {'time_steps': [0, 2, 3]}, {'lam': 1e-6},
           {'lam': {'g': 1e-3}}, {'Q': {'g': 1e-3, 'd': 1e-6}}, {'R': {'r': 0.5}}, {'alignment': 'bounded_state'},
           {'lam': 'limit'}]


class TestFromLinearization:

    @pytest.mark.parametrize('kind', KINDS)
    @pytest.mark.parametrize('transform', [None, 'dxdz_sliding', 'z_function'])
    def test_bit_identical_to_run(self, simulator, trajectory, kind, transform, tmp_path):
        z = {} if transform is None else {'z_function': z_optic_flow, 'z_state_names': ['of', 'd']}
        ran = _run(simulator, trajectory, kind, **z)
        lin = ran.linearization
        extra = {}
        if transform == 'dxdz_sliding':
            extra = {'dxdz_sliding': np.stack(ran.dxdz_sliding), 'z_state_names': ['of', 'd']}
        elif transform == 'z_function':
            extra = {'z_function': z_optic_flow, 'x_sim': trajectory[1], 'z_state_names': ['of', 'd']}
        wrapped = ObservabilityAnalysis.from_linearization(**_fields(lin), method=f'stochastic-{kind}', w=WINDOW_SIZE,
                                                           R=0.1, Q=1e-4, **extra)
        assert wrapped.state_names == ran.state_names and wrapped.model_state_names == ['g', 'd']
        for query in QUERIES:
            if transform is not None and 'states' in query:
                query = {'states': ['of' if s == 'g' else s for s in query['states']]}
            if transform is not None and isinstance(query.get('lam'), dict):
                query = {'lam': {'of': 1e-3}}
            pd.testing.assert_frame_equal(wrapped.min_error_variance(**query), ran.min_error_variance(**query),
                                          check_exact=True)
        np.testing.assert_array_equal(wrapped.fisher_information(), ran.fisher_information())
        for k in (0, 7, -1):
            pd.testing.assert_frame_equal(wrapped.observability_matrix(k), ran.observability_matrix(k),
                                          check_exact=True)
        a = wrapped.save_results(tmp_path / 'a')
        b = ran.save_results(tmp_path / 'b')
        assert Path(a['min_error_variance']).read_text() == Path(b['min_error_variance']).read_text()

    def test_method_names(self, simulator, trajectory):
        lin = _run(simulator, trajectory, 'observability').linearization
        for method in ('stochastic-observability', 'stochastic-observability-classic', 'stochastic-observability-jax'):
            oa = ObservabilityAnalysis.from_linearization(lin.Phi, lin.C, method=method, w=WINDOW_SIZE)
            assert oa.method == method and oa.linearization.bounded == 'initial'
        oa = ObservabilityAnalysis.from_linearization(lin.Phi, lin.C, method='stochastic-constructability', w=3)
        assert oa.linearization.bounded == 'final' and oa.n_windows == N_STEPS_SLIDING - 2
        for method in ('bounds-empirical', 'stochastic', 'stochastic-observability-other', None):
            with pytest.raises(ValueError, match='needs a stochastic method'):
                ObservabilityAnalysis.from_linearization(lin.Phi, lin.C, method=method, w=WINDOW_SIZE)
        with pytest.raises(ValueError, match='does not match method'):
            ObservabilityAnalysis.from_linearization(lin.Phi, lin.C, method='stochastic-observability', w=WINDOW_SIZE,
                                                     bounded='final')

    def test_arrays_shared_read_only_not_copied(self, simulator, trajectory):
        lin = _run(simulator, trajectory, 'observability').linearization
        Phi, C = np.array(lin.Phi), np.array(lin.C)   # writable arrays owned by the caller
        a = ObservabilityAnalysis.from_linearization(Phi, C, method='stochastic-observability', w=WINDOW_SIZE)
        b = ObservabilityAnalysis.from_linearization(Phi, C, method='stochastic-observability', w=3)
        for oa in (a, b):
            assert np.shares_memory(oa.linearization.Phi, Phi) and np.shares_memory(oa.linearization.C, C)
            assert not oa.linearization.Phi.flags.writeable and not oa.linearization.C.flags.writeable
        assert np.shares_memory(a.linearization.Phi, b.linearization.Phi)
        assert Phi.flags.writeable   # the caller's own array is left as it was
        with pytest.raises(ValueError):
            a.linearization.Phi[0, 0, 0] = 1.0

    def test_only_query_settings_change(self, simulator, trajectory):
        lin = _run(simulator, trajectory, 'observability').linearization
        oa = ObservabilityAnalysis.from_linearization(**_fields(lin), method='stochastic-observability',
                                                      w=WINDOW_SIZE, Q=1e-4, R=0.1)
        assert oa.run() is oa and oa.is_computed
        oa.update_settings(Q=1e-3, R=0.2, lam=1e-6, alignment='bounded_state')
        assert oa.is_computed
        for change in ({'w': 3}, {'z_function': z_optic_flow}, {'method': 'stochastic-constructability-classic'}):
            with pytest.raises(ValueError, match=r'only the query settings \(R, lam, Q, alignment\)'):
                oa.update_settings(**change)

    def test_defaults_without_names_or_time(self, simulator, trajectory):
        lin = _run(simulator, trajectory, 'observability').linearization
        oa = ObservabilityAnalysis.from_linearization(lin.Phi, lin.C, method='stochastic-observability', w=None,
                                                      R=0.1, Q=1e-4)
        assert oa.state_names == ['x_0', 'x_1'] and oa.sensor_names == ['y_0']
        assert oa.n_windows == 1 and oa.t_sim is None
        ev = oa.min_error_variance()
        assert list(ev.columns) == ['time', 'time_initial', 'x_0', 'x_1'] and len(ev) == 1

    def test_x_sim_as_array(self, simulator, trajectory):
        ran = _run(simulator, trajectory, 'constructability', z_function=z_optic_flow, z_state_names=['of', 'd'])
        x = np.column_stack([trajectory[1]['g'], trajectory[1]['d']])
        oa = ObservabilityAnalysis.from_linearization(**_fields(ran.linearization), method='stochastic-constructability',
                                                      w=WINDOW_SIZE, z_function=z_optic_flow, x_sim=x,
                                                      z_state_names=['of', 'd'], R=0.1, Q=1e-4)
        pd.testing.assert_frame_equal(oa.min_error_variance(), ran.min_error_variance(), check_exact=True)

    def test_errors(self, simulator, trajectory):
        ran = _run(simulator, trajectory, 'observability', z_function=z_optic_flow, z_state_names=['of', 'd'])
        lin = ran.linearization
        dxdz = np.stack(ran.dxdz_sliding)
        make = ObservabilityAnalysis.from_linearization
        with pytest.raises(ValueError, match='either as dxdz_sliding or as z_function'):
            make(lin.Phi, lin.C, method='stochastic-observability', w=WINDOW_SIZE, dxdz_sliding=dxdz,
                 z_function=z_optic_flow, x_sim=trajectory[1])
        with pytest.raises(ValueError, match='z_function needs x_sim'):
            make(lin.Phi, lin.C, method='stochastic-observability', w=WINDOW_SIZE, z_function=z_optic_flow)
        with pytest.raises(ValueError, match='z_state_names names transformed states'):
            make(lin.Phi, lin.C, method='stochastic-observability', w=WINDOW_SIZE, z_state_names=['of', 'd'])
        with pytest.raises(ValueError, match='dxdz_sliding must have shape'):
            make(lin.Phi, lin.C, method='stochastic-observability', w=3, dxdz_sliding=dxdz)
        with pytest.raises(ValueError, match='C must have shape'):
            make(lin.Phi, lin.C[:-1], method='stochastic-observability', w=WINDOW_SIZE)
        with pytest.raises(ValueError, match='Phi must have shape'):
            make(lin.Phi[:, :1], lin.C, method='stochastic-observability', w=WINDOW_SIZE)
        with pytest.raises(ValueError, match='state names'):
            make(lin.Phi, lin.C, method='stochastic-observability', w=WINDOW_SIZE, state_names=['g'])
        with pytest.raises(ValueError, match='window size'):
            make(lin.Phi, lin.C, method='stochastic-observability', w=N_STEPS_SLIDING + 1)
        with pytest.warns(UserWarning, match='dxdz_sliding is set without z_state_names'):
            oa = make(lin.Phi, lin.C, method='stochastic-observability', w=WINDOW_SIZE, dxdz_sliding=dxdz)
        assert oa.state_names == [0, 1]


class TestLinearizationAccessor:

    @pytest.mark.parametrize('kind, bounded', [('observability', 'initial'), ('constructability', 'final')])
    def test_fields(self, simulator, trajectory, kind, bounded):
        oa = _run(simulator, trajectory, kind, z_function=z_optic_flow, z_state_names=['of', 'd'])
        lin = oa.linearization
        assert isinstance(lin, Linearization)
        assert [f.name for f in dataclasses.fields(lin)] == ['Phi', 'C', 't_sim', 'state_names', 'sensor_names',
                                                             'bounded']
        assert lin.Phi.shape == (N_STEPS_SLIDING, 2, 2) and lin.C.shape == (N_STEPS_SLIDING, 1, 2)
        assert lin.state_names == ('g', 'd') and lin.sensor_names == ('r',) and lin.bounded == bounded
        np.testing.assert_array_equal(lin.t_sim, trajectory[0])
        assert not (lin.Phi.flags.writeable or lin.C.flags.writeable or lin.t_sim.flags.writeable)
        with pytest.raises(dataclasses.FrozenInstanceError):
            lin.Phi = None
        # the model's own names, not the transformed ones
        assert oa.state_names == ['of', 'd'] and oa.model_state_names == ['g', 'd']

    def test_asdict_round_trip(self, simulator, trajectory):
        oa = _run(simulator, trajectory, 'constructability')
        again = ObservabilityAnalysis.from_linearization(**dataclasses.asdict(oa.linearization),
                                                         method='stochastic-constructability', w=WINDOW_SIZE,
                                                         R=0.1, Q=1e-4)
        pd.testing.assert_frame_equal(again.min_error_variance(), oa.min_error_variance(), check_exact=True)
        assert again.linearization.state_names == oa.linearization.state_names

    def test_none_for_bounds_methods(self, simulator, trajectory):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1).run()
        assert oa.linearization is None
        with pytest.raises(RuntimeError, match='call run'):
            ObservabilityAnalysis(simulator, *trajectory).linearization

    def test_model_state_names_every_method(self, simulator, trajectory, seom):
        plain = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1).run()
        assert plain.model_state_names == plain.state_names == ['g', 'd']
        for storage in ('observability', 'fisher'):
            z = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1, storage=storage,
                                      z_function=z_optic_flow, z_state_names=['of', 'd']).run()
            assert z.state_names == ['of', 'd'] and z.model_state_names == ['g', 'd']
        assert ObservabilityAnalysis.from_sliding(seom).model_state_names == ['g', 'd']
        stochastic = _run(simulator, trajectory, 'observability')
        assert stochastic.model_state_names == stochastic.state_names == ['g', 'd']


class TestDeterministicStates:

    def test_constant_parameter(self):
        # g relaxes (its row of A is not zero), d is a constant parameter, t is a clock
        f = lambda X, U: [-0.5 * X[0] + U[0], 0 * U[0], 1 + 0 * U[0]]
        h = lambda X, U: [X[0] / X[1]]
        sim = pybounds.Simulator(f, h, dt=0.01, state_names=['g', 'd', 't'], input_names=['u'],
                                 measurement_names=['r'])
        t, x, u, _ = sim.simulate(x0={'g': 2.0, 'd': 3.0, 't': 0.0}, u={'u': 0.1 * np.ones(20)},
                                  return_full_output=True)
        oa = ObservabilityAnalysis(sim, t, x, u, method='stochastic-observability-classic', w=5).run()
        assert oa.deterministic_states() == ['d', 't']
        assert oa.deterministic_states(atol=1.0) == ['g', 'd', 't']

    def test_model_coordinates_and_bounds_methods(self, simulator, trajectory):
        oa = _run(simulator, trajectory, 'observability', z_function=z_optic_flow, z_state_names=['of', 'd'])
        assert oa.deterministic_states() == ['g', 'd']   # f does not depend on x at all here
        bounds = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1).run()
        with pytest.raises(ValueError, match='does not build'):
            bounds.deterministic_states()


class TestKeepLinearization:

    @pytest.fixture
    def counting(self, monkeypatch):
        calls = []
        originals = {m: analysis._BUILDERS[m] for m in ('stochastic-observability-classic',
                                                        'stochastic-constructability-classic')}

        def counted(original):
            def func(*args, **kwargs):
                calls.append(kwargs)
                return original.func(*args, **kwargs)
            return analysis._Builder(func, original.options)

        for method, original in originals.items():
            monkeypatch.setitem(analysis._BUILDERS, method, counted(original))
        return calls

    @pytest.mark.parametrize('kind', KINDS)
    def test_w_sweep_linearizes_once(self, simulator, trajectory, counting, kind):
        oa = _run(simulator, trajectory, kind)
        for w in (2, 9, None, WINDOW_SIZE):
            oa.update_settings(w=w)
            assert not oa.is_computed
            oa.run()
            fresh = ObservabilityAnalysis(simulator, *trajectory, method=f'stochastic-{kind}-classic', w=w,
                                          R=0.1, Q=1e-4).run()
            pd.testing.assert_frame_equal(oa.min_error_variance(), fresh.min_error_variance(), check_exact=True)
        assert len([c for c in counting]) == 1 + 4   # one for oa, one per fresh analysis

    @pytest.mark.parametrize('kind', KINDS)
    def test_z_function_change_keeps_linearization(self, simulator, trajectory, counting, kind):
        oa = _run(simulator, trajectory, kind)
        oa.update_settings(z_function=z_optic_flow, z_state_names=['of', 'd']).run()
        oa.update_settings(z_state_names=['optic_flow', 'height']).run()
        assert len(counting) == 1 and oa.state_names == ['optic_flow', 'height']
        fresh = _run(simulator, trajectory, kind, z_function=z_optic_flow, z_state_names=['optic_flow', 'height'])
        pd.testing.assert_frame_equal(oa.min_error_variance(), fresh.min_error_variance(), check_exact=True)
        oa.update_settings(z_function=None, z_state_names=None).run()
        assert oa.state_names == ['g', 'd'] and len(counting) == 2

    def test_other_changes_linearize_again(self, simulator, trajectory, counting):
        oa = _run(simulator, trajectory, 'observability')
        oa.update_settings(eps=1e-6).run()
        assert len(counting) == 2
        oa.update_settings(method='stochastic-constructability-classic').run()
        assert len(counting) == 3 and oa.linearization.bounded == 'final'
        oa.update_settings(method='stochastic-constructability-classic', w=4).run()   # same method: kept
        assert len(counting) == 3
        oa.run()   # an explicit run() with nothing changed linearizes again, like the other methods
        assert len(counting) == 4

    def test_switching_to_bounds_discards(self, simulator, trajectory):
        oa = _run(simulator, trajectory, 'observability')
        oa.update_settings(method='bounds-empirical', eps=EPS)
        assert oa._lin is None
        oa.run()
        assert oa.linearization is None and oa.method == 'bounds-empirical'


class TestRegressions:

    def test_settings_saved_from_linearization_load_into_a_simulator_analysis(self, simulator, trajectory, tmp_path):
        """Regression: from_linearization saved its suffix-less method name, which load_settings rejected."""
        lin = _run(simulator, trajectory, 'constructability').linearization
        wrapped = ObservabilityAnalysis.from_linearization(**_fields(lin), method='stochastic-constructability',
                                                           w=WINDOW_SIZE, R=0.1, Q=1e-4)
        path = wrapped.save_settings(tmp_path / 's.yaml')
        assert ObservabilityAnalysis(simulator, *trajectory).load_settings(path).method == \
            'stochastic-constructability-classic'
        keep = ObservabilityAnalysis(simulator, *trajectory, method='stochastic-constructability-jax')
        assert keep.load_settings(path).method == 'stochastic-constructability-jax'   # same kind: backend kept
        loaded = ObservabilityAnalysis(simulator, *trajectory).load_settings(path).run()
        pd.testing.assert_frame_equal(loaded.min_error_variance(), wrapped.min_error_variance(), check_exact=True)
        files = wrapped.save_results(tmp_path / 'out')
        assert ObservabilityAnalysis(simulator, *trajectory).load_settings(files['sidecar']).method == \
            'stochastic-constructability-classic'

    def test_bare_method_names(self, simulator, trajectory):
        oa = ObservabilityAnalysis(simulator, *trajectory, method='stochastic-observability')
        assert oa.method == 'stochastic-observability-classic'
        oa.update_settings(method='stochastic-constructability')
        assert oa.method == 'stochastic-constructability-classic'
        pytest.importorskip('jax')
        jax_sim = pybounds.JaxSimulator(lambda x, u: [u[0], 0.0 * u[0]], lambda x, u: [x[0] / x[1]], dt=0.01,
                                        state_names=['g', 'd'], input_names=['u'], measurement_names=['r'])
        assert ObservabilityAnalysis(jax_sim, *trajectory, method='stochastic-observability').method == \
            'stochastic-observability-jax'

    def test_linearization_compares_by_identity(self, simulator, trajectory):
        """Regression: the dataclass compared array fields element-wise, so == and hash() raised."""
        lin = _run(simulator, trajectory, 'observability').linearization
        copy = dataclasses.replace(lin, Phi=lin.Phi.copy())
        assert lin == lin and lin != copy and lin != dataclasses.replace(lin)
        assert {lin: 1}[lin] == 1 and hash(lin) != hash(copy)

    def test_custom_simulator_without_state_names(self, trajectory):
        """Regression: with no simulator state names, a dict x_sim and a z_function failed: the transform ordered
        x_sim by the default names x_0, x_1 instead of the dict's keys."""
        class Custom:
            dt = 0.01
            measurement_names = ['r']

            @staticmethod
            def f(x, u):
                return [u[0], 0 * u[0]]

            @staticmethod
            def h(x, u):
                return [x[0] / x[1]]

        oa = ObservabilityAnalysis(Custom(), *trajectory, method='stochastic-observability-classic', w=WINDOW_SIZE,
                                   R=0.1, Q=1e-4, z_function=z_optic_flow, z_state_names=['of', 'd']).run()
        assert oa.model_state_names == ['g', 'd'] and oa.state_names == ['of', 'd']
        g, d = trajectory[1]['g'][3], trajectory[1]['d'][3]
        np.testing.assert_allclose(oa.dxdz_sliding[3], np.linalg.inv([[1 / d, -g / d ** 2], [0.0, 1.0]]), rtol=1e-12)
        # from_linearization names the states after a dict x_sim the same way
        wrapped = ObservabilityAnalysis.from_linearization(oa.linearization.Phi, oa.linearization.C,
                                                           method='stochastic-observability', w=WINDOW_SIZE,
                                                           t_sim=trajectory[0], z_function=z_optic_flow,
                                                           x_sim=trajectory[1],
                                                           z_state_names=['of', 'd'], R=0.1, Q=1e-4)
        assert wrapped.model_state_names == ['g', 'd']
        pd.testing.assert_frame_equal(wrapped.min_error_variance(), oa.min_error_variance(), check_exact=True)
