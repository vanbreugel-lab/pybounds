import inspect
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sympy as sp

import pybounds
from pybounds import analysis
from pybounds.analysis import ObservabilityAnalysis, SlidingO
from conftest import N_STEPS_SLIDING, WINDOW_SIZE, EPS, AnalyticSimulator

N_WINDOWS = N_STEPS_SLIDING - WINDOW_SIZE + 1


def z_optic_flow(x):
    """Transform [g, d] -> [g/d, d]."""
    return sp.Matrix([x[0] / x[1], x[1]])


@pytest.fixture(scope='module')
def trajectory(simulation_output):
    t_sim, x_sim, u_sim, _ = simulation_output
    return (t_sim[:N_STEPS_SLIDING],
            {k: v[:N_STEPS_SLIDING] for k, v in x_sim.items()},
            {k: v[:N_STEPS_SLIDING] for k, v in u_sim.items()})


@pytest.fixture(scope='module')
def oa(simulator, trajectory):
    """A computed analysis matching the conftest seom fixture."""
    return ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R={'r': 0.1}).run()


@pytest.fixture
def counting_builder(monkeypatch):
    """Replace the empirical builder with one that counts its calls."""
    calls = []
    original = analysis._BUILDERS['empirical']

    def func(*args, **kwargs):
        calls.append(kwargs)
        return original.func(*args, **kwargs)

    monkeypatch.setitem(analysis._BUILDERS, 'empirical', analysis._Builder(func, original.options))
    return calls


def _two_sensor_sliding(n_windows=5, w=4):
    """Synthetic SlidingO with sensors 'r' and 'a' and states 'g', 'd'."""
    rng = np.random.default_rng(3)
    index = pd.MultiIndex.from_tuples([(s, k) for k in range(w) for s in ('r', 'a')], names=['sensor', 'time_step'])
    O_list = [pd.DataFrame(rng.normal(size=(2 * w, 2)), index=index, columns=['g', 'd']) for _ in range(n_windows)]
    t_sim = 0.1 * np.arange(n_windows + w - 1)
    return SlidingO(O_df_sliding=O_list, t_sim=t_sim, O_index=np.arange(n_windows), w=w)


class TestLazy:

    def test_construction_and_update_do_not_compute(self, simulator, trajectory, counting_builder):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE)
        oa.update_settings(w=4, eps=1e-4, R=0.2)
        assert counting_builder == []
        assert not oa.is_computed

    @pytest.mark.parametrize('query', ['min_error_variance', 'fisher', 'observability_matrix'])
    def test_queries_before_run_raise(self, simulator, trajectory, query):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE)
        with pytest.raises(RuntimeError, match='call run'):
            getattr(oa, query)()
        with pytest.raises(RuntimeError, match='call run'):
            oa.state_names

    def test_run_builds_once_for_many_queries(self, simulator, trajectory, counting_builder):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, R={'r': 0.1}).run()
        oa.min_error_variance()
        oa.min_error_variance(states=['d'])
        oa.min_error_variance(time_steps=[0, 1, 2], lam=1e-6)
        oa.fisher()
        oa.observability_matrix(3)
        assert len(counting_builder) == 1
        assert counting_builder[0]['w'] == WINDOW_SIZE

    @pytest.mark.parametrize('change', [{'w': 4}, {'eps': 1e-3}, {'z_function': z_optic_flow}])
    def test_o_setting_change_discards_results(self, oa_fresh, change):
        oa_fresh.update_settings(**change)
        assert not oa_fresh.is_computed
        with pytest.raises(RuntimeError, match='call run'):
            oa_fresh.min_error_variance()

    @pytest.mark.parametrize('change', [{'R': 0.5}, {'lam': 1e-6}])
    def test_query_setting_change_keeps_o_and_clears_cache(self, oa_fresh, change):
        before = oa_fresh.min_error_variance()
        oa_fresh.update_settings(**change)
        assert oa_fresh.is_computed
        after = oa_fresh.min_error_variance()
        assert not after[['g', 'd']].equals(before[['g', 'd']])

    def test_update_settings_returns_self_for_chaining(self, simulator, trajectory):
        oa = ObservabilityAnalysis(simulator, *trajectory).update_settings(w=WINDOW_SIZE).run()
        assert oa.n_windows == N_WINDOWS


@pytest.fixture
def oa_fresh(simulator, trajectory):
    return ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R={'r': 0.1}).run()


class TestEquivalence:

    def test_matches_manual_pipeline(self, oa, seom):
        expected = pybounds.SlidingFisherObservability(seom.O_df_sliding, time=seom.t_sim, R={'r': 0.1},
                                                       lam=1e-8).get_minimum_error_variance()
        pd.testing.assert_frame_equal(oa.min_error_variance(), expected)

    def test_explicit_lists_match_defaults(self, oa, simulator):
        explicit = oa.min_error_variance(states=simulator.state_names, sensors=simulator.measurement_names,
                                         time_steps=np.arange(WINDOW_SIZE))
        pd.testing.assert_frame_equal(explicit, oa.min_error_variance())

    def test_attributes(self, oa, seom):
        assert oa.method == 'empirical'
        assert oa.w == WINDOW_SIZE
        assert oa.n_windows == N_WINDOWS
        assert oa.state_names == ['g', 'd']
        assert oa.sensor_names == ['r']
        assert oa.time_steps == list(range(WINDOW_SIZE))
        np.testing.assert_array_equal(oa.O_index, np.arange(N_WINDOWS))
        np.testing.assert_allclose(oa.O_time, seom.t_sim[:N_WINDOWS])
        assert isinstance(oa.source, pybounds.SlidingEmpiricalObservabilityMatrix)
        assert set(oa.window_data) == {'t', 'u', 'y', 'y_plus', 'y_minus'}
        assert oa.dxdz_sliding is None
        for O_a, O_b in zip(oa.O_df_sliding, seom.O_df_sliding):
            pd.testing.assert_frame_equal(O_a, O_b)


class TestSelections:

    @pytest.mark.parametrize('kwargs', [{'states': ['d']}, {'states': ['d', 'g']}, {'time_steps': [0, 2, 4]},
                                        {'states': ['g'], 'time_steps': [1, 2, 3]}])
    def test_matches_sliding_fisher(self, oa, seom, kwargs):
        expected = pybounds.SlidingFisherObservability(seom.O_df_sliding, time=seom.t_sim, R={'r': 0.1},
                                                       lam=1e-8, **kwargs).get_minimum_error_variance()
        pd.testing.assert_frame_equal(oa.min_error_variance(**kwargs), expected)

    def test_state_selection_is_conditional(self, oa):
        """Selecting one state treats the other as known, so its variance is lower than jointly."""
        joint = oa.min_error_variance()['d'].dropna()
        conditional = oa.min_error_variance(states=['d'])['d'].dropna()
        assert np.all(conditional.values < joint.values)

    @pytest.mark.parametrize('sensors', [['r'], ['a'], ['a', 'r']])
    def test_sensor_selection(self, sensors):
        sliding = _two_sensor_sliding()
        R = {'r': 0.1, 'a': 2.0}
        oa = ObservabilityAnalysis.from_sliding(sliding, R=R)
        expected = pybounds.SlidingFisherObservability(sliding.O_df_sliding, time=sliding.t_sim, R=R,
                                                       sensors=sensors).get_minimum_error_variance()
        pd.testing.assert_frame_equal(oa.min_error_variance(sensors=sensors), expected)

    def test_single_string_accepted(self, oa):
        pd.testing.assert_frame_equal(oa.min_error_variance(states='d'), oa.min_error_variance(states=['d']))


class TestMethodsAndOptions:

    def test_auto_method_for_simulator(self, simulator, trajectory):
        assert ObservabilityAnalysis(simulator, *trajectory).method == 'empirical'

    def test_auto_method_for_jax_simulator(self, trajectory):
        pytest.importorskip('jax')
        jax_sim = pybounds.JaxSimulator(lambda x, u: np.array([u[0], 0.0 * u[0]]), lambda x, u: x[:1] / x[1:],
                                        dt=0.01, state_names=['g', 'd'], input_names=['u'], measurement_names=['r'])
        assert ObservabilityAnalysis(jax_sim, *trajectory).method == 'jax'

    def test_unknown_method_raises(self, simulator, trajectory):
        with pytest.raises(ValueError, match="unknown method 'bogus'; valid methods: \\['empirical', 'jax'\\]"):
            ObservabilityAnalysis(simulator, *trajectory, method='bogus')

    def test_unsupported_option_raises(self, simulator, trajectory):
        with pytest.raises(TypeError, match=r"method 'jax' does not accept: \['eps'\]"):
            ObservabilityAnalysis(simulator, *trajectory, method='jax', eps=1e-4)
        with pytest.raises(TypeError, match=r"does not accept: \['epsilon'\].*valid options"):
            ObservabilityAnalysis(simulator, *trajectory, epsilon=1e-4)
        oa = ObservabilityAnalysis(simulator, *trajectory)
        with pytest.raises(TypeError, match='does not accept'):
            oa.update_settings(epsilon=1e-4)

    def test_none_removes_method_option(self, simulator, trajectory):
        oa = ObservabilityAnalysis(simulator, *trajectory, eps=1e-4)
        oa.update_settings(eps=None)
        assert 'eps' not in oa.settings

    def test_declared_options_match_native_signatures(self):
        params = inspect.signature(pybounds.SlidingEmpiricalObservabilityMatrix.__init__).parameters
        assert analysis._BUILDERS['empirical'].options <= set(params)
        pytest.importorskip('jax')
        params = inspect.signature(pybounds.JaxSlidingEmpiricalObservabilityMatrix.__init__).parameters
        assert analysis._BUILDERS['jax'].options <= set(params)

    def test_options_are_forwarded(self, trajectory):
        """parallel_sliding with a thread-safe custom simulator gives the sequential result."""
        seq = ObservabilityAnalysis(AnalyticSimulator(), *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1).run()
        par = ObservabilityAnalysis(AnalyticSimulator(), *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1,
                                    parallel_sliding=True).run()
        assert par.source.parallel_sliding is True
        pd.testing.assert_frame_equal(par.min_error_variance(), seq.min_error_variance())

    def test_builder_default_eps_applies(self, simulator, trajectory):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE).run()
        assert oa.source.eps == 1e-5


class TestRAndLam:

    def test_query_overrides_settings(self, oa, seom):
        expected = pybounds.SlidingFisherObservability(seom.O_df_sliding, time=seom.t_sim, R=0.5,
                                                       lam=1e-6).get_minimum_error_variance()
        pd.testing.assert_frame_equal(oa.min_error_variance(R=0.5, lam=1e-6), expected)

    def test_R_none_means_identity(self, oa, seom):
        with pytest.warns(UserWarning, match='R not set'):
            ev = oa.min_error_variance(R=None)
        with pytest.warns(UserWarning, match='R not set'):
            expected = pybounds.SlidingFisherObservability(seom.O_df_sliding, time=seom.t_sim, R=None,
                                                           lam=1e-8).get_minimum_error_variance()
        pd.testing.assert_frame_equal(ev, expected)

    def test_lam_limit(self):
        sliding = _two_sensor_sliding(n_windows=2)
        oa = ObservabilityAnalysis.from_sliding(sliding, R=0.1, lam='limit')
        expected = pybounds.FisherObservability(sliding.O_df_sliding[0], R=0.1, lam='limit').error_variance
        np.testing.assert_allclose(oa.fisher().FO[0].error_variance.values, expected.values)

    def test_force_R_scalar(self, oa):
        pd.testing.assert_frame_equal(oa.min_error_variance(R=0.1, force_R_scalar=True),
                                      oa.min_error_variance(R=0.1))
        with pytest.raises(Exception, match='R must be a scalar'):
            oa.min_error_variance(force_R_scalar=True)   # settings R is a dict

    def test_matrix_R_is_correct_but_not_cached(self, oa_fresh):
        expected = oa_fresh.min_error_variance(R=0.1)
        n_cached = len(oa_fresh._cache)
        pd.testing.assert_frame_equal(oa_fresh.min_error_variance(R=0.1 * np.eye(WINDOW_SIZE)), expected)
        assert len(oa_fresh._cache) == n_cached


class TestValidation:

    def test_unknown_state(self, oa):
        with pytest.raises(ValueError, match=r"unknown states \['x'\]; available states: \['g', 'd'\]"):
            oa.min_error_variance(states=['x'])

    def test_unknown_sensor(self, oa):
        with pytest.raises(ValueError, match=r"unknown sensors \['q'\]"):
            oa.min_error_variance(sensors='q')

    def test_unknown_time_step(self, oa):
        with pytest.raises(ValueError, match=r'unknown time_steps \[6\]'):
            oa.min_error_variance(time_steps=[0, 6])

    @pytest.mark.parametrize('states', [[], ['g', 'g']])
    def test_empty_or_duplicate(self, oa, states):
        with pytest.raises(ValueError, match='must not be empty|duplicate'):
            oa.min_error_variance(states=states)

    def test_dict_R_missing_sensor(self):
        oa = ObservabilityAnalysis.from_sliding(_two_sensor_sliding(), R={'r': 0.1})
        with pytest.raises(ValueError, match=r"R has no noise level for sensors \['a'\]"):
            oa.min_error_variance()
        oa.min_error_variance(sensors=['r'])   # fine when 'a' is not selected


class TestZFunction:

    def test_matches_empirical_z(self, simulator, trajectory):
        kwargs = dict(w=WINDOW_SIZE, eps=EPS, z_function=z_optic_flow, z_state_names=['q', 'd'])
        oa = ObservabilityAnalysis(simulator, *trajectory, R={'r': 0.1}, **kwargs).run()
        seom_z = pybounds.SlidingEmpiricalObservabilityMatrix(simulator, *trajectory, **kwargs)
        for O_a, O_b in zip(oa.O_df_sliding, seom_z.O_df_sliding):
            pd.testing.assert_frame_equal(O_a, O_b)
        assert oa.state_names == ['q', 'd']
        assert len(oa.dxdz_sliding) == N_WINDOWS
        oa.min_error_variance(states=['q'])
        with pytest.raises(ValueError, match='z_function is set, so states use the transformed names'):
            oa.min_error_variance(states=['g'])

    def test_matches_jax_z(self, trajectory):
        jax = pytest.importorskip('jax')
        jnp = jax.numpy
        jax_sim = pybounds.JaxSimulator(lambda x, u: jnp.array([u[0], 0.0 * u[0]]), lambda x, u: jnp.array([x[0] / x[1]]),
                                        dt=0.01, state_names=['g', 'd'], input_names=['u'], measurement_names=['r'])
        kwargs = dict(w=WINDOW_SIZE, z_function=z_optic_flow, z_state_names=['q', 'd'])
        oa = ObservabilityAnalysis(jax_sim, *trajectory, **kwargs).run()
        jax_z = pybounds.JaxSlidingEmpiricalObservabilityMatrix(jax_sim, *trajectory, **kwargs)
        for O_a, O_b in zip(oa.O_df_sliding, jax_z.O_df_sliding):
            np.testing.assert_allclose(O_a.values, O_b.values)
            assert list(O_a.columns) == list(O_b.columns)

    def test_missing_z_state_names_warns(self, simulator, trajectory):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, z_function=z_optic_flow)
        with pytest.warns(UserWarning, match='without z_state_names'):
            oa.run()


class TestCaching:

    def test_returns_copies(self, oa_fresh):
        first = oa_fresh.min_error_variance()
        first.iloc[:, :] = -1.0
        second = oa_fresh.min_error_variance()
        assert not (second[['g', 'd']] == -1.0).any().any()

    def test_state_order_is_part_of_the_key(self, oa_fresh):
        a = oa_fresh.min_error_variance(states=['g', 'd'])
        b = oa_fresh.min_error_variance(states=['d', 'g'])
        assert list(a.columns[-2:]) == ['g', 'd']
        assert list(b.columns[-2:]) == ['d', 'g']

    def test_cache_hits(self, oa_fresh, monkeypatch):
        oa_fresh.min_error_variance(states=['d'])
        calls = []
        monkeypatch.setattr(oa_fresh, 'fisher', lambda *a, **k: calls.append(1))
        oa_fresh.min_error_variance(states=['d'])
        assert calls == []
        oa_fresh.clear_cache()
        with pytest.raises(AttributeError):   # the stubbed fisher() is called again after clearing
            oa_fresh.min_error_variance(states=['d'])


class TestWindows:

    def test_w_none_single_window(self, simulator, trajectory):
        oa = ObservabilityAnalysis(simulator, *trajectory, R={'r': 0.1}).run()
        assert oa.n_windows == 1
        row = oa.min_error_variance().dropna(subset=['g'])
        assert len(row) == 1
        assert np.isclose(row['time'].item(), trajectory[0][N_STEPS_SLIDING // 2])

    def test_w_too_large_raises_at_run(self, simulator, trajectory):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=N_STEPS_SLIDING + 1)
        with pytest.raises(ValueError, match='window size'):
            oa.run()

    def test_non_consecutive_windows_rejected(self):
        sliding = _two_sensor_sliding()
        strided = SlidingO(sliding.O_df_sliding, sliding.t_sim, np.arange(0, 10, 2), sliding.w)
        with pytest.raises(NotImplementedError, match='O_index'):
            ObservabilityAnalysis.from_sliding(strided)


class TestAccessors:

    def test_observability_matrix_is_a_copy(self, oa_fresh):
        O = oa_fresh.observability_matrix(2)
        O.iloc[:, :] = 0.0
        assert oa_fresh.observability_matrix(2).abs().values.sum() > 0

    def test_observability_matrix_selection(self, oa):
        O = oa.observability_matrix(0, states=['d'], time_steps=[1, 2])
        assert O.shape == (2, 1)
        np.testing.assert_allclose(O.values[:, 0], oa.observability_matrix(0).values[1:3, 1])

    def test_out_of_range_window(self, oa):
        with pytest.raises(IndexError):
            oa.observability_matrix(N_WINDOWS)

    def test_plot(self, oa):
        image = oa.plot_observability_matrix(1, states=['g', 'd'])
        image.fig.canvas.draw()
        assert image.ax is not None
        plt.close(image.fig)
        fig, ax = plt.subplots()
        image = oa.plot_observability_matrix(1, ax=ax, cmap='viridis')
        assert image.ax is ax and image.fig is None
        plt.close(fig)

    def test_from_sliding_object(self, seom):
        oa = ObservabilityAnalysis.from_sliding(seom, R={'r': 0.1})
        assert oa.is_computed and oa.method == 'external'
        assert oa.run() is oa
        expected = pybounds.SlidingFisherObservability(seom.O_df_sliding, time=seom.t_sim, R={'r': 0.1},
                                                       lam=1e-8).get_minimum_error_variance()
        pd.testing.assert_frame_equal(oa.min_error_variance(), expected)
        with pytest.raises(ValueError, match='only R and lam can be changed'):
            oa.update_settings(w=3)
        oa.update_settings(lam=1e-6)

    def test_repr(self, simulator, trajectory):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE)
        assert repr(oa) == "ObservabilityAnalysis(method='empirical', w=6, not computed)"


def test_works_without_jax():
    code = ("import sys; sys.modules['jax'] = None\n"
            "import numpy as np, pybounds\n"
            "sim = pybounds.Simulator(lambda X, U: [U[0], 0*U[0]], lambda X, U: [X[0]/X[1]], dt=0.01,\n"
            "                         state_names=['g','d'], input_names=['u'], measurement_names=['r'])\n"
            "t, x, u, _ = sim.simulate(x0={'g': 2., 'd': 3.}, u={'u': 0.1*np.ones(10)}, return_full_output=True)\n"
            "oa = pybounds.ObservabilityAnalysis(sim, t, x, u, w=4, R=0.1)\n"
            "assert oa.method == 'empirical'\n"
            "oa.run().min_error_variance()\n"
            "try:\n"
            "    pybounds.ObservabilityAnalysis(sim, t, x, u, w=4, method='jax').run()\n"
            "except ImportError as e:\n"
            "    assert 'pip install jax' in str(e)\n"
            "else:\n"
            "    raise AssertionError('expected ImportError')\n")
    out = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
