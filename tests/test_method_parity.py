"""Parity between the bounds-* and stochastic-* methods: whatever model or option works with the bounds methods and
can work with the stochastic ones does, and with linearization='flow' the stochastic Gramians reproduce the bounds
methods' Fisher information as Q -> 0 (computed exactly with stochastic.deterministic_observability_gramian)."""

import numpy as np
import pandas as pd
import pytest

import pybounds
from pybounds import stochastic
from pybounds.analysis import ObservabilityAnalysis, Linearization

casadi = pytest.importorskip('casadi')

N, W = 30, 8


def deterministic_fisher(oa, R=1.0):
    """O^T R^-1 O of every window of a stochastic analysis' linearization (the exact Q = 0 Gramian)."""
    lin = oa.linearization
    Rinv = [np.eye(lin.C.shape[1]) / R] * oa.w
    return np.stack([stochastic.deterministic_observability_gramian(lin.Phi[k:k + oa.w], lin.C[k:k + oa.w], Rinv)
                     for k in range(oa.n_windows)])


def max_relative(a, b):
    return float(np.max(np.abs(a - b).max(axis=(1, 2)) / np.abs(b).max(axis=(1, 2))))


def pendulum(k=10.0, dt=0.02, use_casadi=False, n_steps=N):
    sin = casadi.sin if use_casadi else np.sin
    f = lambda X, U: [X[1], -k * sin(X[0]) + U[0]]
    h = lambda X, U: [X[0], X[0] + 0.5 * X[1]]
    sim = pybounds.Simulator(f, h, dt=dt, state_names=['th', 'om'], input_names=['u'], measurement_names=['a', 'b'])
    t, x, u, _ = sim.simulate(x0={'th': 1.2, 'om': 0.0}, u={'u': 0.3 * np.sin(np.arange(n_steps) * 0.2)},
                              return_full_output=True)
    return sim, t, x, u


def bounds_and_stochastic(sim, t, x, u, kind='observability', backend='classic', **options):
    bounds_method = 'bounds-jax' if backend == 'jax' and pybounds._JAX_AVAILABLE and \
        isinstance(sim, getattr(pybounds, 'JaxSimulator', ())) else 'bounds-empirical'
    bounds_options = {} if bounds_method == 'bounds-jax' else {'eps': 1e-6}
    if 'aux_list' in options:
        bounds_options['aux_list'] = options['aux_list']
    bounds = ObservabilityAnalysis(sim, t, x, u, method=bounds_method, w=W, R=1.0, **bounds_options).run()
    stoch = ObservabilityAnalysis(sim, t, x, u, method=f'stochastic-{kind}-{backend}', w=W, R=1.0, Q=1e-4,
                                  **options).run()
    return bounds, stoch


class TestLinearizationParity:

    @pytest.mark.parametrize('use_casadi', [False, True])
    def test_continuous_simulator(self, use_casadi):
        """CasADi-written f works (it used to fail), and the flow linearization matches bounds-empirical."""
        bounds, stoch = bounds_and_stochastic(*pendulum(use_casadi=use_casadi))
        assert stoch.settings.get('linearization') is None   # automatic: 'flow' for a pybounds Simulator
        assert max_relative(deterministic_fisher(stoch), bounds.fisher_information()) < 1e-5

    def test_coarse_time_step_flow_vs_expm(self):
        """Why 'flow' is the default: expm(A dt) freezes A over each step, the integrator's sensitivity does not."""
        sim, t, x, u = pendulum(k=100.0, dt=0.05)
        bounds = ObservabilityAnalysis(sim, t, x, u, w=W, R=1.0, eps=1e-6).run().fisher_information()
        flow = ObservabilityAnalysis(sim, t, x, u, method='stochastic-observability-classic', w=W, R=1.0,
                                     linearization='flow').run()
        expm = ObservabilityAnalysis(sim, t, x, u, method='stochastic-observability-classic', w=W, R=1.0,
                                     linearization='expm').run()
        assert max_relative(deterministic_fisher(flow), bounds) < 1e-5
        assert max_relative(deterministic_fisher(expm), bounds) > 1e-2

    def test_discrete_simulator(self):
        """Regression: a discrete-time Simulator's update map was exponentiated as if it were continuous."""
        A = np.array([[0.9, 0.1], [-0.2, 0.95]])
        f = lambda X, U: [0.9 * X[0] + 0.1 * X[1], -0.2 * X[0] + 0.95 * X[1] + U[0] + 0.01 * X[0] * X[1]]
        h = lambda X, U: [X[0]]
        sim = pybounds.Simulator(f, h, dt=0.1, discrete=True, state_names=['a', 'b'], input_names=['u'],
                                 measurement_names=['y'])
        t, x, u, _ = sim.simulate(x0={'a': 1.0, 'b': 0.5}, u={'u': 0.1 * np.ones(N)}, return_full_output=True)
        bounds, stoch = bounds_and_stochastic(sim, t, x, u)
        Phi = stoch.linearization.Phi
        np.testing.assert_allclose(Phi[0], A + np.array([[0, 0], [0.01 * x['b'][0], 0.01 * x['a'][0]]]), rtol=1e-12)
        assert max_relative(deterministic_fisher(stoch), bounds.fisher_information()) < 1e-8
        with pytest.raises(ValueError, match="does not apply to a discrete-time model"):
            ObservabilityAnalysis(sim, t, x, u, method='stochastic-observability-classic', w=W,
                                  linearization='expm').run()

    def test_constructability_matrix_maps_the_final_state(self):
        sim, t, x, u = pendulum()
        con = ObservabilityAnalysis(sim, t, x, u, method='stochastic-constructability-classic', w=W, R=1.0,
                                    Q=1e-4).run()
        Phi = con.linearization.Phi
        k = 4
        transition = np.eye(2)
        for j in range(W - 1):
            transition = Phi[k + j] @ transition
        initial = stochastic.window_observability_matrix(Phi, con.linearization.C, k, W, 'initial').to_numpy()
        np.testing.assert_allclose(con.observability_matrix(k).to_numpy(), initial @ np.linalg.inv(transition),
                                   rtol=1e-9, atol=1e-12)

    def test_options_and_errors(self, tmp_path):
        sim, t, x, u = pendulum()
        oa = ObservabilityAnalysis(sim, t, x, u, method='stochastic-observability-classic', w=W, R=1.0, Q=1e-4,
                                   linearization='expm')
        loaded = ObservabilityAnalysis(sim, t, x, u).load_settings(oa.save_settings(tmp_path / 's.yaml'))
        assert loaded.settings['linearization'] == 'expm'
        path = tmp_path / 'old.yaml'   # a 0.3.0 settings file, without the key: automatic
        path.write_text('settings:\n  method: stochastic-observability-classic\n  w: 8\n  Q: 1.0e-4\n')
        old = ObservabilityAnalysis(sim, t, x, u, method='stochastic-observability-classic',
                                    linearization='expm').load_settings(path)
        assert 'linearization' not in old.settings
        with pytest.raises(ValueError, match='unknown linearization'):
            ObservabilityAnalysis(sim, t, x, u, method='stochastic-observability-classic', linearization='rk4').run()
        if pybounds._JAX_AVAILABLE:
            with pytest.raises(ValueError, match='cannot differentiate'):
                ObservabilityAnalysis(sim, t, x, u, method='stochastic-observability-jax',
                                      linearization='flow').run()


class CustomSimulator:
    """A simulator known only through f, h and dt (no integrator for the stochastic methods to differentiate)."""
    dt = 0.02
    state_names, input_names, measurement_names = ['g', 'd'], ['u'], ['r']

    @staticmethod
    def f(x, u, aux=None):
        return [u[0] - (0.3 if aux is None else aux) * x[0], 0 * u[0]]

    @staticmethod
    def h(x, u, aux=None):
        return [x[0] / x[1]]


class TestCustomSimulators:

    def _trajectory(self):
        t = 0.02 * np.arange(N)
        return t, np.column_stack([2 + 0.1 * np.sin(t), 3 + 0 * t]), np.ones((N, 1))

    def test_automatic_expm_and_flow_unavailable(self):
        t, x, u = self._trajectory()
        oa = ObservabilityAnalysis(CustomSimulator(), t, x, u, method='stochastic-observability-classic', w=W,
                                   R=0.1, Q=1e-4).run()
        from scipy.linalg import expm
        np.testing.assert_allclose(oa.linearization.Phi[0], expm(np.array([[-0.3, 0], [0, 0]]) * 0.02), rtol=1e-8)
        with pytest.raises(ValueError, match="integrator is unknown"):
            ObservabilityAnalysis(CustomSimulator(), t, x, u, method='stochastic-observability-classic',
                                  linearization='flow').run()

    def test_aux_list(self):
        t, x, u = self._trajectory()
        aux = [0.3 + 0.01 * k for k in range(N)]
        oa = ObservabilityAnalysis(CustomSimulator(), t, x, u, method='stochastic-observability-classic', w=W,
                                   R=0.1, Q=1e-4, aux_list=aux).run()
        from scipy.linalg import expm
        np.testing.assert_allclose(oa.linearization.Phi[10], expm(np.array([[-aux[10], 0], [0, 0]]) * 0.02),
                                   rtol=1e-8)

    def test_discrete_attribute(self):
        class Discrete(CustomSimulator):
            discrete = True

            @staticmethod
            def f(x, u, aux=None):
                return [0.9 * x[0] + u[0], x[1]]

        t, x, u = self._trajectory()
        oa = ObservabilityAnalysis(Discrete(), t, x, u, method='stochastic-observability-classic', w=W, R=0.1,
                                   Q=1e-4).run()
        np.testing.assert_allclose(oa.linearization.Phi[0], [[0.9, 0], [0, 1]], atol=1e-9)

    def test_measurement_only_simulator_is_rejected(self):
        class MeasurementsOnly:
            def simulate(self, x0, u, aux=None):
                return np.zeros((len(u), 1))

        t, x, u = self._trajectory()
        with pytest.raises(TypeError, match='process noise enters the state'):
            ObservabilityAnalysis(MeasurementsOnly(), t, x, u, method='stochastic-observability-classic').run()


class TestJaxSimulatorParity:

    @pytest.fixture(autouse=True)
    def _jax(self):
        pytest.importorskip('jax')

    @staticmethod
    def own_trajectory(sim, x0, u, aux=None):
        """The JaxSimulator's own state trajectory (bounds-jax re-simulates every window with it)."""
        import jax.numpy as jnp
        from pybounds.jax_simulator import _x64, _to_aux
        with _x64():
            xs = [jnp.asarray(x0, dtype=jnp.float64)]
            for k in range(len(u) - 1):
                xs.append(sim._step_jax(xs[-1], jnp.asarray(u[k]), None if aux is None else _to_aux(aux[k])))
            return np.array(xs)

    @pytest.mark.parametrize('integrator, substeps', [('rk4', 3), ('euler', 2)])
    @pytest.mark.parametrize('use_aux', [False, True])
    def test_flow_matches_bounds_jax(self, integrator, substeps, use_aux):
        import jax.numpy as jnp
        if use_aux:
            f = lambda x, u, aux: jnp.array([x[1], -aux['k'] * jnp.sin(x[0]) + u[0]])
            h = lambda x, u, aux: jnp.array([x[0], x[0] + 0.5 * x[1]])
        else:
            f = lambda x, u: jnp.array([x[1], -10.0 * jnp.sin(x[0]) + u[0]])
            h = lambda x, u: jnp.array([x[0], x[0] + 0.5 * x[1]])
        sim = pybounds.JaxSimulator(f, h, dt=0.02, state_names=['th', 'om'], input_names=['u'],
                                    measurement_names=['a', 'b'], integrator=integrator, substeps=substeps)
        u = (0.3 * np.sin(np.arange(N) * 0.2))[:, None]
        aux = [{'k': 10.0}] * N if use_aux else None
        x = self.own_trajectory(sim, [1.2, 0.0], u, aux)
        t = 0.02 * np.arange(N)
        options = {'aux_list': aux} if use_aux else {}
        bounds = ObservabilityAnalysis(sim, t, x, u, method='bounds-jax', w=W, R=1.0, **options).run()
        for backend, tolerance in (('jax', 1e-10), ('classic', 1e-6)):
            stoch = ObservabilityAnalysis(sim, t, x, u, method=f'stochastic-observability-{backend}', w=W, R=1.0,
                                          Q=1e-4, **options).run()
            assert max_relative(deterministic_fisher(stoch), bounds.fisher_information()) < tolerance, backend

    def test_trajectory_mismatch_is_a_real_difference(self):
        """The bounds methods re-simulate every window from its first state; the stochastic methods linearize at
        every given sample. On a trajectory that is not the model's own (here: IDAS instead of RK4), they differ."""
        import jax.numpy as jnp
        sim_jax = pybounds.JaxSimulator(lambda x, u: jnp.array([x[1], -100.0 * jnp.sin(x[0]) + u[0]]),
                                        lambda x, u: jnp.array([x[0]]), dt=0.05, state_names=['th', 'om'],
                                        input_names=['u'], measurement_names=['y'])
        f = lambda X, U: [X[1], -100.0 * np.sin(X[0]) + U[0]]
        sim = pybounds.Simulator(f, lambda X, U: [X[0]], dt=0.05, state_names=['th', 'om'], input_names=['u'],
                                 measurement_names=['y'])
        t, x, u, _ = sim.simulate(x0={'th': 1.5, 'om': 0.0}, u={'u': np.zeros(N)}, return_full_output=True)
        bounds = ObservabilityAnalysis(sim_jax, t, x, u, method='bounds-jax', w=W, R=1.0).run()
        stoch = ObservabilityAnalysis(sim_jax, t, x, u, method='stochastic-observability-jax', w=W, R=1.0,
                                      Q=1e-4).run()
        assert max_relative(deterministic_fisher(stoch), bounds.fisher_information()) > 1e-8


class TestMeasurementNoiseMatrix:

    @pytest.fixture(scope='class')
    def two_sensors(self):
        sim, t, x, u = pendulum()
        return ObservabilityAnalysis(sim, t, x, u, method='stochastic-observability-classic', w=W, R=1.0,
                                     Q=1e-4).run()

    def test_diagonal_matrix_equals_dict(self, two_sensors):
        oa = two_sensors
        expected = oa.min_error_variance(R={'a': 0.1, 'b': 0.2})
        pd.testing.assert_frame_equal(oa.min_error_variance(R=np.diag([0.1, 0.2])), expected, check_exact=True)
        index = pd.MultiIndex.from_arrays([['a', 'b'] * W, np.repeat(np.arange(W), 2)], names=['sensor', 'time_step'])
        full = pd.DataFrame(np.diag([0.1, 0.2] * W), index=index, columns=index)
        np.testing.assert_allclose(oa.fisher_information(R=full), oa.fisher_information(R={'a': 0.1, 'b': 0.2}),
                                   rtol=1e-12)
        shuffled = full.iloc[::-1, ::-1]   # labels, not positions, decide
        np.testing.assert_allclose(oa.fisher_information(R=shuffled), oa.fisher_information(R=full), rtol=1e-12)

    def test_correlated_sensors(self, two_sensors):
        oa = two_sensors
        R = np.array([[0.1, 0.04], [0.04, 0.2]])
        lin = oa.linearization
        expected = stochastic.sliding_gramians(lin.Phi, lin.C, W, np.linalg.inv(1e-4 * np.eye(2)),
                                               [np.linalg.inv(R)] * W, 'initial')
        np.testing.assert_allclose(oa.fisher_information(R=R), expected, rtol=1e-12)
        sub = oa.fisher_information(R=R, sensors=['b'])
        np.testing.assert_allclose(sub, oa.fisher_information(R={'b': 0.2}, sensors=['b']), rtol=1e-12)

    def test_time_varying_blocks(self, two_sensors):
        oa = two_sensors
        scales = np.linspace(1.0, 2.0, W)
        full = np.zeros((2 * W, 2 * W))
        for j, c in enumerate(scales):
            full[2 * j:2 * j + 2, 2 * j:2 * j + 2] = c * np.array([[0.1, 0.0], [0.0, 0.2]])
        lin = oa.linearization
        expected = stochastic.sliding_gramians(lin.Phi, lin.C, W, np.linalg.inv(1e-4 * np.eye(2)),
                                               [np.diag([1 / (0.1 * c), 1 / (0.2 * c)]) for c in scales], 'initial')
        np.testing.assert_allclose(oa.fisher_information(R=full), expected, rtol=1e-12)

    def test_noise_correlated_in_time_is_rejected(self, two_sensors):
        full = np.eye(2 * W) * 0.1
        full[0, 2] = full[2, 0] = 0.01
        with pytest.raises(ValueError, match='white in time'):
            two_sensors.min_error_variance(R=full)
        with pytest.raises(ValueError, match='a matrix R must be'):
            two_sensors.min_error_variance(R=np.eye(3))


class TestOutputs:

    @pytest.fixture(scope='class')
    def con(self):
        sim, t, x, u = pendulum()
        return ObservabilityAnalysis(sim, t, x, u, method='stochastic-constructability-classic', w=W, R=1.0, Q=1e-4,
                                     keep_source=True).run()

    def test_fisher_result(self, con):
        sliding = con.fisher(lam={'th': 1e-6, 'om': 1e-8}, alignment='bounded_state')
        pd.testing.assert_frame_equal(sliding.get_minimum_error_variance(),
                                      con.min_error_variance(lam={'th': 1e-6, 'om': 1e-8}, alignment='bounded_state'),
                                      check_exact=True)
        window = sliding.FO[3]
        F, F_inv, R = window.get_fisher_information()
        np.testing.assert_array_equal(F.to_numpy(), con.fisher_information()[3])
        assert list(window.error_variance.columns) == ['th', 'om'] and R.shape == (2 * W, 2 * W)
        assert sliding.shift_index == W - 1 and sliding.n_window == con.n_windows
        sub = con.fisher(states=['om'], sensors=['b'], time_steps=[0, 2])
        assert sub.FO[0].R.shape == (2, 2) and list(sub.FO[0].F.columns) == ['om']

    def test_matrices_and_export(self, con, tmp_path):
        frames = con.O_df_sliding
        assert len(frames) == con.n_windows
        pd.testing.assert_frame_equal(frames[-1], con.observability_matrix(-1))
        saved = np.load(con.save_results(tmp_path, include_observability_matrices=True)['observability_matrices'])
        np.testing.assert_array_equal(saved['O'], np.stack([f.to_numpy() for f in frames]))
        np.testing.assert_array_equal(saved['C'], con.linearization.C)
        assert list(saved['model_state_names']) == ['th', 'om'] and list(saved['state_names']) == ['th', 'om']

    def test_source_is_the_linearization(self, con):
        assert isinstance(con.source, Linearization) and con.source is con.linearization
        assert con.window_data is None   # no perturbed simulations exist for the stochastic methods
