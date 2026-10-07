"""Stochastic observability / constructability (pybounds.stochastic) and their ObservabilityAnalysis methods."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sympy as sp

import pybounds
from pybounds import analysis, stochastic
from pybounds.analysis import ObservabilityAnalysis
from conftest import N_STEPS_SLIDING, WINDOW_SIZE, EPS

N_WINDOWS = N_STEPS_SLIDING - WINDOW_SIZE + 1
STOCHASTIC = ['stochastic-observability-classic', 'stochastic-observability-jax',
              'stochastic-constructability-classic', 'stochastic-constructability-jax']
inv = np.linalg.inv


# ---------------------------------------------------------------------------------------------
# The duality letter's own LTV system (Burak's observabilityExample.m), and verbatim MATLAB ports
# ---------------------------------------------------------------------------------------------

def letter_system(N=30):
    A = np.array([[2.0, -1.0], [0.0, 1.0]])
    Phis = [A + np.array([[0.0, np.sin(np.pi * k / 18)], [np.cos(np.pi * k / 18), 0.0]]) for k in range(N + 1)]
    Cs = [np.array([[1.0, 0.0]])] * (N + 1)
    Qs = [1e-2 * np.array([[3.6, 1.2], [1.2, 6.0]])] * (N + 1)
    Rs = [np.array([[0.1]])] * (N + 1)
    return Phis, Cs, Qs, Rs


def matlab_observability_gram(As, Cs, Qs, Rs, w):
    """code/stochObservabilityGram.m, 0-indexed."""
    F = Cs[w - 1].T @ (inv(Rs[w - 1]) @ Cs[w - 1])
    for i in range(1, w):
        A, Q = As[w - 1 - i], Qs[w - 1 - i]
        F = (A.T @ inv(Q) @ A - A.T @ inv(Q) @ inv(F + inv(Q)) @ inv(Q) @ A
             + Cs[w - 1 - i].T @ (inv(Rs[w - 1 - i]) @ Cs[w - 1 - i]))
    return F


def matlab_constructability_gram(As, Cs, Qs, Rs, w):
    """code/stochConstructabilityGram.m, 0-indexed, final iterate of a w-sample window."""
    F = Cs[0].T @ (inv(Rs[0]) @ Cs[0])
    for i in range(w - 1):
        A, Q = As[i], Qs[i]
        F = (-inv(Q) @ A @ inv(F + A.T @ inv(Q) @ A) @ A.T @ inv(Q) + inv(Q)
             + Cs[i + 1].T @ (inv(Rs[i + 1]) @ Cs[i + 1]))
    return F


def _relative(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b))


class TestRecursions:

    @pytest.mark.parametrize('w', [1, 2, 5, 11, 25, 31])
    def test_match_matlab(self, w):
        Phis, Cs, Qs, Rs = letter_system()
        Qinvs, Rinvs = [inv(Q) for Q in Qs[:w]], [inv(R) for R in Rs[:w]]
        F_obs = stochastic.stochastic_observability_gramian(Phis[:w], Cs[:w], Qinvs, Rinvs)
        F_con = stochastic.stochastic_constructability_gramian(Phis[:w], Cs[:w], Qinvs, Rinvs)
        assert _relative(F_obs, matlab_observability_gram(Phis, Cs, Qs, Rs, w)) < 1e-12
        assert _relative(F_con, matlab_constructability_gram(Phis, Cs, Qs, Rs, w)) < 1e-12

    def test_converged_value_of_figure_2(self):
        Phis, Cs, Qs, Rs = letter_system()
        F = stochastic.stochastic_observability_gramian(Phis, Cs, [inv(Q) for Q in Qs], [inv(R) for R in Rs])
        np.testing.assert_allclose(F, [[76.931317, -36.700184], [-36.700184, 44.543350]], atol=1e-5)

    @pytest.mark.parametrize('w', [2, 11, 31])
    def test_duality(self, w):
        Phis, Cs, Qs, Rs = letter_system()
        forward, dual = stochastic.duality_check(Phis[:w], Cs[:w], Qs[:w], Rs[:w])
        assert _relative(forward, dual) < 1e-11

    @pytest.mark.parametrize('bounded', ['initial', 'final'])
    def test_batched_equals_per_window(self, bounded):
        Phis, Cs, Qs, Rs = letter_system()
        w, Qinv, Rinvs = 6, inv(Qs[0]), [inv(Rs[0])] * 6
        gramian = (stochastic.stochastic_observability_gramian if bounded == 'initial'
                   else stochastic.stochastic_constructability_gramian)
        batched = stochastic.sliding_gramians(np.stack(Phis), np.stack(Cs), w, Qinv, Rinvs, bounded=bounded)
        per_window = np.stack([gramian(Phis[k:k + w], Cs[k:k + w], [Qinv] * w, Rinvs)
                               for k in range(len(Phis) - w + 1)])
        np.testing.assert_allclose(batched, per_window, rtol=1e-12, atol=0)

    def test_constructability_chains_with_F0(self):
        Phis, Cs, Qs, Rs = letter_system()
        Qinvs, Rinvs = [inv(Q) for Q in Qs], [inv(R) for R in Rs]
        full = stochastic.stochastic_constructability_gramian(Phis[:10], Cs[:10], Qinvs[:10], Rinvs[:10])
        first = stochastic.stochastic_constructability_gramian(Phis[:5], Cs[:5], Qinvs[:5], Rinvs[:5])
        chained = stochastic.stochastic_constructability_gramian(Phis[4:10], Cs[4:10], Qinvs[4:10], Rinvs[4:10],
                                                                 F0=first)
        assert _relative(full, chained) < 1e-12

    def test_deterministic_limit(self):
        Phis, Cs, Qs, Rs = letter_system()
        w = 5
        Rinvs = [inv(R) for R in Rs[:w]]
        transitions = [np.eye(2)]
        for j in range(w - 1):
            transitions.append(Phis[j] @ transitions[-1])
        O = np.vstack([Cs[j] @ transitions[j] for j in range(w)])
        np.testing.assert_allclose(stochastic.deterministic_observability_gramian(Phis[:w], Cs[:w], Rinvs),
                                   O.T @ O / 0.1, rtol=1e-12)
        small = stochastic.stochastic_observability_gramian(Phis[:w], Cs[:w], [np.eye(2) * 1e8] * w, Rinvs)
        assert _relative(small, O.T @ O / 0.1) < 1e-4

    def test_process_covariance(self):
        Q = stochastic.process_covariance(1e-2, ['a', 'b', 'c'], overrides={'b': 1e-6})
        np.testing.assert_array_equal(np.diag(Q), [1e-2, 1e-6, 1e-2])
        with pytest.raises(ValueError, match='not a state'):
            stochastic.process_covariance(1e-2, ['a'], overrides={'z': 1.0})
        with pytest.raises(ValueError, match='strictly positive'):
            stochastic.process_covariance(0.0, ['a'])
        with pytest.warns(RuntimeWarning, match='decades'):
            stochastic.process_covariance(1.0, ['a', 'b'], overrides={'b': 1e-14})

    def test_linearize_lti(self):
        A = np.array([[0.0, 1.0], [-2.0, -0.3]])
        f = lambda x, u: A @ x + np.array([0.0, u[0]])
        h = lambda x, u: [x[0]]
        Phi, C = stochastic.linearize(f, h, np.ones((4, 2)), np.zeros((4, 1)), dt=0.1)
        from scipy.linalg import expm
        np.testing.assert_allclose(Phi[2], expm(A * 0.1), rtol=1e-8)
        np.testing.assert_allclose(C[2], [[1.0, 0.0]], atol=1e-10)


# ---------------------------------------------------------------------------------------------
# ObservabilityAnalysis
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def trajectory(simulation_output):
    t_sim, x_sim, u_sim, _ = simulation_output
    return (t_sim[:N_STEPS_SLIDING],
            {k: v[:N_STEPS_SLIDING] for k, v in x_sim.items()},
            {k: v[:N_STEPS_SLIDING] for k, v in u_sim.items()})


@pytest.fixture(scope='module')
def bounds(simulator, trajectory):
    return ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1).run()


@pytest.fixture(scope='module')
def obs(simulator, trajectory):
    return ObservabilityAnalysis(simulator, *trajectory, method='stochastic-observability-classic',
                                 w=WINDOW_SIZE, R=0.1, Q=1e-4).run()


@pytest.fixture(scope='module')
def con(simulator, trajectory):
    return ObservabilityAnalysis(simulator, *trajectory, method='stochastic-constructability-classic',
                                 w=WINDOW_SIZE, R=0.1, Q=1e-4).run()


def _states(ev):
    return ev.drop(columns=['time', 'time_initial'])


class TestMethods:

    def test_method_names(self):
        assert list(analysis._BUILDERS) == ['bounds-empirical', 'bounds-jax'] + [
            'stochastic-observability-classic', 'stochastic-observability-jax',
            'stochastic-constructability-classic', 'stochastic-constructability-jax']

    @pytest.mark.parametrize('alias, method', [('empirical', 'bounds-empirical'), ('jax', 'bounds-jax')])
    def test_aliases(self, simulator, trajectory, alias, method):
        oa = ObservabilityAnalysis(simulator, *trajectory, method=alias)
        assert oa.method == method
        oa.update_settings(method='empirical')
        assert oa.method == 'bounds-empirical'
        assert oa._settings_document()['settings']['method'] == 'bounds-empirical'

    def test_alias_gives_same_result(self, simulator, trajectory, bounds):
        oa = ObservabilityAnalysis(simulator, *trajectory, method='empirical', w=WINDOW_SIZE, eps=EPS, R=0.1).run()
        pd.testing.assert_frame_equal(oa.min_error_variance(), bounds.min_error_variance())

    def test_old_settings_file_loads(self, simulator, trajectory, tmp_path):
        path = tmp_path / 's.yaml'
        path.write_text("settings: {method: empirical, w: 6, R: 0.1, lam: 1.0e-8}\n")
        oa = ObservabilityAnalysis(simulator, *trajectory, method='stochastic-observability-classic')
        assert oa.load_settings(path).method == 'bounds-empirical'

    @pytest.mark.parametrize('method', STOCHASTIC)
    def test_runs(self, simulator, trajectory, method):
        if method.endswith('-jax'):
            pytest.importorskip('jax')
        oa = ObservabilityAnalysis(simulator, *trajectory, method=method, w=WINDOW_SIZE, R=0.1, Q=1e-4).run()
        assert oa.n_windows == N_WINDOWS
        assert oa.state_names == ['g', 'd'] and oa.sensor_names == ['r']
        assert oa.time_steps == list(range(WINDOW_SIZE))
        ev = oa.min_error_variance()
        assert list(ev.columns) == ['time', 'time_initial', 'g', 'd']
        assert len(ev) == N_STEPS_SLIDING and _states(ev).notna().sum().tolist() == [N_WINDOWS] * 2
        assert oa.fisher_information().shape == (N_WINDOWS, 2, 2)

    @pytest.mark.parametrize('kind', ['observability', 'constructability'])
    def test_classic_matches_jax(self, simulator, trajectory, kind):
        pytest.importorskip('jax')
        results = [ObservabilityAnalysis(simulator, *trajectory, method=f'stochastic-{kind}-{backend}',
                                         w=WINDOW_SIZE, R=0.1, Q=1e-4).run().fisher_information()
                   for backend in ('classic', 'jax')]
        np.testing.assert_allclose(results[0], results[1], rtol=1e-6)

    def test_jax_with_jax_simulator(self, trajectory, obs):
        pytest.importorskip('jax')
        jax_sim = pybounds.JaxSimulator(lambda x, u: [u[0], 0.0 * u[0]], lambda x, u: [x[0] / x[1]], dt=0.01,
                                        state_names=['g', 'd'], input_names=['u'], measurement_names=['r'])
        for method in ('stochastic-observability-jax', 'stochastic-observability-classic'):
            oa = ObservabilityAnalysis(jax_sim, *trajectory, method=method, w=WINDOW_SIZE, R=0.1, Q=1e-4).run()
            np.testing.assert_allclose(oa.fisher_information(), obs.fisher_information(), rtol=1e-6)

    def test_deterministic_limit_matches_bounds(self, obs, bounds):
        """With Q = 0 the linearized Gramian is O^T R^-1 O; this model's dynamics are linear, so Phi is exact."""
        F = np.stack([stochastic.deterministic_observability_gramian(obs._Phi[k:k + WINDOW_SIZE],
                                                                     obs._C[k:k + WINDOW_SIZE],
                                                                     [np.eye(1) / 0.1] * WINDOW_SIZE)
                      for k in range(N_WINDOWS)])
        np.testing.assert_allclose(F, bounds.fisher_information(), rtol=1e-6)
        np.testing.assert_allclose(obs.fisher_information(Q=1e-7), bounds.fisher_information(), rtol=1e-4)

    def test_process_noise_lowers_information(self, obs, con, bounds):
        F_bounds = bounds.fisher_information()
        for oa in (obs, con):
            ev, ev_bounds = _states(oa.min_error_variance()), _states(bounds.min_error_variance())
            assert (ev.dropna().to_numpy() >= ev_bounds.dropna().to_numpy() * (1 - 1e-9)).all()
            assert not np.allclose(oa.fisher_information(), F_bounds)

    def test_aux_list_unsupported(self, simulator, trajectory):
        with pytest.raises(TypeError, match="does not accept: \\['aux_list'\\]"):
            ObservabilityAnalysis(simulator, *trajectory, method='stochastic-observability-classic',
                                  aux_list=[None] * N_STEPS_SLIDING)

    def test_eps_is_classic_only(self, simulator, trajectory):
        ObservabilityAnalysis(simulator, *trajectory, method='stochastic-observability-classic', eps=1e-6)
        with pytest.raises(TypeError, match="does not accept: \\['eps'\\]"):
            ObservabilityAnalysis(simulator, *trajectory, method='stochastic-observability-jax', eps=1e-6)

    @pytest.mark.parametrize('storage', ['fisher', 'fisher_per_sensor'])
    def test_storage_must_be_observability(self, simulator, trajectory, storage):
        with pytest.raises(ValueError, match='does not apply to method'):
            ObservabilityAnalysis(simulator, *trajectory, method='stochastic-observability-classic', storage=storage)
        oa = ObservabilityAnalysis(simulator, *trajectory, storage=storage)
        with pytest.raises(ValueError, match='does not apply to method'):
            oa.update_settings(method='stochastic-constructability-classic')

    def test_simulator_without_model_functions(self, trajectory):
        from conftest import AnalyticSimulator
        with pytest.raises(TypeError, match='callable f'):
            ObservabilityAnalysis(AnalyticSimulator(), *trajectory, method='stochastic-observability-classic').run()

    def test_full_trajectory_window(self, simulator, trajectory):
        oa = ObservabilityAnalysis(simulator, *trajectory, method='stochastic-constructability-classic', R=0.1,
                                   Q=1e-4).run()
        assert oa.n_windows == 1 and oa.w == N_STEPS_SLIDING


class TestQueries:

    def test_query_settings_do_not_rerun(self, simulator, trajectory, monkeypatch):
        calls = []
        original = analysis._BUILDERS['stochastic-observability-classic']

        def func(*args, **kwargs):
            calls.append(kwargs)
            return original.func(*args, **kwargs)

        monkeypatch.setitem(analysis._BUILDERS, 'stochastic-observability-classic',
                            analysis._Builder(func, original.options))
        oa = ObservabilityAnalysis(simulator, *trajectory, method='stochastic-observability-classic',
                                   w=WINDOW_SIZE, R=0.1, Q=1e-4).run()
        first = oa.min_error_variance()
        oa.update_settings(Q=1e-2, R={'r': 0.2}, lam=1e-6, alignment='bounded_state')
        assert oa.is_computed
        changed = oa.min_error_variance()
        oa.min_error_variance(Q=1e-3)
        assert len(calls) == 1
        assert not np.allclose(_states(first).dropna(), _states(changed).dropna())

    def test_per_query_values_match_settings(self, simulator, trajectory, obs):
        other = ObservabilityAnalysis(simulator, *trajectory, method='stochastic-observability-classic',
                                      w=WINDOW_SIZE, R={'r': 0.2}, Q=1e-3, lam=1e-6).run()
        pd.testing.assert_frame_equal(obs.min_error_variance(R={'r': 0.2}, Q=1e-3, lam=1e-6),
                                      other.min_error_variance())

    def test_Q_forms_agree(self, obs):
        expected = obs.fisher_information()
        np.testing.assert_array_equal(obs.fisher_information(Q={'g': 1e-4, 'd': 1e-4}), expected)
        np.testing.assert_array_equal(obs.fisher_information(Q=1e-4 * np.eye(2)), expected)
        np.testing.assert_array_equal(
            obs.fisher_information(Q=pd.DataFrame(1e-4 * np.eye(2), index=['g', 'd'], columns=['g', 'd'])), expected)
        assert not np.allclose(obs.fisher_information(Q={'g': 1e-4, 'd': 1e-8}), expected)

    def test_Q_errors(self, simulator, trajectory, obs, bounds):
        with pytest.raises(ValueError, match='needs the process noise covariance Q'):
            ObservabilityAnalysis(simulator, *trajectory, method='stochastic-observability-classic',
                                  w=WINDOW_SIZE, R=0.1).run().min_error_variance()
        with pytest.raises(ValueError, match="missing \\['d'\\]"):
            obs.min_error_variance(Q={'g': 1e-4})
        with pytest.raises(ValueError, match='positive definite'):
            obs.min_error_variance(Q=np.diag([1e-4, -1.0]))
        with pytest.raises(ValueError, match='strictly positive'):
            obs.min_error_variance(Q=0.0)
        with pytest.raises(ValueError, match='only applies to the stochastic methods'):
            bounds.min_error_variance(Q=1e-4)

    def test_Q_setting_ignored_by_bounds_methods(self, simulator, trajectory, bounds):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1, Q=1e-4).run()
        pd.testing.assert_frame_equal(oa.min_error_variance(), bounds.min_error_variance())

    def test_Q_per_state_vector_and_series(self, obs):
        """Regression: a 1-D Q fell through to float() and raised an unrelated TypeError."""
        expected = obs.fisher_information(Q={'g': 1e-4, 'd': 1e-6})
        np.testing.assert_array_equal(obs.fisher_information(Q=np.array([1e-4, 1e-6])), expected)
        np.testing.assert_array_equal(obs.fisher_information(Q=[1e-4, 1e-6]), expected)
        np.testing.assert_array_equal(obs.fisher_information(Q=pd.Series({'d': 1e-6, 'g': 1e-4})), expected)
        with pytest.raises(ValueError, match='one variance per model state'):
            obs.min_error_variance(Q=np.array([1e-4, 1e-6, 1e-6]))
        with pytest.raises(ValueError, match='strictly positive'):
            obs.min_error_variance(Q=np.array([1e-4, 0.0]))
        with pytest.raises(ValueError, match=r'Q must be a scalar, a dict or 1-D array'):
            obs.min_error_variance(Q='large')

    @pytest.mark.parametrize('Q', [pd.DataFrame(np.diag([1e-4, 1e-6]), index=['g', 'd'], columns=['g', 'd']),
                                   pd.DataFrame([[1e-6, 0.0], [0.0, 1e-4]], index=['d', 'g'], columns=['d', 'g']),
                                   np.diag([1e-4, 1e-6]), np.array([1e-4, 1e-6]), {'g': 1e-4, 'd': 1e-6}, 1e-4])
    def test_Q_yaml_roundtrip(self, simulator, trajectory, obs, tmp_path, Q):
        """Regression: a DataFrame Q was written with R's (sensor, time_step) labels and could not be loaded."""
        import yaml
        oa = ObservabilityAnalysis(simulator, *trajectory, method='stochastic-observability-classic',
                                   w=WINDOW_SIZE, R=0.1, Q=Q)
        loaded = ObservabilityAnalysis(simulator, *trajectory).load_settings(oa.save_settings(tmp_path / 's.yaml'))
        loaded_Q = loaded.settings['Q']
        if isinstance(Q, pd.DataFrame):
            pd.testing.assert_frame_equal(loaded_Q, Q)
        else:
            np.testing.assert_array_equal(np.asarray(loaded_Q if not isinstance(Q, dict) else
                                                     [loaded_Q[k] for k in Q]), np.asarray(
                                                         Q if not isinstance(Q, dict) else list(Q.values())))
        np.testing.assert_array_equal(loaded.run().fisher_information(), obs.fisher_information(Q=Q))
        # the save_results sidecar loads too
        files = obs.save_results(tmp_path / 'out', Q=Q)
        assert ObservabilityAnalysis(simulator, *trajectory).load_settings(files['sidecar']).method == obs.method
        sidecar_Q = yaml.safe_load(open(files['sidecar']))['selection']['Q']
        if isinstance(Q, pd.DataFrame):
            assert sidecar_Q['_names'] == list(Q.index)

    def test_R_errors(self, obs):
        with pytest.raises(ValueError, match='a matrix R is not supported'):
            obs.min_error_variance(R=np.eye(WINDOW_SIZE))
        with pytest.raises(ValueError, match="no noise level for sensors \\['r'\\]"):
            obs.min_error_variance(R={'x': 1.0})

    def test_state_selection_is_conditional(self, obs):
        F = obs.fisher_information()
        ev = obs.min_error_variance(states=['d'], lam=0.0)
        np.testing.assert_allclose(_states(ev).dropna()['d'].to_numpy(), 1 / F[:, 1, 1], rtol=1e-10)
        np.testing.assert_array_equal(obs.fisher_information(states=['d', 'g']), F[:, ::-1][:, :, ::-1])

    def test_time_step_selection(self, obs, con):
        pd.testing.assert_frame_equal(obs.min_error_variance(time_steps=list(range(WINDOW_SIZE))),
                                      obs.min_error_variance())
        assert not obs.min_error_variance(time_steps=[0, 1]).equals(obs.min_error_variance())
        # observability measured only at the window's first step: F = C_0^T R^-1 C_0 exactly
        F = obs._stochastic_fisher(None, None, [0], 0.1, 1e-4, False)
        np.testing.assert_allclose(F, np.swapaxes(obs._C[:N_WINDOWS], 1, 2) @ obs._C[:N_WINDOWS] / 0.1, rtol=1e-12)
        # constructability measured only at the window's last step: F = Q^-1 - ... + C_e^T R^-1 C_e, at least C_e
        F = con._stochastic_fisher(None, None, [WINDOW_SIZE - 1], 0.1, 1e-4, False)
        C_e = con._C[WINDOW_SIZE - 1:]
        np.testing.assert_allclose(F, np.swapaxes(C_e, 1, 2) @ C_e / 0.1, rtol=1e-9)

    def test_lam_limit(self, obs):
        ev = obs.min_error_variance(lam='limit')
        np.testing.assert_allclose(_states(ev).dropna().to_numpy(),
                                   _states(obs.min_error_variance(lam=0.0)).dropna().to_numpy(), rtol=1e-6)

    def test_results_are_cached(self, obs):
        first = obs.min_error_variance(Q=1e-3)
        assert any(key[-2] == 1e-3 for key in obs._cache)
        pd.testing.assert_frame_equal(obs.min_error_variance(Q=1e-3), first)

    def test_fisher_needs_O(self, obs):
        with pytest.raises(ValueError, match='fisher_information'):
            obs.fisher()
        with pytest.raises(ValueError, match='observability_matrix'):
            obs.O_df_sliding

    @pytest.mark.parametrize('kind', ['obs', 'con'])
    def test_observability_matrix(self, kind, obs, con):
        oa = obs if kind == 'obs' else con
        k = 3
        O = oa.observability_matrix(k)
        assert list(O.columns) == ['g', 'd'] and O.index.names == ['sensor', 'time_step']
        # the row of the bounded sample is C there: the first for observability, the last for constructability
        j = 0 if kind == 'obs' else WINDOW_SIZE - 1
        np.testing.assert_allclose(O.loc[('r', j)].to_numpy(), oa._C[k + j][0])
        Phi = oa._Phi[k:k + WINDOW_SIZE]
        expected = stochastic.window_observability_matrix(oa._Phi, oa._C, k, WINDOW_SIZE,
                                                          'initial' if kind == 'obs' else 'final')
        np.testing.assert_array_equal(O.to_numpy(), expected.to_numpy())
        if kind == 'obs':
            F = stochastic.deterministic_observability_gramian(Phi, oa._C[k:k + WINDOW_SIZE],
                                                               [np.eye(1) / 0.1] * WINDOW_SIZE)
            np.testing.assert_allclose(O.to_numpy().T @ O.to_numpy() / 0.1, F, rtol=1e-12)
        assert oa.observability_matrix(k, sensors=['r'], time_steps=[0, 1]).shape == (2, 2)
        image = oa.plot_observability_matrix(k)
        import matplotlib.pyplot as plt
        plt.close(image.fig)

    def test_save_results(self, con, tmp_path):
        import yaml
        files = con.save_results(tmp_path / 'out', alignment='bounded_state')
        sidecar = yaml.safe_load(Path(files['sidecar']).read_text())
        assert sidecar['selection']['Q'] == 1e-4 and sidecar['selection']['alignment'] == 'bounded_state'
        assert 'final time-step' in sidecar['time_alignment']
        assert sidecar['analysis']['settings']['method'] == 'stochastic-constructability-classic'
        with pytest.raises(ValueError, match='does not build'):
            con.save_results(tmp_path / 'other', include_observability_matrices=True)


class TestAlignment:

    def test_default_is_center(self, bounds, obs):
        for oa in (bounds, obs):
            assert oa.settings['alignment'] == 'center'
            pd.testing.assert_frame_equal(oa.min_error_variance(alignment='center'), oa.min_error_variance())
        # center is pybounds' long-standing placement, w // 2 past each window's start
        ev = bounds.min_error_variance()
        assert ev['time_initial'].first_valid_index() == WINDOW_SIZE // 2

    @pytest.mark.parametrize('kind, offset', [('bounds', 0), ('obs', 0), ('con', WINDOW_SIZE - 1)])
    def test_bounded_state(self, kind, offset, bounds, obs, con):
        oa = {'bounds': bounds, 'obs': obs, 'con': con}[kind]
        center, bounded = oa.min_error_variance(), oa.min_error_variance(alignment='bounded_state')
        assert bounded['time_initial'].first_valid_index() == offset
        shift = offset - WINDOW_SIZE // 2
        np.testing.assert_array_equal(_states(bounded).to_numpy()[offset:offset + N_WINDOWS],
                                      _states(center).to_numpy()[offset - shift:offset - shift + N_WINDOWS])
        np.testing.assert_array_equal(bounded['time'].to_numpy(), center['time'].to_numpy())
        np.testing.assert_array_equal(bounded['time_initial'].dropna().to_numpy(),
                                      center['time_initial'].dropna().to_numpy())
        # 'time' is now the time of the state each result bounds
        row = bounded.dropna().iloc[0]
        assert row['time'] == pytest.approx(row['time_initial'] + offset * 0.01)

    def test_setting_and_fisher(self, simulator, trajectory, bounds):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1,
                                   alignment='bounded_state').run()
        pd.testing.assert_frame_equal(oa.min_error_variance(), bounds.min_error_variance(alignment='bounded_state'))
        pd.testing.assert_frame_equal(oa.fisher().get_minimum_error_variance(), oa.min_error_variance())
        # a matrix R takes the per-window path, which must honour the alignment too
        R = 0.1 * np.eye(WINDOW_SIZE)
        np.testing.assert_allclose(_states(oa.min_error_variance(R=R)).to_numpy(),
                                   _states(oa.min_error_variance()).to_numpy(), rtol=1e-10)

    def test_fisher_storage_modes(self, simulator, trajectory, bounds):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1, storage='fisher').run()
        np.testing.assert_allclose(_states(oa.min_error_variance(alignment='bounded_state')).to_numpy(),
                                   _states(bounds.min_error_variance(alignment='bounded_state')).to_numpy(),
                                   rtol=1e-8)

    def test_observability_and_constructability_line_up_when_centered(self, obs, con):
        """For a slowly varying trajectory both describe nearly the same quantity, so centered they agree
        better than placed at the states they bound."""
        def gap(alignment):
            a = _states(obs.min_error_variance(alignment=alignment))
            b = _states(con.min_error_variance(alignment=alignment))
            return np.nanmax(np.abs(np.log(a.to_numpy() / b.to_numpy())))
        assert np.isnan(gap('bounded_state')) or gap('center') <= gap('bounded_state')

    def test_invalid(self, simulator, trajectory, bounds):
        with pytest.raises(ValueError, match="unknown alignment 'start'"):
            ObservabilityAnalysis(simulator, *trajectory, alignment='start')
        with pytest.raises(ValueError, match='unknown alignment'):
            bounds.min_error_variance(alignment='start')

    def test_from_sliding(self, bounds):
        oa = ObservabilityAnalysis.from_sliding(analysis.SlidingO(O=bounds._O, index=bounds._index,
                                                                  state_names=['g', 'd'], t_sim=bounds.t_sim),
                                                R=0.1, alignment='bounded_state')
        pd.testing.assert_frame_equal(oa.min_error_variance(), bounds.min_error_variance(alignment='bounded_state'))
        oa.update_settings(alignment='center')
        pd.testing.assert_frame_equal(oa.min_error_variance(), bounds.min_error_variance())

    def test_yaml_roundtrip(self, simulator, trajectory, tmp_path):
        Q = {'g': 1e-4, 'd': 1e-6}
        oa = ObservabilityAnalysis(simulator, *trajectory, method='stochastic-constructability-classic',
                                   w=WINDOW_SIZE, R=0.1, Q=Q, alignment='bounded_state', eps=1e-6)
        path = oa.save_settings(tmp_path / 's.yaml')
        loaded = ObservabilityAnalysis(simulator, *trajectory).load_settings(path)
        assert loaded.method == 'stochastic-constructability-classic'
        assert loaded.settings['Q'] == Q and loaded.settings['alignment'] == 'bounded_state'
        assert loaded.settings['eps'] == 1e-6
        Q_matrix = np.diag([1e-4, 1e-6])
        oa.update_settings(Q=Q_matrix)
        loaded = ObservabilityAnalysis(simulator, *trajectory).load_settings(oa.save_settings(tmp_path / 'm.yaml'))
        np.testing.assert_array_equal(loaded.settings['Q'], Q_matrix)


class TestZFunction:

    @staticmethod
    def z_optic_flow(x):
        return sp.Matrix([x[0] / x[1], x[1]])

    @pytest.mark.parametrize('kind, offset', [('observability', 0), ('constructability', WINDOW_SIZE - 1)])
    def test_transformed_at_bounded_state(self, simulator, trajectory, kind, offset):
        oa = ObservabilityAnalysis(simulator, *trajectory, method=f'stochastic-{kind}-classic', w=WINDOW_SIZE,
                                   R=0.1, Q=1e-4, z_function=self.z_optic_flow,
                                   z_state_names=['of', 'd']).run()
        assert oa.state_names == ['of', 'd']
        x = np.column_stack([trajectory[1]['g'], trajectory[1]['d']])
        k = 4
        g, d = x[k + offset]
        dzdx = np.array([[1 / d, -g / d ** 2], [0.0, 1.0]])
        np.testing.assert_allclose(oa.dxdz_sliding[k], inv(dzdx), rtol=1e-12)
        untransformed = ObservabilityAnalysis(simulator, *trajectory, method=f'stochastic-{kind}-classic',
                                              w=WINDOW_SIZE, R=0.1, Q=1e-4).run().fisher_information()
        np.testing.assert_allclose(oa.fisher_information()[k], inv(dzdx).T @ untransformed[k] @ inv(dzdx),
                                   rtol=1e-10)
        # exactly the congruence with the stored dx/dz, without re-symmetrizing
        dxdz = np.stack(oa.dxdz_sliding)
        np.testing.assert_array_equal(oa.fisher_information(), np.swapaxes(dxdz, 1, 2) @ untransformed @ dxdz)
        # the transformed matrix's rows are sensitivities to z
        O = oa.observability_matrix(k)
        assert list(O.columns) == ['of', 'd']
        ev = oa.min_error_variance(states=['of'])
        assert list(ev.columns) == ['time', 'time_initial', 'of']
