import numpy as np
import pandas as pd
import pytest
import pybounds
from conftest import N_STEPS, EPS, DT, STATE_NAMES, MEASUREMENT_NAMES


class TestEOMTypes:

    def test_O_is_ndarray(self, eom):
        assert isinstance(eom.O, np.ndarray)

    def test_O_df_is_dataframe(self, eom):
        assert isinstance(eom.O_df, pd.DataFrame)


class TestEOMStructure:

    def test_O_shape(self, eom):
        # p=1 measurement, w=N_STEPS time steps, n=2 states
        assert eom.O.shape == (N_STEPS * 1, 2)

    def test_O_df_columns(self, eom):
        assert list(eom.O_df.columns) == ['g', 'd']

    def test_O_df_index_names(self, eom):
        assert eom.O_df.index.names == ['sensor', 'time_step']

    def test_O_df_sensor_level(self, eom):
        sensors = eom.O_df.index.get_level_values('sensor').unique().tolist()
        assert sensors == ['r']

    def test_y_nominal_shape(self, eom):
        assert eom.y_nominal.shape == (N_STEPS, 1)


class TestEOMStoredAttributes:

    def test_n(self, eom):
        assert eom.n == 2

    def test_p(self, eom):
        assert eom.p == 1

    def test_w(self, eom):
        assert eom.w == N_STEPS

    def test_state_names(self, eom):
        assert list(eom.state_names) == STATE_NAMES

    def test_measurement_names(self, eom):
        assert list(eom.measurement_names) == MEASUREMENT_NAMES


class TestEOMNumericalProperties:

    def test_O_all_finite(self, eom):
        assert np.all(np.isfinite(eom.O))

    def test_O_df_values_match_O_array(self, eom):
        assert np.allclose(eom.O_df.values, eom.O)

    def test_O_g_column_nonzero(self, eom):
        assert np.any(np.abs(eom.O[:, 0]) > 1e-10)

    def test_O_d_column_nonzero(self, eom):
        assert np.any(np.abs(eom.O[:, 1]) > 1e-10)

    def test_eps_sensitivity(self, simulator):
        """O norms with different eps should agree within an order of magnitude."""
        x0 = {'g': 2.0, 'd': 3.0}
        u = {'u': 0.1 * np.ones(30)}
        eom1 = pybounds.EmpiricalObservabilityMatrix(simulator, x0, u, eps=1e-3)
        eom2 = pybounds.EmpiricalObservabilityMatrix(simulator, x0, u, eps=1e-5)
        ratio = np.linalg.norm(eom1.O) / np.linalg.norm(eom2.O)
        assert 0.5 < ratio < 2.0


class AnalyticSimulator:
    """Stateless closed-form version of the conftest system, so it is thread-safe."""

    def simulate(self, x0, u, aux=None):
        g0, d0 = x0
        g = g0 + DT * np.concatenate([[0.0], np.cumsum(np.ravel(u))[:-1]])
        return (g / d0)[:, None]


class TestEOMParallel:

    def test_parallel_with_pybounds_simulator_warns_and_matches_sequential(self, simulator, eom):
        x0 = {'g': 2.0, 'd': 3.0}
        u = {'u': 0.1 * np.ones(N_STEPS)}
        with pytest.warns(RuntimeWarning, match='not thread-safe'):
            eom_par = pybounds.EmpiricalObservabilityMatrix(simulator, x0, u, eps=EPS, parallel=True)
        assert eom_par.parallel is False
        assert np.allclose(eom_par.O, eom.O)

    def test_parallel_with_thread_safe_simulator_runs_threaded(self, recwarn):
        x0 = np.array([2.0, 3.0])
        u = 0.1 * np.ones((N_STEPS, 1))
        eom_seq = pybounds.EmpiricalObservabilityMatrix(AnalyticSimulator(), x0, u, eps=EPS)
        eom_par = pybounds.EmpiricalObservabilityMatrix(AnalyticSimulator(), x0, u, eps=EPS, parallel=True)
        assert eom_par.parallel is True
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
        assert np.allclose(eom_par.O, eom_seq.O)
