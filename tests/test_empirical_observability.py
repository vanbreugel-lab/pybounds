import numpy as np
import pandas as pd
import pytest
import pybounds
from conftest import N_STEPS, EPS, STATE_NAMES, MEASUREMENT_NAMES


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
