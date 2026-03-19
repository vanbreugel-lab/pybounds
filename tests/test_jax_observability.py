"""
Unit tests for the JAX autodiff backend:
  JaxSimulator, JaxEmpiricalObservabilityMatrix, JaxSlidingEmpiricalObservabilityMatrix.

Uses pytest.importorskip so every test is automatically skipped when JAX is
not installed (e.g. in the standard CI job that only runs the core tests).

The mono-camera system (g/d states, r=g/d measurement) from conftest.py is
reused.  Because f and h use only basic arithmetic (no trig), the JAX RK4
integrator and the do_mpc IDAS solver produce numerically identical results
for this system, so we can compare against the legacy classes with tight
tolerances.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pybounds
from pybounds import (JaxSimulator, JaxEmpiricalObservabilityMatrix,
                      JaxSlidingEmpiricalObservabilityMatrix)
from conftest import (STATE_NAMES, INPUT_NAMES, MEASUREMENT_NAMES,
                      DT, N_STEPS, N_STEPS_SLIDING, WINDOW_SIZE, EPS)

N_WINDOWS = N_STEPS_SLIDING - WINDOW_SIZE + 1   # 25


# ---------------------------------------------------------------------------
# JAX-compatible dynamics and measurement (mono-camera)
# ---------------------------------------------------------------------------

def f_jax(x, u):
    return jnp.array([u[0], 0.0 * u[0]])


def h_jax(x, u):
    return jnp.array([x[0] / x[1]])


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def jax_sim():
    return JaxSimulator(f_jax, h_jax, dt=DT,
                        state_names=STATE_NAMES,
                        input_names=INPUT_NAMES,
                        measurement_names=MEASUREMENT_NAMES)


@pytest.fixture(scope='module')
def jax_eom(jax_sim):
    x0 = {'g': 2.0, 'd': 3.0}
    u  = {'u': 0.1 * np.ones(N_STEPS)}
    return JaxEmpiricalObservabilityMatrix(jax_sim, x0, u)


@pytest.fixture(scope='module')
def jax_seom(jax_sim, seom):
    # Reuse the trajectory already computed by the session-scoped seom fixture.
    return JaxSlidingEmpiricalObservabilityMatrix(
        jax_sim, seom.t_sim, seom.x_sim, seom.u_sim, w=WINDOW_SIZE)


# ---------------------------------------------------------------------------
# JaxSimulator tests
# ---------------------------------------------------------------------------

class TestJaxSimulator:
    def test_output_shape(self, jax_sim):
        x0  = np.array([2.0, 3.0])
        u   = np.column_stack([0.1 * np.ones(N_STEPS)])
        y   = jax_sim.simulate(x0, u)
        assert y.shape == (N_STEPS, 1)

    def test_output_is_ndarray(self, jax_sim):
        x0  = np.array([2.0, 3.0])
        u   = np.column_stack([0.1 * np.ones(N_STEPS)])
        y   = jax_sim.simulate(x0, u)
        assert isinstance(y, np.ndarray)

    def test_matches_legacy(self, jax_sim, simulation_output):
        """JAX RK4 and IDAS agree for this linear system."""
        t_sim, x_sim, u_sim, _ = simulation_output
        x0  = np.array([x_sim['g'][0], x_sim['d'][0]])
        u   = np.column_stack([u_sim['u']])
        y_jax    = jax_sim.simulate(x0, u)
        y_legacy = (np.asarray(x_sim['g']) / np.asarray(x_sim['d']))[:, None]
        np.testing.assert_allclose(y_jax, y_legacy, atol=1e-6)

    def test_dict_inputs(self, jax_sim):
        """Accepts dict x0 and dict u_seq."""
        x0 = {'g': 2.0, 'd': 3.0}
        u  = {'u': 0.1 * np.ones(N_STEPS)}
        y  = jax_sim.simulate(x0, u)
        assert y.shape == (N_STEPS, 1)


# ---------------------------------------------------------------------------
# JaxEmpiricalObservabilityMatrix tests
# ---------------------------------------------------------------------------

class TestJaxEmpiricalObservabilityMatrix:
    def test_O_shape(self, jax_eom):
        assert jax_eom.O.shape == (N_STEPS * 1, 2)   # (w*p, n)

    def test_y_nominal_shape(self, jax_eom):
        assert jax_eom.y_nominal.shape == (N_STEPS, 1)

    def test_O_df_columns(self, jax_eom):
        assert list(jax_eom.O_df.columns) == STATE_NAMES

    def test_O_df_index_names(self, jax_eom):
        assert list(jax_eom.O_df.index.names) == ['sensor', 'time_step']

    def test_O_df_sensor_level(self, jax_eom):
        sensors = jax_eom.O_df.index.get_level_values('sensor').unique().tolist()
        assert sensors == MEASUREMENT_NAMES

    def test_O_df_values_match_O(self, jax_eom):
        np.testing.assert_array_equal(jax_eom.O_df.values, jax_eom.O)

    def test_matches_legacy(self, jax_eom, eom):
        """Jacobian from autodiff matches numerical finite-difference (atol=1e-3)."""
        np.testing.assert_allclose(jax_eom.O, eom.O, atol=1e-3)

    def test_stored_attributes(self, jax_eom):
        assert jax_eom.n == 2
        assert jax_eom.p == 1
        assert jax_eom.w == N_STEPS
        assert jax_eom.state_names == STATE_NAMES
        assert jax_eom.measurement_names == MEASUREMENT_NAMES


# ---------------------------------------------------------------------------
# JaxSlidingEmpiricalObservabilityMatrix tests
# ---------------------------------------------------------------------------

class TestJaxSlidingEmpiricalObservabilityMatrix:
    def test_O_df_sliding_is_list(self, jax_seom):
        assert isinstance(jax_seom.O_df_sliding, list)

    def test_O_sliding_is_list(self, jax_seom):
        assert isinstance(jax_seom.O_sliding, list)

    def test_n_windows(self, jax_seom):
        assert len(jax_seom.O_df_sliding) == N_WINDOWS
        assert len(jax_seom.O_sliding) == N_WINDOWS

    def test_window_O_shape(self, jax_seom):
        for O in jax_seom.O_sliding:
            assert O.shape == (WINDOW_SIZE * 1, 2)   # (w*p, n)

    def test_window_df_columns(self, jax_seom):
        for O_df in jax_seom.O_df_sliding:
            assert list(O_df.columns) == STATE_NAMES

    def test_window_df_index_names(self, jax_seom):
        for O_df in jax_seom.O_df_sliding:
            assert list(O_df.index.names) == ['sensor', 'time_step']

    def test_all_finite(self, jax_seom):
        for O in jax_seom.O_sliding:
            assert np.all(np.isfinite(O))

    def test_matches_legacy(self, jax_seom, seom):
        """All windows match the legacy numerical result (atol=1e-3)."""
        for O_jax, O_leg in zip(jax_seom.O_sliding, seom.O_sliding):
            np.testing.assert_allclose(O_jax, O_leg, atol=1e-3)

    def test_get_observability_matrix_returns_copy(self, jax_seom):
        a = jax_seom.get_observability_matrix()
        b = jax_seom.get_observability_matrix()
        assert a is not b
        assert len(a) == N_WINDOWS

    def test_t_sim_stored(self, jax_seom, seom):
        np.testing.assert_array_equal(jax_seom.t_sim, seom.t_sim)

    def test_O_index(self, jax_seom):
        expected = np.arange(0, N_STEPS_SLIDING - WINDOW_SIZE + 1)
        np.testing.assert_array_equal(jax_seom.O_index, expected)
