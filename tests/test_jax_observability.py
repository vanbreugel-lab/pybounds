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
import sympy as sp

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


def z_optic_flow(x):
    """Transform [g, d] -> [g/d, d]."""
    return sp.Matrix([x[0] / x[1], x[1]])


Z_STATE_NAMES = ['r', 'd']


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

    def test_window_data_keys(self, jax_seom):
        assert set(jax_seom.window_data) == {'t', 'u', 'y'}
        for k in ('t', 'u', 'y'):
            assert len(jax_seom.window_data[k]) == N_WINDOWS

    def test_window_data_matches_legacy(self, jax_seom, seom):
        for k in ('t', 'u'):
            for a, b in zip(jax_seom.window_data[k], seom.window_data[k]):
                np.testing.assert_array_equal(a, b)
        for a, b in zip(jax_seom.window_data['y'], seom.window_data['y']):
            assert a.shape == (WINDOW_SIZE, 1)
            np.testing.assert_allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# Coordinate transformation (z_function)
# ---------------------------------------------------------------------------

class TestJaxZFunction:
    def test_eom_matches_legacy(self, jax_sim, simulator):
        x0 = {'g': 2.0, 'd': 3.0}
        u = {'u': 0.1 * np.ones(N_STEPS)}
        jax_eom_z = JaxEmpiricalObservabilityMatrix(jax_sim, x0, u, z_function=z_optic_flow,
                                                    z_state_names=Z_STATE_NAMES)
        eom_z = pybounds.EmpiricalObservabilityMatrix(simulator, x0, u, eps=EPS, z_function=z_optic_flow,
                                                      z_state_names=Z_STATE_NAMES)
        assert list(jax_eom_z.O_df.columns) == Z_STATE_NAMES
        assert jax_eom_z.state_names == Z_STATE_NAMES
        assert jax_eom_z.dxdz is not None
        np.testing.assert_array_equal(jax_eom_z.O, jax_eom_z.O_df.values)
        np.testing.assert_allclose(jax_eom_z.O, eom_z.O, atol=1e-3)

    def test_eom_without_z_has_no_jacobian(self, jax_eom):
        assert jax_eom.dxdz is None
        assert jax_eom.dzdx_sym is None

    @pytest.mark.parametrize('z_state_names', [Z_STATE_NAMES, None])
    def test_sliding_matches_transform_states_per_window(self, jax_sim, jax_seom, seom, z_state_names):
        jax_seom_z = JaxSlidingEmpiricalObservabilityMatrix(
            jax_sim, seom.t_sim, seom.x_sim, seom.u_sim, w=WINDOW_SIZE,
            z_function=z_optic_flow, z_state_names=z_state_names)
        assert len(jax_seom_z.O_df_sliding) == N_WINDOWS
        for i in (0, N_WINDOWS - 1):
            expected, _, _ = pybounds.transform_states(
                O=jax_seom.O_df_sliding[i], z_function=z_optic_flow,
                x0=seom.x_sim[jax_seom.O_index[i]], z_state_names=z_state_names)
            assert list(jax_seom_z.O_df_sliding[i].columns) == list(expected.columns)
            np.testing.assert_allclose(jax_seom_z.O_df_sliding[i].values, expected.values)
            np.testing.assert_array_equal(jax_seom_z.O_sliding[i], jax_seom_z.O_df_sliding[i].values)

    def test_sliding_matches_legacy(self, jax_sim, simulator, seom):
        kwargs = dict(w=WINDOW_SIZE, z_function=z_optic_flow, z_state_names=Z_STATE_NAMES)
        jax_seom_z = JaxSlidingEmpiricalObservabilityMatrix(
            jax_sim, seom.t_sim, seom.x_sim, seom.u_sim, **kwargs)
        seom_z = pybounds.SlidingEmpiricalObservabilityMatrix(
            simulator, seom.t_sim, seom.x_sim, seom.u_sim, eps=EPS, **kwargs)
        assert jax_seom_z.state_names == Z_STATE_NAMES
        for O_jax, O_leg in zip(jax_seom_z.O_sliding, seom_z.O_sliding):
            np.testing.assert_allclose(O_jax, O_leg, atol=1e-3)


# ---------------------------------------------------------------------------
# Auxiliary inputs (aux / aux_list)
# ---------------------------------------------------------------------------

def f_jax_aux(x, u, aux):
    """Mono-camera dynamics with the input scaled by an auxiliary gain."""
    return jnp.array([aux['gain'] * u[0], 0.0 * u[0]])


def h_jax_aux(x, u, aux):
    return jnp.array([x[0] / x[1] + aux['offset']])


class AnalyticAuxSimulator:
    """Closed-form counterpart of f_jax_aux / h_jax_aux for the CasADi-side classes."""

    def simulate(self, x0, u, aux=None):
        g0, d0 = x0
        g = g0 + DT * aux['gain'] * np.concatenate([[0.0], np.cumsum(np.ravel(u))[:-1]])
        return (g / d0 + aux['offset'])[:, None]


@pytest.fixture(scope='module')
def jax_sim_aux():
    return JaxSimulator(f_jax_aux, h_jax_aux, dt=DT,
                        state_names=STATE_NAMES,
                        input_names=INPUT_NAMES,
                        measurement_names=MEASUREMENT_NAMES)


def _aux_list(n):
    return [{'gain': g, 'offset': 0.1 * g} for g in np.linspace(0.5, 2.0, n)]


class TestJaxAux:
    def test_simulate_aux_scales_input(self, jax_sim, jax_sim_aux):
        x0 = np.array([2.0, 3.0])
        u = 0.1 * np.ones((N_STEPS, 1))
        y_aux = jax_sim_aux.simulate(x0, u, aux={'gain': 2.0, 'offset': 0.0})
        np.testing.assert_allclose(y_aux, jax_sim.simulate(x0, 2.0 * u))

    def test_eom_aux_matches_scaled_input(self, jax_sim, jax_sim_aux):
        x0 = np.array([2.0, 3.0])
        u = 0.1 * np.ones((N_STEPS, 1))
        eom_aux = JaxEmpiricalObservabilityMatrix(jax_sim_aux, x0, u, aux={'gain': 2.0, 'offset': 0.5})
        eom_ref = JaxEmpiricalObservabilityMatrix(jax_sim, x0, 2.0 * u)
        np.testing.assert_allclose(eom_aux.O, eom_ref.O)
        np.testing.assert_allclose(eom_aux.y_nominal, eom_ref.y_nominal + 0.5)

    def test_sliding_window_i_uses_aux_list_i(self, jax_sim_aux, seom):
        aux_list = _aux_list(N_STEPS_SLIDING)
        jax_seom_aux = JaxSlidingEmpiricalObservabilityMatrix(
            jax_sim_aux, seom.t_sim, seom.x_sim, seom.u_sim, w=WINDOW_SIZE, aux_list=aux_list)
        for i in (0, N_WINDOWS - 1):
            k = jax_seom_aux.O_index[i]
            eom_i = JaxEmpiricalObservabilityMatrix(
                jax_sim_aux, seom.x_sim[k], seom.u_sim[k:k + WINDOW_SIZE], aux=aux_list[i])
            np.testing.assert_allclose(jax_seom_aux.O_sliding[i], eom_i.O)
            np.testing.assert_allclose(jax_seom_aux.window_data['y'][i], eom_i.y_nominal)

    def test_sliding_aux_matches_legacy(self, jax_sim_aux, seom):
        aux_list = _aux_list(N_STEPS_SLIDING)
        jax_seom_aux = JaxSlidingEmpiricalObservabilityMatrix(
            jax_sim_aux, seom.t_sim, seom.x_sim, seom.u_sim, w=WINDOW_SIZE, aux_list=aux_list)
        seom_aux = pybounds.SlidingEmpiricalObservabilityMatrix(
            AnalyticAuxSimulator(), seom.t_sim, seom.x_sim, seom.u_sim,
            w=WINDOW_SIZE, eps=EPS, aux_list=aux_list)
        for O_jax, O_leg in zip(jax_seom_aux.O_sliding, seom_aux.O_sliding):
            np.testing.assert_allclose(O_jax, O_leg, atol=1e-3)

    def test_sliding_aux_list_wrong_length_raises(self, jax_sim_aux, seom):
        with pytest.raises(ValueError, match='aux_list must have same number of elements'):
            JaxSlidingEmpiricalObservabilityMatrix(
                jax_sim_aux, seom.t_sim, seom.x_sim, seom.u_sim, w=WINDOW_SIZE,
                aux_list=_aux_list(N_STEPS_SLIDING - 1))

    def test_sliding_aux_list_mismatched_shapes_raises(self, jax_sim_aux, seom):
        aux_list = _aux_list(N_STEPS_SLIDING)
        aux_list[1] = {'gain': np.ones(2), 'offset': 0.0}
        with pytest.raises(ValueError, match='same structure and array shapes'):
            JaxSlidingEmpiricalObservabilityMatrix(
                jax_sim_aux, seom.t_sim, seom.x_sim, seom.u_sim, w=WINDOW_SIZE, aux_list=aux_list)


# ---------------------------------------------------------------------------
# Integration substeps
# ---------------------------------------------------------------------------

def f_decay(x, u):
    return jnp.array([-20.0 * x[0] + 0.0 * u[0]])


def h_identity(x, u):
    return jnp.array([x[0]])


def _decay_sim(dt, **kwargs):
    return JaxSimulator(f_decay, h_identity, dt=dt, state_names=['x'], input_names=['u'],
                        measurement_names=['y'], **kwargs)


class TestJaxSubsteps:
    def test_default_is_one(self, jax_sim):
        assert jax_sim.substeps == 1

    @pytest.mark.parametrize('substeps', [4, 7])
    def test_substeps_equals_finer_dt(self, jax_sim, substeps):
        """substeps=k gives the same samples as a k-times finer dt with each input repeated k times."""
        coarse = JaxSimulator(f_jax, h_jax, dt=0.05, state_names=STATE_NAMES, input_names=INPUT_NAMES,
                              measurement_names=MEASUREMENT_NAMES, substeps=substeps)
        fine = JaxSimulator(f_jax, h_jax, dt=0.05 / substeps, state_names=STATE_NAMES,
                            input_names=INPUT_NAMES, measurement_names=MEASUREMENT_NAMES)
        x0 = np.array([2.0, 3.0])
        u = np.linspace(-0.5, 0.5, 20)[:, None]
        y_fine = fine.simulate(x0, np.repeat(u, substeps, axis=0))[::substeps]
        np.testing.assert_allclose(coarse.simulate(x0, u), y_fine, rtol=1e-12)

    def test_substeps_improve_accuracy(self):
        dt, n = 0.1, 20
        u = np.zeros((n, 1))
        y_exact = np.exp(-20.0 * dt * np.arange(n))[:, None]
        err_1 = np.max(np.abs(_decay_sim(dt).simulate([1.0], u) - y_exact))
        err_8 = np.max(np.abs(_decay_sim(dt, substeps=8).simulate([1.0], u) - y_exact))
        assert err_8 < 1e-3 * err_1

    def test_observability_matrix_with_substeps(self):
        """jacfwd through the substep loop: for this linear system dy_k/dx0 = y_k when x0 = 1,
        and both approximate exp(-20 k dt)."""
        dt, n = 0.1, 20
        eom = JaxEmpiricalObservabilityMatrix(_decay_sim(dt, substeps=8), [1.0], np.zeros((n, 1)))
        np.testing.assert_allclose(eom.O[:, 0], eom.y_nominal[:, 0], rtol=1e-12)
        np.testing.assert_allclose(eom.O[:, 0], np.exp(-20.0 * dt * np.arange(n)), atol=1e-4)

    @pytest.mark.parametrize('substeps', [0, -1, 1.5, True, '2'])
    def test_invalid_substeps_raises(self, substeps):
        with pytest.raises(ValueError, match='substeps must be a positive integer'):
            _decay_sim(0.1, substeps=substeps)


# ---------------------------------------------------------------------------
# Non-finite output warning
# ---------------------------------------------------------------------------

def f_const(x, u):
    return jnp.array([0.0 * x[0] + 0.0 * u[0]])


def h_log(x, u):
    """NaN whenever the state is negative."""
    return jnp.array([jnp.log(x[0])])


@pytest.fixture(scope='module')
def jax_sim_log():
    return JaxSimulator(f_const, h_log, dt=DT, state_names=['x'], input_names=['u'],
                        measurement_names=['y'])


def _runtime_warnings(recwarn):
    return [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


class TestJaxNonFiniteWarning:
    def test_simulate_warns(self, jax_sim_log):
        with pytest.warns(RuntimeWarning, match='JaxSimulator.simulate output contains NaN or inf'):
            jax_sim_log.simulate([-1.0], np.zeros((10, 1)))

    def test_eom_warns(self, jax_sim_log):
        with pytest.warns(RuntimeWarning, match='JaxEmpiricalObservabilityMatrix contains NaN or inf'):
            JaxEmpiricalObservabilityMatrix(jax_sim_log, [-1.0], np.zeros((10, 1)))

    def test_sliding_warns_with_bad_window_count(self, jax_sim_log):
        n, w = 12, 3
        x_sim = np.r_[np.ones(8), -np.ones(4)][:, None]   # windows starting at index 8, 9 are negative
        with pytest.warns(RuntimeWarning, match=r'\(2 of 10 windows\) contains NaN or inf'):
            JaxSlidingEmpiricalObservabilityMatrix(
                jax_sim_log, np.arange(n) * DT, x_sim, np.zeros((n, 1)), w=w)

    def test_no_warning_for_finite_output(self, jax_sim_log, recwarn):
        jax_sim_log.simulate([1.0], np.zeros((10, 1)))
        JaxEmpiricalObservabilityMatrix(jax_sim_log, [1.0], np.zeros((10, 1)))
        JaxSlidingEmpiricalObservabilityMatrix(
            jax_sim_log, np.arange(12) * DT, np.ones((12, 1)), np.zeros((12, 1)), w=3)
        assert not _runtime_warnings(recwarn)


# ---------------------------------------------------------------------------
# float64 is scoped to pybounds calls
# ---------------------------------------------------------------------------

class TestJaxPrecisionScope:
    def test_import_does_not_enable_x64_globally(self):
        import subprocess
        import sys
        code = ('import jax.numpy as jnp, pybounds; '
                'assert pybounds._JAX_AVAILABLE; '
                'print(jnp.ones(1).dtype)')
        out = subprocess.run([sys.executable, '-W', 'ignore', '-c', code],
                             capture_output=True, text=True, check=True)
        assert out.stdout.strip() == 'float32'

    def test_results_are_float64_and_global_default_unchanged(self, jax_sim, seom):
        y = jax_sim.simulate([2.0, 3.0], 0.1 * np.ones((5, 1)))
        eom = JaxEmpiricalObservabilityMatrix(jax_sim, [2.0, 3.0], 0.1 * np.ones((5, 1)))
        s = JaxSlidingEmpiricalObservabilityMatrix(jax_sim, seom.t_sim, seom.x_sim, seom.u_sim, w=WINDOW_SIZE)
        assert y.dtype == np.float64
        assert eom.O.dtype == np.float64
        assert s.O_sliding[0].dtype == np.float64
        assert jnp.ones(1).dtype == jnp.float32
