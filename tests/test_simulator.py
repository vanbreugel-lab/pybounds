import numpy as np
import pytest
import pybounds
from conftest import (
    dynamics_f, measurement_h,
    STATE_NAMES, INPUT_NAMES, MEASUREMENT_NAMES,
    DT, N_STEPS,
)


class TestSimulatorConstruction:

    def test_state_names_stored(self, simulator):
        assert simulator.state_names == STATE_NAMES

    def test_input_names_stored(self, simulator):
        assert simulator.input_names == INPUT_NAMES

    def test_measurement_names_stored(self, simulator):
        assert simulator.measurement_names == MEASUREMENT_NAMES

    def test_n_states(self, simulator):
        assert simulator.n == 2

    def test_m_inputs(self, simulator):
        assert simulator.m == 1

    def test_p_measurements(self, simulator):
        assert simulator.p == 1

    def test_dt_stored(self, simulator):
        assert simulator.dt == DT

    def test_raises_if_no_state_names_or_n(self):
        with pytest.raises(ValueError, match='must set state_names or n'):
            pybounds.Simulator(dynamics_f, measurement_h, dt=DT,
                               input_names=['u'])

    def test_raises_if_both_state_names_and_n(self):
        with pytest.raises(ValueError, match='cannot set state_names and n'):
            pybounds.Simulator(dynamics_f, measurement_h, dt=DT,
                               state_names=['g', 'd'], n=2,
                               input_names=['u'])

    def test_raises_if_both_input_names_and_m(self):
        with pytest.raises(ValueError, match='cannot set in and n'):
            pybounds.Simulator(dynamics_f, measurement_h, dt=DT,
                               state_names=['g', 'd'],
                               input_names=['u'], m=1)

    def test_measurement_names_length_mismatch_raises(self):
        with pytest.raises(ValueError, match='measurement_names must have length equal to y'):
            pybounds.Simulator(dynamics_f, measurement_h, dt=DT,
                               state_names=['g', 'd'],
                               input_names=['u'],
                               measurement_names=['r1', 'r2'])

    def test_numeric_n_m_default_names(self):
        sim = pybounds.Simulator(dynamics_f, measurement_h, dt=DT, n=2, m=1)
        assert sim.state_names == ['x_0', 'x_1']
        assert sim.input_names == ['u_0']


class TestSimulateOutputTypes:

    def test_returns_four_tuple(self, simulation_output):
        assert len(simulation_output) == 4

    def test_time_is_ndarray(self, simulation_output):
        t_sim, *_ = simulation_output
        assert isinstance(t_sim, np.ndarray)

    def test_x_sim_is_dict(self, simulation_output):
        _, x_sim, *_ = simulation_output
        assert isinstance(x_sim, dict)

    def test_u_sim_is_dict(self, simulation_output):
        _, _, u_sim, _ = simulation_output
        assert isinstance(u_sim, dict)

    def test_y_sim_is_dict(self, simulation_output):
        *_, y_sim = simulation_output
        assert isinstance(y_sim, dict)

    def test_x_sim_keys(self, simulation_output):
        _, x_sim, *_ = simulation_output
        assert set(x_sim.keys()) == {'g', 'd'}

    def test_u_sim_keys(self, simulation_output):
        _, _, u_sim, _ = simulation_output
        assert set(u_sim.keys()) == {'u'}

    def test_y_sim_keys(self, simulation_output):
        *_, y_sim = simulation_output
        assert set(y_sim.keys()) == {'r'}


class TestSimulateOutputShapes:

    def test_time_length(self, simulation_output):
        t_sim, *_ = simulation_output
        assert len(t_sim) == N_STEPS

    def test_x_state_lengths(self, simulation_output):
        _, x_sim, *_ = simulation_output
        for key in x_sim:
            assert len(x_sim[key]) == N_STEPS

    def test_u_input_lengths(self, simulation_output):
        _, _, u_sim, _ = simulation_output
        for key in u_sim:
            assert len(u_sim[key]) == N_STEPS

    def test_y_measurement_lengths(self, simulation_output):
        *_, y_sim = simulation_output
        for key in y_sim:
            assert len(y_sim[key]) == N_STEPS


class TestSimulateOutputValues:

    def test_time_starts_at_zero(self, simulation_output):
        t_sim, *_ = simulation_output
        assert t_sim[0] == pytest.approx(0.0)

    def test_time_spacing_is_dt(self, simulation_output):
        t_sim, *_ = simulation_output
        diffs = np.diff(t_sim)
        assert np.allclose(diffs, DT, atol=1e-10)

    def test_initial_state_g(self, simulation_output):
        _, x_sim, *_ = simulation_output
        assert x_sim['g'][0] == pytest.approx(2.0, abs=1e-3)

    def test_d_is_constant(self, simulation_output):
        """d_dot = 0, so d stays at its initial value."""
        _, x_sim, *_ = simulation_output
        assert np.allclose(x_sim['d'], x_sim['d'][0], atol=1e-4)

    def test_measurement_equals_g_over_d(self, simulation_output):
        """r = g/d by definition of measurement_h."""
        _, x_sim, _, y_sim = simulation_output
        r_expected = x_sim['g'] / x_sim['d']
        assert np.allclose(y_sim['r'], r_expected, rtol=1e-3, atol=1e-4)

    def test_g_increases_with_positive_input(self, simulator):
        x0 = {'g': 1.0, 'd': 2.0}
        u = {'u': 0.5 * np.ones(50)}
        _, x_sim, *_ = simulator.simulate(x0=x0, u=u, return_full_output=True)
        assert np.all(np.diff(x_sim['g']) > 0)

    def test_g_decreases_with_negative_input(self, simulator):
        x0 = {'g': 3.0, 'd': 2.0}
        u = {'u': -0.5 * np.ones(50)}
        _, x_sim, *_ = simulator.simulate(x0=x0, u=u, return_full_output=True)
        assert np.all(np.diff(x_sim['g']) < 0)

    def test_mpc_and_u_both_set_raises(self, simulator):
        u = {'u': np.ones(10)}
        with pytest.raises(Exception, match='u must be None if running MPC'):
            simulator.simulate(x0={'g': 1.0, 'd': 1.0}, u=u, mpc=True)
