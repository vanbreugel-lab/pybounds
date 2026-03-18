import numpy as np
import pytest
import pybounds

STATE_NAMES = ['g', 'd']
INPUT_NAMES = ['u']
MEASUREMENT_NAMES = ['r']
DT = 0.01
N_STEPS = 50
N_STEPS_SLIDING = 30
WINDOW_SIZE = 6
EPS = 1e-4


def dynamics_f(X, U):
    g, d = X
    u = U[0]
    return [u, 0 * u]


def measurement_h(X, U):
    g, d = X
    return [g / d]


@pytest.fixture(scope='session')
def simulator():
    return pybounds.Simulator(
        dynamics_f, measurement_h,
        dt=DT,
        state_names=STATE_NAMES,
        input_names=INPUT_NAMES,
        measurement_names=MEASUREMENT_NAMES,
    )


@pytest.fixture(scope='session')
def simulation_output(simulator):
    x0 = {'g': 2.0, 'd': 3.0}
    u = {'u': 0.1 * np.ones(N_STEPS)}
    return simulator.simulate(x0=x0, u=u, return_full_output=True)


@pytest.fixture(scope='session')
def eom(simulator):
    x0 = {'g': 2.0, 'd': 3.0}
    u = {'u': 0.1 * np.ones(N_STEPS)}
    return pybounds.EmpiricalObservabilityMatrix(simulator, x0, u, eps=EPS)


@pytest.fixture(scope='session')
def seom(simulator, simulation_output):
    t_sim, x_sim, u_sim, _ = simulation_output
    t_s = t_sim[:N_STEPS_SLIDING]
    x_s = {k: v[:N_STEPS_SLIDING] for k, v in x_sim.items()}
    u_s = {k: v[:N_STEPS_SLIDING] for k, v in u_sim.items()}
    return pybounds.SlidingEmpiricalObservabilityMatrix(
        simulator, t_s, x_s, u_s, w=WINDOW_SIZE, eps=EPS,
    )


@pytest.fixture(scope='session')
def fisher_obs(eom):
    return pybounds.FisherObservability(eom.O_df, R={'r': 0.1}, lam=1e-8)


@pytest.fixture(scope='session')
def sliding_fisher(seom):
    return pybounds.SlidingFisherObservability(
        seom.O_df_sliding,
        time=seom.t_sim,
        lam=1e-8,
        R={'r': 0.1},
        states=['g', 'd'],
        sensors=['r'],
        time_steps=np.arange(0, WINDOW_SIZE),
        w=None,
    )
