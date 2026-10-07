"""Per-state regularization: lam as a dict of state name -> value or a 1-D array, for every method and storage."""

import numpy as np
import pandas as pd
import pytest
import sympy as sp

import pybounds
from pybounds import analysis
from pybounds.analysis import ObservabilityAnalysis, DEFAULT_LAM
from conftest import N_STEPS_SLIDING, WINDOW_SIZE, EPS

LAM = {'g': 1e-3, 'd': 1e-6}


def z_optic_flow(x):
    return sp.Matrix([x[0] / x[1], x[1]])


@pytest.fixture(scope='module')
def trajectory(simulation_output):
    t_sim, x_sim, u_sim, _ = simulation_output
    return (t_sim[:N_STEPS_SLIDING],
            {k: v[:N_STEPS_SLIDING] for k, v in x_sim.items()},
            {k: v[:N_STEPS_SLIDING] for k, v in u_sim.items()})


def _make(simulator, trajectory, kind, **settings):
    if kind.startswith('stochastic'):
        return ObservabilityAnalysis(simulator, *trajectory, method=f'{kind}-classic', w=WINDOW_SIZE, R=0.1, Q=1e-4,
                                     **settings).run()
    return ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1, storage=kind,
                                 **settings).run()


KINDS = ['observability', 'fisher_per_sensor', 'fisher', 'stochastic-observability', 'stochastic-constructability']


def _states(ev):
    return ev.drop(columns=['time', 'time_initial'])


def _expected(oa, lam_vector, states=None):
    """diag((F + diag(lam))^-1) per window from the unregularized Fisher information."""
    F = oa.fisher_information(states=states)
    if oa.linearization is not None:   # the stochastic path clips negative eigenvalues before regularizing
        values, vectors = np.linalg.eigh(F)
        F = (vectors * np.clip(values, 0, None)[:, None, :]) @ np.swapaxes(vectors, 1, 2)
    return np.diagonal(np.linalg.inv(F + np.diag(lam_vector)), axis1=1, axis2=2)


@pytest.mark.parametrize('kind', KINDS)
class TestEveryMethodAndStorage:

    def test_uniform_vector_is_bit_identical_to_scalar(self, simulator, trajectory, kind):
        oa = _make(simulator, trajectory, kind)
        for lam in (DEFAULT_LAM, 1e-5):
            expected = oa.min_error_variance(lam=lam)
            for per_state in (np.full(2, lam), [lam, lam], {'g': lam, 'd': lam}):
                pd.testing.assert_frame_equal(oa.min_error_variance(lam=per_state), expected, check_exact=True)
            pd.testing.assert_frame_equal(oa.min_error_variance(states=['d'], lam={'d': lam}),
                                          oa.min_error_variance(states=['d'], lam=lam), check_exact=True)

    def test_per_state_values(self, simulator, trajectory, kind):
        oa = _make(simulator, trajectory, kind)
        ev = _states(oa.min_error_variance(lam=LAM)).dropna().to_numpy()
        np.testing.assert_allclose(ev, _expected(oa, [LAM['g'], LAM['d']]), rtol=1e-9)
        # an array is in the order of the selected states
        ev_swapped = oa.min_error_variance(states=['d', 'g'], lam=[LAM['d'], LAM['g']])
        np.testing.assert_allclose(_states(ev_swapped)[["g", "d"]].dropna().to_numpy(), ev, rtol=1e-9)
        pd.testing.assert_frame_equal(oa.min_error_variance(states=['d', 'g'], lam=LAM), ev_swapped,
                                      check_exact=True)

    def test_as_setting(self, simulator, trajectory, kind):
        oa = _make(simulator, trajectory, kind)
        expected = oa.min_error_variance(lam=LAM)
        oa.update_settings(lam=LAM)
        assert oa.is_computed
        pd.testing.assert_frame_equal(oa.min_error_variance(), expected, check_exact=True)
        # the setting may name states a query does not select
        pd.testing.assert_frame_equal(oa.min_error_variance(states=['d']),
                                      oa.min_error_variance(states=['d'], lam={'d': LAM['d']}), check_exact=True)


class TestRules:

    @pytest.fixture(scope='class')
    def oa(self, simulator, trajectory):
        return _make(simulator, trajectory, 'observability')

    def test_omitted_states_take_default_lam(self, oa):
        pd.testing.assert_frame_equal(oa.min_error_variance(lam={'g': 1e-3}),
                                      oa.min_error_variance(lam={'g': 1e-3, 'd': DEFAULT_LAM}), check_exact=True)
        pd.testing.assert_frame_equal(oa.min_error_variance(lam={}), oa.min_error_variance(lam=DEFAULT_LAM),
                                      check_exact=True)

    def test_errors(self, oa):
        with pytest.raises(ValueError, match=r"not selected: \['g'\]"):
            oa.min_error_variance(states=['d'], lam=LAM)
        with pytest.raises(ValueError, match=r"not selected: \['x'\]"):
            oa.min_error_variance(lam={'x': 1e-3})
        for bad in ({'g': 0.0}, {'g': -1e-3}, [1e-3, np.nan], [1e-3, 0.0]):
            with pytest.raises(ValueError, match='must be > 0'):
                oa.min_error_variance(lam=bad)
        with pytest.raises(ValueError, match="'limit' is only available as a scalar"):
            oa.min_error_variance(lam={'g': 'limit'})
        with pytest.raises(ValueError, match='one value per selected state'):
            oa.min_error_variance(lam=[1e-3, 1e-3, 1e-3])
        with pytest.raises(ValueError, match='one value per selected state'):
            oa.min_error_variance(states=['d'], lam=[1e-3, 1e-3])
        oa.update_settings(lam={'x': 1e-3})
        try:
            with pytest.raises(ValueError, match=r"not selected: \['x'\]"):
                oa.min_error_variance()
        finally:
            oa.update_settings(lam=DEFAULT_LAM)

    def test_limit_stays_scalar(self, oa):
        ev = oa.min_error_variance(lam='limit')
        expected = oa.fisher(lam='limit').get_minimum_error_variance()
        pd.testing.assert_frame_equal(ev, expected, check_exact=True)
        with pytest.raises(ValueError, match="'limit' is only available as a scalar"):
            oa.min_error_variance(lam=['limit', 'limit'])

    def test_fast_path_is_kept(self, oa, monkeypatch):
        """A per-state lam is answered from the stored array, not per window through pandas."""
        def fail(*args, **kwargs):
            raise AssertionError('fell back to the per-window path')
        monkeypatch.setattr(analysis, 'SlidingFisherObservability', fail)
        oa.clear_cache()
        oa.min_error_variance(lam=LAM)
        oa.min_error_variance(states=['d'], lam=[1e-4])

    def test_matrix_R_uses_per_window_path(self, oa):
        R = 0.1 * np.eye(WINDOW_SIZE)
        np.testing.assert_allclose(_states(oa.min_error_variance(R=R, lam=LAM)).to_numpy(),
                                   _states(oa.min_error_variance(lam=LAM)).to_numpy(), rtol=1e-10)

    def test_cache(self, oa):
        oa.clear_cache()
        first = oa.min_error_variance(lam=LAM)
        assert len(oa._cache) == 1
        pd.testing.assert_frame_equal(oa.min_error_variance(lam=dict(LAM)), first)
        pd.testing.assert_frame_equal(oa.min_error_variance(lam=np.array([LAM['g'], LAM['d']])), first)
        assert len(oa._cache) == 1   # dict and array resolve to the same key
        assert not oa.min_error_variance(lam={'g': 1e-2}).equals(first)
        assert len(oa._cache) == 2

    def test_fisher_and_save_results(self, oa, tmp_path):
        import yaml
        sliding = oa.fisher(lam=LAM)
        pd.testing.assert_frame_equal(sliding.get_minimum_error_variance(), oa.min_error_variance(lam=LAM))
        files = oa.save_results(tmp_path / 'out', states=['d'], lam={'d': 1e-5})
        sidecar = yaml.safe_load(open(files['sidecar']))
        assert sidecar['selection']['lam'] == [1e-5]

    def test_yaml_roundtrip(self, simulator, trajectory, tmp_path):
        oa = ObservabilityAnalysis(simulator, *trajectory, w=WINDOW_SIZE, eps=EPS, R=0.1, lam=LAM)
        loaded = ObservabilityAnalysis(simulator, *trajectory).load_settings(oa.save_settings(tmp_path / 's.yaml'))
        assert loaded.settings['lam'] == LAM
        oa.update_settings(lam=np.array([1e-3, 1e-6]))
        loaded = ObservabilityAnalysis(simulator, *trajectory).load_settings(oa.save_settings(tmp_path / 'a.yaml'))
        assert loaded.settings['lam'] == [1e-3, 1e-6]
        loaded.run()
        pd.testing.assert_frame_equal(loaded.min_error_variance(), oa.run().min_error_variance(lam=LAM))
        path = tmp_path / 'hand.yaml'
        path.write_text('settings:\n  lam:\n    g: 1e-3\n    d: 1e-6\n')
        assert ObservabilityAnalysis(simulator, *trajectory).load_settings(path).settings['lam'] == LAM

    @pytest.mark.parametrize('kind', ['observability', 'stochastic-constructability'])
    def test_transformed_names(self, simulator, trajectory, kind):
        oa = _make(simulator, trajectory, kind, z_function=z_optic_flow, z_state_names=['of', 'd'])
        ev = _states(oa.min_error_variance(lam={'of': 1e-3, 'd': 1e-6})).dropna().to_numpy()
        np.testing.assert_allclose(ev, _expected(oa, [1e-3, 1e-6]), rtol=1e-9)
        with pytest.raises(ValueError, match=r"not selected: \['g'\]"):
            oa.min_error_variance(lam={'g': 1e-3})

    def test_stochastic_clip_before_regularizer(self, simulator, trajectory):
        """With a tiny Q, cancellation can leave slightly negative eigenvalues; they are clipped to 0 and only
        then is diag(lam) added, so every variance stays below its ceiling 1/lam_i."""
        oa = _make(simulator, trajectory, 'stochastic-observability')
        lam = {'g': 1e-6, 'd': 1e-9}
        ev = _states(oa.min_error_variance(Q=1e-11, lam=lam)).dropna()
        assert (ev['g'] > 0).all() and (ev['g'] <= 1e6).all() and (ev['d'] > 0).all() and (ev['d'] <= 1e9).all()
        F = oa.fisher_information(Q=1e-11)
        values, vectors = np.linalg.eigh(F)
        F_clipped = (vectors * np.clip(values, 0, None)[:, None, :]) @ np.swapaxes(vectors, 1, 2)
        expected = np.diagonal(np.linalg.inv(F_clipped + np.diag([1e-6, 1e-9])), axis1=1, axis2=2)
        np.testing.assert_array_equal(ev.to_numpy(), expected)


def test_fisher_observability_accepts_vector_lam(eom):
    scalar = pybounds.FisherObservability(eom.O_df, R=0.1, lam=1e-6)
    vector = pybounds.FisherObservability(eom.O_df, R=0.1, lam=np.array([1e-6, 1e-6]))
    pd.testing.assert_frame_equal(vector.error_variance, scalar.error_variance, check_exact=True)
    per_state = pybounds.FisherObservability(eom.O_df, R=0.1, lam=np.array([1e-2, 1e-6]))
    np.testing.assert_allclose(per_state.F_inv.to_numpy(),
                               np.linalg.inv(scalar.F.to_numpy() + np.diag([1e-2, 1e-6])), rtol=1e-12)
