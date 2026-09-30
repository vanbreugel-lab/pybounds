from pathlib import Path
"""ObservabilityAnalysis storage modes: 'observability' (default), 'fisher_per_sensor', 'fisher'."""
import numpy as np
import pandas as pd
import pytest
import yaml

from pybounds import ObservabilityAnalysis
from pybounds.analysis import SlidingO
from test_analysis_memory import LinearSim, z_fn

EPS64 = np.finfo(float).eps
SIM = LinearSim(6, 5)
T, X, U = SIM.trajectory(40)
STATES, SENSORS = SIM.state_names, SIM.measurement_names
R_DICT = {s: 0.1 * (j + 1) for j, s in enumerate(SENSORS)}


def _pair(storage, **kwargs):
    """The same analysis with the default storage and with `storage`."""
    fisher_sensors = kwargs.pop('fisher_sensors', None)
    base = dict(w=kwargs.pop('w', 6), eps=1e-4, R=0.1)
    base.update(kwargs)
    return (ObservabilityAnalysis(SIM, T, X, U, **base).run(),
            ObservabilityAnalysis(SIM, T, X, U, storage=storage, fisher_sensors=fisher_sensors, **base).run())


def _assert_agree(oa, fa, **query):
    """F agrees to rounding; error variance within 10 * cond(F + lam I) * eps (see the design note)."""
    F_o = oa.fisher_information(states=query.get('states'), sensors=query.get('sensors'), R=query.get('R', 0.1))
    F_f = fa.fisher_information(states=query.get('states'), sensors=query.get('sensors'), R=query.get('R', 0.1))
    scale = np.max(np.abs(F_o), axis=(1, 2), keepdims=True)
    assert np.all(np.abs(F_f - F_o) <= 1e-12 * scale)

    ev_o, ev_f = oa.min_error_variance(**query), fa.min_error_variance(**query)
    assert list(ev_f.columns) == list(ev_o.columns)
    pd.testing.assert_frame_equal(ev_f[['time', 'time_initial']], ev_o[['time', 'time_initial']])
    lam = query.get('lam', 1e-8)
    cond = np.array([np.linalg.cond(F + lam * np.eye(F.shape[0])) for F in F_o])
    names = [c for c in ev_o.columns if c not in ('time', 'time_initial')]
    a, b = ev_o[names].dropna().to_numpy(), ev_f[names].dropna().to_numpy()
    assert a.shape == b.shape
    assert np.all(np.abs(b - a) <= 10 * cond[:, None] * EPS64 * np.abs(a))


QUERIES = [dict(), dict(states=['x2']), dict(states=['x4', 'x0']), dict(sensors=['s3', 's0']),
           dict(states=['x1', 'x5', 'x3'], sensors=['s1', 's2', 's4'], R=R_DICT, lam=1e-6), dict(R=0.37),
           dict(R=R_DICT)]


class TestFisherPerSensor:

    @pytest.mark.parametrize('query', QUERIES)
    @pytest.mark.parametrize('w', [3, 12])
    def test_agrees_with_observability(self, w, query):
        oa, fa = _pair('fisher_per_sensor', w=w)
        _assert_agree(oa, fa, **query)

    def test_z_function(self):
        names = [f'z{i}' for i in range(6)]
        oa, fa = _pair('fisher_per_sensor', z_function=z_fn, z_state_names=names)
        assert fa.state_names == names
        for query in (dict(), dict(states=['z0', 'z3'], sensors=['s2', 's4'], R=R_DICT)):
            _assert_agree(oa, fa, **query)

    def test_from_sliding(self):
        oa = ObservabilityAnalysis(SIM, T, X, U, w=6, eps=1e-4, R=0.1).run()
        sliding = SlidingO(O=oa._O, index=oa._index, state_names=oa.state_names, t_sim=T, w=6)
        fa = ObservabilityAnalysis.from_sliding(sliding, R=0.1, storage='fisher_per_sensor')
        assert fa._O is None
        _assert_agree(oa, fa, states=['x0', 'x1'], sensors=['s0', 's4'], R=R_DICT)

    def test_lam_limit(self):
        oa, fa = _pair('fisher_per_sensor', w=3)
        a = oa.min_error_variance(states=['x0', 'x1'], lam='limit')[['x0', 'x1']].dropna().to_numpy()
        b = fa.min_error_variance(states=['x0', 'x1'], lam='limit')[['x0', 'x1']].dropna().to_numpy()
        np.testing.assert_allclose(b, a, rtol=1e-6)

    def test_packed_size(self):
        _, fa = _pair('fisher_per_sensor', w=6)
        assert fa._F.shape == (35, 5, 6 * 7 // 2)   # (n_windows, p, n(n+1)/2)


class TestFisherSummed:

    @pytest.mark.parametrize('query', [dict(), dict(states=['x3']), dict(states=['x5', 'x1'], R=0.37, lam=1e-6)])
    def test_agrees_with_observability(self, query):
        oa, fa = _pair('fisher')
        _assert_agree(oa, fa, **query)

    def test_fisher_sensors_subset(self):
        oa, fa = _pair('fisher', fisher_sensors=['s1', 's3'])
        _assert_agree(oa, fa, sensors=['s1', 's3'], states=['x0', 'x2'])
        _assert_agree(oa, fa, sensors=['s3', 's1'])   # same set, any order
        assert fa._F.shape == (35, 1, 21)

    def test_sensor_selection_raises(self):
        _, fa = _pair('fisher')
        with pytest.raises(ValueError, match="storage='fisher_per_sensor' or storage='observability'"):
            fa.min_error_variance(sensors=['s0'])

    def test_dict_R_raises(self):
        _, fa = _pair('fisher')
        with pytest.raises(ValueError, match="R must be a scalar; use storage='fisher_per_sensor'"):
            fa.min_error_variance(R=R_DICT)

    def test_unknown_fisher_sensors(self):
        with pytest.raises(ValueError, match="unknown fisher_sensors \\['q'\\]"):
            ObservabilityAnalysis(SIM, T, X, U, w=6, storage='fisher', fisher_sensors=['q']).run()


@pytest.mark.parametrize('storage', ['fisher_per_sensor', 'fisher'])
class TestUnsupported:
    """Queries a Fisher mode cannot answer raise and name the setting to change."""

    def test_time_steps(self, storage):
        _, fa = _pair(storage)
        with pytest.raises(ValueError, match="selecting time_steps .* use storage='observability'"):
            fa.min_error_variance(time_steps=[0, 1])

    def test_matrix_R(self, storage):
        _, fa = _pair(storage)
        with pytest.raises(ValueError, match="a matrix R .* use a scalar or per-sensor dict R, or storage='observability'"):
            fa.min_error_variance(R=0.1 * np.eye(30))

    def test_fisher_objects(self, storage):
        _, fa = _pair(storage)
        with pytest.raises(ValueError, match="use fisher_information\\(\\) .* or storage='observability'"):
            fa.fisher()

    def test_O_df_sliding(self, storage):
        _, fa = _pair(storage)
        with pytest.raises(ValueError, match="O_df_sliding needs the observability matrices"):
            fa.O_df_sliding

    def test_save_observability_matrices(self, storage, tmp_path):
        _, fa = _pair(storage)
        with pytest.raises(ValueError, match="include_observability_matrices needs"):
            fa.save_results(tmp_path, include_observability_matrices=True)
        files = fa.save_results(tmp_path)   # the error variance itself can be saved
        assert yaml.safe_load(Path(files['sidecar']).read_text())['storage'] == storage


class TestRecomputeWindow:

    @pytest.mark.parametrize('storage', ['fisher_per_sensor', 'fisher'])
    @pytest.mark.parametrize('z', [False, True])
    def test_matches_stored_O(self, storage, z):
        kwargs = dict(z_function=z_fn, z_state_names=[f'z{i}' for i in range(6)]) if z else {}
        oa, fa = _pair(storage, **kwargs)
        for k in (0, 17, -1):
            pd.testing.assert_frame_equal(fa.observability_matrix(k), oa.observability_matrix(k), check_exact=True)
        pd.testing.assert_frame_equal(fa.observability_matrix(3, states=[oa.state_names[1]], sensors=['s2']),
                                      oa.observability_matrix(3, states=[oa.state_names[1]], sensors=['s2']))

    def test_out_of_range(self):
        _, fa = _pair('fisher')
        with pytest.raises(IndexError):
            fa.observability_matrix(35)

    def test_external_cannot_recompute(self):
        oa = ObservabilityAnalysis(SIM, T, X, U, w=6, eps=1e-4).run()
        fa = ObservabilityAnalysis.from_sliding(SlidingO(O=oa._O, index=oa._index, state_names=oa.state_names),
                                                storage='fisher_per_sensor')
        with pytest.raises(ValueError, match='cannot be recomputed'):
            fa.observability_matrix(0)


class TestFisherInformation:

    @pytest.mark.parametrize('query', [dict(), dict(states=['x3', 'x0'], sensors=['s1', 's4'], R=R_DICT)])
    def test_matches_fisher_objects_exactly(self, query):
        oa = ObservabilityAnalysis(SIM, T, X, U, w=6, eps=1e-4, R=0.1).run()
        F = oa.fisher_information(**query)
        fisher = oa.fisher(**query)
        assert F.shape == (35, len(query.get('states', STATES)), len(query.get('states', STATES)))
        for k in (0, 20, 34):
            np.testing.assert_array_equal(F[k], fisher.FO[k].F.to_numpy())


class TestSettings:

    def test_unknown_storage(self):
        with pytest.raises(ValueError, match="unknown storage 'O'; valid storage modes"):
            ObservabilityAnalysis(SIM, T, X, U, storage='O')

    def test_fisher_sensors_needs_fisher_storage(self):
        with pytest.raises(ValueError, match="fisher_sensors only applies to storage='fisher'"):
            ObservabilityAnalysis(SIM, T, X, U, fisher_sensors=['s0'])

    def test_changing_storage_discards_results(self):
        oa = ObservabilityAnalysis(SIM, T, X, U, w=6, eps=1e-4).run()
        oa.update_settings(storage='fisher_per_sensor')
        assert not oa.is_computed
        oa.run()
        assert oa._O is None and oa._F is not None
        assert 'fisher_per_sensor' in repr(oa)

    def test_yaml_roundtrip(self, tmp_path):
        oa = ObservabilityAnalysis(SIM, T, X, U, w=6, storage='fisher', fisher_sensors=['s0', 's2'])
        path = oa.save_settings(tmp_path / 's.yaml')
        loaded = ObservabilityAnalysis(SIM, T, X, U).load_settings(path)
        assert loaded.settings['storage'] == 'fisher' and loaded.settings['fisher_sensors'] == ['s0', 's2']


def test_jax_fisher_per_sensor():
    jax = pytest.importorskip('jax')
    import pybounds
    Ad = SIM.Ad

    jsim = pybounds.JaxSimulator(lambda x, u: jax.numpy.asarray((Ad - np.eye(6)) / 0.01) @ x,
                                 lambda x, u: jax.numpy.asarray(SIM.C) @ x, dt=0.01, state_names=STATES,
                                 input_names=['u'], measurement_names=SENSORS)
    oa = ObservabilityAnalysis(jsim, T, X, U, w=6, R=0.1).run()
    fa = ObservabilityAnalysis(jsim, T, X, U, w=6, R=0.1, storage='fisher_per_sensor').run()
    _assert_agree(oa, fa, states=['x0', 'x3'], sensors=['s1', 's2'], R=R_DICT)
    np.testing.assert_allclose(fa.observability_matrix(4).values, oa.observability_matrix(4).values, rtol=1e-12)
