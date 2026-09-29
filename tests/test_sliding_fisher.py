import numpy as np
import pandas as pd
import pytest
import pybounds
from conftest import N_STEPS_SLIDING, WINDOW_SIZE


class TestSlidingFisherOutputType:

    def test_returns_dataframe(self, sliding_fisher):
        ev = sliding_fisher.get_minimum_error_variance()
        assert isinstance(ev, pd.DataFrame)

    def test_returns_copy(self, sliding_fisher):
        ev1 = sliding_fisher.get_minimum_error_variance()
        ev2 = sliding_fisher.get_minimum_error_variance()
        assert ev1 is not ev2


class TestSlidingFisherColumns:

    def test_has_time_column(self, sliding_fisher):
        ev = sliding_fisher.get_minimum_error_variance()
        assert 'time' in ev.columns

    def test_has_time_initial_column(self, sliding_fisher):
        ev = sliding_fisher.get_minimum_error_variance()
        assert 'time_initial' in ev.columns

    def test_has_g_column(self, sliding_fisher):
        ev = sliding_fisher.get_minimum_error_variance()
        assert 'g' in ev.columns

    def test_has_d_column(self, sliding_fisher):
        ev = sliding_fisher.get_minimum_error_variance()
        assert 'd' in ev.columns


class TestSlidingFisherValues:

    def test_error_variance_g_positive(self, sliding_fisher):
        ev = sliding_fisher.get_minimum_error_variance()
        g_vals = ev['g'].dropna().values
        assert np.all(g_vals > 0)

    def test_error_variance_d_positive(self, sliding_fisher):
        ev = sliding_fisher.get_minimum_error_variance()
        d_vals = ev['d'].dropna().values
        assert np.all(d_vals > 0)

    def test_time_column_monotonic(self, sliding_fisher):
        ev = sliding_fisher.get_minimum_error_variance()
        t = ev['time'].dropna().values
        assert np.all(np.diff(t) >= 0)

    def test_shift_index(self, sliding_fisher):
        expected = WINDOW_SIZE // 2
        assert sliding_fisher.shift_index == expected

    def test_fo_list_length(self, sliding_fisher):
        expected = N_STEPS_SLIDING - WINDOW_SIZE + 1
        assert len(sliding_fisher.FO) == expected


class TestSlidingFisherDefaults:

    def test_default_lam_matches_explicit_1e_8(self, seom):
        sfo_default = pybounds.SlidingFisherObservability(seom.O_df_sliding, R={'r': 0.1})
        sfo_explicit = pybounds.SlidingFisherObservability(seom.O_df_sliding, R={'r': 0.1}, lam=1e-8)
        assert np.allclose(sfo_default.EV[['g', 'd']].values, sfo_explicit.EV[['g', 'd']].values)


class TestSlidingFisherTimeAlignment:

    @pytest.mark.parametrize('w', [3, 4, 5, 6, 7, 9])
    def test_error_variance_stamped_at_window_center(self, w):
        n_window, dt = 8, 0.1
        time = np.arange(n_window + w - 1) * dt
        index = pd.MultiIndex.from_tuples([('r', k) for k in range(w)], names=['sensor', 'time_step'])
        rng = np.random.default_rng(1)
        O_list = [pd.DataFrame(rng.normal(size=(w, 2)), index=index, columns=['g', 'd']) for _ in range(n_window)]
        sfo = pybounds.SlidingFisherObservability(O_list, R=0.1, time=time)
        assert sfo.shift_index == w // 2
        ev = sfo.get_minimum_error_variance()
        first = ev.dropna(subset=['g']).iloc[0]
        assert np.isclose(first['time_initial'], 0.0)
        assert np.isclose(first['time'], (w // 2) * dt)


class TestSlidingFisherSingleWindow:

    def test_compute_observability_single_window_has_time(self, simulator):
        n = 8
        t, x, u, _ = simulator.simulate(x0={'g': 2.0, 'd': 3.0}, u={'u': 0.1 * np.ones(n)},
                                        return_full_output=True)
        ev = pybounds.compute_observability(simulator, t, x, u, R={'r': 0.1}, w=n)
        assert {'time', 'time_initial', 'g', 'd'} <= set(ev.columns)
        row = ev.dropna(subset=['g'])
        assert len(row) == 1
        assert np.isclose(row['time'].item(), t[n // 2])

    def test_single_window_without_time(self):
        index = pd.MultiIndex.from_tuples([('r', k) for k in range(4)], names=['sensor', 'time_step'])
        O = pd.DataFrame(np.random.default_rng(2).normal(size=(4, 2)), index=index, columns=['g', 'd'])
        ev = pybounds.SlidingFisherObservability([O], R=0.1).get_minimum_error_variance()
        assert list(ev.columns[:2]) == ['time', 'time_initial']
        assert ev['time'].item() == 2


class TestSlidingFisherForceRScalar:

    def test_force_R_scalar_matches_matrix_R(self, seom):
        kwargs = dict(R=0.1, time=seom.t_sim, lam=1e-8)
        ev = pybounds.SlidingFisherObservability(seom.O_df_sliding, **kwargs).EV_aligned
        ev_forced = pybounds.SlidingFisherObservability(seom.O_df_sliding, force_R_scalar=True, **kwargs).EV_aligned
        pd.testing.assert_frame_equal(ev_forced, ev)

    def test_force_R_scalar_rejects_dict_R(self, seom):
        with pytest.raises(Exception, match='R must be a scalar'):
            pybounds.SlidingFisherObservability(seom.O_df_sliding, R={'r': 0.1}, force_R_scalar=True)
