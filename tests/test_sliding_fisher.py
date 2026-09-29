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
