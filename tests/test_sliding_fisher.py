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
        expected = int(np.round(0.5 * WINDOW_SIZE))
        assert sliding_fisher.shift_index == expected

    def test_fo_list_length(self, sliding_fisher):
        expected = N_STEPS_SLIDING - WINDOW_SIZE + 1
        assert len(sliding_fisher.FO) == expected


class TestSlidingFisherDefaults:

    def test_default_lam_matches_explicit_1e_8(self, seom):
        sfo_default = pybounds.SlidingFisherObservability(seom.O_df_sliding, R={'r': 0.1})
        sfo_explicit = pybounds.SlidingFisherObservability(seom.O_df_sliding, R={'r': 0.1}, lam=1e-8)
        assert np.allclose(sfo_default.EV[['g', 'd']].values, sfo_explicit.EV[['g', 'd']].values)
