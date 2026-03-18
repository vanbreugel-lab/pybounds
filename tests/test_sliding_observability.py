import numpy as np
import pytest
import pybounds
from conftest import N_STEPS_SLIDING, WINDOW_SIZE, EPS


class TestSEOMTypes:

    def test_O_df_sliding_is_list(self, seom):
        assert isinstance(seom.O_df_sliding, list)

    def test_O_sliding_is_list(self, seom):
        assert isinstance(seom.O_sliding, list)

    def test_t_sim_stored(self, seom):
        assert len(seom.t_sim) == N_STEPS_SLIDING


class TestSEOMWindowCount:

    def test_number_of_windows(self, seom):
        expected = N_STEPS_SLIDING - WINDOW_SIZE + 1
        assert len(seom.O_df_sliding) == expected

    def test_O_sliding_count_matches_df_count(self, seom):
        assert len(seom.O_sliding) == len(seom.O_df_sliding)


class TestSEOMWindowShapes:

    def test_each_O_shape(self, seom):
        for i, O in enumerate(seom.O_sliding):
            assert O.shape == (WINDOW_SIZE * 1, 2), \
                f"Window {i}: expected ({WINDOW_SIZE}, 2), got {O.shape}"

    def test_each_O_df_columns(self, seom):
        for i, df in enumerate(seom.O_df_sliding):
            assert list(df.columns) == ['g', 'd'], f"Window {i} wrong columns"

    def test_each_O_df_index_names(self, seom):
        for i, df in enumerate(seom.O_df_sliding):
            assert df.index.names == ['sensor', 'time_step'], \
                f"Window {i} wrong index names"

    def test_each_O_finite(self, seom):
        for i, O in enumerate(seom.O_sliding):
            assert np.all(np.isfinite(O)), f"Window {i} has non-finite values"


class TestSEOMGetObservabilityMatrix:

    def test_returns_list(self, seom):
        result = seom.get_observability_matrix()
        assert isinstance(result, list)

    def test_returns_copy(self, seom):
        result = seom.get_observability_matrix()
        assert result is not seom.O_df_sliding

    def test_correct_length(self, seom):
        result = seom.get_observability_matrix()
        assert len(result) == len(seom.O_df_sliding)

    def test_elements_are_dataframes(self, seom):
        import pandas as pd
        result = seom.get_observability_matrix()
        for df in result:
            assert isinstance(df, pd.DataFrame)


class TestSEOMValidation:

    def test_raises_if_window_exceeds_trajectory(self, simulator, simulation_output):
        t_sim, x_sim, u_sim, _ = simulation_output
        t_s = t_sim[:10]
        x_s = {k: v[:10] for k, v in x_sim.items()}
        u_s = {k: v[:10] for k, v in u_sim.items()}
        with pytest.raises(ValueError, match='window size must be smaller'):
            pybounds.SlidingEmpiricalObservabilityMatrix(
                simulator, t_s, x_s, u_s, w=20, eps=EPS,
            )

    def test_raises_if_t_x_size_mismatch(self, simulator, simulation_output):
        t_sim, x_sim, u_sim, _ = simulation_output
        t_s = t_sim[:20]
        x_s = {k: v[:15] for k, v in x_sim.items()}  # shorter than t_s
        u_s = {k: v[:20] for k, v in u_sim.items()}
        with pytest.raises(ValueError, match='t_sim & x_sim must have same number of rows'):
            pybounds.SlidingEmpiricalObservabilityMatrix(
                simulator, t_s, x_s, u_s, w=6, eps=EPS,
            )
