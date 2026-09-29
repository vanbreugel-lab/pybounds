import numpy as np
import pytest
import pybounds
from conftest import N_STEPS_SLIDING, WINDOW_SIZE, EPS, AnalyticSimulator


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
        with pytest.raises(ValueError, match='window size.*must be smaller'):
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


class TestSEOMParallel:

    @staticmethod
    def _trajectory(simulation_output):
        t_sim, x_sim, u_sim, _ = simulation_output
        return (t_sim[:N_STEPS_SLIDING],
                {k: v[:N_STEPS_SLIDING] for k, v in x_sim.items()},
                {k: v[:N_STEPS_SLIDING] for k, v in u_sim.items()})

    def test_parallel_sliding_with_pybounds_simulator_warns_and_matches_sequential(
            self, simulator, simulation_output, seom):
        t_s, x_s, u_s = self._trajectory(simulation_output)
        with pytest.warns(RuntimeWarning, match='running windows sequentially'):
            seom_par = pybounds.SlidingEmpiricalObservabilityMatrix(
                simulator, t_s, x_s, u_s, w=WINDOW_SIZE, eps=EPS, parallel_sliding=True)
        assert seom_par.parallel_sliding is False
        for O_par, O_seq in zip(seom_par.O_sliding, seom.O_sliding):
            assert np.allclose(O_par, O_seq)

    def test_parallel_sliding_with_thread_safe_simulator_runs_threaded(self, simulation_output, recwarn):
        t_s, x_s, u_s = self._trajectory(simulation_output)
        seom_seq = pybounds.SlidingEmpiricalObservabilityMatrix(
            AnalyticSimulator(), t_s, x_s, u_s, w=WINDOW_SIZE, eps=EPS)
        seom_par = pybounds.SlidingEmpiricalObservabilityMatrix(
            AnalyticSimulator(), t_s, x_s, u_s, w=WINDOW_SIZE, eps=EPS, parallel_sliding=True)
        assert seom_par.parallel_sliding is True
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
        for O_par, O_seq in zip(seom_par.O_sliding, seom_seq.O_sliding):
            assert np.allclose(O_par, O_seq)
