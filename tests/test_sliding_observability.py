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


class TestSEOMDictOrder:

    def test_dict_key_order_does_not_matter(self, simulator, simulation_output, seom):
        t_sim, x_sim, u_sim, _ = simulation_output
        x_rev = {k: x_sim[k][:N_STEPS_SLIDING] for k in ['d', 'g']}
        u_s = {k: v[:N_STEPS_SLIDING] for k, v in u_sim.items()}
        seom_rev = pybounds.SlidingEmpiricalObservabilityMatrix(
            simulator, t_sim[:N_STEPS_SLIDING], x_rev, u_s, w=WINDOW_SIZE, eps=EPS)
        for O_rev, O in zip(seom_rev.O_sliding, seom.O_sliding):
            assert np.allclose(O_rev, O)


def _module_level_factory():
    return AnalyticSimulator()


class TestSpawnPicklabilityCheck:
    """parallel_sliding with a simulator_factory must fail fast, not hang, for callables workers can't load."""

    @pytest.fixture
    def interactive_main(self, monkeypatch):
        """Pretend we're in a notebook: __main__ has no __file__, and a factory is defined there."""
        import sys
        import types
        main = types.ModuleType('__main__')
        monkeypatch.setitem(sys.modules, '__main__', main)

        def make_sim():
            return AnalyticSimulator()
        make_sim.__module__ = '__main__'
        make_sim.__qualname__ = 'make_sim'
        main.make_sim = make_sim
        return make_sim

    def _seom(self, simulation_output, **kwargs):
        t_sim, x_sim, u_sim, _ = simulation_output
        return pybounds.SlidingEmpiricalObservabilityMatrix(
            AnalyticSimulator(), t_sim[:N_STEPS_SLIDING],
            {k: v[:N_STEPS_SLIDING] for k, v in x_sim.items()},
            {k: v[:N_STEPS_SLIDING] for k, v in u_sim.items()},
            w=WINDOW_SIZE, eps=EPS, parallel_sliding=True, **kwargs)

    def test_factory_from_interactive_main_raises(self, simulation_output, interactive_main):
        with pytest.raises(ValueError, match="simulator_factory uses 'make_sim'.*interactive session"):
            self._seom(simulation_output, simulator_factory=interactive_main)

    def test_partial_wrapping_interactive_function_raises(self, simulation_output, interactive_main):
        import functools
        with pytest.raises(ValueError, match="interactive session"):
            self._seom(simulation_output, simulator_factory=functools.partial(interactive_main))

    def test_z_function_from_interactive_main_raises(self, simulation_output, interactive_main):
        with pytest.raises(ValueError, match="z_function uses 'make_sim'"):
            self._seom(simulation_output, simulator_factory=_module_level_factory, z_function=interactive_main)

    def test_lambda_factory_raises(self, simulation_output):
        with pytest.raises(ValueError, match='simulator_factory must be picklable'):
            self._seom(simulation_output, simulator_factory=lambda: AnalyticSimulator())

    def test_module_level_factory_passes_check(self, interactive_main):
        from pybounds.observability import _check_spawn_picklable
        _check_spawn_picklable(_module_level_factory, 'simulator_factory')  # no error
