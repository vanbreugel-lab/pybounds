import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pybounds
from conftest import WINDOW_SIZE


class TestObservabilityMatrixImageConstruction:

    def test_construction_from_df(self, seom):
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0], cmap='bwr')
        assert OI is not None

    def test_n_sensor(self, seom):
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0])
        assert OI.n_sensor == 1

    def test_n_time_step(self, seom):
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0])
        assert OI.n_time_step == WINDOW_SIZE

    def test_pw(self, seom):
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0])
        assert OI.pw == WINDOW_SIZE * 1  # p=1, w=WINDOW_SIZE

    def test_n_states(self, seom):
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0])
        assert OI.n == 2

    def test_state_names_default(self, seom):
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0])
        assert 'g' in OI.state_names_default
        assert 'd' in OI.state_names_default

    def test_numpy_array_raises_type_error(self, seom):
        with pytest.raises(TypeError):
            pybounds.ObservabilityMatrixImage(seom.O_sliding[0])

    def test_state_names_wrong_length_raises_type_error(self, seom):
        with pytest.raises(TypeError):
            pybounds.ObservabilityMatrixImage(
                seom.O_df_sliding[0], state_names=['g', 'd', 'extra']
            )


class TestObservabilityMatrixImagePlot:

    def test_plot_runs_without_error(self, seom):
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0], cmap='bwr')
        OI.plot(scale=1.0)
        plt.close('all')

    def test_plot_stores_figure(self, seom):
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0])
        OI.plot(scale=1.0)
        assert isinstance(OI.fig, plt.Figure)
        plt.close('all')

    def test_plot_stores_ax(self, seom):
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0])
        OI.plot(scale=1.0)
        assert OI.ax is not None
        plt.close('all')

    def test_plot_with_external_ax_stores_none_fig(self, seom):
        """When an external ax is passed, OI.fig should remain None."""
        fig, ax = plt.subplots()
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0])
        OI.plot(ax=ax)
        assert OI.fig is None
        plt.close('all')
