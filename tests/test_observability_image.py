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


def _underscore_sensor_O():
    """O with the default-style sensor name 'y_0' (and a state name with an underscore)."""
    sim = pybounds.Simulator(lambda X, U: [U[0], 0 * U[0]], lambda X, U: [X[0] / X[1]], dt=0.01,
                             state_names=['g', 'd'], input_names=['u'])   # default measurement name 'y_0'
    return pybounds.EmpiricalObservabilityMatrix(sim, {'g': 2.0, 'd': 3.0}, {'u': 0.1 * np.ones(4)}, eps=1e-4).O_df


class TestObservabilityMatrixImageLabels:

    @pytest.mark.parametrize('kwargs', [{}, {'sensor_names': ['y_a']}, {'state_names': ['x_s']}])
    def test_names_with_underscores_render(self, kwargs):
        O = _underscore_sensor_O()
        assert set(O.index.get_level_values('sensor')) == {'y_0'}
        OI = pybounds.ObservabilityMatrixImage(O, **kwargs)
        OI.plot()
        OI.fig.canvas.draw()   # mathtext labels are only parsed when drawn
        plt.close(OI.fig)


class TestObservabilityMatrixImageClipping:

    def test_vmin_ratio_does_not_modify_O(self, seom):
        O = seom.O_df_sliding[0]
        OI = pybounds.ObservabilityMatrixImage(O)
        before = OI.O.values.copy()
        OI.plot(vmin_ratio=0.9)
        OI.fig.canvas.draw()
        plt.close(OI.fig)
        np.testing.assert_array_equal(OI.O.values, before)
        np.testing.assert_array_equal(O.values, before)


class TestObservabilityMatrixImageNameTypes:

    def test_tuple_state_and_sensor_names(self, seom):
        """EmpiricalObservabilityMatrix stores state_names as a tuple when z_function is used."""
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0], state_names=('a', 'b'), sensor_names=('s',))
        assert len(OI.state_names) == 2
        OI.plot()
        OI.fig.canvas.draw()
        plt.close(OI.fig)
