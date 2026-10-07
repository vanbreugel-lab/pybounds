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


def _two_sensor_image_O(sensor_major):
    import pandas as pd
    if sensor_major:
        rows = [(s, k) for s in ('r', 'a') for k in range(3)]
    else:
        rows = [(s, k) for k in range(3) for s in ('r', 'a')]
    index = pd.MultiIndex.from_tuples(rows, names=['sensor', 'time_step'])
    return pd.DataFrame(np.arange(12.0).reshape(6, 2), index=index, columns=['g', 'd'])


class TestObservabilityMatrixImageRowLabels:

    @pytest.mark.parametrize('sensor_major', [False, True])
    def test_each_row_labeled_from_its_index(self, sensor_major):
        O = _two_sensor_image_O(sensor_major)
        OI = pybounds.ObservabilityMatrixImage(O)
        assert OI.sensor_names_default == ['r', 'a']
        assert OI.measurement_names == ['${%s}_{,k=%d}$' % (s, k) for s, k in O.index]

    def test_time_major_labels_unchanged(self):
        O = _two_sensor_image_O(sensor_major=False)
        assert pybounds.ObservabilityMatrixImage(O, sensor_names=['sa', 'sb']).measurement_names == \
            ['$sa,_{k=0}$', '$sb,_{k=0}$', '$sa,_{k=1}$', '$sb,_{k=1}$', '$sa,_{k=2}$', '$sb,_{k=2}$']
        assert pybounds.ObservabilityMatrixImage(O, sensor_names=['s']).measurement_names == \
            ['${s}_{0,k=0}$', '${s}_{1,k=0}$', '${s}_{0,k=1}$', '${s}_{1,k=1}$', '${s}_{0,k=2}$', '${s}_{1,k=2}$']


class TestObservabilityMatrixImageSettings:
    """Regression for #16: the constructor's cmap / vmin_ratio / vmax_percentile were overwritten by plot()."""

    @staticmethod
    def _drawn(OI):
        return np.asarray(OI.ax.images[0].get_array())

    def test_constructor_values_reach_plot(self, seom):
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0], cmap='viridis', vmin_ratio=0.5,
                                               vmax_percentile=90)
        OI.plot()
        assert OI.ax.images[0].get_cmap().name == 'viridis'
        O = np.abs(seom.O_df_sliding[0].to_numpy())
        crange = np.percentile(O, 90)
        assert OI.crange == crange
        drawn = np.abs(self._drawn(OI))
        assert drawn[O > 1e-6].min() >= 0.5 * crange * (1 - 1e-12)   # small entries raised to the floor
        plt.close(OI.fig)

    def test_plot_arguments_override_and_persist(self, seom):
        OI = pybounds.ObservabilityMatrixImage(seom.O_df_sliding[0], cmap='viridis')
        OI.plot(cmap='magma')
        assert OI.ax.images[0].get_cmap().name == 'magma'
        plt.close(OI.fig)
        OI.plot()
        assert OI.ax.images[0].get_cmap().name == 'magma'   # a value given to plot() replaces the stored one
        plt.close(OI.fig)

    def test_default_plot_is_unchanged(self, seom):
        """Defaults: bwr, no clipping (vmin_ratio 0), color range the largest |O|, as before."""
        O = seom.O_df_sliding[0]
        OI = pybounds.ObservabilityMatrixImage(O)
        OI.plot()
        assert OI.ax.images[0].get_cmap().name == 'bwr' and OI.vmin_ratio == 0.0
        assert OI.crange == np.abs(O.to_numpy()).max()
        np.testing.assert_array_equal(self._drawn(OI), O.to_numpy())
        plt.close(OI.fig)

    def test_clipping_keeps_sign_and_skips_tiny_values(self):
        import pandas as pd
        index = pd.MultiIndex.from_tuples([('r', 0), ('r', 1), ('r', 2)], names=['sensor', 'time_step'])
        O = pd.DataFrame([[10.0, -0.5], [1e-9, 0.2], [0.0, -10.0]], index=index, columns=['g', 'd'])
        OI = pybounds.ObservabilityMatrixImage(O, vmin_ratio=0.1)
        OI.plot()
        drawn = self._drawn(OI)
        np.testing.assert_array_equal(drawn[:, 0], [10.0, 1e-9, 0.0])   # above the floor, or below 1e-6: untouched
        np.testing.assert_array_equal(drawn[:, 1], [-1.0, 1.0, -10.0])  # raised to 0.1 * 10, sign kept
        plt.close(OI.fig)

    def test_messages_and_names(self, seom):
        with pytest.raises(TypeError, match="O must be a pandas DataFrame with a \\('sensor', 'time_step'\\)"):
            pybounds.ObservabilityMatrixImage(seom.O_sliding[0])
        OI = pybounds.ObservabilityMatrixImage(_two_sensor_image_O(sensor_major=False), sensor_names=['s'])
        assert OI.sensor_names == ['{s}_{0}', '{s}_{1}']   # numbered like the row labels, no stray '$'
        assert OI.measurement_names[:2] == ['${s}_{0,k=0}$', '${s}_{1,k=0}$']
