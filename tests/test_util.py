import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import pytest

from pybounds.util import FixedKeysDict, SetDict, LatexStates, colorline, plot_heatmap_log_timeseries


class TestFixedKeysDict:

    def test_set_existing_key(self):
        d = FixedKeysDict({'a': 1, 'b': 2})
        d['a'] = 99
        assert d['a'] == 99

    def test_add_new_key_raises(self):
        d = FixedKeysDict({'a': 1})
        with pytest.raises(KeyError):
            d['new_key'] = 5

    def test_delete_raises(self):
        d = FixedKeysDict({'a': 1})
        with pytest.raises(KeyError):
            del d['a']

    def test_pop_raises(self):
        d = FixedKeysDict({'a': 1})
        with pytest.raises(KeyError):
            d.pop('a')

    def test_clear_raises(self):
        d = FixedKeysDict({'a': 1})
        with pytest.raises(KeyError):
            d.clear()

    def test_update_existing_key(self):
        d = FixedKeysDict({'a': 1, 'b': 2})
        d.update({'a': 10})
        assert d['a'] == 10

    def test_update_new_key_raises(self):
        d = FixedKeysDict({'a': 1})
        with pytest.raises(KeyError):
            d.update({'new_key': 5})

    def test_is_dict_subclass(self):
        d = FixedKeysDict({'a': 1})
        assert isinstance(d, dict)


class TestSetDict:

    def test_overwrite_changes_existing_value(self):
        target = {'a': 1, 'b': 2}
        SetDict().set_dict_with_overwrite(target, {'a': 99})
        assert target['a'] == 99

    def test_preserve_keeps_existing_value(self):
        target = {'a': 1, 'b': 2}
        SetDict().set_dict_with_preserve(target, {'a': 99})
        assert target['a'] == 1

    def test_overwrite_adds_new_key_from_source(self):
        target = {'a': 1}
        SetDict().set_dict_with_overwrite(target, {'b': 2})
        assert target['b'] == 2

    def test_preserve_adds_new_key_from_source(self):
        target = {'a': 1}
        SetDict().set_dict_with_preserve(target, {'b': 2})
        assert target['b'] == 2


class TestLatexStates:

    def test_known_state_d_converts(self):
        lc = LatexStates()
        result = lc.convert_to_latex(['d'])
        assert result == [r'$d$']

    def test_known_state_phi_converts(self):
        lc = LatexStates()
        result = lc.convert_to_latex(['phi'])
        assert result == [r'$\phi$']

    def test_single_string_returns_string(self):
        lc = LatexStates()
        result = lc.convert_to_latex('phi')
        assert isinstance(result, str)
        assert result == r'$\phi$'

    def test_unknown_state_unchanged(self):
        lc = LatexStates()
        result = lc.convert_to_latex(['my_custom_state'])
        assert result == ['my_custom_state']

    def test_remove_dollar_signs(self):
        lc = LatexStates()
        result = lc.convert_to_latex(['phi'], remove_dollar_signs=True)
        assert '$' not in result[0]

    def test_list_length_preserved(self):
        lc = LatexStates()
        result = lc.convert_to_latex(['phi', 'my_custom_state'])
        assert len(result) == 2


class TestColorline:

    def test_returns_line_collection(self):
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, 20)
        y = np.sin(x)
        z = x
        lc = colorline(x, y, z, ax=ax)
        assert isinstance(lc, mcoll.LineCollection)
        plt.close('all')

    def test_scalar_z_does_not_raise(self):
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, 10)
        y = np.zeros(10)
        lc = colorline(x, y, z=0.5, ax=ax)
        assert lc is not None
        plt.close('all')

    def test_creates_ax_if_none_given(self):
        x = np.linspace(0, 1, 10)
        y = np.zeros(10)
        lc = colorline(x, y, z=x)
        assert lc is not None
        plt.close('all')


class TestPlotHeatmapLogTimeseries:

    def test_returns_three_tuple(self):
        data = np.abs(np.random.randn(5, 3)) + 0.01
        result = plot_heatmap_log_timeseries(data)
        assert len(result) == 3
        plt.close('all')

    def test_cnorm_not_none(self):
        data = np.abs(np.random.randn(4, 2)) + 0.01
        cnorm, cmap, ticks = plot_heatmap_log_timeseries(data)
        assert cnorm is not None
        plt.close('all')
