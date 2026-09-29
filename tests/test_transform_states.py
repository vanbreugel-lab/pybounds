import numpy as np
import sympy as sp
import pytest
import pybounds


def _identity_z(x):
    return sp.Matrix([x[0], x[1]])


def _optic_flow_z(x):
    """Transform [g, d] → [g/d, d]."""
    return sp.Matrix([x[0] / x[1], x[1]])


X0 = np.array([2.0, 3.0])


class TestTransformStatesReturnType:

    def test_returns_three_tuple(self, eom):
        result = pybounds.transform_states(
            O=eom.O_df, z_function=_identity_z, x0=X0,
        )
        assert len(result) == 3

    def test_O_z_is_dataframe(self, eom):
        import pandas as pd
        O_z, _, _ = pybounds.transform_states(
            O=eom.O_df, z_function=_optic_flow_z, x0=X0,
        )
        assert isinstance(O_z, pd.DataFrame)

    def test_dxdz_is_ndarray(self, eom):
        _, dxdz, _ = pybounds.transform_states(
            O=eom.O_df, z_function=_optic_flow_z, x0=X0,
        )
        assert isinstance(dxdz, np.ndarray)

    def test_dzdx_sym_is_sympy_matrix(self, eom):
        _, _, dzdx_sym = pybounds.transform_states(
            O=eom.O_df, z_function=_optic_flow_z, x0=X0,
        )
        assert isinstance(dzdx_sym, sp.matrices.MatrixBase)


class TestTransformStatesShapes:

    def test_O_z_shape_preserved(self, eom):
        O_z, _, _ = pybounds.transform_states(
            O=eom.O_df, z_function=_optic_flow_z, x0=X0,
        )
        assert O_z.shape == eom.O_df.shape

    def test_dxdz_is_square_n_by_n(self, eom):
        _, dxdz, _ = pybounds.transform_states(
            O=eom.O_df, z_function=_optic_flow_z, x0=X0,
        )
        assert dxdz.shape == (2, 2)


class TestTransformStatesValues:

    def test_identity_transform_preserves_O(self, eom):
        """z = x (identity) should leave O unchanged."""
        O_z, _, _ = pybounds.transform_states(
            O=eom.O_df, z_function=_identity_z, x0=X0,
        )
        assert np.allclose(O_z.values, eom.O_df.values, atol=1e-8)

    def test_z_state_names_applied_to_columns(self, eom):
        O_z, _, _ = pybounds.transform_states(
            O=eom.O_df,
            z_function=_optic_flow_z,
            x0=X0,
            z_state_names=['optic_flow', 'height'],
        )
        assert list(O_z.columns) == ['optic_flow', 'height']


def _scale_z(x):
    """z = [2 x_0, x_1], so dz/dx = diag(2, 1) and dx/dz = diag(0.5, 1)."""
    return sp.Matrix([2 * x[0], x[1]])


class TestTransformStatesJacobianLabels:

    def test_returned_jacobians_match_their_names(self, eom):
        O_z, dxdz, dzdx_sym = pybounds.transform_states(O=eom.O_df, z_function=_scale_z, x0=X0)
        np.testing.assert_allclose(dxdz, np.diag([0.5, 1.0]))
        assert dzdx_sym == sp.diag(2, 1)
        np.testing.assert_allclose(O_z.values, eom.O_df.values @ dxdz)

    def test_eom_attributes_and_deprecated_aliases(self, simulator):
        x0 = {'g': 2.0, 'd': 3.0}
        u = {'u': 0.1 * np.ones(20)}
        eom_z = pybounds.EmpiricalObservabilityMatrix(simulator, x0, u, z_function=_scale_z)
        np.testing.assert_allclose(eom_z.dxdz, np.diag([0.5, 1.0]))
        assert eom_z.dzdx_sym == sp.diag(2, 1)
        with pytest.warns(DeprecationWarning, match='Use dxdz instead'):
            np.testing.assert_allclose(eom_z.dzdx, eom_z.dxdz)
        with pytest.warns(DeprecationWarning, match='Use dzdx_sym instead'):
            assert eom_z.dxdz_sym == eom_z.dzdx_sym
