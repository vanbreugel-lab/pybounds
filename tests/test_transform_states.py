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

    def test_dzdx_is_ndarray(self, eom):
        _, dzdx, _ = pybounds.transform_states(
            O=eom.O_df, z_function=_optic_flow_z, x0=X0,
        )
        assert isinstance(dzdx, np.ndarray)

    def test_dxdz_sym_is_sympy_matrix(self, eom):
        _, _, dxdz_sym = pybounds.transform_states(
            O=eom.O_df, z_function=_optic_flow_z, x0=X0,
        )
        assert isinstance(dxdz_sym, sp.matrices.MatrixBase)


class TestTransformStatesShapes:

    def test_O_z_shape_preserved(self, eom):
        O_z, _, _ = pybounds.transform_states(
            O=eom.O_df, z_function=_optic_flow_z, x0=X0,
        )
        assert O_z.shape == eom.O_df.shape

    def test_dzdx_is_square_n_by_n(self, eom):
        _, dzdx, _ = pybounds.transform_states(
            O=eom.O_df, z_function=_optic_flow_z, x0=X0,
        )
        assert dzdx.shape == (2, 2)


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
