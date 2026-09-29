import numpy as np
import sympy as sp
import pytest
import pybounds


def _make_identity_jac():
    x0, x1 = sp.symbols('x_0 x_1')
    def f(x):
        return sp.Matrix([x[0], x[1]])
    return pybounds.SymbolicJacobian(f, [x0, x1])


def _make_squared_jac():
    x0, x1 = sp.symbols('x_0 x_1')
    def f(x):
        return sp.Matrix([x[0] ** 2, x[1] ** 2])
    return pybounds.SymbolicJacobian(f, [x0, x1])


def _make_optic_flow_jac():
    g, d = sp.symbols('x_0 x_1')
    def h(x):
        return sp.Matrix([x[0] / x[1]])
    return pybounds.SymbolicJacobian(h, [g, d])


class TestSymbolicJacobianConstruction:

    def test_construction_from_function(self):
        x0, x1 = sp.symbols('x_0 x_1')
        def f(x):
            return sp.Matrix([x[0] ** 2, x[0] * x[1]])
        SJ = pybounds.SymbolicJacobian(f, [x0, x1])
        assert SJ.jacobian_symbolic is not None

    def test_jacobian_shape_two_outputs(self):
        x0, x1 = sp.symbols('x_0 x_1')
        def f(x):
            return sp.Matrix([x[0] ** 2, x[0] * x[1]])
        SJ = pybounds.SymbolicJacobian(f, [x0, x1])
        assert SJ.jacobian_symbolic.shape == (2, 2)

    def test_jacobian_shape_one_output(self):
        x0, x1 = sp.symbols('x_0 x_1')
        def h(x):
            return sp.Matrix([x[0] / x[1]])
        SJ = pybounds.SymbolicJacobian(h, [x0, x1])
        assert SJ.jacobian_symbolic.shape == (1, 2)

    def test_get_jacobian_function_returns_callable(self):
        SJ = _make_identity_jac()
        jac_func = SJ.get_jacobian_function()
        assert callable(jac_func)


class TestSymbolicJacobianNumericalValues:

    def test_identity_jacobian(self):
        """Jacobian of f(x) = x is the identity matrix."""
        SJ = _make_identity_jac()
        jac_func = SJ.get_jacobian_function()
        result = np.array(jac_func(np.array([1.0, 2.0])), dtype=float)
        assert np.allclose(result, np.eye(2), atol=1e-10)

    def test_squared_jacobian_at_2_3(self):
        """Jacobian of [x0^2, x1^2] at (2,3) is diag(4, 6)."""
        SJ = _make_squared_jac()
        jac_func = SJ.get_jacobian_function()
        result = np.array(jac_func(np.array([2.0, 3.0])), dtype=float)
        expected = np.diag([4.0, 6.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_optic_flow_jacobian_at_2_3(self):
        """Jacobian of g/d at (g=2, d=3): [1/d, -g/d^2] = [1/3, -2/9]."""
        SJ = _make_optic_flow_jac()
        jac_func = SJ.get_jacobian_function()
        result = np.array(jac_func(np.array([2.0, 3.0])), dtype=float)
        expected = np.array([[1.0 / 3.0, -2.0 / 9.0]])
        assert np.allclose(result, expected, atol=1e-10)

    def test_output_shape_one_output(self):
        SJ = _make_optic_flow_jac()
        jac_func = SJ.get_jacobian_function()
        result = np.array(jac_func(np.array([2.0, 3.0])), dtype=float)
        assert result.shape == (1, 2)

    def test_output_shape_two_outputs(self):
        SJ = _make_squared_jac()
        jac_func = SJ.get_jacobian_function()
        result = np.array(jac_func(np.array([2.0, 3.0])), dtype=float)
        assert result.shape == (2, 2)
