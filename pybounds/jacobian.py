import sympy as sp
import numpy as np


class SymbolicJacobian:
    def __init__(self, func, state_vars):
        """
        Initialize the Jacobian calculator with a function or symbolic expression for f(x).

        :param func: A Python function or sympy Matrix representing f(x).
                     If func is a Python function, it should take in x and u (e.g., f(x))
                     and return a sympy Matrix for xdot.
                     If func is a sympy Matrix, it directly represents xdot.
        :param state_vars: List of sympy symbols for variables (e.g., [x1, x2]).
        """

        self.state_vars = sp.Matrix(state_vars)  # state variables as sympy Matrix

        # Detect if func is symbolic (sympy.Matrix) or a Python function
        if isinstance(func, sp.Matrix):
            # If func is a symbolic Matrix, use it directly as xdot
            self.z_sym = func
        else:
            # If func is a Python function, call it with symbolic state and control variables
            x_sym = self.state_vars
            self.z_sym = func(x_sym)  # get the symbolic expression from the function

        # Calculate the Jacobian of xdot_sym with respect to the state variables
        self.jacobian_symbolic = self.z_sym.jacobian(self.state_vars)
        self.jacobian_symbolic = sp.simplify(self.jacobian_symbolic)

    def get_jacobian_function(self):
        """
        Returns a numerical function to evaluate the Jacobian at specific values of x and u.

        :return: A function that calculates the Jacobian matrix given values of x and u.
                 The function takes numpy arrays x and u as inputs.
        """
        # Convert the symbolic Jacobian to a numerical function using lambdify
        jacobian_func = sp.lambdify(self.state_vars, self.jacobian_symbolic, 'numpy')

        # Return a wrapper function that accepts x as numpy array
        def jacobian_numerical(x):
            return jacobian_func(*np.array(x))

        return jacobian_numerical
