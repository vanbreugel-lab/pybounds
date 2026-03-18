"""
JAX-based simulator and observability matrix for pybounds.

Provides drop-in alternatives to ``Simulator`` and ``EmpiricalObservabilityMatrix``
that use JAX's forward-mode autodiff (``jax.jacfwd``) instead of numerical
perturbations.  All classes produce the same output formats as their legacy
counterparts so downstream ``FisherObservability`` / ``SlidingFisherObservability``
require no changes.

Requirements
------------
- JAX must be installed: ``pip install "jax[cpu]"``
- ``f_jax`` and ``h_jax`` must use ``jax.numpy`` (not ``numpy``) for math
  operations so that JAX can trace through them.  Plain Python arithmetic
  operators (``+``, ``-``, ``*``, ``/``) work with both backends as-is.

Pipeline hand-off
-----------------
do_mpc ``Simulator`` remains the entry point for MPC trajectory reconstruction
(finding control inputs from a measured trajectory).  Once ``(t_sim, x_sim,
u_sim)`` are known, ``JaxSimulator`` takes over for the observability analysis:

    do_mpc Simulator  →  MPC  →  (t_sim, x_sim, u_sim)
                                        ↓
                            JaxSimulator / JaxEmpiricalObservabilityMatrix
                                        ↓
                              O_df  (same format as legacy)
                                        ↓
                      FisherObservability / SlidingFisherObservability  (unchanged)
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)   # use float64 to match do_mpc precision


# ---------------------------------------------------------------------------
# JaxSimulator
# ---------------------------------------------------------------------------

class JaxSimulator:
    """Open-loop forward simulator implemented in pure JAX.

    Uses ``jax.lax.scan`` over RK4 (or Euler) time steps so the simulation
    function is fully JAX-traceable.  This makes the measurement trajectory
    Y differentiable with respect to the initial state x₀ via ``jax.jacfwd``.

    Parameters
    ----------
    f_jax : callable
        Dynamics function ``f_jax(x, u) -> x_dot``.
        ``x`` and ``u`` are 1-D ``jnp`` arrays; return value must also be a
        ``jnp`` array of shape ``(n,)``.
    h_jax : callable
        Measurement function ``h_jax(x, u) -> y``.
        Returns a ``jnp`` array of shape ``(p,)``.
    dt : float
        Integration time step (seconds).
    state_names : list of str
        Names of state variables (length n).
    input_names : list of str
        Names of input variables (length m).
    measurement_names : list of str
        Names of measurement variables (length p).
    integrator : {'rk4', 'euler'}
        Numerical integration scheme.  ``'rk4'`` is more accurate and
        recommended; ``'euler'`` is faster but first-order.
    """

    def __init__(self, f_jax, h_jax, dt,
                 state_names, input_names, measurement_names,
                 integrator='rk4'):
        self.f_jax = f_jax
        self.h_jax = h_jax
        self.dt = float(dt)
        self.state_names = list(state_names)
        self.input_names = list(input_names)
        self.measurement_names = list(measurement_names)
        self.n = len(state_names)
        self.m = len(input_names)
        self.p = len(measurement_names)
        self.integrator = integrator

        # Build the pure JAX simulation function once and store it.
        self._simulate_jax = self._build_simulate()

    def _build_simulate(self):
        """Return a pure JAX function  simulate(x0, u_seq) -> y_traj.

        x0    : shape (n,)
        u_seq : shape (w, m)  — one input vector per time step
        y_traj: shape (w, p)  — measurement at every time step
        """
        dt = self.dt
        f = self.f_jax
        h = self.h_jax
        integrator = self.integrator

        def euler_step(x, u):
            return x + dt * jnp.asarray(f(x, u))

        def rk4_step(x, u):
            k1 = jnp.asarray(f(x,              u))
            k2 = jnp.asarray(f(x + dt / 2 * k1, u))
            k3 = jnp.asarray(f(x + dt / 2 * k2, u))
            k4 = jnp.asarray(f(x + dt * k3,      u))
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        step_fn = rk4_step if integrator == 'rk4' else euler_step

        def simulate(x0, u_seq):
            """Integrate forward and return measurement trajectory.

            Measurement at step k is evaluated *before* integrating step k,
            matching the convention used by ``Simulator.simulate()``.
            """
            def scan_fn(x, u):
                y = jnp.asarray(h(x, u))
                x_next = step_fn(x, u)
                return x_next, y

            x0_arr = jnp.asarray(x0, dtype=jnp.float64)
            u_arr = jnp.asarray(u_seq, dtype=jnp.float64)
            _, y_traj = jax.lax.scan(scan_fn, x0_arr, u_arr)
            return y_traj   # shape (w, p)

        return simulate

    def simulate(self, x0, u_seq):
        """Run the open-loop simulation.

        Parameters
        ----------
        x0 : array-like or dict
            Initial state, shape ``(n,)`` or dict mapping state name → value.
        u_seq : array-like or dict
            Input sequence, shape ``(w, m)`` or dict mapping input name →
            array of length ``w``.

        Returns
        -------
        y_traj : np.ndarray, shape (w, p)
            Measurement trajectory.
        """
        x0_arr = _to_array(x0, self.state_names)
        u_arr = _to_u_array(u_seq, self.input_names)
        y = self._simulate_jax(x0_arr, u_arr)
        return np.array(y)


# ---------------------------------------------------------------------------
# JaxEmpiricalObservabilityMatrix
# ---------------------------------------------------------------------------

class JaxEmpiricalObservabilityMatrix:
    """Observability matrix via JAX forward-mode autodiff.

    Replaces the 2n numerical perturbation simulations used by
    ``EmpiricalObservabilityMatrix`` with a single ``jax.jacfwd`` call,
    giving the *exact* Jacobian dY/dx₀ in one forward pass.

    Output attributes match those of ``EmpiricalObservabilityMatrix`` so this
    class can be used as a drop-in replacement.

    Parameters
    ----------
    jax_simulator : JaxSimulator
        A configured ``JaxSimulator`` instance.
    x0 : array-like or dict
        Initial state, shape ``(n,)`` or dict.
    u_seq : array-like or dict
        Input sequence, shape ``(w, m)`` or dict.
    eps : float, optional
        Accepted for API compatibility but not used (the Jacobian is exact).
    """

    def __init__(self, jax_simulator, x0, u_seq, eps=None):
        self.jax_simulator = jax_simulator
        self.eps = eps  # kept for API compat; not used

        x0_arr = _to_array(x0, jax_simulator.state_names)
        u_arr = _to_u_array(u_seq, jax_simulator.input_names)

        self.x0 = x0_arr
        self.u = u_arr
        self.n = jax_simulator.n
        self.p = jax_simulator.p
        self.w = u_arr.shape[0]
        self.state_names = jax_simulator.state_names
        self.measurement_names = jax_simulator.measurement_names

        # Nominal trajectory
        self.y_nominal = np.array(jax_simulator._simulate_jax(x0_arr, u_arr))  # (w, p)

        # Jacobian: dY/dx0, shape (w, p, n)
        jac_fn = jax.jit(jax.jacfwd(jax_simulator._simulate_jax, argnums=0))
        jac = np.array(jac_fn(x0_arr, u_arr))   # (w, p, n)

        # Reshape to (w*p, n) matching EmpiricalObservabilityMatrix.O
        # Row order: [sensor_0 t=0, sensor_1 t=0, ..., sensor_p t=0,
        #             sensor_0 t=1, ...]
        self.O = jac.reshape(self.w * self.p, self.n)

        # Build MultiIndex DataFrame matching EmpiricalObservabilityMatrix.O_df
        measurement_labels = self.measurement_names * self.w
        time_labels = np.repeat(np.arange(self.w), self.p).astype(int)

        self.O_df = pd.DataFrame(
            self.O,
            columns=self.state_names,
            index=measurement_labels,
        )
        self.O_df['time_step'] = time_labels
        self.O_df = self.O_df.set_index('time_step', append=True)
        self.O_df.index.names = ['sensor', 'time_step']


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(x, names):
    """Convert x0 (dict or array-like) to a 1-D float64 jnp array."""
    if isinstance(x, dict):
        return jnp.array([x[k] for k in names], dtype=jnp.float64)
    return jnp.array(x, dtype=jnp.float64).ravel()


def _to_u_array(u, names):
    """Convert u (dict or array-like) to a 2-D float64 jnp array (w, m)."""
    if isinstance(u, dict):
        cols = [np.asarray(u[k]).ravel() for k in names]
        return jnp.array(np.column_stack(cols), dtype=jnp.float64)
    arr = np.asarray(u)
    if arr.ndim == 1:
        arr = arr[:, None]
    return jnp.array(arr, dtype=jnp.float64)
