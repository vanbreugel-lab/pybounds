"""
Stochastic observability and constructability Gramians, accounting for process noise.

    B. Boyacioglu & F. van Breugel, "Duality of Stochastic Observability and Constructability and
    their Relation to the Fisher Information", IEEE L-CSS (2025), doi:10.1109/LCSYS.2025.3547297.

These are reachable through ``ObservabilityAnalysis`` as the methods
``'stochastic-observability-classic'``, ``'stochastic-observability-jax'``,
``'stochastic-constructability-classic'`` and ``'stochastic-constructability-jax'``; the functions
here are the building blocks, usable on their own for a linear time-varying system.

What this adds over the empirical method
----------------------------------------

The empirical method computes ``F = O^T R^-1 O`` for a window of measurements. That is the Fisher
information of the window's **initial** state, and it assumes **no process noise**: each
measurement makes an independent addition to the information, so a longer window always knows
more, without bound.

With process noise the measurements are no longer independent -- uncertainty injected at step k
propagates into every later measurement -- and the information *saturates*: a measurement far from
the state of interest tells you almost nothing about it, because the state has since been kicked
around by noise you cannot see. That is what the two recursions here compute.

Two Gramians, two ends of the window
------------------------------------

- **Observability** (``stochastic_observability_gramian``, the letter's Lemma 1 / Eq. 33) is the FIM
  with respect to the window's **initial** state ``x_0``. It runs **backward** through the window,
  seeded at the last sample. Its inverse Cramer-Rao-bounds a fixed-point smoother of ``x_0``.
- **Constructability** (``stochastic_constructability_gramian``, Eq. 30, originally Tichavsky 1998)
  is the FIM with respect to the window's **final** state ``x_{w-1}``. It runs **forward**, seeded at
  the first sample. Its inverse is the posterior Cramer-Rao bound -- what a Kalman filter's error
  covariance tracks. Put ``F_k = (P_k^+)^-1`` and the recursion below *is* a Kalman filter's
  information form.

So the empirical method and the stochastic observability Gramian both describe the state at the
**start** of their window; the constructability Gramian describes the state at the **end**. Plotting
all of them at the window centre (``ObservabilityAnalysis(alignment='center')``, the default) puts
them on one axis, but the same feature then appears at the same time even though the states they
bound are ``w-1`` samples apart. ``alignment='bounded_state'`` plots each at the state it bounds.

The letter proves the two are duals under time reversal (Theorem 1): reverse the trajectory, invert
each transition, conjugate ``Q``, and one recursion becomes the other. ``duality_check`` asserts it.

Requirements and gotchas
------------------------

- **Q must be strictly positive definite.** Both recursions form ``Q^-1`` at every step. A state
  whose true process noise is zero -- a constant parameter -- still has to be given some, and it will
  then slowly "forget" across the window. Give such states a much smaller ``Q`` than the dynamic ones.
- **Do not push Q to machine zero to recover the empirical answer.** The bracket
  ``Q^-1 - Q^-1 (F + Q^-1)^-1 Q^-1`` is a difference of two enormous matrices as ``Q -> 0``, and it
  cancels catastrophically. Use ``deterministic_observability_gramian``, the exact ``Q = 0`` formula,
  for that comparison. The per-state version of this trap is the *spread* of Q, not its absolute
  value: see ``MAX_Q_SPREAD``.
- **The linearization is the other half of the method.** The recursions are for a discrete-time
  *linear* time-varying system, so a nonlinear model is linearized at every sample of the realized
  trajectory. ``ObservabilityAnalysis`` takes ``Phi_k`` from the simulator's own integrator step by
  default (``linearization='flow'``), and the stochastic Gramians then reproduce the empirical
  method's Fisher information as ``Q -> 0``, to integrator and finite-difference accuracy, on the
  model's own trajectory. ``linearize`` (for a model given only as ``f`` and ``h``) uses
  ``Phi_k = expm(A_k dt)`` instead, which freezes ``A`` over each step.

Transcription notes
-------------------

- The letter's Eq. (33) defines ``phi = Phi^T_{N-k,N-k-1}`` and then uses ``phi`` in the positions
  where its own LTI specialization (Eq. 34) has a plain ``Phi``, so the two disagree by a transpose.
  The LTI form is the one that reduces correctly to the deterministic recursion as ``Q -> 0``, and it
  is what Burak's MATLAB implements, so that is what is implemented here (``Phi^T bracket Phi``).
- Both initializations are written ``C R^-1 C`` in the letter with no transpose. Every other equation
  in it, the MATLAB, and dimensional consistency say ``C^T R^-1 C``.
- The intermediate iterates of Eq. (33) are **not** the observability Gramians of shorter windows --
  the letter's Remark 1 is explicit that the dual system depends on the window length. Only the final
  value means anything, which is why no ``return_sequence`` option is offered.

Batching
--------

Both recursions accept a leading batch axis on every matrix (e.g. ``Phis[j]`` of shape
``(n_windows, n, n)``), so that all sliding windows can be computed in one pass of ``w`` steps. With
2-D matrices they are exactly the per-window recursions.

The recursions were checked against Burak's MATLAB (``validation/stochastic_duality_fig2.ipynb``).
"""

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import expm


# ---------------------------------------------------------------------------------------------
# Jacobians along the trajectory
# ---------------------------------------------------------------------------------------------

def _fd_jacobian(func, x, u, eps):
    """ Central-difference Jacobian of ``func(x, u)`` with respect to x.

    The step is ``eps``, absolute and identical for every state, exactly as the empirical method
    perturbs to build O, so ``eps`` means the same thing on both paths. On a model whose states span
    many decades, prefer the JAX backend, which differentiates exactly.

    :return: array of shape (len(func(x, u)), len(x))
    """

    x = np.asarray(x, dtype=float)
    n = x.size
    f0 = np.asarray(func(x, u), dtype=float).ravel()
    jacobian = np.zeros((f0.size, n))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        jacobian[:, i] = (np.asarray(func(x + dx, u), dtype=float).ravel()
                          - np.asarray(func(x - dx, u), dtype=float).ravel()) / (2 * eps)

    return jacobian


def _with_aux(func, aux):
    """func(x, u) when aux is None, else x, u -> func(x, u, aux) (the JaxSimulator convention)."""
    return func if aux is None else (lambda x, u: func(x, u, aux))


def _jacobians_finite_difference(f, h, x_traj, u_traj, eps, aux_list=None):
    """ (df/dx, dh/dx) at every sample by central differences. Shapes (N, n_f, n) and (N, p, n). """

    aux_list = [None] * len(x_traj) if aux_list is None else aux_list
    A = np.stack([_fd_jacobian(_with_aux(f, aux_list[k]), x_traj[k], u_traj[k], eps=eps) for k in range(len(x_traj))])
    C = np.stack([_fd_jacobian(_with_aux(h, aux_list[k]), x_traj[k], u_traj[k], eps=eps) for k in range(len(x_traj))])

    return A, C


def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
        from .jax_simulator import _x64, _to_aux
    except ImportError:
        raise ImportError('the stochastic *-jax methods need JAX. Install it with: pip install jax[cpu]') from None
    return jax, jnp, _x64, _to_aux


def _stack_aux(aux_list):
    """aux_list (one pytree per sample) stacked along a leading axis for vmap, or None."""
    if aux_list is None:
        return None
    jax, jnp, _, _to_aux = _jax_modules()
    try:
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *[_to_aux(a) for a in aux_list])
    except (ValueError, TypeError) as error:
        raise ValueError('aux_list entries must all have the same structure and array shapes so they can be '
                         'batched across samples') from error


def _jacobians_jax(f, h, x_traj, u_traj, aux_list=None):
    """ (df/dx, dh/dx) at every sample by forward-mode autodiff, batched over the trajectory.

    ``f`` and ``h`` must be written with ``jax.numpy``. They often return a Python list of scalars,
    which is stacked with ``jnp.asarray``. float64 is enabled only for this computation, as in
    ``JaxSimulator``. With aux_list, sample k calls ``f(x, u, aux_list[k])``.
    """

    jax, jnp, _x64, _ = _jax_modules()

    def as_vector(func):
        def wrapped(x, u, aux):
            return jnp.ravel(jnp.asarray(func(x, u) if aux is None else func(x, u, aux)))
        return wrapped

    with _x64():
        x_batch = jnp.asarray(x_traj, dtype=jnp.float64)
        u_batch = jnp.asarray(u_traj, dtype=jnp.float64)
        aux_batch = _stack_aux(aux_list)
        in_axes = (0, 0, None if aux_batch is None else 0)
        jac_f = jax.jit(jax.vmap(jax.jacfwd(as_vector(f), argnums=0), in_axes=in_axes))
        jac_h = jax.jit(jax.vmap(jax.jacfwd(as_vector(h), argnums=0), in_axes=in_axes))
        try:
            A, C = np.array(jac_f(x_batch, u_batch, aux_batch)), np.array(jac_h(x_batch, u_batch, aux_batch))
        except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError) as error:
            raise TypeError('JAX could not trace f or h: write them with jax.numpy (jnp) instead of numpy, '
                            'or use a *-classic method') from error

    return A, C


def linearize(f, h, x_traj, u_traj, dt, backend='classic', eps=1e-5, discrete=False, aux_list=None):
    """ Linearize a model given as functions at every sample of a realized trajectory.

    Returns the per-step **discrete** transition matrices and measurement Jacobians of the letter's
    Eq. (22), ``dx_{k+1} = Phi_k dx_k + w_k``, ``dy_k = C_k dx_k + v_k``. For continuous dynamics::

        A_k = df/dx |(x_k, u_k),   Phi_k = expm(A_k * dt),   C_k = dh/dx |(x_k, u_k)

    which freezes ``A`` at the left endpoint of each step, an ``O(dt^2)`` local error when the
    linearization moves quickly within one sample. For a discrete-time model (``discrete=True``, f is the
    update map ``x_{k+1} = f(x_k, u_k)``) ``Phi_k = df/dx`` exactly.

    ``ObservabilityAnalysis`` uses this for simulators it only knows through ``f`` and ``h``. For a
    pybounds ``Simulator`` or a ``JaxSimulator`` it differentiates the simulator's own integrator step
    instead (``linearization='flow'``, the default there), which has no frozen-``A`` error and matches the
    bounds-* methods as ``Q -> 0``.

    :param callable f: continuous-time dynamics, x_dot = f(x, u), or the discrete update map
    :param callable h: measurements, y = h(x, u)
    :param x_traj: (N, n) states along the trajectory
    :param u_traj: (N, m) inputs along the trajectory
    :param float dt: sample time (unused for a discrete model)
    :param str backend: 'classic' (central finite differences) or 'jax' (forward-mode autodiff;
        f and h must use jax.numpy)
    :param float eps: finite-difference step ('classic' only)
    :param bool discrete: f is a discrete-time update map
    :param list aux_list: optional auxiliary data, one entry per sample; sample k calls ``f(x, u, aux_list[k])``
        and ``h(x, u, aux_list[k])``
    :return: (Phi, C) of shapes (N, n, n) and (N, p, n)
    """

    x_traj = np.asarray(x_traj, dtype=float)
    u_traj = np.asarray(u_traj, dtype=float).reshape(x_traj.shape[0], -1)

    if backend == 'jax':
        A, C = _jacobians_jax(f, h, x_traj, u_traj, aux_list)
    elif backend == 'classic':
        A, C = _jacobians_finite_difference(f, h, x_traj, u_traj, eps, aux_list)
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'classic' or 'jax'")

    Phi = A if discrete else np.stack([expm(A_k * dt) for A_k in A])

    return Phi, C


def _linearize_casadi_simulator(simulator, x_traj, u_traj, linearization='flow', eps=1e-5):
    """ (Phi, C) of a pybounds ``Simulator`` at every sample, from its CasADi model: exact derivatives.

    The model is rebuilt symbolically from ``simulator.f`` the way the Simulator itself builds it (lists of
    CasADi SX variables), so any f the Simulator accepts works, including one written with CasADi functions.

    - discrete model: ``Phi_k = df/dx`` of the update map;
    - continuous, ``'flow'``: ``Phi_k`` is the sensitivity of one sample of the simulator's own integrator
      (``params_simulator``: integration_tool, abstol, reltol), the exact transition matrix of the step the
      bounds-* methods simulate;
    - continuous, ``'expm'``: ``Phi_k = expm(A_k dt)`` with the exact ``A_k = df/dx``.

    ``C_k = dh/dx`` symbolically when h can be evaluated on CasADi symbols, else by central differences of
    the numeric h with step ``eps`` (the Simulator only ever calls h numerically, so h need not be symbolic).
    """

    import casadi as ca

    x_traj = np.asarray(x_traj, dtype=float)
    N = x_traj.shape[0]
    u_traj = np.asarray(u_traj, dtype=float).reshape(N, -1)
    n, m = x_traj.shape[1], u_traj.shape[1]

    X, U = ca.SX.sym('x', n), ca.SX.sym('u', m)
    X_list, U_list = [X[i] for i in range(n)], [U[i] for i in range(m)]
    rhs = ca.vertcat(*simulator.f(X_list, U_list))

    if simulator.model.model_type == 'discrete':
        transition = ca.jacobian(rhs, X)
    elif linearization == 'flow':
        params = getattr(simulator, 'params_simulator', None) or {}
        options = {key: params[key] for key in ('abstol', 'reltol') if key in params}
        step = ca.integrator('step', params.get('integration_tool', 'idas'), {'x': X, 'p': U, 'ode': rhs},
                             0.0, float(simulator.dt), options)
        transition = ca.jacobian(step(x0=X, p=U)['xf'], X)
    else:
        transition = ca.jacobian(rhs, X)

    def evaluate(expression, rows):
        """expression(x_k, u_k) for every sample: (N, rows, n)."""
        values = np.array(ca.Function('J', [X, U], [expression]).map(N)(x_traj.T, u_traj.T))
        return values.reshape(rows, N, n).transpose(1, 0, 2)

    Phi = evaluate(transition, n)
    if simulator.model.model_type != 'discrete' and linearization != 'flow':
        Phi = np.stack([expm(A_k * simulator.dt) for A_k in Phi])

    h = getattr(simulator.h, '__wrapped__', simulator.h)
    try:
        y = ca.vertcat(*h(X_list, U_list))
        if not isinstance(y, (ca.SX, ca.MX)) or y.shape[0] == 0:
            raise TypeError('h did not return CasADi expressions')
        C = evaluate(ca.jacobian(y, X), y.shape[0])
    except Exception:   # h is numeric-only (e.g. uses np.unwrap or Python branching): differentiate numerically
        C = np.stack([_fd_jacobian(simulator.h, x_traj[k], u_traj[k], eps=eps) for k in range(N)])

    return Phi, C


def _linearize_jax_simulator(simulator, x_traj, u_traj, backend='jax', eps=1e-5, aux_list=None):
    """ (Phi, C) of a ``JaxSimulator`` at every sample, from its own integrator step (``'flow'``).

    ``Phi_k`` is the Jacobian of one sample of the simulator's integration (its integrator and ``substeps``),
    by forward-mode autodiff (``backend='jax'``) or central differences (``'classic'``). ``C_k = dh/dx``.
    """

    jax, jnp, _x64, _ = _jax_modules()
    x_traj = np.asarray(x_traj, dtype=float)
    N = x_traj.shape[0]
    u_traj = np.asarray(u_traj, dtype=float).reshape(N, -1)
    step = simulator._step_jax

    def h_vector(x, u, aux=None):
        return jnp.ravel(jnp.asarray(simulator.h_jax(x, u) if aux is None else simulator.h_jax(x, u, aux)))

    with _x64():
        if backend == 'jax':
            x_batch = jnp.asarray(x_traj, dtype=jnp.float64)
            u_batch = jnp.asarray(u_traj, dtype=jnp.float64)
            aux_batch = _stack_aux(aux_list)
            in_axes = (0, 0, None if aux_batch is None else 0)
            Phi = np.array(jax.jit(jax.vmap(jax.jacfwd(step, argnums=0), in_axes=in_axes))(x_batch, u_batch, aux_batch))
            C = np.array(jax.jit(jax.vmap(jax.jacfwd(h_vector, argnums=0), in_axes=in_axes))(x_batch, u_batch, aux_batch))
        else:
            step_jit, h_jit = jax.jit(step), jax.jit(h_vector)
            aux_list = [None] * N if aux_list is None else aux_list
            Phi = np.stack([_fd_jacobian(lambda x, u, a=aux_list[k]: np.asarray(step_jit(x, u, a)),
                                         x_traj[k], u_traj[k], eps=eps) for k in range(N)])
            C = np.stack([_fd_jacobian(lambda x, u, a=aux_list[k]: np.asarray(h_jit(x, u, a)),
                                       x_traj[k], u_traj[k], eps=eps) for k in range(N)])

    return Phi, C


# ---------------------------------------------------------------------------------------------
# Process noise
# ---------------------------------------------------------------------------------------------

# The largest ratio between the biggest and smallest diagonal entry of Q that the recursions can be
# trusted at. Measured on a 33-state vehicle model (window 6): the result held flat over twelve
# decades of spread and then failed, within two more decades, by nine orders of magnitude -- in every
# state, not only the retuned ones. The constructability recursion sets the limit; observability
# survives to a spread of ~1e16 on the same run.
MAX_Q_SPREAD = 1e12


def process_covariance(q, state_names, overrides=None):
    """ The per-step process covariance: ``q * I``, with per-state values substituted on the diagonal.

    This is the **per-step discrete** covariance ``Q_k`` of the letter's Eq. (22), used verbatim at
    every step. It is not a continuous spectral density and is never scaled by ``dt``.

    Per-state values matter for constant parameters, which physically have *no* process noise but
    must be given some because both recursions form ``Q^-1``. A uniform ``q`` makes the model slowly
    "forget" its parameters across the window, which discounts exactly the distant measurements they
    are estimated from; giving them a much smaller ``q`` is the fix.

    :param float q: the value used for every state not named in ``overrides``
    :param state_names: the model's state names, in state-vector order
    :param dict overrides: state name -> variance, for the states that should differ from ``q``. A
        name that is not a state raises rather than being ignored
    :return: (n, n) diagonal array
    """

    names = list(state_names)
    diagonal = np.full(len(names), _positive_q(q, 'Q'))

    for name, value in (overrides or {}).items():
        if name not in names:
            raise ValueError('Cannot set Q for %r: not a state of this model. Its states are: %s'
                             % (name, ', '.join(map(str, names))))
        diagonal[names.index(name)] = _positive_q(value, 'Q for %r' % (name,))

    if diagonal.max() / diagonal.min() > MAX_Q_SPREAD:
        warnings.warn(_q_spread_warning(diagonal, names), RuntimeWarning, stacklevel=2)

    return np.diag(diagonal)


def _q_spread_warning(diagonal, names):
    """ The message behind ``MAX_Q_SPREAD``.

    Driving one diagonal entry of Q towards zero while the others stay put hits the same
    catastrophic cancellation as driving all of Q to zero, so for a diagonal Q the quantity to watch
    is the ratio of its largest to smallest entry, which is ``cond(Q)``. A Q that has gone too far
    corrupts the whole answer, not just the entry that was retuned.
    """

    low, high = float(np.min(diagonal)), float(np.max(diagonal))
    spread = high / low
    lowest = [name for name, value in zip(names, diagonal) if value == low]
    named = ', '.join(map(str, lowest[:4])) + (', ...' if len(lowest) > 4 else '')

    return ('Q spans %.3g decades, %.3g down to %.3g (on %s) -- a cond(Q) of %.3g against a '
            'trustworthy limit of %.3g. Both recursions form Q^-1, and a spread this wide makes '
            'the bracket Q^-1 - Q^-1 (F + Q^-1)^-1 Q^-1 cancel catastrophically -- which corrupts '
            'every state, including those whose Q was left alone, not merely the retuned ones. '
            'Narrow the spread, or use a bounds-* method for the no-process-noise answer.'
            % (np.log10(spread), high, low, named, spread, MAX_Q_SPREAD))


def _positive_q(value, what):
    """ Both recursions form ``Q^-1`` at every step, so every diagonal entry has to be usable. """

    if not value > 0:
        raise ValueError('%s must be strictly positive: both recursions form Q^-1 at every step. '
                         'Got %r. For the Q = 0 answer use deterministic_observability_gramian, '
                         'or a bounds-* method.' % (what, value))

    return float(value)


# ---------------------------------------------------------------------------------------------
# The recursions
# ---------------------------------------------------------------------------------------------

def _T(M):
    """ Transpose of the last two axes (a plain transpose for a 2-D matrix). """
    return np.swapaxes(M, -1, -2)


def _solve(A, B):
    """ ``np.linalg.solve(A, B)`` for a matrix right-hand side, with batch axes broadcast explicitly
    (numpy < 2 would read an (n, n) B against a batched A as a stack of vectors). """
    if B.ndim < A.ndim:
        B = np.broadcast_to(B, A.shape[:-2] + B.shape[-2:])
    return np.linalg.solve(A, B)


def stochastic_observability_gramian(Phis, Cs, Qinvs, Rinvs):
    """ The w-step stochastic observability Gramian ``F^{x_0}_{down,w}`` -- the letter's Eq. (33).

    Fisher information of the window's measurements with respect to its **initial** state. Stepping
    backward from the end of the window:

        F_j = Phi_j^T [ Q_j^-1 - Q_j^-1 (F_{j+1} + Q_j^-1)^-1 Q_j^-1 ] Phi_j + C_j^T R_j^-1 C_j

    initialized ``F_{w-1} = C_{w-1}^T R_{w-1}^-1 C_{w-1}``. By Sherman-Morrison-Woodbury the bracket
    is ``(Q_j + F_{j+1}^-1)^-1``, so this is the information-form backward pass of a smoother; as
    ``Q -> 0`` it reduces to ``F_j = Phi_j^T F_{j+1} Phi_j + C_j^T R_j^-1 C_j``, which telescopes into
    ``O^T blkdiag(R)^-1 O``.

    :param Phis: length-w sequence of transition matrices; ``Phis[j]`` maps x_j -> x_{j+1}. The last
        entry is never used.
    :param Cs: length-w sequence of measurement Jacobians, already row-sliced to the chosen sensors
    :param Qinvs: length-w sequence of ``Q_j^-1`` (inverted by the caller: Q is usually the same at
        every step)
    :param Rinvs: length-w sequence of ``R_j^-1``
    :return: (n, n) array, or (..., n, n) with batched inputs
    """

    w = len(Cs)
    F = _T(Cs[-1]) @ Rinvs[-1] @ Cs[-1]

    for j in range(w - 2, -1, -1):
        Qinv = Qinvs[j]
        # bracket = Qinv - Qinv (F + Qinv)^-1 Qinv  ==  (Q_j + F^-1)^-1  by SMW
        bracket = Qinv - Qinv @ _solve(F + Qinv, Qinv)
        F = _T(Phis[j]) @ bracket @ Phis[j] + _T(Cs[j]) @ Rinvs[j] @ Cs[j]
        F = 0.5 * (F + _T(F))

    return F


def stochastic_constructability_gramian(Phis, Cs, Qinvs, Rinvs, F0=None):
    """ The w-step stochastic constructability Gramian ``F^{x_{w-1}}`` -- the letter's Eq. (30).

    Fisher information of the window's measurements with respect to its **final** state, so its
    inverse is the posterior Cramer-Rao bound: the one that lines up with a Kalman filter's error
    covariance. Stepping forward, with ``Phi = Phi_{k+1,k}``:

        F_{k+1} = (Q_k + Phi F_k^-1 Phi^T)^-1 + C_{k+1}^T R_{k+1}^-1 C_{k+1}

    written in the SMW form that avoids ``F^-1``, which is singular at the seed whenever there are
    fewer measurements than states:

        (Q + Phi F^-1 Phi^T)^-1 = Qinv - Qinv Phi (F + Phi^T Qinv Phi)^-1 Phi^T Qinv

    initialized ``F_0 = C_0^T R_0^-1 C_0``. Arguments are as ``stochastic_observability_gramian``.

    :param F0: optional information about x_0 ALREADY accumulated, including x_0's own measurement;
        it replaces the ``C_0^T R_0^-1 C_0`` seed (so ``Cs[0]`` and ``Rinvs[0]`` are then unused).
        Feeding each result back in as the next call's ``F0`` is the same recursion carried on, so a
        prior (e.g. the inverse of an initial covariance P0) enters here.
    """

    w = len(Cs)
    F = _T(Cs[0]) @ Rinvs[0] @ Cs[0] if F0 is None else np.array(F0, dtype=float)

    for k in range(w - 1):
        Phi, Qinv = Phis[k], Qinvs[k]
        M = _T(Phi) @ Qinv @ Phi
        bracket = Qinv - Qinv @ Phi @ _solve(F + M, _T(Phi) @ Qinv)
        F = bracket + _T(Cs[k + 1]) @ Rinvs[k + 1] @ Cs[k + 1]
        F = 0.5 * (F + _T(F))

    return F


def deterministic_observability_gramian(Phis, Cs, Rinvs):
    """ The ``Q -> 0`` limit, ``F = sum_j (C_j Phi_{j,0})^T R_j^-1 (C_j Phi_{j,0})``.

    Exactly ``O^T blkdiag(R)^-1 O`` for the linearized system, so this is what to compare against the
    empirical method -- rather than driving ``Q`` towards zero in the recursion, where the bracket
    cancels catastrophically.
    """

    w = len(Cs)
    n = Cs[0].shape[-1]
    F = np.zeros((n, n))
    Phi_j0 = np.eye(n)

    for j in range(w):
        CP = Cs[j] @ Phi_j0
        F = F + _T(CP) @ Rinvs[j] @ CP
        if j < w - 1:
            Phi_j0 = Phis[j] @ Phi_j0

    return F


def duality_check(Phis, Cs, Qds, Rs):
    """ Theorem 1 as an assertion: the dual system's constructability is this system's observability.

    Builds the dual of the letter's Eq. (36) -- ``Phi_bar = Phi^-1`` in reverse order, ``C_bar`` and
    ``R_bar`` reversed, ``Q_bar = Phi^-1 Q Phi^-T`` -- runs the *forward* constructability recursion on
    it, and returns that alongside the *backward* observability recursion on the original. The two
    must agree. This exercises every term of both recursions, so an error in either one breaks it.

    :param Phis: length-w sequence of (n, n) transition matrices
    :param Cs: length-w sequence of (p, n) measurement Jacobians
    :param Qds: length-w sequence of (n, n) process noise covariances (not inverses)
    :param Rs: length-w sequence of (p, p) measurement noise covariances (not inverses)
    :return: (F_observability, F_constructability_of_dual)
    """

    w = len(Cs)
    Qinvs = [np.linalg.inv(Q) for Q in Qds]
    Rinvs = [np.linalg.inv(R) for R in Rs]

    forward = stochastic_observability_gramian(Phis, Cs, Qinvs, Rinvs)

    # Dual system, indexed 0..w-1 in reverse time: bar_k corresponds to original index w-1-k.
    Phi_inv = [np.linalg.inv(P) for P in Phis]
    dual_Phis = [Phi_inv[w - 2 - k] for k in range(w - 1)] + [np.eye(Phis[0].shape[0])]
    dual_Cs = [Cs[w - 1 - k] for k in range(w)]
    dual_Rinvs = [Rinvs[w - 1 - k] for k in range(w)]
    dual_Qinvs = [np.linalg.inv(Phi_inv[w - 2 - k] @ Qds[w - 2 - k] @ Phi_inv[w - 2 - k].T)
                  for k in range(w - 1)] + [np.eye(Phis[0].shape[0])]

    dual = stochastic_constructability_gramian(dual_Phis, dual_Cs, dual_Qinvs, dual_Rinvs)

    return forward, dual


# ---------------------------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------------------------

def sliding_gramians(Phi, C, w, Qinv, Rinvs, bounded='initial'):
    """ The stochastic Gramian of every sliding window of a linearized trajectory, all at once.

    :param Phi: (N, n, n) transition matrices along the trajectory
    :param C: (N, p, n) measurement Jacobians, already row-sliced to the chosen sensors
    :param int w: window size; windows start at 0, 1, ..., N - w
    :param Qinv: (n, n) inverse process noise covariance, the same at every step
    :param Rinvs: length-w sequence of (p, p) inverse measurement noise covariances, one per step of
        the window (a zero matrix leaves that step unmeasured)
    :param str bounded: 'initial' (stochastic observability) or 'final' (stochastic constructability)
    :return: (N - w + 1, n, n) array
    """

    n_windows = Phi.shape[0] - w + 1
    Phis = [Phi[j:j + n_windows] for j in range(w)]
    Cs = [C[j:j + n_windows] for j in range(w)]
    Qinvs = [Qinv] * w

    if bounded == 'initial':
        return stochastic_observability_gramian(Phis, Cs, Qinvs, Rinvs)
    if bounded == 'final':
        return stochastic_constructability_gramian(Phis, Cs, Qinvs, Rinvs)
    raise ValueError(f"bounded must be 'initial' or 'final', got {bounded!r}")


def window_observability_matrix(Phi, C, k, w, bounded='initial', sensor_names=None, state_names=None):
    """ The deterministic (Q = 0) observability or constructability matrix of one window.

    The stochastic methods build no observability matrix -- they recurse on ``Phi`` and ``C`` directly
    -- but the equivalent noise-free matrix can be assembled from the same linearization. With
    ``Q = 0``, ``O^T R^-1 O`` of this matrix is the Fisher information.

    Observability rows are ``C_{k+j} Phi_{j<-k}``, relating each measurement to the window's
    **initial** state. Constructability rows are ``C_{k+j} Phi^-1_{k+j<-e}``, relating them to the
    **final** state ``e = k + w - 1``. Both are in **forward** time order, so a row's ``time_step`` is
    its position in the window.

    :param Phi: (N, n, n) transition matrices along the trajectory
    :param C: (N, p, n) measurement Jacobians
    :param int k: index of the window's first sample
    :param int w: window size
    :param str bounded: 'initial' or 'final'
    :return: DataFrame with a (sensor, time_step) MultiIndex and one column per state
    """

    n = Phi.shape[1]
    p = C.shape[1]
    sensor_names = list(sensor_names) if sensor_names is not None else ['y_%d' % i for i in range(p)]
    state_names = list(state_names) if state_names is not None else ['x_%d' % i for i in range(n)]

    maps = [None] * w
    Psi = np.eye(n)
    if bounded == 'final':
        # Psi[j] maps the window's final state back to sample k+j, so it accumulates backwards.
        for j in range(w - 1, -1, -1):
            maps[j] = Psi
            if j > 0:
                try:
                    Psi = np.linalg.solve(Phi[k + j - 1], Psi)
                except np.linalg.LinAlgError:
                    raise ValueError(f'the transition matrix at sample {k + j - 1} is singular (possible for a '
                                     'discrete-time model), so the constructability matrix, which maps the final '
                                     'state back through it, does not exist; the constructability Gramian itself '
                                     'does not need it') from None
    elif bounded == 'initial':
        # Psi[j] maps the window's initial state forward to sample k+j.
        for j in range(w):
            maps[j] = Psi
            if j < w - 1:
                Psi = Phi[k + j] @ Psi
    else:
        raise ValueError(f"bounded must be 'initial' or 'final', got {bounded!r}")

    block = np.concatenate([C[k + j] @ maps[j] for j in range(w)], axis=0)
    index = pd.MultiIndex.from_arrays([sensor_names * w, np.repeat(np.arange(w), p).astype(int)],
                                      names=['sensor', 'time_step'])

    return pd.DataFrame(block, index=index, columns=state_names)
