# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install locally (editable)
pip install -e ".[dev]"   # includes pytest and pytest-cov

# Install from PyPI
pip install pybounds

# Install with JAX backend
pip install "pybounds[jax]"   # or: pip install jax[cpu]
```

## Tests

```bash
pytest tests/                                    # all fast tests
pytest tests/test_diagnostic_pdf.py -v -s        # generates tests/diagnostic_report.pdf (slow, ~40s)
pytest tests/test_jax_observability.py           # JAX backend tests (requires jax installed)
pytest tests/test_simulator.py::TestSimulator::test_output_shape  # single test
```

No linter is configured. Functionality is also demonstrated through Jupyter notebooks in [examples/](examples/).

## Architecture

**pybounds** computes empirical observability of nonlinear dynamical systems with sensor noise. The core workflow is: define system dynamics + measurements → simulate → analyze observability.

### Key modules

- [pybounds/simulator.py](pybounds/simulator.py) — `Simulator` class wraps user-defined dynamics `f(x, u, t)` and measurement functions `h(x, u, t)`. Uses do_mpc/CasADi (IDAS/CVODES solvers) for numerical integration. The simulator returns pandas DataFrames with labeled states, inputs, and outputs.

- [pybounds/observability.py](pybounds/observability.py) — Core analysis layer. Main classes:
  - `EmpiricalObservabilityMatrix` — Builds observability matrix by numerically perturbing initial conditions (finite differences) and comparing output trajectories.
  - `SlidingEmpiricalObservabilityMatrix` — Sliding window version that evaluates observability along a trajectory.
  - `FisherObservability` / `SlidingFisherObservability` — Information-theoretic alternative using Fisher information matrix.
  - `ObservabilityMatrixImage` — Visualizes observability matrices as heatmaps.

- [pybounds/jacobian.py](pybounds/jacobian.py) — `SymbolicJacobian` uses SymPy for symbolic differentiation with numerical evaluation.

- [pybounds/util.py](pybounds/util.py) — `FixedKeysDict`, `SetDict`, `LatexStates`, and plotting utilities (`colorline`, `plot_heatmap_log_timeseries`).

### Data flow

1. User defines `f(x, u)` (dynamics) and `h(x, u)` (measurements) as plain Python functions.
2. `Simulator` integrates the system over time via do_mpc/CasADi, returning labeled dicts.
3. Observability classes perturb initial states by epsilon, re-simulate, and construct the empirical observability matrix O.
4. `FisherObservability` / `SlidingFisherObservability` compute F = OᵀR⁻¹O and invert for minimum error variance.
5. Results can be projected into transformed coordinates via `transform_states()`.

### JAX backend

`JaxSimulator`, `JaxEmpiricalObservabilityMatrix`, and `JaxSlidingEmpiricalObservabilityMatrix` (in [pybounds/jax_simulator.py](pybounds/jax_simulator.py)) replace finite-difference Jacobians with exact autodiff via `jax.jacfwd` + `jax.vmap`. Requires `f` and `h` to use `jax.numpy` instead of `numpy`. The `compute_observability()` helper accepts `use_jax=True` to route through this backend.

### Dependencies

do_mpc and CasADi are central — CasADi provides symbolic math and solvers; do_mpc wraps model definition and simulation. NumPy/SciPy/Pandas handle numerical operations and data; SymPy is used only in `SymbolicJacobian`. JAX is optional.
