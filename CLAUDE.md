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
pytest tests/test_stochastic.py                  # stochastic observability/constructability (jax parts skip without jax)
pytest tests/test_simulator.py::TestSimulator::test_output_shape  # single test
```

No linter is configured. Functionality is also demonstrated through Jupyter notebooks in [examples/](examples/). [validation/stochastic_duality_fig2.ipynb](validation/stochastic_duality_fig2.ipynb) checks the stochastic recursions against Burak Boyacioglu's MATLAB reference (verbatim ports in the notebook; do not "improve" them) and redraws Fig. 2 of the duality letter; re-run it (`jupyter nbconvert --to notebook --execute --inplace`) after touching either recursion.

## Architecture

**pybounds** computes empirical observability of nonlinear dynamical systems with sensor noise. The core workflow is: define system dynamics + measurements → simulate → analyze observability.

### Key modules

- [pybounds/simulator.py](pybounds/simulator.py) — `Simulator` class wraps user-defined dynamics `f(x, u, t)` and measurement functions `h(x, u, t)`. Uses do_mpc/CasADi (IDAS/CVODES solvers) for numerical integration. The simulator returns pandas DataFrames with labeled states, inputs, and outputs.

- [pybounds/observability.py](pybounds/observability.py) — Core analysis layer. Main classes:
  - `EmpiricalObservabilityMatrix` — Builds observability matrix by numerically perturbing initial conditions (finite differences) and comparing output trajectories.
  - `SlidingEmpiricalObservabilityMatrix` — Sliding window version that evaluates observability along a trajectory.
  - `FisherObservability` / `SlidingFisherObservability` — Information-theoretic alternative using Fisher information matrix.
  - `ObservabilityMatrixImage` — Visualizes observability matrices as heatmaps.

- [pybounds/stochastic.py](pybounds/stochastic.py) — stochastic observability (Eq. 33, bounds the window's initial state) and constructability (Eq. 30, final state) Gramians with process noise Q, from Boyacioglu & van Breugel, IEEE L-CSS 2025; `linearize` (Phi = expm(A dt), C by finite differences or `jax.jacfwd`), `sliding_gramians` (all windows batched, looping over w only), `duality_check`, `deterministic_observability_gramian` (exact Q = 0 limit). Do not drive Q towards 0 in the recursions: the Q^-1 bracket cancels catastrophically.

- [pybounds/jacobian.py](pybounds/jacobian.py) — `SymbolicJacobian` uses SymPy for symbolic differentiation with numerical evaluation.

- [pybounds/util.py](pybounds/util.py) — `FixedKeysDict`, `SetDict`, `LatexStates`, and plotting utilities (`colorline`, `plot_heatmap_log_timeseries`).

### Data flow

1. User defines `f(x, u)` (dynamics) and `h(x, u)` (measurements) as plain Python functions.
2. `Simulator` integrates the system over time via do_mpc/CasADi, returning labeled dicts.
3. Observability classes perturb initial states by epsilon, re-simulate, and construct the empirical observability matrix O.
4. `FisherObservability` / `SlidingFisherObservability` compute F = OᵀR⁻¹O and invert for minimum error variance.
5. Results can be projected into transformed coordinates via `transform_states()`.

### ObservabilityAnalysis

[pybounds/analysis.py](pybounds/analysis.py) — `ObservabilityAnalysis` is the high-level API: configure → `run()` → query. Nothing is computed until `run()`; queries (`min_error_variance`, `fisher`, `observability_matrix`, `plot_observability_matrix`, `save_results`) raise until then. After `run()` it keeps every window's O, so states/sensors/time_steps can be sub-selected (state selection is conditional: other states treated as known) and R/lam changed without rebuilding O. Changing an O-building setting (`method`, `w`, `aux_list`, `z_function`, `z_state_names`, method options) discards results. Settings save/load as YAML (`save_settings` / `load_settings`, never importing code from the file); `save_results` writes a CSV + YAML sidecar (+ optional `.npz` of all O's). How O is built is pluggable: `_BUILDERS` maps a method name to a function returning a `SlidingO` (`'bounds-empirical'`, `'bounds-jax'`; the old names `'empirical'`/`'jax'` are aliases in `_METHOD_ALIASES`, canonicalized on input so `.method` and saved YAML hold the new names); a new way of building O is one builder plus a registry entry. The four `'stochastic-{observability,constructability}-{classic,jax}'` builders instead return a `_Linearization` (Phi, C along the trajectory); `_store_linearization` keeps it and every query runs the batched recursion, so the query settings `Q` (process noise, model coordinates) and `R`/`lam` apply without `run()`. They support only `storage='observability'`, no `aux_list`, scalar/dict R; `observability_matrix(k)` returns the equivalent noise-free matrix. `alignment` (query setting, `'center'` default = `w // 2`, or `'bounded_state'` = first sample, last for constructability) sets where each window's result is placed, via `_shift_index`; `z_function` is evaluated at the bounded state. Stochastic builders return a public frozen `Linearization` (Phi, C, t_sim, model state_names, sensor_names, bounded; independent of w), exposed as `analysis.linearization` with read-only, uncopied arrays; `_store_linearization(lin, dxdz_sliding=None)` derives windows/transform from the current settings, so `update_settings` of only `w`/`z_function`/`z_state_names` keeps `self._lin` and the next `run()` re-derives instead of re-linearizing. `ObservabilityAnalysis.from_linearization(Phi, C, *, method, w, ...)` wraps a precomputed linearization (no simulator, `_external`, query settings only) and must stay bit-identical to `run()`. `model_state_names` (every method) = names Q is keyed by; `deterministic_states()` = rows of Phi equal to e_i. `lam` may be per-state (dict by selected/transformed state name, omitted -> DEFAULT_LAM; or a 1-D array in selected-state order), resolved by `_resolve_lam` to an array that flows through `_fisher_inverse` (diag(lam)); a uniform vector must stay bit-identical to the scalar, and the bounds fast path must not fall back for it. The class applies `z_function` itself. `compute_observability()` is a thin wrapper around it.

Memory: `run()` streams windows (builders return a `_WindowStream`, or the JAX batched array) straight into one read-only `(n_windows, w*p, n)` array plus a shared row index, or into packed Fisher information; per-window DataFrames are built on access. `keep_source=True` falls back to the full (memory-heavy) builder object and keeps `source`/`window_data`. `storage='fisher_per_sensor'` / `'fisher'` store (per-sensor / summed) Fisher information instead of O and raise for queries they can't answer. The default O mode must stay bit-identical to 7b69d66: `tests/reference_7b69d66.py` is a verbatim copy of the old Fisher classes that the tests compare against. `FisherObservability` computes F for scalar/dict R as `(O.T * r_inv) @ O` (exactly equal to the dense product) and builds `R`/`R_inv` lazily; its `O.copy()` must stay (under pandas < 3 it normalizes memory layout, which keeps the matmul bit-identical). Queries use `_FastWindows` (row/column positions computed once, windows sliced in Fortran order) with a window-0 check against `FisherObservability` that falls back on any difference. JAX `batch_size` chunks windows (repeatable per batch size, but can differ from unbatched in the last bit). See [docs/design/observability_storage.md](docs/design/observability_storage.md).

### JAX backend

`JaxSimulator`, `JaxEmpiricalObservabilityMatrix`, and `JaxSlidingEmpiricalObservabilityMatrix` (in [pybounds/jax_simulator.py](pybounds/jax_simulator.py)) replace finite-difference Jacobians with exact autodiff via `jax.jacfwd` + `jax.vmap`. Requires `f` and `h` to use `jax.numpy` instead of `numpy`. The `compute_observability()` helper accepts `use_jax=True` to route through this backend.

### Dependencies

do_mpc and CasADi are central — CasADi provides symbolic math and solvers; do_mpc wraps model definition and simulation. NumPy/SciPy/Pandas handle numerical operations and data; SymPy is used only in `SymbolicJacobian`. JAX is optional.

## Possible future work

Ideas discussed but deliberately not implemented yet. Don't build these unless asked.

- **Stability warning for the JAX integrator.** `JaxSimulator` uses fixed-step RK4 (or Euler) with step `dt / substeps`. On stiff systems a too-large step makes RK4 diverge, but the values usually stay finite at typical window lengths (e.g. max |O| ≈ 4e21 at w=20 for a 500/s relaxation at dt=0.01), so the existing NaN/inf warning doesn't fire. A possible check: at each window's initial state, compute the eigenvalues of ∂f/∂x (one vmapped `jax.jacfwd(f)` call) and warn when max|λ| · dt/substeps exceeds the stability limit (≈2.8 for RK4, 2 for Euler), suggesting the number of substeps needed. It is a heuristic, since it only linearizes at each window's start.
