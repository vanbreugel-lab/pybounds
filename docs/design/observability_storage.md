# Memory use in `ObservabilityAnalysis`: storage design

## Problem
At commit 7b69d66, `ObservabilityAnalysis` used several times more memory than the observability matrices (O) it needed.

- **O was held about 5 times after `run()`:**
  - the builder object's `O_sliding` arrays and `O_df_sliding` DataFrames, plus the analysis' own copy of every window;
  - with the finite-difference builder, `window_data['y_plus'/'y_minus']`, which is 2× O again.
- **Every query built dense matrices.** `FisherObservability` built a dense (w·p)×(w·p) `R` and `R_inv` per window, even for a scalar R, and `SlidingFisherObservability` kept every window's object until the query returned. At w = 100 a single query peaked at about 4× O.

## Changes

### A. O is stored once
- **One array:** `ObservabilityAnalysis` keeps one read-only `(n_windows, w·p, n)` float array plus one shared row index. Per-window DataFrames are built only when accessed.
- **Builder objects are released:** after `run()`, the builder's native object and its `window_data` are dropped unless `keep_source=True`.
- **`from_sliding`:** a list of DataFrames is copied once into the array, and the caller's list is not kept. An array passed as `SlidingO(O=..., index=..., state_names=...)` is used without a copy.

### B. Diagonal R without dense matrices
- **Scalar, per-sensor dict and `None` R:** kept as one variance per row, and F is computed as `(Oᵀ · r⁻¹) @ O`. Because `Oᵀ @ diag(r⁻¹)` has one nonzero term per element, this equals the old dense product exactly.
- **`R` / `R_inv`:** still available, built on first access.
- **Matrix R:** unchanged.
- **Per-window objects:** `SlidingFisherObservability(keep_windows=False)` drops each window's `FisherObservability` after use, and `min_error_variance` uses it.

A and B change no results: tests compare against a verbatim copy of the 7b69d66 classes (`tests/reference_7b69d66.py`).

### C. Optional Fisher-information storage (`storage=`)
Per window, the representations below were compared.

| | `'observability'` (default) | `'fisher_per_sensor'` | `'fisher'` |
|---|---|---|---|
| Stored per window | O: w·p·n numbers | p unit-noise matrices F_s = O_sᵀO_s, packed: p·n(n+1)/2 | one F over `fisher_sensors`, unit noise: n(n+1)/2 |
| Select states (conditional crop) | yes | yes | yes |
| Select sensors | yes | yes | no |
| Select time_steps | yes | no | no |
| Scalar R | yes | yes (F / r) | yes (F / r) |
| Per-sensor (dict) R | yes | yes (Σ_s F_s / r_s) | no |
| Correlated or time-varying (matrix) R | yes | no | no |
| `lam`, including `'limit'` | yes | yes | yes |
| `z_function` | yes | yes (applied to O before forming F) | yes |
| `fisher()` objects | yes | no, use `fisher_information()` | no, use `fisher_information()` |
| `observability_matrix(k)` and plots | stored | recomputed for that one window | recomputed for that one window |
| Bit-identical to 7b69d66 | yes | no, rounding-level | no, rounding-level |

**Memory** (8 bytes per number):
- `'observability'`: 8·n_windows·w·p·n.
- `'fisher_per_sensor'`: 8·n_windows·p·n(n+1)/2, a ratio of (n+1)/(2w) to O. It is larger than O when w < (n+1)/2 and smaller for longer windows.
- `'fisher'`: 8·n_windows·n(n+1)/2, always far smaller.

**Numerics:**
- **F:** summing per-sensor F gives the same numbers in a different summation order. Measured differences are below 1e-14 relative.
- **Error variance:** inversion amplifies that by the condition number κ of F + λI. The measured difference is at most about 3·κ·ε (ε = 2.2e-16). That is 1e-14 to 1e-11 for well-conditioned windows, and 3e-6 relative for a nearly unobservable window with κ = 3e10.
- **Tests:** they allow 10·κ·ε.

**Unsupported queries** raise a `ValueError` naming the storage to use; they are never approximated.

**Recomputing a window:** in the Fisher modes, `observability_matrix(k)` rebuilds window k from the simulator. That matches the stored O exactly for the empirical builder, and to rounding for JAX. It is not possible for `from_sliding` analyses, which raise.

## Measurements
Setup: 250 samples, 40 states, 66 sensors, a linear system with the finite-difference builder, measured with tracemalloc. Each query selects 10 states and 7 sensors (states only for `'fisher'`), with a scalar R.

| w | Version / storage | One O | Held after `run()` | Peak, `min_error_variance` | Peak, `fisher()` / `fisher_information()` |
|---|---|---|---|---|---|
| 5 | 7b69d66 | 26 MB | 134 MB (5.2× O) | 15.4 MB | 14.3 MB |
| 5 | `'observability'` | 26 MB | 26 MB (1.0×) | 1.8 MB | 7.1 MB |
| 5 | `'fisher_per_sensor'` | 26 MB | 107 MB (4.1×) | 5.8 MB | 5.8 MB |
| 5 | `'fisher'` | 26 MB | 1.7 MB (0.06×) | 5.8 MB | 5.8 MB |
| 100 | 7b69d66 | 319 MB | 1618 MB (5.1×) | 1222 MB | 1221 MB |
| 100 | `'observability'` | 319 MB | 319 MB (1.0×) | 5.3 MB | 32.5 MB |
| 100 | `'fisher_per_sensor'` | 319 MB | 65 MB (0.21×) | 3.5 MB | 3.5 MB |
| 100 | `'fisher'` | 319 MB | 1.1 MB (0.003×) | 3.5 MB | 3.5 MB |

`fisher()` in the O mode still returns one light `FisherObservability` per window (its API), including each window's cropped O. Use `fisher_information()` when only F is needed. `tests/test_memory_benchmark.py` runs a smaller version of this table in CI.

## Recommendation
- **Keep `storage='observability'` as the default.** It answers every query, is bit-identical to earlier results, and after A and B holds exactly one copy of O, with query memory independent of the number of windows.
- **Use `'fisher_per_sensor'` for long windows (w > (n+1)/2) when memory matters** and the queries are state or sensor selections with a scalar or per-sensor R. For example, at w = 100 and n = 40 it stores 5× less than O.
- **Use `'fisher'` when the sensor set and a scalar R are fixed in advance.**
- **For a per-state λ,** read `fisher_information(states, sensors, R)` and invert F + diag(λ) directly. This works with every storage mode.

## Remaining limitation
Peak memory **during** `run()` is unchanged, at about 5× O for the finite-difference builder in every mode. `SlidingEmpiricalObservabilityMatrix` builds `O_sliding`, `O_df_sliding` and `y_plus`/`y_minus` for all windows before the analysis keeps one copy. The fix is a builder that writes each window straight into the array, or into its Fisher information, and doesn't keep the perturbation trajectories. That would bring the peak down to about 1× O, or to the Fisher size.

## API changes
- **`source` and `window_data`** now require `keep_source=True`; otherwise they raise a `RuntimeError` saying so.
- **`O_df_sliding`** builds new DataFrames on each access (a full copy of O); use `observability_matrix(k)` for one window.
- **Added:**
  - `storage`, `fisher_sensors`, `keep_source` settings (also saved in YAML);
  - `fisher_information()`;
  - `SlidingO(O=..., index=..., state_names=...)`;
  - `SlidingFisherObservability(keep_windows=...)`.
- **`FisherObservability.R` / `R_inv`** are now properties with the same values.
