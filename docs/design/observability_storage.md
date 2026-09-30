# Memory use in `ObservabilityAnalysis`: storage design

## Problem
At commit 7b69d66, `ObservabilityAnalysis` used several times more memory than the observability matrices (O) it needed.

- **O was held about 5 times after `run()`:**
  - the builder object's `O_sliding` arrays and `O_df_sliding` DataFrames, plus the analysis' own copy of every window;
  - with the finite-difference builder, `window_data['y_plus'/'y_minus']`, which is 2× O again.
- **Every query built dense matrices.** `FisherObservability` built a dense (w·p)×(w·p) `R` and `R_inv` per window, even for a scalar R, and `SlidingFisherObservability` kept every window's object until the query returned. At w = 100 a single query peaked at about 4× O.

## Changes

### A. O is stored once, and built one window at a time
- **One array:** `ObservabilityAnalysis` keeps one read-only `(n_windows, w·p, n)` float array plus one shared row index. Per-window DataFrames are built only when accessed.
- **Streaming `run()`:** each window is written into that array, or into its Fisher information (see C), as soon as it is computed.
  - The finite-difference builder computes one window at a time. The process pool uses `imap`, and the thread path keeps a bounded number of windows in flight.
  - The JAX builder hands over its batched Jacobian array directly, sharing JAX's buffer read-only.
  - So all windows' builder objects, DataFrames and `y_plus`/`y_minus` never exist at once.
- **`keep_source=True`:** keeps the full builder object and `window_data`, at the old memory cost.
- **`from_sliding`:** a list of DataFrames is copied once into the array, and the caller's list is not kept. An array passed as `SlidingO(O=..., index=..., state_names=...)` is used without a copy.

### B. Diagonal R without dense matrices
- **Scalar, per-sensor dict and `None` R:** kept as one variance per row, and F is computed as `(Oᵀ · r⁻¹) @ O`. Because `Oᵀ @ diag(r⁻¹)` has one nonzero term per element, this equals the old dense product exactly.
- **`R` / `R_inv`:** still available, built on first access.
- **Matrix R:** unchanged.
- **Per-window objects:** `SlidingFisherObservability(keep_windows=False)` drops each window's `FisherObservability` after use, and `min_error_variance` uses it.

A and B change no results: tests compare against a verbatim copy of the 7b69d66 classes (`tests/reference_7b69d66.py`). A separate check against the 7b69d66 package matched 357 result frames exactly, under both pandas 3 and pandas 2.3, including the process- and thread-parallel paths. `FisherObservability` still copies each window's O before selecting from it: under pandas < 3 that copy also normalizes the memory layout, which keeps the matrix products bit-identical.

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
Setup: 250 samples, 40 states, 66 sensors, a linear system with the finite-difference builder. Each query selects 10 states and 7 sensors (states only for `'fisher'`), with a scalar R.

**Allocations** (tracemalloc):

| w | Version / storage | One O | Peak during `run()` | Held after `run()` | Peak, `min_error_variance` | Peak, `fisher()` / `fisher_information()` |
|---|---|---|---|---|---|---|
| 5 | 7b69d66 | 26 MB | 134 MB (5.2× O) | 134 MB (5.2×) | 15.4 MB | 14.3 MB |
| 5 | `'observability'` | 26 MB | 28 MB (1.06×) | 26 MB (1.0×) | 1.7 MB | 7.6 MB |
| 5 | `'fisher_per_sensor'` | 26 MB | 108 MB (4.2×) | 107 MB (4.1×) | 5.8 MB | 5.8 MB |
| 5 | `'fisher'` | 26 MB | 3 MB (0.12×) | 1.7 MB | 5.8 MB | 5.8 MB |
| 100 | 7b69d66 | 319 MB | 1618 MB (5.1×) | 1618 MB (5.1×) | 1222 MB | 1221 MB |
| 100 | `'observability'` | 319 MB | 343 MB (1.08×) | 319 MB (1.0×) | 5.7 MB | 33.1 MB |
| 100 | `'fisher_per_sensor'` | 319 MB | 90 MB (0.28×) | 65 MB (0.21×) | 3.5 MB | 3.5 MB |
| 100 | `'fisher'` | 319 MB | 25 MB (0.08×) | 1.1 MB | 3.5 MB | 3.5 MB |

**Process memory** (RSS), which is what the operating system sees. Memory freed inside the process isn't necessarily returned to the OS, so this is the number that decides whether a run fits.

| w | Version / storage | After `run()` | After a query | Process peak |
|---|---|---|---|---|
| 5 | 7b69d66 | 145 MB | 162 MB | 162 MB |
| 5 | `'observability'` | 27 MB | 29 MB | 29 MB |
| 5 | `'fisher'` | 3 MB | 8 MB | 8 MB |
| 100 | 7b69d66 | 1633 MB | 1926 MB | 2872 MB |
| 100 | `'observability'` | 344 MB | 345 MB | 345 MB |
| 100 | `'fisher_per_sensor'` | 91 MB | 92 MB | 92 MB |
| 100 | `'fisher'` | 25 MB | 29 MB | 29 MB |

**Several analyses in one process** (w = 5): holding 1, 2, 3 and 4 analyses took 145, 292, 438 and 585 MB before, and takes 27, 53, 79 and 105 MB now. That is about 1× O each.

**JAX backend** (w = 100, same system): memory after `run()` went from 1338 MB to 382 MB in RSS. The JAX classes compute every window in one batched call, so a Fisher storage mode doesn't lower the JAX peak below about 1× O.

**Other notes:**
- `fisher()` in the O mode still returns one light `FisherObservability` per window (its API), including each window's cropped O. Use `fisher_information()` when only F is needed.
- Run times are unchanged.
- `tests/test_memory_benchmark.py` runs a smaller version of the allocation table in CI.

## Recommendation
- **Keep `storage='observability'` as the default.** It answers every query and is bit-identical to earlier results. After A and B, it needs about one copy of O at its peak and afterwards, and query memory doesn't depend on the number of windows.
- **Use `'fisher_per_sensor'` for long windows (w > (n+1)/2) when memory matters** and the queries are state or sensor selections with a scalar or per-sensor R. For example, at w = 100 and n = 40 its peak and stored size are 3.5–5× below O's, with the finite-difference builder.
- **Use `'fisher'` when the sensor set and a scalar R are fixed in advance.**
- **For a per-state λ,** read `fisher_information(states, sensors, R)` and invert F + diag(λ) directly. This works with every storage mode.

## Remaining limitations
- **JAX peak:** the JAX backend still computes all windows at once. Its peak is about 1× O plus JAX's own working memory, whatever the storage mode. A future `batch_size` option could process windows in chunks. It would be opt-in, because a different batch size may change results at the last-bit level.
- **`keep_source=True`:** uses the full builder object, at the 7b69d66 memory cost.

## API changes
- **`source` and `window_data`** now require `keep_source=True`; otherwise they raise a `RuntimeError` saying so.
- **`O_df_sliding`** builds new DataFrames on each access (a full copy of O); use `observability_matrix(k)` for one window.
- **Added:**
  - `storage`, `fisher_sensors`, `keep_source` settings (also saved in YAML);
  - internal streaming hooks on the builder classes (`_prepared`, `_iter_windows`, `_compute`), with their public behavior unchanged;
  - `fisher_information()`;
  - `SlidingO(O=..., index=..., state_names=...)`;
  - `SlidingFisherObservability(keep_windows=...)`.
- **`FisherObservability.R` / `R_inv`** are now properties with the same values.
