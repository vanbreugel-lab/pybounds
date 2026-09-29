"""
Diagnostic test that generates a PDF report with:
  - Page 1: System equations (dynamics f and measurement h)
  - Page 2: Pipeline timing summary
  - Page 3: Observability over time for states g and d (replicates the
            final plot from examples/mono_camera_example.ipynb)
  - Page 4: Parallel vs serial benchmark for SlidingEmpiricalObservabilityMatrix
  - Page 5: JAX autodiff vs. legacy numerical observability comparison

Run with:  pytest tests/test_diagnostic_pdf.py -v -s
Output:    tests/diagnostic_report.pdf
"""

import os
import textwrap
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pytest
import pybounds
from pybounds import colorline, SlidingEmpiricalObservabilityMatrix, SlidingFisherObservability

PDF_PATH = 'tests/diagnostic_report.pdf'

# ---------------------------------------------------------------------------
# JAX-compatible dynamics and measurement for the mono-camera system.
# These are module-level so they are picklable and JAX-traceable.
# ---------------------------------------------------------------------------
try:
    import jax.numpy as jnp
    from pybounds import JaxSimulator, JaxEmpiricalObservabilityMatrix, JaxSlidingEmpiricalObservabilityMatrix

    def _dynamics_f_jax(x, u):
        return jnp.array([u[0], 0.0 * u[0]])

    def _measurement_h_jax(x, u):
        return jnp.array([x[0] / x[1]])

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


# Module-level factory for process-based parallel benchmark (must be picklable).
def _make_diagnostic_simulator():
    """Create a fresh Simulator for the mono-camera system (used by parallel workers)."""
    from conftest import dynamics_f, measurement_h
    return pybounds.Simulator(dynamics_f, measurement_h, dt=0.01,
                              state_names=['g', 'd'], input_names=['u'],
                              measurement_names=['r'])


# ---------------------------------------------------------------------------
# Helper: equations page
# ---------------------------------------------------------------------------

def _make_equations_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    lines = [
        ('title',  'pybounds Diagnostic Report — Mono-Camera System'),
        ('spacer', ''),
        ('head',   'States'),
        ('body',   r'$g$ — ground speed'),
        ('body',   r'$d$ — distance above ground'),
        ('spacer', ''),
        ('head',   r'Dynamics   $\dot{\mathbf{x}} = f(\mathbf{x}, u)$'),
        ('body',   r'$\dot{g} = u$'),
        ('body',   r'$\dot{d} = 0$'),
        ('spacer', ''),
        ('head',   r'Measurement   $\mathbf{y} = h(\mathbf{x})$'),
        ('body',   r'$r = g \, / \, d \quad$ (ventral optic flow)'),
        ('spacer', ''),
        ('head',   'Simulation parameters'),
        ('body',   r'$x_0 = [g_0, d_0] = [2.0,\ 3.0]$'),
        ('body',   r'$dt = 0.01\ \mathrm{s}$'),
        ('body',   r'$u(t) = \sin(3t)$ (with near-zero segment $3 \leq t < 6\ \mathrm{s}$)'),
        ('spacer', ''),
        ('head',   'Observability parameters'),
        ('body',   r'Window size $w = 6$ time-steps'),
        ('body',   r'Perturbation $\varepsilon = 10^{-4}$'),
        ('body',   r'Sensor noise $R = \{r: 0.1\}$'),
        ('body',   r'Regularisation $\lambda = 10^{-8}$'),
    ]

    font_sizes = {'title': 16, 'head': 12, 'body': 11, 'spacer': 6}
    font_weights = {'title': 'bold', 'head': 'bold', 'body': 'normal', 'spacer': 'normal'}
    y = 0.95
    dy = {'title': 0.045, 'head': 0.038, 'body': 0.032, 'spacer': 0.015}

    for kind, text in lines:
        ax.text(0.08, y, text,
                transform=ax.transAxes,
                fontsize=font_sizes[kind],
                fontweight=font_weights[kind],
                va='top')
        y -= dy[kind]

    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper: timing page
# ---------------------------------------------------------------------------

def _make_timing_page(pdf, timings, n_steps, n_windows):
    """Bar chart + table of wall-clock times for each pipeline stage."""
    fig = plt.figure(figsize=(8.5, 11))

    # --- title text ---
    fig.text(0.08, 0.96, 'Pipeline Timing Summary',
             fontsize=16, fontweight='bold', va='top')
    fig.text(0.08, 0.92,
             f'Trajectory: {n_steps} time-steps   |   '
             f'Sliding windows: {n_windows}   |   '
             f'Window size: 6',
             fontsize=10, va='top', color='#444444')

    labels = [t['label'] for t in timings]
    durations = [t['duration'] for t in timings]
    total = sum(durations)

    # --- horizontal bar chart ---
    # left=0.30 gives enough room for the y-axis step-name labels
    ax = fig.add_axes([0.30, 0.55, 0.55, 0.30])
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(labels)))
    bars = ax.barh(labels[::-1], durations[::-1], color=colors[::-1])
    ax.set_xlabel('Wall-clock time (s)', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    for bar, dur in zip(bars, durations[::-1]):
        ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{dur:.3f} s', va='center', fontsize=8)

    # --- table ---
    # Column widths as fractions of table width; description gets the most space
    col_widths = [0.20, 0.10, 0.10, 0.60]
    col_labels = ['Step', 'Time (s)', '% of total', 'Description']
    # Wrap description text to fit the column (~45 chars at this width)
    WRAP = 45
    rows = []
    for t, dur in zip(timings, durations):
        rows.append([
            t['label'],
            f"{dur:.3f}",
            f"{100 * dur / total:.1f}%",
            textwrap.fill(t['description'], WRAP),
        ])
    rows.append(['TOTAL', f"{total:.3f}", '100%', ''])

    ax2 = fig.add_axes([0.05, 0.05, 0.90, 0.44])
    ax2.axis('off')
    tbl = ax2.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc='left',
        loc='upper left',
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    # Adjust row heights to fit wrapped text (each extra line needs more height)
    base_h = 1 / (len(rows) + 2)      # rough default height per row
    for row_idx, row in enumerate(rows, start=1):   # row 0 is the header
        n_lines = max(len(str(cell).split('\n')) for cell in row)
        tbl[row_idx, 0].set_height(base_h * max(1, n_lines))
        for col_idx in range(len(col_labels)):
            tbl[row_idx, col_idx].set_height(base_h * max(1, n_lines))

    # Style header row
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor('#2c5f8a')
        tbl[0, col].set_text_props(color='white', fontweight='bold')
    # Style total row
    for col in range(len(col_labels)):
        tbl[len(rows), col].set_facecolor('#e8e8e8')
        tbl[len(rows), col].set_text_props(fontweight='bold')

    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper: observability plot page (mirrors notebook final cell)
# ---------------------------------------------------------------------------

def _make_observability_page(pdf, t_sim, x_sim, EV_no_nan, states):
    n_state = len(states)
    fig, ax = plt.subplots(n_state, 2, figsize=(8.5, 11), dpi=150)
    ax = np.atleast_2d(ax)

    cmap = 'inferno_r'
    ev_data = EV_no_nan[states].values
    min_ev = np.nanmin(ev_data[ev_data > 0])
    max_ev = np.nanmax(ev_data)
    log_low = int(np.floor(np.log10(min_ev)))
    log_high = int(np.ceil(np.log10(max_ev)))
    cnorm = mpl.colors.LogNorm(10 ** log_low, 10 ** log_high)

    for n, state_name in enumerate(states):
        colorline(t_sim, x_sim[state_name], EV_no_nan[state_name].values,
                  ax=ax[n, 0], cmap=cmap, norm=cnorm)
        colorline(t_sim, EV_no_nan[state_name].values, EV_no_nan[state_name].values,
                  ax=ax[n, 1], cmap=cmap, norm=cnorm)

        # Colorbar placed within the reserved right margin
        cax = ax[n, -1].inset_axes([1.05, 0.0, 0.06, 1.0])
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap), cax=cax,
            ticks=np.logspace(log_low, log_high, log_high - log_low + 1))
        cbar.set_label(f'min. error variance: {state_name}',
                       rotation=270, fontsize=7, labelpad=12)
        cbar.ax.tick_params(labelsize=6)

        x_vals = x_sim[state_name]
        ax[n, 0].set_ylim(np.min(x_vals) - 0.1, np.max(x_vals) + 0.1)
        ax[n, 0].set_ylabel(f'state: {state_name}', fontsize=8)

        ax[n, 1].set_ylim(10 ** log_low, 10 ** log_high)
        ax[n, 1].set_yscale('log')
        ax[n, 1].set_ylabel(f'min. error variance: {state_name}', fontsize=8)
        ax[n, 1].set_yticks(np.logspace(log_low, log_high, log_high - log_low + 1))

    for a in ax.flat:
        a.tick_params(axis='both', labelsize=6)
        a.set_xlabel('time (s)', fontsize=7)
        offset = t_sim[-1] * 0.05
        a.set_xlim(-offset, t_sim[-1] + offset)

    fig.suptitle('Minimum Error Variance over Time\n(mono-camera system)',
                 fontsize=11, fontweight='bold')
    # right=0.78 leaves room for the colorbar strip + rotated label
    fig.subplots_adjust(left=0.10, right=0.78, top=0.92, bottom=0.08,
                        wspace=0.35, hspace=0.45)

    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper: parallel benchmark page
# ---------------------------------------------------------------------------

def _make_benchmark_page(pdf, serial_time, parallel_time, n_windows,
                         results_match, max_abs_diff, n_threads,
                         parallel_crashed=False):
    """Bar chart comparing serial vs parallel SEOM wall-clock time."""
    fig = plt.figure(figsize=(8.5, 11))

    fig.text(0.08, 0.96, 'Parallel vs Serial Benchmark',
             fontsize=16, fontweight='bold', va='top')
    fig.text(0.08, 0.92,
             f'SlidingEmpiricalObservabilityMatrix  |  {n_windows} windows  '
             f'|  parallel_sliding=True  |  simulator_factory=<callable>  '
             f'|  n_workers={n_threads} (multiprocessing.Pool, spawn)',
             fontsize=9, va='top', color='#444444', wrap=True)

    speedup = serial_time / parallel_time if parallel_time > 0 else float('nan')
    correctness_color = '#2ca02c' if results_match else '#d62728'
    if parallel_crashed:
        correctness_label = 'CRASH — CasADi/IDAS fatal signal (not thread-safe)'
    elif results_match:
        correctness_label = 'PASS — results match serial'
    else:
        correctness_label = f'FAIL — max |diff| = {max_abs_diff:.2e}'

    # --- bar chart ---
    ax = fig.add_axes([0.25, 0.68, 0.50, 0.18])
    bars = ax.barh(['parallel', 'serial'], [parallel_time, serial_time],
                   color=['#1f77b4', '#ff7f0e'])
    ax.set_xlabel('Wall-clock time (s)', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    for bar, val in zip(bars, [parallel_time, serial_time]):
        ax.text(bar.get_width() + max(serial_time, parallel_time) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f} s', va='center', fontsize=9)

    # --- summary text ---
    summary_y = 0.60
    lines = [
        ('Serial time',    f'{serial_time:.3f} s'),
        ('Parallel time',  f'{parallel_time:.3f} s'),
        ('Speedup',        f'{speedup:.2f}×'),
        ('Correctness',    correctness_label),
    ]
    for label, value in lines:
        fig.text(0.10, summary_y, label + ':', fontsize=11, fontweight='bold', va='top')
        color = correctness_color if label == 'Correctness' else 'black'
        fig.text(0.38, summary_y, value, fontsize=11, va='top', color=color)
        summary_y -= 0.055

    # --- notes ---
    notes_y = 0.36
    fig.text(0.08, notes_y, 'Notes', fontsize=12, fontweight='bold', va='top')
    notes_y -= 0.04
    note_lines = [
        '• Simulator.simulate() mutates shared state (self.simulator, self.x, …) on each call.',
        '  CasADi/IDAS is NOT thread-safe: ThreadPoolExecutor causes fatal crashes.',
        '• The new simulator_factory parameter passes a zero-arg callable to SEOM.',
        '  Each worker process calls factory() once to get its own Simulator.',
        '  This eliminates all shared state between workers.',
        '• Uses multiprocessing.Pool with start method "spawn" (macOS/Windows default).',
        '  "spawn" avoids fork-safety issues with CasADi shared libraries.',
        '• Each worker incurs ~0.02 s Simulator setup overhead (one-time, amortized',
        '  across all windows assigned to that worker).',
    ]
    for line in note_lines:
        fig.text(0.08, notes_y, line, fontsize=9, va='top', color='#333333')
        notes_y -= 0.032

    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: JAX vs legacy comparison page
# ---------------------------------------------------------------------------

def _make_jax_comparison_page(pdf, t_sim_full, y_legacy_full, y_jax_full,
                               O_legacy, O_jax, timing_rows):
    """Page comparing JAX autodiff against legacy numerical observability.

    Layout (3 rows × 2 cols):
      Row 1: y-trajectory overlay (left) | residual |y_jax - y_legacy| (right)
      Row 2: O matrix heatmap legacy (left) | O matrix heatmap JAX (right)
      Row 3: Timing table (spans both columns)
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.97, 'JAX Autodiff vs Legacy Numerical Observability',
             fontsize=14, fontweight='bold', va='top')
    fig.text(0.08, 0.94,
             'Mono-camera system  |  RK4 integrator  |  jacfwd Jacobian vs ±ε perturbations',
             fontsize=9, va='top', color='#444444')

    # ---- Row 1: simulation trajectories ----
    ax_y = fig.add_axes([0.08, 0.73, 0.38, 0.17])
    ax_r = fig.add_axes([0.56, 0.73, 0.38, 0.17])

    ax_y.plot(t_sim_full, y_legacy_full.ravel(), color='#1f77b4',
              linewidth=1.5, label='do_mpc (IDAS)')
    ax_y.plot(t_sim_full, y_jax_full.ravel(), color='#ff7f0e',
              linewidth=1.2, linestyle='--', label='JAX RK4')
    ax_y.set_ylabel('y = g/d (optic flow)', fontsize=7)
    ax_y.set_xlabel('time (s)', fontsize=7)
    ax_y.tick_params(labelsize=6)
    ax_y.legend(fontsize=6, loc='upper right')
    ax_y.set_title('Simulation output: y trajectory', fontsize=8, fontweight='bold')
    ax_y.spines[['top', 'right']].set_visible(False)

    residual = np.abs(y_jax_full.ravel() - y_legacy_full.ravel())
    ax_r.semilogy(t_sim_full, residual + 1e-20, color='#d62728', linewidth=1.2)
    ax_r.set_ylabel('|y_JAX − y_legacy|', fontsize=7)
    ax_r.set_xlabel('time (s)', fontsize=7)
    ax_r.tick_params(labelsize=6)
    ax_r.set_title('Residual (log scale)', fontsize=8, fontweight='bold')
    ax_r.spines[['top', 'right']].set_visible(False)

    # ---- Row 2: O matrix heatmaps ----
    for col, (O_mat, title) in enumerate([(O_legacy, 'O legacy (numerical ±ε)'),
                                          (O_jax,    'O JAX (autodiff jacfwd)')]):
        left = 0.08 + col * 0.48
        ax_o = fig.add_axes([left, 0.42, 0.38, 0.22])
        vmax = np.max(np.abs(O_mat)) or 1.0
        im = ax_o.imshow(O_mat, aspect='auto', cmap='bwr',
                         vmin=-vmax, vmax=vmax)
        ax_o.set_title(title, fontsize=8, fontweight='bold')
        ax_o.set_xlabel('State (g, d)', fontsize=7)
        ax_o.set_ylabel('Sensor × time-step', fontsize=7)
        ax_o.set_xticks([0, 1])
        ax_o.set_xticklabels(['g', 'd'], fontsize=7)
        ax_o.tick_params(axis='y', labelsize=5)
        fig.colorbar(im, ax=ax_o, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # Difference heatmap between O_jax and O_legacy
    ax_diff = fig.add_axes([0.08, 0.28, 0.84, 0.10])
    diff = O_jax - O_legacy
    vmax_d = max(np.max(np.abs(diff)), 1e-12)
    im_d = ax_diff.imshow(diff.T, aspect='auto', cmap='bwr',
                          vmin=-vmax_d, vmax=vmax_d)
    ax_diff.set_title(f'O_JAX − O_legacy  (max |diff| = {vmax_d:.2e})',
                      fontsize=8, fontweight='bold')
    ax_diff.set_xlabel('Sensor × time-step', fontsize=7)
    ax_diff.set_yticks([0, 1])
    ax_diff.set_yticklabels(['g', 'd'], fontsize=7)
    ax_diff.tick_params(axis='x', labelsize=5)
    fig.colorbar(im_d, ax=ax_diff, fraction=0.02, pad=0.01).ax.tick_params(labelsize=6)

    # ---- Row 3: timing table ----
    ax_t = fig.add_axes([0.05, 0.02, 0.90, 0.22])
    ax_t.axis('off')
    col_widths = [0.35, 0.13, 0.13, 0.39]
    tbl = ax_t.table(
        cellText=timing_rows,
        colLabels=['Step', 'Time (s)', 'Speedup vs legacy', 'Notes'],
        colWidths=col_widths,
        cellLoc='left',
        loc='upper left',
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for col in range(4):
        tbl[0, col].set_facecolor('#2c5f8a')
        tbl[0, col].set_text_props(color='white', fontweight='bold')

    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper: JAX vmap sliding benchmark page
# ---------------------------------------------------------------------------

def _make_jax_sliding_page(pdf, serial_time, parallel_time, vmap_warm_time,
                            vmap_hot_time, n_windows, results_match, max_abs_diff,
                            n_workers):
    """Bar chart + summary comparing serial, process-parallel, and JAX vmap SEOM."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.96, 'JAX vmap Sliding Observability Benchmark',
             fontsize=16, fontweight='bold', va='top')
    fig.text(0.08, 0.92,
             f'SlidingEmpiricalObservabilityMatrix  |  {n_windows} windows  |  window size: 6',
             fontsize=9, va='top', color='#444444')

    labels = ['JAX vmap (hot)', 'JAX vmap (JIT warmup)',
              f'process-parallel ({n_workers} workers)', 'serial (legacy)']
    times = [vmap_hot_time, vmap_warm_time, parallel_time, serial_time]

    # --- bar chart ---
    ax = fig.add_axes([0.32, 0.69, 0.55, 0.20])
    colors = ['#2ca02c', '#98df8a', '#1f77b4', '#ff7f0e']
    bars = ax.barh(labels, times, color=colors)
    ax.set_xlabel('Wall-clock time (s)', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    xmax = max(times)
    for bar, val in zip(bars, times):
        ax.text(bar.get_width() + xmax * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f} s', va='center', fontsize=9)

    # --- summary text ---
    correctness_color = '#2ca02c' if results_match else '#d62728'
    correctness_label = (
        f'PASS — max |O_jax − O_serial| = {max_abs_diff:.2e}'
        if results_match else
        f'FAIL — max |diff| = {max_abs_diff:.2e}')

    def _su(ref, val):
        return f'{ref / val:.1f}×' if val > 0 else '—'

    summary_y = 0.63
    lines = [
        ('Serial (legacy)',             f'{serial_time:.3f} s',     '1.0× (baseline)'),
        (f'Process-parallel ({n_workers}w)', f'{parallel_time:.3f} s', _su(serial_time, parallel_time)),
        ('JAX vmap (JIT warmup)',       f'{vmap_warm_time:.3f} s',  _su(serial_time, vmap_warm_time)),
        ('JAX vmap (hot)',              f'{vmap_hot_time:.4f} s',   _su(serial_time, vmap_hot_time)),
        ('Correctness',                 correctness_label,           ''),
    ]
    for label, value, speedup in lines:
        fig.text(0.08, summary_y, label + ':',  fontsize=10, fontweight='bold', va='top')
        color = correctness_color if label == 'Correctness' else 'black'
        fig.text(0.40, summary_y, value,        fontsize=10, va='top', color=color)
        fig.text(0.72, summary_y, speedup,      fontsize=10, va='top', color='#333333')
        summary_y -= 0.052

    # --- notes ---
    notes_y = 0.35
    fig.text(0.08, notes_y, 'How it works', fontsize=12, fontweight='bold', va='top')
    notes_y -= 0.04
    note_lines = [
        '• JaxSlidingEmpiricalObservabilityMatrix extracts all window initial states and',
        '  input sequences into two batched arrays:',
        '    x0_batch  shape (n_windows, n)        — one x0 per window',
        '    u_batch   shape (n_windows, w, m)     — one input sequence per window',
        '• A single vmapped+JIT-compiled jacfwd call computes all Jacobians at once:',
        '    vmapped_jac = jax.jit(jax.vmap(jax.jacfwd(sim, argnums=0)))',
        '    jac_batch   = vmapped_jac(x0_batch, u_batch)   # (n_windows, w, p, n)',
        '• On CPU this becomes a single parallelisable XLA computation; on GPU it would',
        '  map to a single kernel launch, giving further speedups.',
        '• No process spawning, no Simulator cloning — pure JAX array operations.',
        '• JIT warmup cost is paid once; all subsequent calls use the compiled kernel.',
    ]
    for line in note_lines:
        fig.text(0.08, notes_y, line, fontsize=9, va='top', color='#333333',
                 fontfamily='monospace' if line.startswith('    ') else None)
        notes_y -= 0.030

    pdf.savefig(fig)
    plt.close(fig)


class _Timer:
    def __init__(self):
        self.start = None
        self.duration = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.duration = time.perf_counter() - self.start


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_generate_diagnostic_pdf(simulator):
    """Generate a three-page diagnostic PDF replicating the mono_camera_example notebook."""

    timings = []

    # --- Step 1: Simulator setup (already done via fixture; time a fresh one) ---
    from conftest import dynamics_f, measurement_h
    with _Timer() as t:
        pybounds.Simulator(dynamics_f, measurement_h, dt=0.01,
                           state_names=['g', 'd'], input_names=['u'],
                           measurement_names=['r'])
    timings.append(dict(label='Simulator setup',
                        duration=t.duration,
                        description='Construct Simulator: build do_mpc model, MPC controller'))

    # --- Step 2: Simulation ---
    dt = 0.01
    tsim1 = np.arange(0, 3, step=dt)
    tsim2 = np.arange(3, 6, step=dt)
    tsim3 = np.arange(6, 9, step=dt)
    u1 = np.sin(3 * tsim1)
    u2 = 1e-5 * np.sin(3 * tsim2)
    u3 = np.sin(3 * tsim3)
    x0 = {'g': 2.0, 'd': 3.0}
    u = dict(u=np.hstack((u1, u2, u3)))

    with _Timer() as t:
        t_sim, x_sim, u_sim, _ = simulator.simulate(x0=x0, u=u, return_full_output=True)
    n_steps = len(t_sim)
    timings.append(dict(label='Simulation',
                        duration=t.duration,
                        description=f'Open-loop simulate {n_steps} steps '
                                    f'({n_steps * dt:.1f} s of simulated time)'))

    # --- Step 3: Sliding EOM ---
    w = 6
    with _Timer() as t:
        SEOM = SlidingEmpiricalObservabilityMatrix(
            simulator, t_sim, x_sim, u_sim, w=w, eps=1e-4)
    n_windows = len(SEOM.O_df_sliding)
    timings.append(dict(label='Sliding EOM',
                        duration=t.duration,
                        description=f'SlidingEmpiricalObservabilityMatrix: '
                                    f'{n_windows} windows × {2 * 2} simulations each '
                                    f'(2 perturbations × {2} states)'))

    # --- Step 4: Sliding Fisher ---
    R = {'r': 0.1}
    o_states = ['g', 'd']
    with _Timer() as t:
        SFO = SlidingFisherObservability(
            SEOM.O_df_sliding, time=SEOM.t_sim, lam=1e-8, R=R,
            states=o_states, sensors=['r'],
            time_steps=np.arange(0, w), w=None)
    timings.append(dict(label='Sliding Fisher',
                        duration=t.duration,
                        description=f'SlidingFisherObservability: F = OᵀR⁻¹O and '
                                    f'(F + λI)⁻¹ for {n_windows} windows'))

    # --- Step 5: Extract minimum error variance ---
    with _Timer() as t:
        EV_aligned = SFO.get_minimum_error_variance()
        EV_no_nan = EV_aligned.bfill().ffill()
    timings.append(dict(label='Error variance',
                        duration=t.duration,
                        description='get_minimum_error_variance() + NaN fill'))

    # --- Step 6: Time the plot generation using an in-memory buffer ---
    import io
    with _Timer() as t:
        buf = io.BytesIO()
        with PdfPages(buf) as pdf_buf:
            _make_equations_page(pdf_buf)
            _make_timing_page(pdf_buf, timings, n_steps, n_windows)
            _make_observability_page(pdf_buf, t_sim, x_sim, EV_no_nan, o_states)
    timings.append(dict(label='PDF generation',
                        duration=t.duration,
                        description='Render equations page, timing page, observability plot'))

    # --- Step 7: Parallel SEOM benchmark (process-based, safe with CasADi) ---
    import os
    N_WORKERS = min(n_windows, os.cpu_count() or 4)
    with _Timer() as t_par:
        SEOM_par = SlidingEmpiricalObservabilityMatrix(
            simulator, t_sim, x_sim, u_sim, w=w, eps=1e-4,
            parallel_sliding=True, simulator_factory=_make_diagnostic_simulator,
            n_workers=N_WORKERS)

    # Correctness check: compare every window's O matrix against serial result
    serial_stack = np.vstack([o for o in SEOM.O_sliding])
    parallel_stack = np.vstack([o for o in SEOM_par.O_sliding])
    max_abs_diff = float(np.max(np.abs(serial_stack - parallel_stack)))
    results_match = max_abs_diff < 1e-6
    parallel_crashed = False

    # --- Step 8: JAX autodiff comparison (single window) ---
    jax_timing_rows = []
    jax_page_data = None

    if _JAX_AVAILABLE:
        jax_sim = JaxSimulator(
            _dynamics_f_jax, _measurement_h_jax, dt=dt,
            state_names=['g', 'd'], input_names=['u'], measurement_names=['r'])

        # Time JAX forward simulation (full trajectory, first call = JIT warmup)
        x0_arr = np.array([2.0, 3.0])
        u_arr_full = np.column_stack([np.hstack((u1, u2, u3))])
        with _Timer() as t_jax_sim_warm:
            y_jax_full = jax_sim.simulate(x0_arr, u_arr_full)
        with _Timer() as t_jax_sim_hot:
            y_jax_full = jax_sim.simulate(x0_arr, u_arr_full)

        # Legacy y for full trajectory: r = g/d
        y_legacy_full = (np.asarray(x_sim['g']) / np.asarray(x_sim['d']))[:, None]  # (n_steps, 1)

        # Pick first window for O matrix comparison
        win_x0 = np.array([x_sim['g'][0], x_sim['d'][0]])
        win_u = np.column_stack([u_sim['u'][:w]])

        # Legacy O (first window, already computed inside SEOM)
        O_legacy_win = SEOM.O_sliding[0]

        # JAX O (first call = JIT warmup, second call = hot)
        with _Timer() as t_jax_eom_warm:
            eom_jax = JaxEmpiricalObservabilityMatrix(jax_sim, win_x0, win_u)
        with _Timer() as t_jax_eom_hot:
            eom_jax = JaxEmpiricalObservabilityMatrix(jax_sim, win_x0, win_u)

        O_jax_win = eom_jax.O
        O_diff_max = float(np.max(np.abs(O_jax_win - O_legacy_win)))

        # Legacy single-window O timing: total SEOM / n_windows
        t_legacy_win = timings[2]['duration'] / n_windows

        def _speedup(ref, val):
            return f"{ref / val:.1f}×" if val > 0 else "—"

        jax_timing_rows = [
            ['do_mpc simulate (full traj)',
             f'{timings[1]["duration"]:.3f}', '1.0× (baseline)',
             f'{n_steps} steps, IDAS solver'],
            ['JAX RK4 simulate (JIT warmup)',
             f'{t_jax_sim_warm.duration:.3f}',
             _speedup(timings[1]['duration'], t_jax_sim_warm.duration),
             'First call: traces + compiles XLA kernel'],
            ['JAX RK4 simulate (hot)',
             f'{t_jax_sim_hot.duration:.4f}',
             _speedup(timings[1]['duration'], t_jax_sim_hot.duration),
             'Subsequent calls: runs compiled kernel'],
            ['Legacy O, single window',
             f'{t_legacy_win:.4f}',
             '1.0× (baseline)',
             f'4 perturbation sims (2 states × ±ε)'],
            ['JAX O = jacfwd (JIT warmup)',
             f'{t_jax_eom_warm.duration:.3f}',
             _speedup(t_legacy_win, t_jax_eom_warm.duration),
             'First call: traces jacfwd + compiles'],
            ['JAX O = jacfwd (hot)',
             f'{t_jax_eom_hot.duration:.4f}',
             _speedup(t_legacy_win, t_jax_eom_hot.duration),
             f'Hot call; max |O_diff| = {O_diff_max:.2e}'],
        ]

        jax_page_data = dict(
            t_sim_full=t_sim,
            y_legacy_full=y_legacy_full,
            y_jax_full=y_jax_full,
            O_legacy=O_legacy_win,
            O_jax=O_jax_win,
            timing_rows=jax_timing_rows,
        )

        print(f"\n{'─' * 60}")
        print(f"  JAX autodiff comparison (single window)")
        print(f"{'─' * 60}")
        for row in jax_timing_rows:
            print(f"  {row[0]:<38} {row[1]:>8} s  speedup {row[2]}")
        print(f"  {'Max |O_jax - O_legacy|':<38} {O_diff_max:.2e}")
        print(f"{'─' * 60}")

    # --- Step 9: JAX vmap sliding benchmark ---
    jax_sliding_page_data = None
    if _JAX_AVAILABLE:
        # JIT warmup: first call traces + compiles the vmapped kernel
        with _Timer() as t_jax_slide_warm:
            JSEOM_warm = JaxSlidingEmpiricalObservabilityMatrix(
                jax_sim, t_sim, x_sim, u_sim, w=w)
        # Hot call: uses compiled kernel
        with _Timer() as t_jax_slide_hot:
            JSEOM = JaxSlidingEmpiricalObservabilityMatrix(
                jax_sim, t_sim, x_sim, u_sim, w=w)

        # Correctness: compare all windows against serial legacy
        jax_stack = np.vstack([o for o in JSEOM.O_sliding])
        serial_stack_full = np.vstack([o for o in SEOM.O_sliding])
        jax_max_diff = float(np.max(np.abs(jax_stack - serial_stack_full)))
        jax_correct = jax_max_diff < 1e-6

        jax_sliding_page_data = dict(
            serial_time=timings[2]['duration'],
            parallel_time=t_par.duration,
            vmap_warm_time=t_jax_slide_warm.duration,
            vmap_hot_time=t_jax_slide_hot.duration,
            n_windows=n_windows,
            results_match=jax_correct,
            max_abs_diff=jax_max_diff,
            n_workers=N_WORKERS,
        )

        print(f"\n{'─' * 60}")
        print(f"  JAX vmap sliding benchmark ({n_windows} windows)")
        print(f"{'─' * 60}")
        print(f"  {'Serial (legacy)':<35} {timings[2]['duration']:>7.3f} s  1.0×")
        print(f"  {'Process-parallel':<35} {t_par.duration:>7.3f} s  "
              f"{timings[2]['duration'] / t_par.duration:.1f}×")
        print(f"  {'JAX vmap (JIT warmup)':<35} {t_jax_slide_warm.duration:>7.3f} s  "
              f"{timings[2]['duration'] / t_jax_slide_warm.duration:.1f}×")
        print(f"  {'JAX vmap (hot)':<35} {t_jax_slide_hot.duration:>7.4f} s  "
              f"{timings[2]['duration'] / t_jax_slide_hot.duration:.1f}×")
        print(f"  {'Correctness (max |diff|)':<35} {jax_max_diff:.2e}  "
              f"({'PASS' if jax_correct else 'FAIL'})")
        print(f"{'─' * 60}")

    # --- Final PDF write (includes JAX page if available) ---
    with PdfPages(PDF_PATH) as pdf:
        _make_equations_page(pdf)
        _make_timing_page(pdf, timings, n_steps, n_windows)
        _make_observability_page(pdf, t_sim, x_sim, EV_no_nan, o_states)
        _make_benchmark_page(pdf,
                             serial_time=timings[2]['duration'],
                             parallel_time=t_par.duration,
                             n_windows=n_windows,
                             results_match=results_match,
                             max_abs_diff=max_abs_diff,
                             n_threads=N_WORKERS,
                             parallel_crashed=parallel_crashed)
        if jax_page_data is not None:
            _make_jax_comparison_page(pdf, **jax_page_data)
        if jax_sliding_page_data is not None:
            _make_jax_sliding_page(pdf, **jax_sliding_page_data)

    assert os.path.exists(PDF_PATH), f"PDF not created at {PDF_PATH}"
    assert os.path.getsize(PDF_PATH) > 1000, "PDF appears empty"

    # Print timing summary to stdout
    total = sum(t['duration'] for t in timings)
    print(f"\n{'─' * 60}")
    print(f"  Pipeline timing summary")
    print(f"{'─' * 60}")
    for t in timings:
        print(f"  {t['label']:<25} {t['duration']:>7.3f} s  "
              f"({100 * t['duration'] / total:.1f}%)")
    print(f"{'─' * 60}")
    print(f"  {'TOTAL':<25} {total:>7.3f} s")
    print(f"{'─' * 60}")

    # Print parallel benchmark results
    serial_eom_time = timings[2]['duration']
    speedup = serial_eom_time / t_par.duration if t_par.duration > 0 else float('nan')
    print(f"\n{'─' * 60}")
    print(f"  Parallel SEOM benchmark (parallel_sliding=True, process-based)")
    print(f"{'─' * 60}")
    print(f"  {'Workers':<25} {N_WORKERS}")
    print(f"  {'Serial SEOM':<25} {serial_eom_time:>7.3f} s")
    print(f"  {'Parallel SEOM':<25} {t_par.duration:>7.3f} s")
    print(f"  {'Speedup':<25} {speedup:>7.2f}×")
    if parallel_crashed:
        print(f"  {'Status':<25} CRASH — CasADi/IDAS fatal signal (not thread-safe)")
    else:
        print(f"  {'Max |diff|':<25} {max_abs_diff:.2e}  "
              f"({'PASS' if results_match else 'FAIL — race condition detected'})")
    print(f"{'─' * 60}")
    print(f"\n  Diagnostic PDF saved to: {PDF_PATH}")
