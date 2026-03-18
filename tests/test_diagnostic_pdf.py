"""
Diagnostic test that generates a PDF report with:
  - Page 1: System equations (dynamics f and measurement h)
  - Page 2: Pipeline timing summary
  - Page 3: Observability over time for states g and d (replicates the
            final plot from examples/mono_camera_example.ipynb)

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
# Timing context manager
# ---------------------------------------------------------------------------

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

    # --- Step 6: Plot generation ---
    with _Timer() as t:
        with PdfPages(PDF_PATH) as pdf:
            _make_equations_page(pdf)
            _make_timing_page(pdf, timings, n_steps, n_windows)
            _make_observability_page(pdf, t_sim, x_sim, EV_no_nan, o_states)
    timings.append(dict(label='PDF generation',
                        duration=t.duration,
                        description='Render equations page, timing page, observability plot'))

    # --- Re-write PDF now that we have the plot timing too ---
    with PdfPages(PDF_PATH) as pdf:
        _make_equations_page(pdf)
        _make_timing_page(pdf, timings, n_steps, n_windows)
        _make_observability_page(pdf, t_sim, x_sim, EV_no_nan, o_states)

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
    print(f"\n  Diagnostic PDF saved to: {PDF_PATH}")
