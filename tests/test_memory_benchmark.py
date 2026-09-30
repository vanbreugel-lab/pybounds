"""Memory benchmark (tracemalloc): peak bytes during run(), bytes held after it, and peak bytes during a query.

Run with `pytest tests/test_memory_benchmark.py -s` to print the table. Sizes are kept small enough for CI;
docs/design/observability_storage.md has numbers for a larger system.
"""
import gc
import tracemalloc

import pytest

from pybounds import ObservabilityAnalysis
from test_analysis_memory import LinearSim

N_STATES, N_SENSORS, N_SAMPLES = 20, 30, 150
STATES = [f'x{i}' for i in range(8)]
SENSORS = [f's{j}' for j in range(5)]


def _measure(w, storage):
    sim = LinearSim(N_STATES, N_SENSORS)
    t, x, u = sim.trajectory(N_SAMPLES)
    oa = ObservabilityAnalysis(sim, t, x, u, w=w, eps=1e-4, R=0.1, storage=storage)
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    oa.run()
    run_peak = tracemalloc.get_traced_memory()[1] - base
    gc.collect()
    held = tracemalloc.get_traced_memory()[0] - base

    query = dict(states=STATES, R=0.2, lam=1e-6)
    if storage != 'fisher':
        query['sensors'] = SENSORS
    peaks = {}
    for name, call in [('min_error_variance', lambda: oa.min_error_variance(**query)),
                       ('fisher_information', lambda: oa.fisher_information(states=STATES, R=0.2))]:
        gc.collect()
        current = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        result = call()
        peaks[name] = tracemalloc.get_traced_memory()[1] - current
        del result
    tracemalloc.stop()
    return oa, run_peak, held, peaks


@pytest.mark.parametrize('w', [5, 100])
@pytest.mark.parametrize('storage', ['observability', 'fisher_per_sensor', 'fisher'])
def test_memory(w, storage):
    oa, run_peak, held, peaks = _measure(w, storage)
    n_windows, n = N_SAMPLES - w + 1, N_STATES
    O_bytes = 8 * n_windows * w * N_SENSORS * n
    packed = n * (n + 1) // 2
    expected = {'observability': O_bytes,
                'fisher_per_sensor': 8 * n_windows * N_SENSORS * packed,
                'fisher': 8 * n_windows * packed}[storage]
    print(f'\nw={w:<3} storage={storage:<17} O={O_bytes / 1e6:7.2f} MB | run peak {run_peak / 1e6:7.2f} MB '
          f'| held {held / 1e6:7.2f} MB '
          f'(expected {expected / 1e6:.2f}) | peak min_error_variance {peaks["min_error_variance"] / 1e6:.2f} MB, '
          f'fisher_information {peaks["fisher_information"] / 1e6:.2f} MB')

    assert held <= 1.15 * expected + 1_000_000   # plus a small constant for the index and bookkeeping
    one_window = 8 * w * N_SENSORS * n
    assert run_peak <= expected + 12 * one_window + 1_000_000   # windows are streamed into storage
    # a query never needs more than a few windows' worth of O plus the per-window results
    assert peaks['min_error_variance'] < max(0.25 * O_bytes, 4 * w * N_SENSORS * n * 8 + 2_000_000)
    if storage == 'fisher_per_sensor':
        assert (held < O_bytes) == ((n + 1) / 2 < w)   # smaller than O exactly when (n+1)/2 < w
