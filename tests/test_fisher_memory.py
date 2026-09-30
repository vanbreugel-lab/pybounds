"""FisherObservability / SlidingFisherObservability: diagonal R without dense matrices, bit-identical to 7b69d66."""
import tracemalloc

import numpy as np
import pandas as pd
import pytest

import pybounds
import reference_7b69d66 as ref

SENSORS = [f's{j}' for j in range(7)]
STATES = [f'x{i}' for i in range(40)]
W = 100


def _O(w=W, sensors=SENSORS, states=STATES, seed=0):
    """A large O (w*p = 700 rows, 40 states) so BLAS uses its blocked kernels."""
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_tuples([(s, k) for k in range(w) for s in sensors], names=['sensor', 'time_step'])
    return pd.DataFrame(rng.normal(size=(len(index), len(states))), index=index, columns=states)


R_DICT = {s: 0.05 * (j + 1) for j, s in enumerate(SENSORS)}
SELECTIONS = [{}, {'states': STATES[:10]}, {'sensors': ['s3', 's0', 's5']}, {'time_steps': [0, 7, 42, 99]},
              {'states': STATES[5:25], 'sensors': SENSORS[:4], 'time_steps': list(range(0, W, 3))}]


def _assert_same(new, old):
    for attr in ('F', 'F_inv', 'error_variance', 'O'):
        pd.testing.assert_frame_equal(getattr(new, attr), getattr(old, attr), check_exact=True)
    assert new.lam == old.lam and new.pw == old.pw and new.n == old.n


class TestBitIdentical:

    @pytest.mark.parametrize('selection', SELECTIONS)
    @pytest.mark.parametrize('R', [0.1, 3, np.float32(0.25), R_DICT, None])
    def test_diagonal_R(self, selection, R):
        O = _O()
        with pytest.warns(UserWarning) if R is None else _nullcontext():
            new = pybounds.FisherObservability(O, R=R, lam=1e-6, **selection)
        with pytest.warns(UserWarning) if R is None else _nullcontext():
            old = ref.FisherObservability(O, R=R, lam=1e-6, **selection)
        _assert_same(new, old)
        # R / R_inv are built on access and equal the old dense ones
        pd.testing.assert_frame_equal(new.R, old.R, check_exact=True)
        pd.testing.assert_frame_equal(new.R_inv, old.R_inv, check_exact=True)
        for a, b in zip(new.get_fisher_information(), old.get_fisher_information()):
            pd.testing.assert_frame_equal(a, b, check_exact=True)

    @pytest.mark.parametrize('selection', SELECTIONS[:3])
    def test_matrix_R(self, selection):
        O = _O(w=20)
        rng = np.random.default_rng(1)
        A = rng.normal(size=(len(O), len(O)))
        for R in (np.diag(rng.uniform(0.1, 1, len(O))), 0.01 * A @ A.T + np.eye(len(O)),
                  pd.DataFrame(np.diag(rng.uniform(0.1, 1, len(O))), index=O.index, columns=O.index)):
            _assert_same(pybounds.FisherObservability(O, R=R, **selection), ref.FisherObservability(O, R=R, **selection))

    def test_force_R_scalar(self):
        O = _O()
        _assert_same(pybounds.FisherObservability(O, R=0.2, force_R_scalar=True),
                     ref.FisherObservability(O, R=0.2, force_R_scalar=True))

    def test_lam_limit(self):
        O = _O(w=4, states=STATES[:3])
        _assert_same(pybounds.FisherObservability(O, R=R_DICT, lam='limit'),
                     ref.FisherObservability(O, R=R_DICT, lam='limit'))

    @pytest.mark.parametrize('R', [0.1, R_DICT])
    def test_sliding(self, R):
        O_list = [_O(w=10, seed=k) for k in range(12)]
        time = 0.1 * np.arange(21)
        kwargs = dict(R=R, lam=1e-7, time=time, states=STATES[:15], sensors=SENSORS[1:6])
        new = pybounds.SlidingFisherObservability(O_list, **kwargs)
        old = ref.SlidingFisherObservability(O_list, **kwargs)
        pd.testing.assert_frame_equal(new.get_minimum_error_variance(), old.get_minimum_error_variance(),
                                      check_exact=True)
        for a, b in zip(new.FO, old.FO):
            pd.testing.assert_frame_equal(a.F, b.F, check_exact=True)
        light = pybounds.SlidingFisherObservability(O_list, keep_windows=False, **kwargs)
        assert light.FO == []
        pd.testing.assert_frame_equal(light.get_minimum_error_variance(), old.get_minimum_error_variance(),
                                      check_exact=True)

    def test_input_O_not_modified(self):
        O = _O(w=5)
        before = O.copy()
        pybounds.FisherObservability(O, R=0.1, states=STATES[:3], sensors=['s1'])
        pd.testing.assert_frame_equal(O, before, check_exact=True)


class TestMemory:

    @pytest.mark.parametrize('R', [0.1, R_DICT])
    def test_no_dense_R_for_diagonal_R(self, R):
        """A (w*p x w*p) float matrix is 700*700*8 = 3.9 MB; the diagonal path stays well below that."""
        O = _O()
        tracemalloc.start()
        FO = pybounds.FisherObservability(O, R=R)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        dense_bytes = len(O) ** 2 * 8
        assert peak < dense_bytes / 4
        assert FO._R is None and FO._R_inv is None   # nothing dense was built

    def test_sliding_keep_windows_false_does_not_grow(self):
        O_list = [_O(w=20, seed=k) for k in range(30)]
        peaks = {}
        for keep in (True, False):
            tracemalloc.start()
            sfo = pybounds.SlidingFisherObservability(O_list, R=0.1, keep_windows=keep)
            peaks[keep] = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            del sfo
        assert peaks[False] < peaks[True] / 5


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
