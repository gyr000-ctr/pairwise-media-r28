"""Hidden-style acceptance checks for the bitcrush constraints.

The shipped ``test_core.py`` holds the 15 smoke cases; this file adds the
quantitative properties from the task statement (level count, theoretical SNR,
TPDF variance, in-band SNR gain from noise shaping, exact hold reference,
boundedness, error classification, thread safety and the latency budget).

Everything here is deterministic.  NumPy-only checks skip cleanly if NumPy is
not installed, so the suite stays runnable in a stdlib-only environment.
"""

import math
import threading
import time

import pytest

import bitcrush
from bitcrush import (
    BitCrush,
    InvalidBitsError,
    InvalidHoldError,
    NonFiniteSampleError,
)


# --------------------------------------------------------------------------- #
# Quantisation levels / SNR
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_exact_level_count_is_two_pow_bits(bits):
    n = 120_000
    ramp = [-1.0 + 2.0 * i / (n - 1) for i in range(n)]
    out = BitCrush(bits, 1).process(ramp)
    assert len(set(out)) == 2 ** bits


def test_level_step_is_two_over_levels():
    bc = BitCrush(8, 1)
    assert bc.levels == 256
    assert abs(bc.step - 2.0 / 256) <= 1e-15


@pytest.mark.parametrize("bits", [8, 12, 16])
def test_no_dither_snr_matches_theory_within_1db(bits):
    np = pytest.importorskip("numpy")
    n = 1 << 17
    t = np.arange(n)
    x = np.sin(2.0 * np.pi * 0.017 * t + 0.7)
    y = np.asarray(BitCrush(bits, 1).process(x.tolist()))
    snr = 10.0 * np.log10(np.mean(x * x) / np.mean((y - x) ** 2))
    assert abs(snr - (6.02 * bits + 1.76)) <= 1.0


# --------------------------------------------------------------------------- #
# TPDF dither statistics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_tpdf_variance_is_theoretical_within_1pct(seed):
    np = pytest.importorskip("numpy")
    d = np.asarray(BitCrush(8, 1, True, seed, shaping=False).tpdf(400_000))
    # d = U1 + U2 - 1 in LSB units -> var = 1/12 + 1/12 = 1/6.
    assert abs(d.var() - 1.0 / 6.0) / (1.0 / 6.0) <= 0.01
    # Support is [-1, +1] LSB -> 2 LSB peak-to-peak.
    assert d.min() >= -1.005 and d.max() <= 1.005


def test_dither_changes_output_and_is_seed_sensitive():
    s = [0.1, 0.2, 0.3, 0.4]
    assert BitCrush(4, 1, True, 1).process(s) != BitCrush(4, 1, False, 1).process(s)
    assert BitCrush(8, 1, True, 1).process(s) != BitCrush(8, 1, True, 2).process(s)
    assert BitCrush(8, 1, True, 1).process(s) == BitCrush(8, 1, True, 1).process(s)


# --------------------------------------------------------------------------- #
# Noise shaping / in-band SNR gain
# --------------------------------------------------------------------------- #
def _band_snr_db(np, x, y, fmax):
    X = np.fft.rfft(x)
    Y = np.fft.rfft(np.asarray(y, dtype=float))
    k = max(2, int(fmax * len(x)))
    sig = np.sum(np.abs(X[1:k]) ** 2)
    noise = max(1e-30, np.sum(np.abs(Y - X)[:k] ** 2))
    return 10.0 * np.log10(sig / noise)


def test_dithered_noise_shaping_improves_average_in_band_snr_by_3db():
    np = pytest.importorskip("numpy")
    n = 1 << 16
    t = np.arange(n)
    fmax = 0.06

    def in_band_noise_db(x, y):
        E = np.abs(np.fft.rfft(np.asarray(y, float) - x)) ** 2
        k = max(2, int(fmax * n))
        return 10.0 * np.log10(E[1:k].mean())

    # Dither benefit is statistical and can vanish for a single correlated
    # phase, so average the in-band noise reduction over fixed seeds/phases.
    reductions = []
    for seed in range(1, 9):
        x = 0.5 * np.sin(2.0 * np.pi * 0.01 * t + 0.4 * seed)
        plain = BitCrush(8, 1, False).process(x.tolist())
        shaped = BitCrush(8, 1, True, seed).process(x.tolist())
        reductions.append(in_band_noise_db(x, plain) - in_band_noise_db(x, shaped))
    assert sum(reductions) / len(reductions) >= 3.0


def test_single_realisation_shows_positive_in_band_gain():
    np = pytest.importorskip("numpy")
    n = 1 << 16
    t = np.arange(n)
    x = 0.5 * np.sin(2.0 * np.pi * 0.01 * t + 1.6)
    plain = BitCrush(8, 1, False).process(x.tolist())
    shaped = BitCrush(8, 1, True, 7).process(x.tolist())
    gain = _band_snr_db(np, x, shaped, 0.06) - _band_snr_db(np, x, plain, 0.06)
    assert gain >= 3.0


def test_noise_shaping_pushes_energy_to_high_frequencies():
    np = pytest.importorskip("numpy")
    n = 4096
    dc = 0.37
    y = np.asarray(BitCrush(8, 1, True, 1234).process([dc] * n))
    e = y - dc
    E = np.abs(np.fft.rfft(e)) ** 2

    def power(f1, f2):
        return float(np.sum(E[int(f1 * n):int(f2 * n)]))

    drop = 10.0 * math.log10(power(0.025, 0.10) / power(0.40, 0.50))
    assert drop <= -6.0


# --------------------------------------------------------------------------- #
# Hold / downsampling reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hold", [1, 2, 3, 4, 7, 16])
def test_hold_matches_per_sample_reference(hold):
    bits = 10
    m = float(1 << (bits - 1))
    n = 1007
    # Deterministic pseudo-random samples without numpy.
    x = [(((i * 1103515245 + 12345) & 0x7FFFFFFF) / 0x7FFFFFFF) * 1.6 - 0.8
         for i in range(n)]
    y = BitCrush(bits, hold).process(x)
    for i, got in enumerate(y):
        src = max(-1.0, min(1.0, x[(i // hold) * hold]))
        k = math.floor(src * m)
        k = max(-int(m), min(int(m) - 1, k))
        ref = (k + 0.5) / m
        assert abs(got - ref) <= 1e-12


@pytest.mark.parametrize("hold", [2, 5])
def test_hold_blocks_are_constant(hold):
    x = [((i * 37) % 101) / 101.0 * 2.0 - 1.0 for i in range(503)]
    y = BitCrush(8, hold).process(x)
    for i, v in enumerate(y):
        assert v == y[(i // hold) * hold]


# --------------------------------------------------------------------------- #
# Boundedness / types / mutation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_output_is_finite_and_bounded(bits):
    x = [2.5, -2.5, 1e308, -1e308, 1.0, -1.0, 0.0, 0.999999]
    y = BitCrush(bits, 3, True, 2, order=2).process(x)
    assert all(isinstance(v, float) and math.isfinite(v) and -1.0 <= v <= 1.0 for v in y)


def test_process_returns_float_list_and_does_not_mutate():
    s = [0.1, 0.2, 0.3]
    snap = list(s)
    out = BitCrush(8, 2, True, 1).process(s)
    assert isinstance(out, list) and all(isinstance(v, float) for v in out)
    assert s == snap
    assert BitCrush(8, 2).process((0.5, 0.5)) == BitCrush(8, 2).process([0.5, 0.5])
    assert BitCrush(8, 1).process([]) == []


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #
def test_invalid_bits_and_hold_are_distinguishable_types():
    with pytest.raises(InvalidBitsError):
        BitCrush(1, 1)
    with pytest.raises(InvalidHoldError):
        BitCrush(8, 0)
    assert not issubclass(InvalidBitsError, InvalidHoldError)
    assert not issubclass(InvalidHoldError, InvalidBitsError)
    for cls in (InvalidBitsError, InvalidHoldError, NonFiniteSampleError):
        assert issubclass(cls, ValueError)


def test_invalid_bits_and_hold_messages_name_parameter():
    try:
        BitCrush(1, 1)
    except ValueError as e:
        assert "bits" in str(e).lower()
    try:
        BitCrush(8, 0)
    except ValueError as e:
        assert "hold" in str(e).lower()


@pytest.mark.parametrize("bad", [0, -3, 1.5, "8", None])
def test_bad_bits_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        BitCrush(bad, 1)


@pytest.mark.parametrize("bad", [0, -2, 2.0, "2", None])
def test_bad_hold_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        BitCrush(8, bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_input_classified(value):
    with pytest.raises(NonFiniteSampleError):
        BitCrush(8, 1).process([0.1, value])


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
def test_shared_instance_is_thread_safe():
    samples = [((i * 37) % 101) / 101.0 * 2.0 - 1.0 for i in range(4000)]
    ref = BitCrush(8, 1, True, 5).process(samples)
    shared = BitCrush(8, 1, True, 5)
    results = []
    lock = threading.Lock()

    def worker(bc):
        r = bc.process(samples)
        with lock:
            results.append(r)

    threads = [
        threading.Thread(target=worker,
                         args=(shared if i % 2 else BitCrush(8, 1, True, 5),))
        for i in range(8)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(results) == 8 and all(r == ref for r in results)


# --------------------------------------------------------------------------- #
# Performance budget
# --------------------------------------------------------------------------- #
def test_one_million_samples_under_500ms():
    samples = [
        ((i * 2654435761) % 1_000_003) / 1_000_003.0 * 2.0 - 1.0
        for i in range(1_000_000)
    ]
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        BitCrush(8, 2, True, 0).process(samples)
        best = min(best, time.perf_counter() - t0)
    assert len(BitCrush(8, 1).process(samples[:10])) == 10
    assert best <= 0.5, f"best of 3 = {best * 1000:.0f} ms"
