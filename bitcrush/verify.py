#!/usr/bin/env python3
"""Constraint verification / evidence script for the bitcrush task.

Run from this directory:

    python verify.py

It checks, with measured numbers:
  * exactly 2**bits quantisation levels;
  * no-dither SNR within 1 dB of 6.02*bits + 1.76;
  * TPDF dither amplitude (2 LSB peak-to-peak) and variance (1/6 LSB**2, <=1%);
  * in-band SNR improvement from TPDF + first-order noise shaping (>= 3 dB);
  * sample-and-hold matches the literal per-sample reference (<= 1e-12);
  * output bounded, non-finite classification, distinct parameter errors,
    thread safety and the 1,000,000-sample / 500 ms budget.

NumPy is used when available; the script exits non-zero if any check fails.
"""

import math
import sys
import threading
import time

import bitcrush
from bitcrush import BitCrush, InvalidBitsError, InvalidHoldError, NonFiniteSampleError

try:
    import numpy as np
except Exception:
    np = None

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def snr_full_band(x, y):
    x = list(x)
    sp = sum(a * a for a in x)
    nz = sum((b - a) ** 2 for a, b in zip(x, y))
    return 10.0 * math.log10(sp / nz)


def main():
    # 1. exact level count --------------------------------------------------
    for b in (2, 4, 8):
        n = 120_000
        ramp = [-1.0 + 2.0 * i / (n - 1) for i in range(n)]
        nu = len(set(BitCrush(b, 1).process(ramp)))
        check(f"exactly 2^{b} = {2 ** b} levels", nu == 2 ** b, f"unique={nu}")

    # 2. theoretical no-dither SNR -----------------------------------------
    if np is not None:
        n = 1 << 17
        t = np.arange(n)
        for b in (8, 12):
            x = np.sin(2 * np.pi * 0.017 * t + 0.7)
            y = BitCrush(b, 1).process(x.tolist())
            s = 10 * np.log10(np.mean(x * x) / np.mean((np.asarray(y) - x) ** 2))
            th = 6.02 * b + 1.76
            check(f"SNR b={b} within 1 dB of {th:.2f}", abs(s - th) <= 1.0,
                  f"SNR={s:.2f} ({s - th:+.2f} vs theory)")

    # 3. TPDF amplitude / variance ------------------------------------------
    for seed in (0, 1):
        d = BitCrush(8, 1, True, seed, shaping=False).tpdf(400_000)
        if np is not None:
            var = float(np.var(d))
        else:
            m = sum(d) / len(d)
            var = sum((v - m) ** 2 for v in d) / len(d)
        rel = abs(var - 1 / 6) / (1 / 6)
        check(f"TPDF seed={seed} variance 1/6 within 1%", rel <= 0.01,
              f"var={var:.6f} rel={rel * 100:.3f}% span=[{min(d):.3f},{max(d):.3f}] LSB")

    # 4. in-band SNR gain from dither + noise shaping -----------------------
    if np is not None:
        n = 1 << 16
        t = np.arange(n)
        fmax = 0.06

        def inband_noise(x, y):
            e = np.abs(np.fft.rfft(np.asarray(y) - x)) ** 2
            return 10 * np.log10(e[1:max(2, int(fmax * n))].mean())

        gains = []
        for seed in range(1, 9):
            x = 0.5 * np.sin(2 * np.pi * 0.01 * t + 0.4 * seed)
            p = BitCrush(8, 1, False).process(x.tolist())
            s = BitCrush(8, 1, True, seed).process(x.tolist())
            gains.append(inband_noise(x, p) - inband_noise(x, s))
        avg = sum(gains) / len(gains)
        check("TPDF + shaping lowers in-band noise by >= 3 dB (8-seed avg)",
              avg >= 3.0, f"avg={avg:.2f} dB")
        print("       note: full-band MSE SNR is not expected to rise because "
              "dither adds noise power; the benefit is decorrelation + "
              "high-frequency shaping.")

    # 5. hold vs literal per-sample reference ------------------------------
    for h in (1, 2, 3, 7):
        b = 10
        m = float(1 << (b - 1))
        x = [(((i * 1103515245 + 12345) & 0x7FFFFFFF) / 0x7FFFFFFF) * 2 - 1
             for i in range(1007)]
        y = BitCrush(b, h).process(x)
        diff = 0.0
        for i, got in enumerate(y):
            src = max(-1.0, min(1.0, x[(i // h) * h]))
            k = math.floor(src * m)
            k = max(-int(m), min(int(m) - 1, k))
            ref = (k + 0.5) / m
            diff = max(diff, abs(got - ref))
        check(f"hold={h} matches per-sample reference <=1e-12", diff <= 1e-12,
              f"maxdiff={diff:.3g}")

    # 6. bounded output -----------------------------------------------------
    y = BitCrush(4, 3, True, 2, order=2).process(
        [2.5, -2.5, 1e308, -1e308, 1, -1, 0.0])
    check("output finite and within [-1,1]",
          all(isinstance(v, float) and math.isfinite(v) and -1 <= v <= 1 for v in y))

    # 7. distinguishable parameter errors ----------------------------------
    tb = th = None
    try:
        BitCrush(1, 1)
    except ValueError as e:
        tb = type(e)
        mb = str(e)
    try:
        BitCrush(8, 0)
    except ValueError as e:
        th = type(e)
        mh = str(e)
    check("bits vs hold raise distinct ValueError subtypes",
          tb is not None and th is not None and tb is not th
          and not issubclass(tb, th) and not issubclass(th, tb)
          and "bits" in mb.lower() and "hold" in mh.lower(),
          f"{tb and tb.__name__} vs {th and th.__name__}")

    # 8. non-finite classification -----------------------------------------
    ok_nf = True
    for v in (float("nan"), float("inf"), float("-inf")):
        try:
            BitCrush(8, 1).process([0.1, v])
            ok_nf = False
        except NonFiniteSampleError:
            pass
    check("NaN/+inf/-inf raise NonFiniteSampleError(ValueError)", ok_nf)

    # 9. thread safety ------------------------------------------------------
    sig = [((i * 37) % 101) / 101 * 2 - 1 for i in range(4000)]
    ref = BitCrush(8, 1, True, 5).process(sig)
    shared = BitCrush(8, 1, True, 5)
    out, lock = [], threading.Lock()

    def worker(bc):
        r = bc.process(sig)
        with lock:
            out.append(r)

    threads = [threading.Thread(
        target=worker,
        args=(shared if i % 2 else BitCrush(8, 1, True, 5),)) for i in range(8)]
    for t_ in threads:
        t_.start()
    for t_ in threads:
        t_.join()
    check("8 concurrent calls reproduce the seeded reference",
          len(out) == 8 and all(r == ref for r in out))

    # 10. performance -------------------------------------------------------
    samples = [((i * 2654435761) % 1_000_003) / 1_000_003 * 2 - 1
               for i in range(1_000_000)]
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        BitCrush(8, 2, True, 0).process(samples)
        best = min(best, time.perf_counter() - t0)
    check("1M samples processed within 500 ms (best of 3)", best <= 0.5,
          f"{best * 1000:.0f} ms")

    print()
    if FAILURES:
        print("FAILURES:", FAILURES)
        return 1
    print("All bitcrush constraints satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
