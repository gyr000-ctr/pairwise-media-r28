"""waveshape 规范与边界测试。

运行：cd waveshape && python -m pytest tests -q
"""

import concurrent.futures
import math
import time

import numpy as np
import pytest

import waveshape


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def sine(freq, n, amp=0.5, phase=0.0):
    t = np.arange(n, dtype=np.float64)
    return amp * np.sin(2.0 * math.pi * freq * t + phase)


def rms(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def steady(x, trim):
    x = np.asarray(x, dtype=np.float64)
    return x[trim:-trim]


# --- 理想参考：零插值 + 砖墙低通（高倍过采样）+ 砖墙抗混叠 + 抽取 -------
def _ideal_up(x, os, cutoff=0.47):
    n = len(x)
    m = n * os
    u = np.zeros(m // 2 + 1, dtype=complex)
    u[: n // 2 + 1] = np.fft.rfft(x)
    return np.fft.irfft(u, m) * os


def _ideal_down(y, os, cutoff=0.47):
    yf = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y)) * os
    yf[f > cutoff] = 0.0
    return np.fft.irfft(yf, len(y))[::os]


def ideal(x, drive, os=64, cutoff=0.47):
    """带限意义下的理想整形结果（两端需 trim 掉瞬态后使用）。"""
    return np.clip(_ideal_down(np.tanh(drive * _ideal_up(x, os)), os, cutoff), -1.0, 1.0)


def alias_drop_db(x, drive, trim=1200, os=64):
    """1x 与 4x 相对理想参考的 RMS 误差之比，单位 dB。"""
    ref = ideal(x, drive, os=os)
    e1 = rms(steady(np.clip(np.tanh(drive * x), -1, 1), trim), steady(ref, trim))
    e4 = rms(steady(np.array(waveshape.WaveShape(drive, 4).process(x.tolist())), trim),
             steady(ref, trim))
    return 20.0 * math.log10(e1 / e4)


S = [0.2] * 100


# ---------------------------------------------------------------------------
# 题面 15 条（保留原样，并额外加固边界）
# ---------------------------------------------------------------------------
CASES = [
    ("odd", lambda: abs(waveshape.WaveShape(1.0).shape(0.3) + waveshape.WaveShape(1.0).shape(-0.3)) < 1e-12),
    ("bounded", lambda: all(abs(v) <= 1.0 for v in waveshape.WaveShape(8.0).process(S))),
    ("len", lambda: len(waveshape.WaveShape(1.0).process([0.1] * 10)) == 10),
    ("bad_drive", lambda: _raises(lambda: waveshape.WaveShape(0.0))),
    ("bad_oversample", lambda: _raises(lambda: waveshape.WaveShape(1.0, 3))),
    ("empty", lambda: waveshape.WaveShape(1.0).process([]) == []),
    ("type", lambda: isinstance(waveshape.WaveShape(1.0).process(S), list)),
    ("det", lambda: waveshape.WaveShape(1.0).process(S) == waveshape.WaveShape(1.0).process(S)),
    ("monotone", lambda: waveshape.WaveShape(1.0).shape(0.2) <= waveshape.WaveShape(1.0).shape(0.6)),
    ("nan", lambda: _raises(lambda: waveshape.WaveShape(1.0).process([float("nan")]))),
    ("no_mut", lambda: waveshape.WaveShape(1.0).process(list(S)) is not None),
    ("drive_harmonics", lambda: waveshape.WaveShape(8.0).shape(0.5) >= waveshape.WaveShape(1.0).shape(0.5)),
    ("float", lambda: isinstance(waveshape.WaveShape(1.0).shape(0.1), float)),
    ("shape_type", lambda: isinstance(waveshape.WaveShape(1.0).process(S)[0], float)),
    ("zero", lambda: abs(waveshape.WaveShape(4.0).shape(0.0)) < 1e-12),
]


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True


# ---------------------------------------------------------------------------
# 奇对称 & 单调 & 有界（强化）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("drive", [0.01, 0.5, 1.0, 3.0, 8.0, 100.0, 1e6])
def test_shape_is_exactly_odd(drive):
    ws = waveshape.WaveShape(drive)
    xs = np.linspace(-20.0, 20.0, 4001)
    for x in xs:
        assert abs(ws.shape(x) + ws.shape(-x)) <= 1e-12


@pytest.mark.parametrize("drive", [0.01, 0.5, 1.0, 8.0, 1e6])
def test_shape_strictly_monotone(drive):
    ws = waveshape.WaveShape(drive)
    xs = np.linspace(-30.0, 30.0, 20001)
    vals = [ws.shape(x) for x in xs]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("drive", [0.01, 1.0, 8.0, 1e6])
def test_shape_bounded(drive):
    ws = waveshape.WaveShape(drive)
    xs = np.linspace(-100.0, 100.0, 20001)
    assert all(-1.0 <= ws.shape(x) <= 1.0 for x in xs)


def test_process_bounded_all_oversampling():
    rng = np.random.default_rng(0)
    x = rng.uniform(-50.0, 50.0, 3000).tolist()
    for oversample in (1, 2, 4, 8):
        y = waveshape.WaveShape(8.0, oversample).process(x)
        assert max(abs(v) for v in y) <= 1.0


# ---------------------------------------------------------------------------
# 4 倍过采样混叠下降 >= 12 dB
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "freq,drive",
    [
        (0.05, 4.0),
        (0.10, 2.0),
        (0.15, 2.0),
        (0.20, 1.0),
        (0.30, 1.0),
    ],
)
def test_aliasing_reduced_at_least_12db(freq, drive):
    x = sine(freq, 16384)
    drop = alias_drop_db(x, drive, trim=1500)
    assert drop >= 12.0, "4x oversampling reduced aliasing by only %.2f dB" % drop


def test_aliasing_reduced_for_two_tone_imd():
    n = 16384
    t = np.arange(n, dtype=np.float64)
    x = 0.3 * np.sin(2 * math.pi * 0.11 * t) + 0.3 * np.sin(2 * math.pi * 0.17 * t)
    for drive in (2.0, 4.0, 8.0):
        drop = alias_drop_db(x, drive, trim=1500)
        assert drop >= 12.0, "two-tone IMD aliasing drop only %.2f dB" % drop


# ---------------------------------------------------------------------------
# 驱动参数单调提升谐波
# ---------------------------------------------------------------------------
def _high_harmonic_energy(drive, freq=0.031, n=16384):
    x = sine(freq, n)
    y = np.asarray(waveshape.WaveShape(drive, 4).process(x.tolist()))
    spec = np.abs(np.fft.rfft(y * np.kaiser(n, 12))) ** 2
    k0 = int(round(freq * n))
    band = np.zeros(len(spec), dtype=bool)
    for k in range(1, 10):
        centre = k * k0
        if centre < len(spec):
            band[max(0, centre - 3):centre + 4] = True
    return float(spec[~band].sum() / spec.sum())


def test_drive_monotonically_increases_harmonics():
    drives = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    energies = [_high_harmonic_energy(d) for d in drives]
    for a, b in zip(energies, energies[1:]):
        assert b > a, "harmonic energy must grow with drive: %r" % (energies,)


def test_drive_harmonics_on_single_shape():
    # 高驱动在固定幅度上的输出不会低于低驱动
    drives = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    vals = [waveshape.WaveShape(d).shape(0.5) for d in drives]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# 反奇对称 f(-x) = -f(x)，<= 1e-12（含 process 路径）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("oversample", [1, 2, 4, 8])
def test_process_is_odd(oversample):
    rng = np.random.default_rng(1)
    x = rng.uniform(-0.8, 0.8, 512)
    ws = waveshape.WaveShape(2.5, oversample)
    y_pos = np.asarray(ws.process(x.tolist()))
    y_neg = np.asarray(ws.process((-x).tolist()))
    assert np.max(np.abs(y_pos + y_neg)) <= 1e-12


# ---------------------------------------------------------------------------
# 输出有界 & 长度 & 类型
# ---------------------------------------------------------------------------
def test_length_preserved():
    for n in (0, 1, 2, 7, 64, 1000):
        x = [0.1] * n
        assert len(waveshape.WaveShape(2.0).process(x)) == n


def test_returns_list_of_float():
    y = waveshape.WaveShape(2.0).process([0.1, 0.2, 0.3])
    assert isinstance(y, list)
    assert all(isinstance(v, float) for v in y)


def test_empty_input_returns_empty_list():
    assert waveshape.WaveShape(3.0).process([]) == []


def test_input_not_mutated():
    x = [0.1, -0.2, 0.3] * 40
    snapshot = list(x)
    waveshape.WaveShape(2.0).process(x)
    assert x == snapshot


# ---------------------------------------------------------------------------
# 性能：100 万样本 <= 600 ms
# ---------------------------------------------------------------------------
def test_million_samples_under_600ms():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1.0, 1.0, 1_000_000).tolist()
    ws = waveshape.WaveShape(4.0, 4)
    ws.process(x[:1024])  # 预热
    # 取三次中的最好成绩，避免共享机器上的偶发调度抖动
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        ws.process(x)
        timings.append((time.perf_counter() - start) * 1000.0)
    best_ms = min(timings)
    assert best_ms <= 600.0, "1M samples took %.1f ms" % best_ms


# ---------------------------------------------------------------------------
# 非法驱动等可区分报错
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("drive", [0, 0.0, -1, -0.5, float("nan"), float("inf"), float("-inf")])
def test_invalid_drive_rejected(drive):
    with pytest.raises(waveshape.InvalidDriveError):
        waveshape.WaveShape(drive)


@pytest.mark.parametrize("drive", ["1", None, [1], object(), True])
def test_invalid_drive_type_rejected(drive):
    with pytest.raises(waveshape.InvalidDriveError):
        waveshape.WaveShape(drive)


@pytest.mark.parametrize("oversample", [0, 3, 5, 16, -1])
def test_invalid_oversample_rejected(oversample):
    with pytest.raises(waveshape.InvalidOversampleError):
        waveshape.WaveShape(1.0, oversample)


@pytest.mark.parametrize("oversample", [2.0, "4", None, True])
def test_invalid_oversample_type_rejected(oversample):
    with pytest.raises(waveshape.InvalidOversampleError):
        waveshape.WaveShape(1.0, oversample)


def test_error_types_are_distinguishable():
    with pytest.raises(waveshape.InvalidDriveError) as drive_err:
        waveshape.WaveShape(0)
    with pytest.raises(waveshape.InvalidOversampleError) as os_err:
        waveshape.WaveShape(1.0, 3)
    assert type(drive_err.value) is not type(os_err.value)
    # 同时是 ValueError，便于通用捕获
    assert isinstance(drive_err.value, ValueError)
    assert isinstance(os_err.value, ValueError)


# ---------------------------------------------------------------------------
# 非有限输入分类报错
# ---------------------------------------------------------------------------
def test_nan_input_raises_nan_error():
    with pytest.raises(waveshape.NaNInputError):
        waveshape.WaveShape(1.0).process([0.1, float("nan"), 0.2])


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_inf_input_raises_inf_error(bad):
    with pytest.raises(waveshape.InfInputError):
        waveshape.WaveShape(1.0).process([0.1, bad])


def test_non_finite_is_subtype_of_documented_error():
    assert issubclass(waveshape.NaNInputError, waveshape.NonFiniteInputError)
    assert issubclass(waveshape.InfInputError, waveshape.NonFiniteInputError)
    assert issubclass(waveshape.NonFiniteInputError, waveshape.WaveShapeError)


def test_nan_and_inf_are_distinguishable():
    with pytest.raises(waveshape.NonFiniteInputError) as nan_exc:
        waveshape.WaveShape(1.0).process([float("nan")])
    with pytest.raises(waveshape.NonFiniteInputError) as inf_exc:
        waveshape.WaveShape(1.0).process([float("inf")])
    assert type(nan_exc.value) is not type(inf_exc.value)


# ---------------------------------------------------------------------------
# 非法样本输入
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["abc", None, 3.0, {"a": 1}, [[0.1, 0.2]], ["x"], [1 + 2j]])
def test_invalid_samples_rejected(bad):
    with pytest.raises(waveshape.InvalidSampleError):
        waveshape.WaveShape(1.0).process(bad)


def test_accepts_tuples_and_numpy_arrays():
    ws = waveshape.WaveShape(1.0)
    assert len(ws.process((0.1, 0.2, 0.3))) == 3
    assert len(ws.process(np.array([0.1, 0.2, 0.3]))) == 3
    assert len(ws.process(np.array([0.1, 0.2], dtype=np.float32))) == 2


# ---------------------------------------------------------------------------
# 并发安全
# ---------------------------------------------------------------------------
def test_concurrent_process_matches_serial():
    rng = np.random.default_rng(3)
    xs = [rng.uniform(-1.0, 1.0, 2000).tolist() for _ in range(8)]
    ws = waveshape.WaveShape(3.0, 4)
    serial = [ws.process(x) for x in xs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        parallel = list(pool.map(ws.process, xs))
    assert serial == parallel


def test_concurrent_construction_and_processing():
    def work(seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1.0, 1.0, 500).tolist()
        return waveshape.WaveShape(1.0 + seed % 4, 4).process(x)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(16)))
    assert all(len(r) == 500 for r in results)


# ---------------------------------------------------------------------------
# 确定性 & 实现细节
# ---------------------------------------------------------------------------
def test_deterministic():
    x = [0.1, -0.2, 0.3] * 100
    assert waveshape.WaveShape(2.0, 4).process(x) == waveshape.WaveShape(2.0, 4).process(x)


def test_ouptut_implements_shape_end_to_end():
    # 常值输入经过完整过采样链后应接近 shape(value)
    value = 0.3
    y = waveshape.WaveShape(4.0, 4).process([value] * 2000)
    expected = waveshape.WaveShape(4.0).shape(value)
    assert abs(y[1000] - expected) <= 1e-3


def test_shape_rejects_non_finite():
    ws = waveshape.WaveShape(1.0)
    with pytest.raises(waveshape.NaNInputError):
        ws.shape(float("nan"))
    with pytest.raises(waveshape.InfInputError):
        ws.shape(float("inf"))
