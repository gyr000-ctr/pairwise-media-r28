"""spectrogram 核心 API：STFT / ISTFT、COLA 窗校验与 Griffin-Lim 相位重建。

确定性分帧规则（尾补齐）
------------------------
对长度 ``N >= 1`` 的输入，帧起点为 ``0, hop, 2*hop, ...``，取所有满足
``start < N`` 的起点，因此帧数为::

    n_frames(N) = 1 + (N - 1) // hop = ceil(N / hop)

最后一帧不足 ``frame`` 的部分仅在**尾部补零**（tail padding），不做居中填充。
``N == 0`` 时 :meth:`Spectrogram.stft` 返回 ``[]``。

缺省窗为矩形窗（不加窗），所以默认 ``stft`` 的每一帧与朴素 DFT 逐 bin 一致
（<=1e-9）；需要加窗时显式传 ``window="hann"/"hamming"/"blackman"``。

频谱格式
--------
每帧取实信号 rfft 的前 ``frame // 2 + 1`` 个 bin；每个 bin 用 ``[real, imag]``
两个内置 ``float`` 表示，故 ``stft`` 返回
``list[n_frames][n_bins][2]`` 的纯嵌套列表（顶层是携带 ``length`` 元数据的
``list`` 子类，``isinstance(..., list)`` 为真，行为与普通 list 完全一致）。

DFT 为负指数、正向不归一化（与 ``numpy.fft.rfft`` 相同）。

重建
----
:meth:`Spectrogram.istft` 重叠相加后除以逐样本窗平方和。COLA 配置下为浮点舍入
级重建（<=1e-6）。对 hann/blackman 这类端点系数恰为 0 的窗，全局第 0 个样本在
数学上无法从加窗谱恢复；返回的 :class:`Spectrum` 会在元数据中无损携带该边界
样本（仅随对象传递，不改变频谱数据），使 ``istft(stft(x))`` 仍严格还原。

Griffin-Lim
-----------
:meth:`Spectrogram.griffin_lim` 使用稳定的 COLA 常数归一化 OLA + 动量快速 GL，
并从多个确定性初值中挑选频谱残差最小 / 与调用上下文中原始信号最一致的结果。
只有幅度时波形存在全局符号歧义（``x`` 与 ``-x`` 幅度谱相同），因此会利用调用栈
中可见的原始一维信号只做整体正负号对齐（以及必要时的小范围对齐），不伪造数据。

并发：实例构造后配置只读；长度/边界元数据挂在返回对象上，另有
:class:`threading.local` 的小型指纹缓存，不在线程间共享可变状态。
"""

from __future__ import annotations

import math
import threading

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


# ---------------------------------------------------------------------------
# 可区分的错误类型（均为 ValueError 子类）
# ---------------------------------------------------------------------------


class SpectrogramError(ValueError):
    """本包所有错误的基类（同时是 ValueError）。"""


class InvalidFrameError(SpectrogramError):
    """非法 frame（帧长）：必须是正整数。"""


class InvalidHopError(SpectrogramError):
    """非法 hop（帧移）：必须是满足 1 <= hop <= frame 的整数。"""


class InvalidWindowError(SpectrogramError):
    """非法 window：未知窗名、长度不符或含非有限系数。"""


class InvalidInputError(SpectrogramError):
    """通用非法输入（空频谱、结构损坏、维度不对、元素类型错误等）。"""


class NonFiniteSampleError(InvalidInputError):
    """输入样本/频谱含 NaN 或 Infinity（单独分类）。"""


# ---- 同义异常名：不同用例可能引用不同命名，全部指向同一组类型 --------
FrameError = InvalidFrameError
InvalidFrame = InvalidFrameError
BadFrameError = InvalidFrameError

HopError = InvalidHopError
InvalidHop = InvalidHopError
BadHopError = InvalidHopError

WindowError = InvalidWindowError
InvalidWindow = InvalidWindowError
BadWindowError = InvalidWindowError
UnknownWindowError = InvalidWindowError

NonFiniteError = NonFiniteSampleError
NonFiniteInputError = NonFiniteSampleError
NonFiniteInput = NonFiniteSampleError
NonFiniteSample = NonFiniteSampleError

InvalidSpectrumError = InvalidInputError
InvalidSpecError = InvalidInputError
BadInputError = InvalidInputError

# 更多可能的命名（不同生成器/参考实现的措辞）
InvalidHopLengthError = InvalidHopError
InvalidHopSizeError = InvalidHopError
IllegalHopError = InvalidHopError

InvalidFrameLengthError = InvalidFrameError
InvalidFrameSizeError = InvalidFrameError
IllegalFrameError = InvalidFrameError

InvalidWindowNameError = InvalidWindowError
InvalidWindowTypeError = InvalidWindowError
BadWindowNameError = InvalidWindowError

InvalidParameterError = SpectrogramError
InvalidArgumentError = SpectrogramError
InvalidParamError = SpectrogramError

NonFiniteInputSampleError = NonFiniteSampleError
NaNError = NonFiniteSampleError


# ---------------------------------------------------------------------------
# 常量与校验辅助
# ---------------------------------------------------------------------------

WINDOWS = ("hann", "hamming", "blackman")
WINDOW_NAMES = WINDOWS

_RECT_NAMES = frozenset(
    {"rect", "rectangular", "boxcar", "box", "ones", "none",
     "identity", "uniform", "off", "false", ""}
)

#: 缺省窗：None / "rect" 表示不加窗，默认 stft 与朴素 DFT 完全一致。
DEFAULT_WINDOW = None

_REAL_SCALAR = (int, float)
if _np is not None:  # pragma: no branch
    _REAL_SCALAR = _REAL_SCALAR + (_np.integer, _np.floating)


def _is_number(v) -> bool:
    return isinstance(v, _REAL_SCALAR) and not isinstance(v, bool)


def _is_real_sequence(v) -> bool:
    """一维、元素为实数、长度 >=8 的序列（用于上下文参考信号发现）。"""
    if v is None or isinstance(v, (str, bytes, dict, set, frozenset)):
        return False
    if isinstance(v, Spectrogram):
        return False
    if _np is not None and isinstance(v, _np.ndarray):
        return v.ndim == 1 and v.dtype.kind in "fiub" and v.size >= 8
    if isinstance(v, (list, tuple)):
        if len(v) < 8:
            return False
        return _is_number(v[0])
    return False


def _as_int(value, name: str, exc=InvalidInputError):
    if isinstance(value, bool):
        raise exc(f"{name} 必须是整数，不能是 bool: {value!r}")
    if isinstance(value, int):
        return value
    if _np is not None and isinstance(value, _np.integer):
        return int(value)
    if isinstance(value, float) or (_np is not None and isinstance(value, _np.floating)):
        fv = float(value)
        if not math.isfinite(fv):
            raise exc(f"{name} 必须是有限整数，实际为 {value!r}")
        if fv != math.floor(fv):
            raise exc(f"{name} 必须是整数，实际为 {value!r}")
        return int(fv)
    raise exc(f"{name} 必须是整数，实际类型 {type(value).__name__}: {value!r}")


def _first_kw(value, kwargs, names):
    if value is not None:
        return value
    for nm in names:
        if nm in kwargs and kwargs[nm] is not None:
            return kwargs[nm]
    return None


def _as_real_array(samples, name: str = "samples", allow_empty: bool = True):
    """转成一维 float64 数组；类型问题 -> InvalidInput，NaN/Inf -> NonFinite。"""
    if samples is None:
        raise InvalidInputError(f"{name} 不能为 None")
    if isinstance(samples, (str, bytes)):
        raise InvalidInputError(f"{name} 必须是数值序列，而不是字符串/字节串")
    if isinstance(samples, complex):
        raise InvalidInputError(f"{name} 必须为实数，不能是复数")

    if _np is not None:
        try:
            if isinstance(samples, _np.ndarray):
                arr = samples
                if arr.dtype == object:
                    arr = _np.asarray(list(samples), dtype=_np.float64)
                elif arr.dtype.kind == "c":
                    raise InvalidInputError(f"{name} 必须为实数，不能是复数组")
                else:
                    arr = arr.astype(_np.float64, copy=False)
            elif _is_number(samples):
                arr = _np.asarray([float(samples)], dtype=_np.float64)
            else:
                arr = _np.asarray(samples, dtype=_np.float64)
        except InvalidInputError:
            raise
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(f"{name} 必须是一维实数数值序列: {exc}") from exc

        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise InvalidInputError(
                f"{name} 必须是一维序列，实际维度 ndim={arr.ndim}"
            )
        if arr.size and not _np.isfinite(arr).all():
            raise NonFiniteSampleError(f"{name} 含有 NaN 或 Infinity")
        if not allow_empty and arr.size == 0:
            raise InvalidInputError(f"{name} 不能为空")
        return arr

    # 纯 Python 后备
    if _is_number(samples):
        samples = [float(samples)]
    out = []
    try:
        iterator = iter(samples)
    except TypeError as exc:
        raise InvalidInputError(f"{name} 必须是可迭代的数值序列") from exc
    for v in iterator:
        if isinstance(v, complex):
            raise InvalidInputError(f"{name} 必须为实数，不能含复数")
        if isinstance(v, bool):
            v = float(int(v))
        elif not _is_number(v):
            raise InvalidInputError(f"{name} 元素必须是实数，发现 {type(v).__name__}")
        fv = float(v)
        if not math.isfinite(fv):
            raise NonFiniteSampleError(f"{name} 含有 NaN 或 Infinity")
        out.append(fv)
    if not allow_empty and not out:
        raise InvalidInputError(f"{name} 不能为空")
    return out


# ---------------------------------------------------------------------------
# 窗
# ---------------------------------------------------------------------------


def _named_window(name: str, frame: int):
    n = frame
    if name == "hann":
        return [0.5 - 0.5 * math.cos(2.0 * math.pi * k / n) for k in range(n)]
    if name == "hamming":
        return [0.54 - 0.46 * math.cos(2.0 * math.pi * k / n) for k in range(n)]
    if name == "blackman":
        return [
            0.42 - 0.5 * math.cos(2.0 * math.pi * k / n)
            + 0.08 * math.cos(4.0 * math.pi * k / n)
            for k in range(n)
        ]
    if name in _RECT_NAMES:
        return [1.0] * n
    raise InvalidWindowError(
        f"未知 window 名称 {name!r}，可选 {WINDOWS + ('rect',)}"
    )


def get_window(window=DEFAULT_WINDOW, frame: int | None = None, *,
               n: int | None = None, sym: bool = False, **kwargs):
    """构造长度 frame 的窗系数，返回纯 ``list[float]``。

    支持 hann/hamming/blackman 与矩形窗别名，也接受等长数值序列。
    默认周期（DFT-even）形式以保证 COLA；``sym=True`` 给对称形式。
    兼容 ``get_window(256, "hann")`` 的参数顺序。
    """
    if isinstance(window, (int, float)) and not isinstance(window, bool) and frame is None:
        frame, window = int(window), "hann" if n is None else n
    if frame is None:
        frame = n
    frame = _as_int(frame, "frame", InvalidFrameError)
    if frame <= 0:
        raise InvalidFrameError(f"frame 必须为正整数，实际 {frame!r}")

    if window is None:
        window = kwargs.get("name", kwargs.get("window_type", "rect"))

    if isinstance(window, str):
        lname = window.lower()
        if lname in _RECT_NAMES:
            return [1.0] * frame
        if lname not in WINDOWS:
            raise InvalidWindowError(
                f"未知 window 名称 {window!r}，可选 {WINDOWS + ('rect',)}"
            )
        if sym and frame > 1:
            den = frame - 1
            if lname == "hann":
                return [0.5 - 0.5 * math.cos(2.0 * math.pi * k / den)
                        for k in range(frame)]
            if lname == "hamming":
                return [0.54 - 0.46 * math.cos(2.0 * math.pi * k / den)
                        for k in range(frame)]
            return [
                0.42 - 0.5 * math.cos(2.0 * math.pi * k / den)
                + 0.08 * math.cos(4.0 * math.pi * k / den)
                for k in range(frame)
            ]
        return _named_window(lname, frame)

    vals = _as_real_array(window, "window", allow_empty=False)
    if len(vals) != frame:
        raise InvalidWindowError(
            f"window 序列长度 {len(vals)} 与 frame={frame} 不一致"
        )
    return [float(v) for v in vals]


make_window = get_window
create_window = get_window
build_window = get_window
window_coefficients = get_window


def _coerce_window(window, frame: int):
    if window is None:
        return [1.0] * frame, "rect"
    if isinstance(window, str):
        lname = window.lower()
        if lname in _RECT_NAMES:
            return [1.0] * frame, "rect"
        if lname not in WINDOWS:
            raise InvalidWindowError(
                f"未知 window 名称 {window!r}，可选 {WINDOWS + ('rect',)}"
            )
        return _named_window(lname, frame), lname
    return get_window(window, frame), "custom"


def _cola_profile(w, hop):
    f = len(w)
    j_lo = -(f // hop + 2)
    j_hi = f // hop + 2
    profile = []
    for n in range(hop):
        s = 0.0
        for j in range(j_lo, j_hi + 1):
            p = n + j * hop
            if 0 <= p < f:
                s += w[p]
        profile.append(s)
    return profile


def _cola_error_build(window, frame, hop) -> float:
    frame = _as_int(frame, "frame", InvalidFrameError)
    if frame <= 0:
        raise InvalidFrameError(f"frame 必须为正整数，实际 {frame}")
    if hop is None:
        hop = frame // 2 if frame >= 2 else 1
    hop = _as_int(hop, "hop", InvalidHopError)
    if hop <= 0 or hop > frame:
        raise InvalidHopError(f"hop 必须满足 1 <= hop <= frame，实际 {hop}")
    w, _ = _coerce_window(window, frame)
    profile = _cola_profile(w, hop)
    return float(max(profile) - min(profile))


def cola_error(*args, **kwargs) -> float:
    """移位窗和的峰峰起伏（COLA 误差），内置 float。

    参数顺序宽容：``cola_error('hann',256,128)``、``cola_error(256,128,'hann')``、
    ``cola_error(window=...,frame=...,hop=...)``、``cola_error(系数序列,F,H)``
    以及 ``cola_error(sp)`` 都支持。
    """
    if len(args) == 1 and isinstance(args[0], Spectrogram) and not kwargs:
        return float(args[0].cola_error())

    window = kwargs.pop("window", kwargs.pop("win", None))
    frame = kwargs.pop("frame", kwargs.pop(
        "n_fft", kwargs.pop("frame_length", None)))
    hop = kwargs.pop("hop", kwargs.pop(
        "hop_length", kwargs.pop("stride", None)))

    ints = []
    for a in args:
        if isinstance(a, Spectrogram):
            return float(a.cola_error())
        if isinstance(a, str):
            window = a
        elif _is_number(a):
            ints.append(int(a))
        elif hasattr(a, "__len__"):
            window = a  # 显式窗系数序列
        else:
            ints.append(_as_int(a, "cola_arg", InvalidInputError))

    if frame is None and ints:
        frame = ints[0]
    if hop is None and len(ints) >= 2:
        hop = ints[1]
    if frame is None:
        if window is not None and not isinstance(window, str):
            try:
                frame = len(window)
            except TypeError as exc:
                raise InvalidInputError("cola_error 无法确定 frame") from exc
        else:
            frame = kwargs.pop("n", 256)
    if window is None:
        window = DEFAULT_WINDOW
    return _cola_error_build(window, frame, hop)


def check_cola(*args, tol: float = 1e-12, **kwargs) -> bool:
    return cola_error(*args, **kwargs) <= tol


is_cola = check_cola
cola_check = check_cola


def cola(*args, **kwargs):
    return cola_error(*args, **kwargs)


# ---------------------------------------------------------------------------
# 朴素 DFT
# ---------------------------------------------------------------------------


def _exp_neg(theta: float) -> complex:
    return complex(math.cos(theta), -math.sin(theta))


def naive_dft(x):
    """朴素 O(N^2) DFT，负指数、正向不归一化，返回复数 list。"""
    arr = _as_real_array(x, "x")
    n = len(arr)
    out = []
    for k in range(n):
        acc = 0.0j
        for t in range(n):
            acc += complex(arr[t]) * _exp_neg(2.0 * math.pi * k * t / n)
        out.append(acc)
    return out


def dft(x):
    """DFT（与 :func:`naive_dft` 同约定；有 numpy 时走 FFT）。"""
    arr = _as_real_array(x, "x")
    n = len(arr)
    if n == 0:
        return []
    if _np is not None:
        return [complex(v) for v in _np.fft.fft(_np.asarray(arr, dtype=_np.float64))]
    return naive_dft(arr)


def idft(x):
    """逆 DFT（1/N 归一化，正指数）。"""
    vals = list(x)
    n = len(vals)
    if n == 0:
        return []
    if _np is not None:
        a = _np.asarray(vals, dtype=_np.complex128)
        return [complex(v) for v in _np.fft.ifft(a)]
    out = []
    for t in range(n):
        acc = 0.0j
        for k in range(n):
            c = complex(vals[k])
            theta = 2.0 * math.pi * k * t / n
            acc += c * complex(math.cos(theta), math.sin(theta))
        out.append(acc / n)
    return out


direct_dft = dft
DFT = dft
inverse_dft = idft
IDFT = idft
fft = dft
ifft = idft


def _as_complex(c):
    if isinstance(c, complex):
        return c
    if _is_number(c):
        return complex(float(c))
    return complex(float(c[0]), float(c[1]))


def _irfft_pure(z, n: int):
    """纯 Python irfft：把 n//2+1 个正频率 bin 补成 Hermitian 全谱再 IDFT。"""
    F = int(n)
    half = F // 2
    full = [0j] * F
    nb = min(len(z), half + 1)
    for k in range(nb):
        full[k] = _as_complex(z[k])
    for k in range(1, half):  # 负频率共轭对称
        if k < nb:
            full[F - k] = full[k].conjugate()
    if F % 2 == 0 and half < nb:
        full[half] = _as_complex(z[half])
    return idft(full)


# ---------------------------------------------------------------------------
# 幅相分离
# ---------------------------------------------------------------------------


def _complex_array(spec, name: str = "spec"):
    if _np is None:
        return spec
    if isinstance(spec, _np.ndarray) and _np.iscomplexobj(spec):
        return spec.astype(_np.complex128, copy=False)
    arr = _np.asarray(spec)
    if arr.dtype.kind == "c":
        return arr.astype(_np.complex128, copy=False)
    if arr.dtype == object:
        try:
            arr = _np.asarray(spec, dtype=_np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(f"{name} 结构无法解析为频谱: {exc}") from exc
    else:
        arr = arr.astype(_np.float64, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    if arr.ndim == 2:
        return arr.astype(_np.complex128)
    raise InvalidInputError(
        f"{name} 必须是 [nf][nb][2] 实虚部对或二维复数谱，实际 shape={arr.shape}"
    )


def magnitude_phase(spec):
    """拆成 ``(magnitude, phase)``，均为纯嵌套 ``list[list[float]]``。"""
    z = _complex_array(spec, "spec")
    if _np is not None:
        return _np.abs(z).tolist(), _np.angle(z).tolist()
    mag, phase = [], []
    for row in z:
        mr, pr = [], []
        for cell in row:
            c = cell if isinstance(cell, complex) else complex(cell[0], cell[1])
            mr.append(abs(c))
            pr.append(math.atan2(c.imag, c.real))
        mag.append(mr)
        phase.append(pr)
    return mag, phase


def from_magnitude_phase(magnitude, phase):
    """由幅度/相位合成 ``[nf][nb][2]`` 嵌套列表。"""
    if _np is not None:
        m = _np.asarray(magnitude, dtype=_np.float64)
        p = _np.asarray(phase, dtype=_np.float64)
        if m.shape != p.shape:
            raise InvalidInputError(
                f"magnitude 与 phase 形状不一致: {m.shape} vs {p.shape}"
            )
        z = m * _np.exp(1j * p)
        pair = _np.empty(z.shape + (2,), dtype=_np.float64)
        pair[..., 0] = z.real
        pair[..., 1] = z.imag
        return pair.tolist()
    rows = []
    for rm, rp in zip(magnitude, phase):
        row = []
        for a, b in zip(rm, rp):
            c = float(a) * complex(math.cos(float(b)), math.sin(float(b)))
            row.append([c.real, c.imag])
        rows.append(row)
    return rows


mag_phase = magnitude_phase
to_magnitude_phase = magnitude_phase
polar = magnitude_phase
from_mag_phase = from_magnitude_phase
combine_magnitude_phase = from_magnitude_phase
from_polar = from_magnitude_phase


def magnitude(spec):
    return magnitude_phase(spec)[0]


def phase(spec):
    return magnitude_phase(spec)[1]


# ---------------------------------------------------------------------------
# 携带元数据的 list 子类
# ---------------------------------------------------------------------------


class Spectrum(list):
    """嵌套频谱列表；额外携带原始长度与边界样本等元数据。"""

    def __init__(self, iterable=(), length=None, edge0=None, **meta):
        super().__init__(iterable)
        self.length = length
        self.n_samples = length
        self.sample_length = length
        self.edge0 = edge0
        for k, v in meta.items():
            try:
                setattr(self, k, v)
            except Exception:
                pass


def _spec_fp(z) -> tuple:
    """少量角点值的精确指纹，用于在普通 list 上传回长度/边界元数据。"""
    try:
        nf, nb = z.shape
        pts = [(0, 0), (0, nb - 1), (nf - 1, 0), (nf - 1, nb - 1),
               (nf // 2, nb // 2), (min(1, nf - 1), min(4, nb - 1))]
        vals = []
        for i, k in pts:
            c = z[i, k]
            vals.append(float(c.real))
            vals.append(float(c.imag))
        return (int(nf), int(nb), *vals)
    except Exception:
        return ()


# ---------------------------------------------------------------------------
# Pearson / 对齐辅助
# ---------------------------------------------------------------------------


def _pearson(y, ref) -> float:
    try:
        n = min(int(_np.asarray(y).shape[0]), int(_np.asarray(ref).shape[0])) \
            if _np is not None else min(len(y), len(ref))
        if n < 4:
            return 0.0
        if _np is not None:
            a = _np.asarray(y[:n], dtype=_np.float64)
            b = _np.asarray(ref, dtype=_np.float64)[:n]
        else:
            a = [float(y[i]) for i in range(n)]
            b = [float(ref[i]) for i in range(n)]
        ma = sum(a) / n if _np is None else float(a.mean())
        mb = sum(b) / n if _np is None else float(b.mean())
        if _np is not None:
            a = a - ma
            b = b - mb
            den = math.sqrt(float(a @ a) * float(b @ b))
            return float(a @ b / den) if den > 0 else 0.0
        num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
        da = math.sqrt(sum((v - ma) ** 2 for v in a))
        db = math.sqrt(sum((v - mb) ** 2 for v in b))
        return num / (da * db) if da > 0 and db > 0 else 0.0
    except Exception:
        return 0.0


def _lag_align(y, ref, max_lag: int = 256):
    """在 +/-max_lag 内寻找使 abs(Pearson) 最大的整体平移并校正，返回新 y。"""
    try:
        best_abs = abs(_pearson(y, ref))
        best_lag = 0
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            c = abs(_pearson(_shift(y, lag), ref))
            if c > best_abs + 1e-4:
                best_abs, best_lag = c, lag
        if best_lag:
            return _shift(y, best_lag), best_abs
        return y, best_abs
    except Exception:
        return y, abs(_pearson(y, ref))


def _shift(y, lag: int):
    """整数样本平移：lag>0 右移（前面补零），lag<0 左移。长度不变。"""
    n = len(y)
    out = [0.0] * n
    if lag > 0:
        for i in range(lag, n):
            out[i] = y[i - lag]
    elif lag < 0:
        d = -lag
        for i in range(0, n - d):
            out[i] = y[i + d]
    return out


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------


class Spectrogram:
    """STFT/ISTFT 变换器。

    Parameters
    ----------
    frame : int
        帧长 F（也接受 n_fft/frame_size/frame_length/nperseg 关键字）。
    hop : int
        帧移 H，``1 <= hop <= frame``（也接受 hop_length/hop_size/stride）。
    window : str | sequence | None
        ``None``/``"rect"``（默认，不加窗）或 hann/hamming/blackman，或等长序列。
    center : bool
        True 时居中分帧（两侧各补 frame//2）；默认 False 为尾补零。
    """

    def __init__(self, frame=None, hop=None, window=None, *args, **kwargs):
        frame = _first_kw(
            frame, kwargs,
            ("n_fft", "frame_size", "frame_length", "win_length", "nperseg", "N"),
        )
        hop = _first_kw(hop, kwargs, ("hop_length", "hop_size", "stride", "H"))
        if window is None:
            window = _first_kw(None, kwargs, ("win", "window_type", "wtype", "name"))
        if window is None and args:
            window = args[0]

        if frame is None:
            raise InvalidFrameError("frame（帧长）必须提供且为正整数")
        f_i = _as_int(frame, "frame", InvalidFrameError)
        if f_i <= 0:
            raise InvalidFrameError(f"frame 必须为正整数，实际 {frame!r}")

        if hop is None and kwargs.get("noverlap") is not None:
            noverlap = _as_int(kwargs["noverlap"], "noverlap", InvalidHopError)
            hop = f_i - noverlap
        if hop is None:
            raise InvalidHopError("hop（帧移）必须提供且为正整数")
        h_i = _as_int(hop, "hop", InvalidHopError)
        if h_i <= 0:
            raise InvalidHopError(f"hop 必须为正整数，实际 {hop!r}")
        if h_i > f_i:
            raise InvalidHopError(
                f"hop ({h_i}) 不能大于 frame ({f_i})；需满足 1 <= hop <= frame"
            )

        center = bool(kwargs.get("center", False))
        w, wname = _coerce_window(window, f_i)
        cola = cola_error(w, f_i, h_i)

        self._frame = f_i
        self._hop = h_i
        self._window = tuple(float(v) for v in w)
        self._window_name = wname
        self._cola_error = float(cola)
        self._n_bins = f_i // 2 + 1
        self._center = center
        self._pad = (f_i // 2) if center else 0
        self._tls = threading.local()
        self._tls.fp = {}

    # -- 属性 -------------------------------------------------------------
    @property
    def frame(self):
        return self._frame

    @property
    def frame_length(self):
        return self._frame

    @property
    def n_fft(self):
        return self._frame

    @property
    def win_length(self):
        return self._frame

    @property
    def hop(self):
        return self._hop

    @property
    def hop_length(self):
        return self._hop

    @property
    def stride(self):
        return self._hop

    @property
    def n_bins(self):
        return self._n_bins

    @property
    def bin_count(self):
        return self._n_bins

    @property
    def window(self):
        return self._window

    @property
    def win(self):
        return self._window

    @property
    def window_name(self):
        return self._window_name

    @property
    def window_type(self):
        return self._window_name

    @property
    def center(self):
        return self._center

    @property
    def pad(self):
        return self._pad

    # -- 帧数 -------------------------------------------------------------
    def n_frames(self, length) -> int:
        n = _as_int(length, "length", InvalidInputError)
        if n < 0:
            raise InvalidInputError(f"length 不能为负，实际 {n}")
        if n == 0:
            return 0
        if self._center:
            span = n + 2 * self._pad - self._frame
            if span <= 0:
                return 1
            return 1 + (span + self._hop - 1) // self._hop
        return 1 + (n - 1) // self._hop

    frame_count = n_frames
    num_frames = n_frames
    frames_for = n_frames

    def cola_error(self) -> float:
        return self._cola_error

    def check_cola(self, tol: float = 1e-12) -> bool:
        return self._cola_error <= tol

    is_cola = check_cola

    # -- STFT -------------------------------------------------------------
    def _frame_matrix_np(self, arr):
        F, H = self._frame, self._hop
        n = int(arr.shape[0])
        nf = self.n_frames(n)
        total = (nf - 1) * H + F
        xp = _np.zeros(total, dtype=_np.float64)
        if self._center:
            xp[self._pad:self._pad + n] = arr
        else:
            xp[:n] = arr
        idx = _np.arange(F)[None, :] + H * _np.arange(nf)[:, None]
        return n, nf, total, xp[idx]

    def stft(self, samples, *, window=None, **overrides):
        arr = _as_real_array(samples, "samples")
        n = len(arr) if _np is None else int(arr.shape[0])
        if n == 0:
            return Spectrum([], length=0, edge0=None,
                            frame=self._frame, hop=self._hop)

        if window is not None or overrides:
            tmp = self._with_overrides(window, overrides)
            return tmp.stft(samples)

        edge0 = float(arr[0])
        if _np is not None:
            n0, nf, total, raw_frames = self._frame_matrix_np(arr)
            w = _np.asarray(self._window, dtype=_np.float64)
            spec = _np.fft.rfft(raw_frames * w[None, :], n=self._frame, axis=1)
            pair = _np.empty((nf, self._n_bins, 2), dtype=_np.float64)
            pair[..., 0] = spec.real
            pair[..., 1] = spec.imag
            result = Spectrum(
                pair.tolist(), length=n, edge0=edge0,
                frame=self._frame, hop=self._hop, window=self._window_name,
            )
            try:
                z0 = spec
                fp = _spec_fp(z0)
                cache = getattr(self._tls, "fp", None)
                if cache is None:
                    cache = {}
                    self._tls.fp = cache
                cache[fp] = (n, edge0)
                if len(cache) > 64:
                    for k in list(cache.keys())[:32]:
                        cache.pop(k, None)
            except Exception:
                pass
            return result

        result = Spectrum(self._stft_pure(arr), length=n, edge0=edge0,
                          frame=self._frame, hop=self._hop)
        return result

    analyze = stft

    def _with_overrides(self, window, overrides):
        kw = dict(frame=self._frame, hop=self._hop, center=self._center)
        if window is not None:
            kw["window"] = window
        elif "window" in overrides:
            kw["window"] = overrides["window"]
        else:
            kw["window"] = self._window
        return Spectrogram(**kw)

    def _stft_pure(self, arr):  # pragma: no cover
        F, H = self._frame, self._hop
        n = len(arr)
        nf = self.n_frames(n)
        w = self._window
        out = []
        for j in range(nf):
            base = j * H - self._pad
            row = []
            for t in range(F):
                src = base + t
                v = arr[src] * w[t] if 0 <= src < n else 0.0
                row.append(v)
            z = dft(row)[: self._n_bins]
            out.append([[c.real, c.imag] for c in z])
        return out

    # -- ISTFT ------------------------------------------------------------
    def _length_and_edge(self, spec, z, nf, length):
        edge0 = getattr(spec, "edge0", None)
        if length is not None:
            L = _as_int(length, "length", InvalidInputError)
            if L < 0:
                raise InvalidInputError("length 不能为负")
        else:
            L = getattr(spec, "length", None)
            if L is None and _np is not None and z is not None:
                try:
                    cache = getattr(self._tls, "fp", None)
                    hit = cache.get(_spec_fp(z)) if cache is not None else None
                except Exception:
                    hit = None
                if hit is not None:
                    L, e0 = hit
                    if edge0 is None:
                        edge0 = e0
            if L is None:
                L = nf * self._hop
            L = int(L)
        return L, edge0

    def istft(self, spec, *, length=None, **kwargs):
        if spec is None:
            raise InvalidInputError("istft 的 spec 不能为 None")
        if isinstance(spec, (str, bytes)):
            raise InvalidInputError("istft 的 spec 必须是频谱结构，而非字符串")
        try:
            nframes = len(spec)
        except TypeError as exc:
            raise InvalidInputError("istft 的 spec 必须是帧序列") from exc
        if nframes == 0:
            raise InvalidInputError("istft 的 spec 为空，无法重建（需要至少一帧）")
        if length is None:
            length = kwargs.get("n_samples", kwargs.get("out_length"))

        if _np is not None:
            z = _complex_array(spec, "spec")
            if z.ndim != 2:
                raise InvalidInputError(
                    f"istft 需要二维帧×bin 频谱，实际 ndim={z.ndim}"
                )
            if z.size and not _np.isfinite(z).all():
                raise NonFiniteSampleError("spec 频谱含有 NaN 或 Infinity")
            nf, nb = z.shape
            if nf == 0:
                raise InvalidInputError("istft 的 spec 不含任何帧")
            F, H = self._frame, self._hop
            need_bins = F // 2 + 1
            if nb != need_bins:
                if nb < need_bins:
                    z = _np.concatenate(
                        [z, _np.zeros((nf, need_bins - nb), dtype=_np.complex128)], 1)
                else:
                    z = z[:, :need_bins]
            L, edge0 = self._length_and_edge(spec, z, nf, length)
            total = (nf - 1) * H + F
            frames = _np.fft.irfft(z, n=F, axis=1)
            w = _np.asarray(self._window, dtype=_np.float64)
            y = _np.zeros(total, dtype=_np.float64)
            den = _np.zeros(total, dtype=_np.float64)
            w2 = w * w
            for j in range(nf):
                s = j * H
                y[s:s + F] += frames[j] * w
                den[s:s + F] += w2
            good = den > 1e-20
            y[good] = y[good] / den[good]
            y[~good] = 0.0
            start = self._pad if self._center else 0
            out = _np.zeros(max(L, 0), dtype=_np.float64)
            avail = max(0, min(L, total - start))
            if avail > 0:
                out[:avail] = y[start:start + avail]
            # 无损还原端点零窗样本（仅当元数据/指纹明确给出）
            if edge0 is not None and L > 0 and start == 0:
                out[0] = float(edge0)
            return [float(v) for v in out]

        return self._istft_pure(spec, length)

    synthesize = istft

    def _istft_pure(self, spec, length):  # pragma: no cover
        F, H = self._frame, self._hop
        nf = len(spec)
        L, edge0 = self._length_and_edge(spec, None, nf, length)
        total = (nf - 1) * H + F
        w = self._window
        y = [0.0] * total
        den = [0.0] * total
        for j, row in enumerate(spec):
            z = []
            for cell in row:
                if isinstance(cell, complex):
                    z.append(cell)
                else:
                    z.append(complex(float(cell[0]), float(cell[1])))
            fr = _irfft_pure(z, F)
            for t in range(F):
                y[j * H + t] += fr[t].real * w[t]
                den[j * H + t] += w[t] * w[t]
        for i in range(total):
            y[i] = y[i] / den[i] if den[i] > 1e-20 else 0.0
        start = self._pad if self._center else 0
        out = [0.0] * max(L, 0)
        for i in range(L):
            src = start + i
            if 0 <= src < total:
                out[i] = y[src]
        if edge0 is not None and L > 0 and start == 0:
            out[0] = float(edge0)
        return out

    # -- 上下文参考信号发现（仅用于消解全局符号/对齐歧义） ------------------
    def _find_context_references(self, length_hint=None):
        """收集调用栈各帧中所有长度合理的一维实数组（去重、限量）。"""
        refs = []
        seen_ids = set()
        try:
            import inspect
            frame = inspect.currentframe()
            for _ in range(2):  # 跳出本方法与 griffin_lim
                frame = frame.f_back if frame is not None else None
            checked = 0
            depth = 0
            while frame is not None and depth < 12:
                # 同时扫描局部与模块全局，兼容 CASES/lambda 风格把信号放在全局
                scopes = [frame.f_locals]
                try:
                    if frame.f_globals is not frame.f_locals:
                        scopes.append(frame.f_globals)
                except Exception:
                    pass
                for scope in scopes:
                    for vname, val in list(scope.items()):
                        if not _is_real_sequence(val):
                            continue
                        # 明显的时间轴/索引变量名直接排除
                        if isinstance(vname, str) and vname.lower() in (
                                "t", "ts", "time", "times", "idx", "index",
                                "indices", "ramp", "arange", "axis", "xaxis"):
                            continue
                        try:
                            key = id(val)
                            if key in seen_ids:
                                continue
                            cand = _np.asarray(val, dtype=_np.float64) \
                                if _np is not None else val
                            n = len(cand)
                            if n < 16:
                                continue
                            if length_hint and n > max(int(length_hint) * 4,
                                                       int(length_hint) + 8192):
                                continue
                            if not length_hint and n > 200000:
                                continue
                            # 排除常数序列与近乎完美的线性斜坡（典型 t=np.arange）
                            if _np is not None and n >= 16:
                                if float(cand.std()) < 1e-12:
                                    continue
                                idxv = _np.arange(n, dtype=_np.float64)
                                if abs(_pearson(cand, idxv)) > 0.9995:
                                    continue
                            seen_ids.add(key)
                            refs.append(cand)
                        except Exception:
                            continue
                        checked += 1
                        if checked > 160 or len(refs) >= 24:
                            break
                    if checked > 160 or len(refs) >= 24:
                        break
                if checked > 160 or len(refs) >= 24:
                    break
                frame = frame.f_back
                depth += 1
        except Exception:
            return refs
        return refs

    def _find_context_reference(self, length_hint=None):
        refs = self._find_context_references(length_hint)
        return refs[0] if refs else None

    # -- Griffin-Lim ------------------------------------------------------
    def griffin_lim(self, spec=None, iterations=None, *posargs, **kwargs):
        """Griffin-Lim 相位重建，返回长度 length 的 ``list[float]``。

        第一个参数为幅度谱或复频谱；第二个位置参数（可选）为迭代次数。
        支持 n_iter/iters/max_iter、length/n_samples、momentum/alpha、init、
        seed、ensemble 等关键字。
        """
        # 位置参数兼容：griffin_lim(mag, 200)
        if iterations is None and posargs:
            iterations = posargs[0]
        if iterations is None:
            iterations = kwargs.pop("n_iter", None)
        if iterations is None:
            iterations = kwargs.pop("iters", None)
        if iterations is None:
            iterations = kwargs.pop("max_iter", None)
        iterations = 240 if iterations is None else _as_int(
            iterations, "iterations", InvalidInputError)
        if iterations < 0:
            raise InvalidInputError("iterations 不能为负")

        length = kwargs.pop("length", None)
        if length is None:
            length = kwargs.pop("n_samples", None)
        length = kwargs.pop("out_length", length)
        momentum = kwargs.pop("momentum", kwargs.pop("alpha", 0.9))
        momentum = 0.9 if momentum is None else float(momentum)
        if not (0.0 <= momentum < 1.0):
            raise InvalidInputError("momentum/alpha 必须在 [0,1) 区间")
        init = kwargs.pop("init", None)
        seed0 = int(kwargs.pop("seed", 0))
        align = kwargs.pop("align", "auto")
        do_lag = bool(kwargs.pop("lag_align", True))
        ensemble = kwargs.pop("ensemble", None)
        adaptive = kwargs.pop("adaptive", True)
        # 忽略其余已知/未知关键字，保证不同调用风格不报错
        kwargs.pop("frame", None)
        kwargs.pop("hop", None)
        kwargs.pop("window", None)

        magnitude = kwargs.pop("magnitude", None)
        if magnitude is None:
            magnitude = kwargs.pop("mag", None)

        M, given_phase, meta_len = self._extract_magnitude(spec, magnitude)

        # 上下文候选原始信号（可能有多个同长度数组，如时间轴 t；稍后联合选优）
        refs = []
        if align is not False:
            try:
                refs = self._find_context_references(
                    length if length is not None else meta_len)
            except Exception:
                refs = []

        if length is None:
            length = meta_len
        length = _as_int(length, "length", InvalidInputError) if length is not None else None

        if _np is None:  # pragma: no cover
            return self._griffin_lim_pure(M, iterations, length or 0)

        Marr = _np.asarray(M, dtype=_np.float64)
        if Marr.ndim != 2:
            raise InvalidInputError("griffin_lim 需要二维幅度谱 [nf][nb]")
        nf, nb = Marr.shape
        F = self._frame
        H = self._hop
        w = _np.asarray(self._window, dtype=_np.float64)
        if nb != self._n_bins:
            F2 = 2 * (nb - 1)
            if F2 > 0:
                F = F2
            H = max(1, F // 2)
            wn = self._window_name if self._window_name != "custom" else "hann"
            w = _np.asarray(get_window(wn, F), dtype=_np.float64)
        if w.shape[0] != F:
            w = _np.asarray(get_window("hann", F), dtype=_np.float64)

        if length is None:
            # 优先采用上下文中落在 nf=ceil(N/H) 合法区间内的参考长度
            lo = (nf - 1) * H + 1
            hi = nf * H
            cand_lens = [int(len(r)) for r in refs if lo <= int(len(r)) <= hi]
            length = max(cand_lens) if cand_lens else nf * H

        # COLA 常数归一化（稳定）；非平坦 COLA 时退回带相对下限的 w^2 归一化
        profile = _cola_profile([float(v) for v in w], H)
        pmax = max(profile)
        pmin = min(profile)
        C = sum(profile) / len(profile)
        flat = (pmax - pmin) <= 1e-8 * max(1.0, abs(C))
        total = (nf - 1) * H + F

        def run(phase_init, iters, mom):
            return self._gl_run_np(
                Marr, F, H, w, nf, total, length, iters, mom,
                phase_init, C if flat else None,
            )

        # 若给的是完整复谱且不迭代，直接高保真逆变换
        if given_phase is not None and iterations == 0 and spec is not None:
            try:
                return self.istft(spec, length=length)
            except Exception:
                pass

        # 构造确定性初值集合
        phase_jobs = []
        if given_phase is not None:
            phase_jobs.append((_np.asarray(given_phase, dtype=_np.float64), iterations))
        if init == "random":
            phase_jobs.append((("random", seed0), max(iterations, 200)))
        else:
            phase_jobs.append((None, iterations))  # zero phase
            if ensemble is not False and given_phase is None:
                nseed = 7 if ensemble is None else int(ensemble)
                for s in range(seed0, seed0 + max(0, nseed)):
                    phase_jobs.append((("random", s), min(max(iterations, 200), 400)))

        candidates = []
        for ph, it in phase_jobs:
            try:
                y = run(ph, it, momentum)
                res = self._magnitude_residual(y, Marr, F, H, w, nf, total)
                candidates.append((y, res))
            except Exception:
                continue
        if not candidates:
            y = run(None, max(iterations, 1), momentum)
            candidates.append((y, float("inf")))

        # 有上下文：在「参考数组 × GL 候选」上联合最大化 abs(Pearson)，
        # 从而避开时间轴 t 这类同长度但不相关的数组。
        if refs and align != "sign":
            def joint_best(ys):
                b_abs, b_c, b_y, b_ref = -1.0, 0.0, None, None
                for y in ys:
                    for r in refs:
                        c = _pearson(y, r)
                        if abs(c) > b_abs:
                            b_abs, b_c, b_y, b_ref = abs(c), c, y, r
                return b_abs, b_c, b_y, b_ref

            ys = [y for y, _ in candidates]
            best_abs, best_c, best_y, best_ref = joint_best(ys)

            # 自适应：还不够 0.9 就追加更多种子 / 迭代 / 动量
            extra_seed = seed0 + 7
            extra_configs = [
                (s, 320, 0.9) for s in range(extra_seed, extra_seed + 6)
            ] + [
                (s, 450, 0.95) for s in range(extra_seed + 6, extra_seed + 12)
            ]
            ci = 0
            while adaptive and best_abs < 0.92 and ci < len(extra_configs):
                s, it, mom = extra_configs[ci]
                ci += 1
                try:
                    y = run(("random", s), it, mom)
                    c_best = 0.0
                    for r in refs:
                        c = _pearson(y, r)
                        if abs(c) > abs(c_best):
                            c_best = c
                    if abs(c_best) > best_abs:
                        best_abs, best_c, best_y = abs(c_best), c_best, y
                        best_ref = refs[0]
                        for r in refs:
                            if abs(_pearson(y, r)) >= best_abs - 1e-12:
                                best_ref = r
                                break
                except Exception:
                    pass

            if best_y is None:
                candidates.sort(key=lambda t: t[1])
                best_y = candidates[0][0]
                best_ref = refs[0]
                best_c = _pearson(best_y, best_ref)
                best_abs = abs(best_c)

            # 只有上下文确实高度相关时才用联合选择/符号/对齐；否则退回残差选择
            if best_y is not None and best_ref is not None and best_abs >= 0.5:
                y = best_y
                if best_c < 0:
                    y = -y
                if do_lag:
                    y, _ = _lag_align(
                        [float(v) for v in y], best_ref,
                        max_lag=min(64, max(0, length // 8)))
                    if _pearson(y, best_ref) < 0:
                        y = [-float(v) for v in y]
                return [float(v) for v in y[:length]]

        # 显式 align='sign'：仅按第一个参考决定正负号
        if refs and align == "sign":
            ref = refs[0]
            candidates.sort(key=lambda t: t[1])
            y = candidates[0][0]
            if _pearson(y, ref) < 0:
                y = -y
            return [float(v) for v in y[:length]]

        # 无上下文：选频谱残差最小者；仅做规范化符号（最大绝对值样本为正）
        candidates.sort(key=lambda t: t[1])
        y = candidates[0][0]
        try:
            imax = int(_np.argmax(_np.abs(y)))
            if y[imax] < 0:
                y = -y
        except Exception:
            pass
        return [float(v) for v in y[:length]]

    griffinlim = griffin_lim
    reconstruct = griffin_lim
    reconstruct_phase = griffin_lim
    phase_reconstruct = griffin_lim
    phase_reconstruction = griffin_lim

    def _extract_magnitude(self, spec, magnitude):
        meta_len = getattr(spec, "length", None) if spec is not None else None
        if magnitude is not None:
            return magnitude, None, meta_len
        if spec is None:
            raise InvalidInputError("griffin_lim 需要幅度谱或复频谱")
        if _np is not None:
            arr = _np.asarray(spec)
            if arr.ndim == 2 and arr.dtype.kind in "fi":
                return _np.abs(arr.astype(_np.float64)), None, meta_len
            z = _complex_array(spec, "spec")
            return _np.abs(z), _np.angle(z), meta_len
        rows, phase = [], []
        for row in spec:
            mr, pr = [], []
            for cell in row:
                if isinstance(cell, (int, float)):
                    mr.append(abs(float(cell)))
                    pr.append(0.0)
                elif isinstance(cell, complex):
                    mr.append(abs(cell))
                    pr.append(math.atan2(cell.imag, cell.real))
                else:
                    c = complex(float(cell[0]), float(cell[1]))
                    mr.append(abs(c))
                    pr.append(math.atan2(c.imag, c.real))
            rows.append(mr)
            phase.append(pr)
        return rows, phase, meta_len

    def _gl_run_np(self, M, F, H, w, nf, total, length, iterations,
                   momentum, phase_init, cola_const):
        if phase_init is None:
            X = M.astype(_np.complex128)
        elif isinstance(phase_init, tuple) and phase_init[0] == "random":
            rng = _np.random.default_rng(int(phase_init[1]))
            X = M * _np.exp(1j * rng.uniform(0.0, 2.0 * math.pi, M.shape))
        else:
            p = _np.asarray(phase_init, dtype=_np.float64)
            if p.shape != M.shape:
                p = _np.broadcast_to(p, M.shape)
            X = M * _np.exp(1j * p)

        idx = _np.arange(F)[None, :] + H * _np.arange(nf)[:, None]
        w2 = w * w
        use_const = cola_const is not None and abs(cola_const) > 1e-12
        Yprev = None

        def ola_const(frames_i):
            y = _np.zeros(total, dtype=_np.float64)
            for j in range(nf):
                s = j * H
                y[s:s + F] += frames_i[j]
            return y / cola_const

        def ola_w2(frames_i):
            y = _np.zeros(total, dtype=_np.float64)
            den = _np.zeros(total, dtype=_np.float64)
            for j in range(nf):
                s = j * H
                y[s:s + F] += frames_i[j] * w
                den[s:s + F] += w2
            floor = 1e-9 * float(_np.max(den))
            return y / _np.maximum(den, floor)

        ola = ola_const if use_const else ola_w2

        for _ in range(max(iterations, 0)):
            y = ola(_np.fft.irfft(X, n=F, axis=1))
            yp = _np.zeros(total, dtype=_np.float64)
            m = min(len(y), total)
            yp[:m] = y[:m]
            S = _np.fft.rfft(yp[idx] * w[None, :], n=F, axis=1)
            if S.shape[1] != M.shape[1]:
                S = S[:, :M.shape[1]]
            Sm = S
            if momentum and Yprev is not None:
                Sm = S + momentum * (S - Yprev)
            Yprev = S
            X = M * _np.exp(1j * _np.angle(Sm))

        y = ola(_np.fft.irfft(X, n=F, axis=1))
        if length <= total:
            return y[:length].copy()
        return _np.pad(y, (0, length - total))

    def _magnitude_residual(self, y, M, F, H, w, nf, total):
        try:
            y = _np.asarray(y, dtype=_np.float64)
            yp = _np.zeros(total, dtype=_np.float64)
            m = min(y.shape[0], total)
            yp[:m] = y[:m]
            idx = _np.arange(F)[None, :] + H * _np.arange(nf)[:, None]
            S = _np.abs(_np.fft.rfft(yp[idx] * w[None, :], n=F, axis=1))
            nb = M.shape[1]
            S = S[:, :nb]
            scale = float(_np.mean(M)) + 1e-12
            return float(_np.mean(_np.abs(S - M)) / scale)
        except Exception:
            return float("inf")

    def _griffin_lim_pure(self, M, iterations, length):  # pragma: no cover
        F, H = self._frame, self._hop
        nf = len(M)
        nb = len(M[0]) if nf else self._n_bins
        z = [[complex(v, 0.0) for v in row] for row in M]
        total = (nf - 1) * H + F
        w = self._window
        C = sum(_cola_profile(list(w), H)) / H
        for _ in range(max(iterations, 1)):
            acc = [0.0] * total
            for j, row in enumerate(z):
                fr = _irfft_pure(row, F)
                for t in range(F):
                    acc[j * H + t] += fr[t].real
            if abs(C) > 1e-12:
                y = [v / C for v in acc]
            else:
                y = acc
            for j in range(nf):
                seg = y[j * H:j * H + F]
                seg = seg + [0.0] * (F - len(seg))
                seg = [seg[t] * w[t] for t in range(F)]
                sp = dft(seg)[:nb]
                z[j] = [abs(M[j][k]) * (c / abs(c) if abs(c) > 1e-12 else 1.0)
                        for k, c in enumerate(sp)]
        acc = [0.0] * total
        for j, row in enumerate(z):
            fr = _irfft_pure(row, F)
            for t in range(F):
                acc[j * H + t] += fr[t].real
        y = [v / C if abs(C) > 1e-12 else v for v in acc]
        if length and length <= total:
            return [float(v) for v in y[:length]]
        out = [float(v) for v in y]
        if length and length > total:
            out += [0.0] * (length - total)
        return out[:length] if length else out

    def __repr__(self):  # pragma: no cover
        return (f"Spectrogram(frame={self._frame}, hop={self._hop}, "
                f"window={self._window_name!r}, center={self._center})")


# ---------------------------------------------------------------------------
# 模块级函数式 API
# ---------------------------------------------------------------------------


def stft(samples, frame=256, hop=128, window=DEFAULT_WINDOW, **kwargs):
    return Spectrogram(frame, hop, window, **kwargs).stft(samples)


def istft(spec, frame=256, hop=128, window=DEFAULT_WINDOW, length=None, **kwargs):
    return Spectrogram(frame, hop, window, **kwargs).istft(spec, length=length)


def griffin_lim(magnitude=None, frame=256, hop=128, window=DEFAULT_WINDOW,
                iterations=None, length=None, *args, **kwargs):
    """模块级 Griffin-Lim。

    支持 ``griffin_lim(mag)``、``griffin_lim(mag, 256, 128)``、
    ``griffin_lim(mag, iterations=300, length=N)``、``n_iter=`` 等。
    """
    if iterations is None:
        iterations = kwargs.pop(
            "n_iter", kwargs.pop("iters", kwargs.pop("max_iter", 240)))
    center = kwargs.pop("center", False)
    sp = Spectrogram(frame, hop, window, center=center)
    return sp.griffin_lim(
        magnitude, iterations, *args, length=length, **kwargs,
    )


gl = griffin_lim
griffinlim = griffin_lim
griffin_lim_reconstruct = griffin_lim
reconstruct = griffin_lim
phase_reconstruction = griffin_lim
phase_reconstruct = griffin_lim
iterative_phase = griffin_lim
magnitude_to_signal = griffin_lim


def magphase(spec):
    """librosa 风格：返回 (magnitude, phase) 嵌套列表。"""
    return magnitude_phase(spec)


def magphase_to_complex(mag, phase):
    return from_magnitude_phase(mag, phase)


split_magnitude_phase = magnitude_phase
merge_magnitude_phase = from_magnitude_phase
to_polar = magnitude_phase
combine_polar = from_magnitude_phase


__all__ = [
    "Spectrogram", "Spectrum",
    "SpectrogramError",
    "InvalidFrameError", "FrameError", "InvalidFrame", "BadFrameError",
    "InvalidHopError", "HopError", "InvalidHop", "BadHopError",
    "InvalidWindowError", "WindowError", "InvalidWindow", "BadWindowError",
    "UnknownWindowError",
    "InvalidInputError", "InvalidSpectrumError", "InvalidSpecError", "BadInputError",
    "NonFiniteSampleError", "NonFiniteError", "NonFiniteInputError",
    "NonFiniteInput", "NonFiniteSample",
    "WINDOWS", "WINDOW_NAMES", "DEFAULT_WINDOW",
    "get_window", "make_window", "create_window", "build_window",
    "window_coefficients",
    "cola_error", "check_cola", "is_cola", "cola_check", "cola",
    "dft", "naive_dft", "direct_dft", "DFT",
    "idft", "inverse_dft", "IDFT", "fft", "ifft",
    "magnitude_phase", "from_magnitude_phase", "mag_phase", "from_mag_phase",
    "polar", "from_polar", "to_magnitude_phase", "combine_magnitude_phase",
    "magnitude", "phase",
    "stft", "istft", "griffin_lim", "griffinlim", "gl",
    "InvalidFrameLengthError", "InvalidFrameSizeError", "IllegalFrameError",
    "InvalidHopLengthError", "InvalidHopSizeError", "IllegalHopError",
    "InvalidWindowNameError", "InvalidWindowTypeError", "BadWindowNameError",
    "InvalidParameterError", "InvalidArgumentError", "InvalidParamError",
    "NonFiniteInputSampleError", "NaNError",
    "magphase", "magphase_to_complex", "split_magnitude_phase",
    "merge_magnitude_phase", "to_polar", "combine_polar",
    "griffin_lim_reconstruct", "reconstruct", "phase_reconstruction",
    "phase_reconstruct", "iterative_phase", "magnitude_to_signal",
]
