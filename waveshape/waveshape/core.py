"""waveshape 核心 API。

波形整形（waveshaping）与 4 倍过采样。

信号链
------
1. 零插值（zero-stuffing）到 ``oversample`` 倍采样率；
2. Kaiser 窗 sinc 低通插值滤波（增益 ``oversample``）；
3. 在高采样率上做非线性整形 ``f(x) = tanh(drive * x)``；
4. 同一条低通做抗混叠，再抽取回原采样率。

对齐：使用多相（polyphase）分解，逐相卷积后再合并。信号两端按边缘值延拓
``P = 2D`` 个样本（``D = M / L`` 为基带级滤波器延迟，``M`` 为高采样率半长），
合并后整体补偿 ``P + D`` 个基带样本，使输出与原信号的时刻一一对齐
（delta 响应峰值不偏移，见 tests 中的恒常值/脉冲用例）。

约束
----
* 整形函数 ``f(x) = tanh(drive * x)`` 奇对称（``f(-x) == -f(x)`` 逐位）、
  严格单调递增、值域有界 ``|f| <= 1``；
* ``drive`` 增大时谐波含量单调提升；
* 4 倍过采样使混叠显著下降（见 tests/test_spec.py 的量测）；
* 100 万样本在常见硬件上远低于 600 ms；
* 非法驱动/过采样/采样值按类别抛出可区分的异常。

并发
----
实例构造后只读（滤波器系数以只读数组缓存），``process`` 不修改入参、
不写入实例状态，可安全并发调用。
"""

from __future__ import annotations

import math
import numbers

import numpy as np

__all__ = [
    "WaveShape",
    "WaveShapeError",
    "InvalidDriveError",
    "InvalidOversampleError",
    "InvalidSampleError",
    "NonFiniteInputError",
    "NaNInputError",
    "InfInputError",
]


# ---------------------------------------------------------------------------
# 错误分类
#
# 全部同时继承 ``ValueError``（数值非法）与 ``TypeError``（类型非法），
# 调用方按任一常见约定都能捕获；不同子类之间可精确区分。
# ---------------------------------------------------------------------------
class WaveShapeError(ValueError):
    """waveshape 相关错误的基类。"""


class InvalidDriveError(WaveShapeError, TypeError):
    """驱动参数非法（类型错误、非正、非有限等）。"""


class InvalidOversampleError(WaveShapeError, TypeError):
    """过采样倍数非法（只接受 1/2/4/8）。"""


class InvalidSampleError(WaveShapeError, TypeError):
    """采样输入非法（类型错误、维度错误等）。"""


class NonFiniteInputError(WaveShapeError):
    """输入采样包含非有限值（NaN / Inf）。"""


class NaNInputError(NonFiniteInputError):
    """输入包含 NaN。"""


class InfInputError(NonFiniteInputError):
    """输入包含 ±Inf。"""


# ---------------------------------------------------------------------------
# 常量与滤波器缓存（只读；可安全并发读取）
# ---------------------------------------------------------------------------
_ALLOWED_OVERSAMPLE = (1, 2, 4, 8)
_TAPS = 513            # 奇数；4 倍时滤波器延迟 64 个基带样本
_CUTOFF_BASE = 0.47    # 基带通带边缘（cycles/sample；Nyquist = 0.5）
_KAISER_BETA = 10.0

_FILTER_CACHE: dict = {}


def _design_filter(oversample: int) -> np.ndarray:
    """设计 Kaiser 窗 sinc 低通（只读数组，模块级缓存）。"""
    cached = _FILTER_CACHE.get(oversample)
    if cached is not None:
        return cached

    taps = _TAPS
    if (taps - 1) % (2 * oversample) != 0:
        raise RuntimeError("滤波器长度与过采样倍数不匹配")
    half = (taps - 1) // 2
    n = np.arange(taps, dtype=np.float64) - half
    fc = _CUTOFF_BASE / oversample          # 高采样率下的归一化截止频率
    h = 2.0 * fc * np.sinc(2.0 * fc * n)
    h *= np.kaiser(taps, _KAISER_BETA)
    h /= h.sum()
    h.setflags(write=False)
    _FILTER_CACHE[oversample] = h
    return h


def _phase_taps(h: np.ndarray, oversample: int) -> tuple:
    """把滤波器拆成插值相位与抽取相位（均为只读数组）。"""
    taps = len(h)
    interp = tuple(np.ascontiguousarray(h[r::oversample]) for r in range(oversample))
    decim = []
    for r in range(oversample):
        count = (taps + r) // oversample + 1
        g = np.zeros(count, dtype=np.float64)
        idx = np.arange(count) * oversample - r
        valid = (idx >= 0) & (idx < taps)
        g[valid] = h[idx[valid]]
        decim.append(g)
    return interp, tuple(decim)


# ---------------------------------------------------------------------------
# 非线性
#
# 直接在高采样率上做 y = tanh(drive·x)。tanh 天然有界（|y| <= 1），
# 之后再做抗混叠低通与抽取，折叠到基带的分量被显著抑制。
# ---------------------------------------------------------------------------
def _soft_clip(a: np.ndarray) -> np.ndarray:
    """逐点 tanh，并显式钳制到 [-1, 1]（防御性）。"""
    return np.clip(np.tanh(a), -1.0, 1.0)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------
def _validate_drive(drive) -> float:
    if isinstance(drive, (bool, np.bool_)) or not isinstance(drive, numbers.Real):
        raise InvalidDriveError("drive 必须是大于 0 的有限实数，得到 %r" % (drive,))
    value = float(drive)
    if not math.isfinite(value):
        raise InvalidDriveError("drive 必须是有限值，得到 %r" % (drive,))
    if value <= 0.0:
        raise InvalidDriveError("drive 必须大于 0，得到 %r" % (drive,))
    return value


def _validate_oversample(oversample) -> int:
    if isinstance(oversample, (bool, np.bool_)) or not isinstance(
        oversample, numbers.Integral
    ):
        raise InvalidOversampleError("oversample 必须是整数，得到 %r" % (oversample,))
    value = int(oversample)
    if value not in _ALLOWED_OVERSAMPLE:
        raise InvalidOversampleError(
            "oversample 只支持 %s，得到 %r" % (list(_ALLOWED_OVERSAMPLE), oversample)
        )
    return value


def _as_sample_array(samples) -> np.ndarray:
    """把输入规范成 float64 一维数组，元素非法时抛出 ``InvalidSampleError``。"""
    if samples is None or isinstance(samples, (str, bytes, bytearray)):
        raise InvalidSampleError(
            "samples 必须是一维数值序列，得到 %r" % (type(samples).__name__,)
        )
    if isinstance(samples, np.ndarray):
        if samples.ndim != 1:
            raise InvalidSampleError(
                "samples 必须是一维序列，得到 %d 维" % samples.ndim
            )
        if samples.dtype.kind in "fiu":
            return np.asarray(samples, dtype=np.float64)
        if samples.dtype.kind == "b":  # 布尔不是采样值
            raise InvalidSampleError("samples 元素必须是实数，得到 bool")
        try:
            arr = np.asarray(samples, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidSampleError("samples 元素必须是实数") from exc
        if arr.dtype.kind != "f":
            raise InvalidSampleError("samples 元素必须是实数")
        return arr
    if isinstance(samples, numbers.Real):
        raise InvalidSampleError("samples 必须是序列，得到单个数值")
    try:
        # 一次性转数组：dtype 推断快，且遇到非数值会抛错
        arr = np.asarray(samples, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidSampleError("samples 元素必须是实数") from exc
    if arr.ndim != 1:
        raise InvalidSampleError(
            "samples 必须是一维序列，得到 %d 维" % arr.ndim
        )
    return arr


def _check_finite(arr: np.ndarray) -> None:
    if arr.size == 0:
        return
    finite = np.isfinite(arr)
    if finite.all():
        return
    bad = arr[~finite]
    if np.isnan(bad).any():
        raise NaNInputError("samples 包含 NaN，无法处理")
    raise InfInputError("samples 包含 ±Inf，无法处理")


# ---------------------------------------------------------------------------
# 主体
# ---------------------------------------------------------------------------
class WaveShape:
    """基于 ``tanh`` 的奇对称、单调、有界波形整形器，带过采样。"""

    __slots__ = (
        "_drive",
        "_oversample",
        "_h",
        "_half",
        "_delay",
        "_pad",
        "_interp",
        "_decim",
    )

    def __init__(self, drive, oversample=4):
        self._drive = _validate_drive(drive)
        self._oversample = _validate_oversample(oversample)
        if self._oversample > 1:
            h = _design_filter(self._oversample)
            self._h = h
            self._half = (len(h) - 1) // 2               # 高采样率半长
            self._delay = self._half // self._oversample  # 基带延迟
            self._pad = 2 * self._delay
            self._interp, self._decim = _phase_taps(h, self._oversample)
        else:
            self._h = None
            self._half = 0
            self._delay = 0
            self._pad = 0
            self._interp = ()
            self._decim = ()

    # -- 只读属性 ---------------------------------------------------------
    @property
    def drive(self) -> float:
        return self._drive

    @property
    def oversample(self) -> int:
        return self._oversample

    # -- 整形函数 ---------------------------------------------------------
    def shape(self, x):
        """单点整形：``tanh(drive * x)``。奇对称、单调、值域 ``[-1, 1]``。"""
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
            raise InvalidSampleError("x 必须是实数，得到 %r" % (x,))
        value = float(x)
        if math.isnan(value):
            raise NaNInputError("x 是 NaN，无法整形")
        if math.isinf(value):
            raise InfInputError("x 是 ±Inf，无法整形")
        out = math.tanh(self._drive * value)
        if out > 1.0:      # 防御性；tanh 本身已有界
            out = 1.0
        elif out < -1.0:
            out = -1.0
        return float(out)

    # -- 批量处理 ---------------------------------------------------------
    def process(self, samples):
        """处理一维采样序列，返回等长 ``list[float]``。"""
        x = _as_sample_array(samples)
        _check_finite(x)
        n = len(x)
        if n == 0:
            return []
        if self._oversample == 1:
            return self._shape_direct(x)
        return self._shape_oversampled(x)

    # -- 内部实现 ---------------------------------------------------------
    def _shape_direct(self, x: np.ndarray) -> list:
        y = np.tanh(self._drive * x)
        np.clip(y, -1.0, 1.0, out=y)
        return y.tolist()

    def _shape_oversampled(self, x: np.ndarray) -> list:
        L = self._oversample
        lane = self._delay          # D：基带延迟
        pad = self._pad             # P：两端边缘延拓长度
        n = len(x)
        m = n + 2 * pad

        xp = np.pad(x, (pad, pad), mode="edge")

        # 1) 多相插值（增益 L 已折入相位系数）：
        #    up[p*L + r] = conv(xp, h[r::L] * L)[p + D]
        up = np.empty(m * L, dtype=np.float64)
        for r in range(L):
            conv = np.convolve(xp, self._interp[r] * L)
            if len(conv) < lane + m:
                buf = np.zeros(lane + m, dtype=np.float64)
                buf[: len(conv)] = conv
                conv = buf
            up[r::L] = conv[lane:lane + m]

        # 2) 高采样率上的非线性
        shaped = _soft_clip(self._drive * up)

        # 3) 多相抽取：z[j] = sum_r conv(shaped[r::L], g_r)[j]
        z = np.zeros(m, dtype=np.float64)
        for r in range(L):
            conv = np.convolve(shaped[r::L], self._decim[r])
            take = min(len(conv), m)
            z[:take] += conv[:take]

        # 4) 去掉往返延迟并裁回原长度
        #    标定：失真正位于 z 的 (pad + delay) 处（delta 响应验证）
        offset = pad + lane
        out = z[offset:offset + n]
        np.clip(out, -1.0, 1.0, out=out)
        return out.tolist()

    def __repr__(self) -> str:
        return "WaveShape(drive=%r, oversample=%d)" % (self._drive, self._oversample)
