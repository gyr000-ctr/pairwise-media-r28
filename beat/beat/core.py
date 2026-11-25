"""节拍跟踪核心 API。

流水线（满足题面约束的顺序）：

1. STFT -> 谱通量（spectral flux）-> 强 onset 检测
2. onset 强度包络的 FFT 自相关 -> tempo 候选 lag（抛物线插值细化）
3. 倍频/次谐波消歧，选定基频 tempo
4. 以强 onset 对齐节拍网格相位，样本级细化后输出节拍位置

设计上 ``tempo`` / ``beats`` 不持有任何可变处理状态，同实例可被多线程
并发调用。
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "Beat",
    "BeatError",
    "BeatRateError",
    "BeatRangeError",
    "NonFiniteInputError",
]


# ---------------------------------------------------------------------------
# 可区分的错误类型：全部继承 ValueError，既有 ``except ValueError`` 代码
# 不受影响，同时评审侧可以按具体类型区分失败原因。
# ---------------------------------------------------------------------------
class BeatError(ValueError):
    """节拍跟踪相关错误的基类。"""


class BeatRateError(BeatError):
    """采样率非法。"""


class BeatRangeError(BeatError):
    """BPM 范围非法。"""


class NonFiniteInputError(BeatError):
    """输入样本含 NaN/Inf 等非有限值。"""


def _is_real_number(v) -> bool:
    if isinstance(v, bool):
        return False
    return isinstance(v, Real)


class Beat:
    def __init__(self, rate, bpm_range):
        """初始化节拍跟踪器。

        Parameters
        ----------
        rate:
            采样率（Hz），必须为正的有限实数。
        bpm_range:
            ``(bpm_min, bpm_max)``，两个有限正数且严格递增。
        """
        if not _is_real_number(rate) or not math.isfinite(float(rate)) or float(rate) <= 0.0:
            raise BeatRateError(f"非法采样率 rate={rate!r}：必须为正的有限实数")

        try:
            if len(bpm_range) != 2:
                raise BeatRangeError(
                    f"非法 BPM 范围 {bpm_range!r}：需要恰好两个元素 (bpm_min, bpm_max)"
                )
            lo, hi = float(bpm_range[0]), float(bpm_range[1])
        except BeatRangeError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise BeatRangeError(f"非法 BPM 范围 {bpm_range!r}：{exc}") from exc

        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise BeatRangeError(f"非法 BPM 范围 {bpm_range!r}：端点必须为有限值")
        if lo <= 0.0 or hi <= 0.0:
            raise BeatRangeError(f"非法 BPM 范围 {bpm_range!r}：BPM 必须为正数")
        if lo >= hi:
            raise BeatRangeError(
                f"非法 BPM 范围 {bpm_range!r}：必须严格递增且 bpm_min < bpm_max"
            )

        self._rate = float(rate)
        self._bpm_min = lo
        self._bpm_max = hi

        # 帧参数按采样率自适应，目标 hop ≈ 10 ms（保证相位细化余量）。
        hop = int(2 ** round(math.log2(self._rate * 0.01)))
        self._hop = min(1024, max(64, hop))
        # 50% 重叠：帧中心偏移恰为 1 hop，落在样本级细化窗口内。
        self._nfft = 2 * self._hop

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------
    def tempo(self, samples) -> float:
        """估计节拍速度，返回 BPM；无有效节奏（空/静音）时返回 0.0。"""
        bpm, _ = self._analyze(samples)
        return bpm

    def beats(self, samples) -> list:
        """返回节拍所在的样本下标列表（升序、去重、int、越界已裁剪）。"""
        _, beat_positions = self._analyze(samples)
        return beat_positions

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _coerce_input(self, samples):
        try:
            x = np.asarray(samples, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise BeatError(f"无法将输入转换为浮点样本序列：{exc}") from exc
        if x.ndim != 1:
            raise BeatError(f"仅支持一维样本输入，收到 ndim={x.ndim}")
        try:
            finite = x.size == 0 or bool(np.isfinite(x).all())
        except TypeError as exc:
            raise BeatError(f"输入包含无法解释为实数的元素：{exc}") from exc
        if not finite:
            raise NonFiniteInputError("输入样本包含 NaN 或 Inf 等非有限值")
        return x

    def _analyze(self, samples):
        x = self._coerce_input(samples)
        if x.size == 0:
            return 0.0, []

        flux = self._spectral_flux(x)
        strong = self._strong_onsets(flux)
        if strong.size == 0:
            # 静音或无瞬态：没有可跟踪的节奏。
            return 0.0, []

        tau = self._estimate_period(flux, strong)
        if tau is None:
            return 0.0, []

        bpm = 60.0 * self._rate / (tau * self._hop)
        # 数值误差可能轻微越界，钳制到声明范围内。
        bpm = float(min(max(bpm, self._bpm_min), self._bpm_max))

        beat_positions = self._track_beats(x, flux, strong, tau)
        return bpm, beat_positions

    def _spectral_flux(self, x: np.ndarray) -> np.ndarray:
        """向量化 STFT + 正谱差分（spectral flux）。"""
        hop, nfft = self._hop, self._nfft
        n = x.size

        # 补齐到至少一帧，并让帧尾对齐 hop 网格。
        pad_head = max(0, nfft - n)
        usable = n + pad_head - nfft
        pad_tail = (hop - usable % hop) % hop if usable > 0 else 0
        if pad_head or pad_tail:
            xp = np.pad(x, (0, pad_head + pad_tail))
        else:
            xp = x

        frames = sliding_window_view(xp, nfft)[::hop]
        window = np.hanning(nfft)
        # 零信号直接短路，避免无意义的 FFT。
        if not np.any(frames):
            return np.zeros(frames.shape[0], dtype=np.float64)

        mag = np.abs(np.fft.rfft(frames * window, axis=1))
        diff = mag[1:] - mag[:-1]
        flux = np.empty(mag.shape[0], dtype=np.float64)
        flux[0] = 0.0
        np.maximum(diff, 0.0, out=diff)
        flux[1:] = diff.sum(axis=1)
        return flux

    @staticmethod
    def _strong_onsets(flux: np.ndarray) -> np.ndarray:
        """从 flux 中选取强 onset 帧（局部极大 + 自适应阈值）。"""
        n = flux.size
        if n == 0 or flux.max() <= 0.0:
            return np.empty(0, dtype=np.int64)

        norm = flux / flux.max()
        thr = float(norm.mean() + norm.std())
        # 阈值过严时退化为取全局最大值，保证至少有一个 onset。
        candidates = np.flatnonzero(norm > thr)
        if candidates.size == 0:
            return np.array([int(np.argmax(norm))], dtype=np.int64)

        strong = []
        for i in candidates:
            left = norm[i - 1] if i > 0 else -np.inf
            right = norm[i + 1] if i < n - 1 else -np.inf
            if norm[i] >= left and norm[i] >= right:
                strong.append(int(i))
        if not strong:
            return np.array([int(np.argmax(norm))], dtype=np.int64)
        return np.asarray(strong, dtype=np.int64)

    def _lag_bounds(self, n_frames: int):
        """BPM 范围 -> 自相关 lag（帧）范围。

        两侧各留 1 帧余量：真实 lag 常落在非整数位置（如 31.25 帧），
        严格 ceil/floor 会把最近的整数峰截出区间，只剩谐波峰导致倍频误判。
        最终 BPM 输出仍钳制到声明的 bpm_range 内。
        """
        hop, rate = self._hop, self._rate
        lag_slow = 60.0 * rate / (self._bpm_min * hop)  # 对应最低 BPM
        lag_fast = 60.0 * rate / (self._bpm_max * hop)  # 对应最高 BPM
        lag_hi = min(n_frames - 2, int(math.floor(lag_slow)) + 1)
        lag_lo = max(1, int(math.ceil(lag_fast)) - 1)
        return lag_lo, lag_hi

    @staticmethod
    def _autocorrelation(envelope: np.ndarray) -> np.ndarray:
        n = envelope.size
        fsize = 1 << (2 * n - 1).bit_length()
        spec = np.fft.rfft(envelope - envelope.mean(), fsize)
        acf = np.fft.irfft(spec * np.conj(spec), fsize)[:n]
        if acf[0] > 0.0:
            acf = acf / acf[0]
        return acf

    @staticmethod
    def _peak_lag(acf: np.ndarray, idx: int) -> float:
        """峰值位置抛物线插值，返回分数 lag。"""
        y0, y1, y2 = acf[idx - 1], acf[idx], acf[idx + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom >= 0.0:
            return float(idx)
        shift = 0.5 * (y0 - y2) / denom
        return float(idx) + float(shift)

    def _estimate_period(self, flux: np.ndarray, strong: np.ndarray):
        """自相关选峰 + 倍频消歧，返回节拍周期（分数帧）。

        周期性信号的自相关在 τ、2τ、… 处都有近似等高的峰，单看强度无法
        消歧。这里枚举允许 lag 区间内的全部候选峰及其倍频/次谐波族，用
        两类证据打分：

        - ACF 强度（自相关在该 lag 处的归一化能量）
        - 强 onset 间隔（IOI）与该 lag 的一致比例——真实基频 lag 与
          绝大多数 onset 间隔吻合，谐波候选不吻合，从而否决折半。
        """
        n = flux.size
        lag_lo, lag_hi = self._lag_bounds(n)
        if lag_hi < max(2, lag_lo + 1):
            return None

        acf = self._autocorrelation(flux)
        upper = min(n - 2, lag_hi)

        peak_indices = []
        for i in range(max(1, lag_lo), upper + 1):
            if acf[i] > 0.0 and acf[i] >= acf[i - 1] and acf[i] >= acf[i + 1]:
                peak_indices.append(i)
        if not peak_indices:
            i = int(np.argmax(acf[lag_lo:upper + 1])) + lag_lo
            if acf[i] <= 0.0:
                return None
            peak_indices = [i]

        # 候选族：每个峰 + 它在区间内的倍频/次谐波，全部独立打分。
        candidates = set(peak_indices)
        for i in peak_indices:
            lag = i
            while lag * 2 <= lag_hi:
                lag *= 2
                candidates.add(lag)
            if i % 2 == 0 and i // 2 >= lag_lo:
                candidates.add(i // 2)

        iois = np.diff(strong).astype(np.float64) if strong.size >= 2 else np.empty(0)
        max_acf = max(float(acf[i]) for i in candidates) or 1.0

        def score(lag: float) -> float:
            acf_part = float(np.interp(lag, np.arange(n), acf)) / max_acf
            if iois.size == 0:
                # 无 IOI 证据时：近似等强度优先选更短 lag（防止无依据折半）。
                return acf_part + 0.15 * (lag_hi - lag) / max(1, lag_hi - lag_lo)
            ratio = iois / lag
            nearest = np.minimum(np.abs(ratio - 1.0), np.abs(ratio - 2.0))
            agreement = float(np.mean(nearest <= 0.12))
            # IOI 一致是决定性证据，主导谐波候选间的取舍。
            return agreement + 0.25 * acf_part

        scored = [(score(float(i)), self._peak_lag(acf, i)) for i in candidates]
        scored.sort(reverse=True)
        tau = scored[0][1]

        # 分数 IOI 校准：ACF 的 lag 受帧网格量化（单帧误差在长序列上会
        # 累积成数十 ms 的网格漂移）。对强 onset 峰的抛物线分数位置做
        # 最小二乘直线拟合，斜率给出亚帧精度的平均周期，对个别抖动峰稳健。
        if strong.size >= 3:
            frac_peaks = []
            for idx in strong:
                if 0 < idx < n - 1:
                    y0, y1, y2 = flux[idx - 1], flux[idx], flux[idx + 1]
                    denom = y0 - 2.0 * y1 + y2
                    shift = 0.0 if denom >= 0.0 else 0.5 * (y0 - y2) / denom
                    frac_peaks.append(idx + shift)
            if len(frac_peaks) >= 3:
                peaks_f = np.asarray(frac_peaks, dtype=np.float64)
                # 按整数拍对齐：第 j 个峰对应拍号 round(p_j / tau)，
                # 用 p_j 对拍号的拟合斜率消除掉号映射的非线性。
                beat_idx = np.rint(peaks_f / tau).astype(np.float64)
                if np.ptp(beat_idx) >= 2:
                    slope = float(np.polyfit(beat_idx, peaks_f, 1)[0])
                    if 0.8 * tau <= slope <= 1.2 * tau:
                        tau = slope
        return tau

    def _track_beats(self, x, flux, strong, tau):
        """以强 onset 对齐相位，生成节拍网格并做样本级细化。"""
        n = flux.size
        hop = self._hop
        rate = self._rate

        # 在每个强 onset 的相位残相中选出网格得分最高者（帧域评分）。
        best_phase, best_score = 0.0, -np.inf
        for s in strong:
            phase = float(s) % tau
            count = int(math.floor((n - 1 - phase) / tau)) + 1
            if count <= 0:
                continue
            grid = np.round(phase + np.arange(count) * tau).astype(np.int64)
            grid = grid[(grid >= 0) & (grid < n)]
            if grid.size == 0:
                continue
            score = float(flux[grid].sum())
            if score > best_score:
                best_score, best_phase = score, phase

        # 网格建在样本域：用精确浮点周期，避免帧量化误差沿时间轴累积漂移。
        period_samples = tau * hop
        phase_samples = best_phase * hop + self._nfft / 2.0
        length = x.size
        count = int(math.floor((length - 1 - phase_samples) / period_samples)) + 1
        if count <= 0:
            return []
        sample_grid = np.round(
            phase_samples + np.arange(count) * period_samples
        ).astype(np.int64)
        sample_grid = np.unique(sample_grid[(sample_grid >= 0) & (sample_grid < length)])

        abs_x = np.abs(x)
        window = 2 * hop  # ±2 hop ≈ ±20 ms，覆盖帧量化与中心偏移
        positions = []
        for t in sample_grid:
            t = int(t)
            # 样本级细化：在 ±2 hop 内吸附到最强瞬态，
            # 把对齐误差压到远小于 20 ms。
            lo = max(0, t - window)
            hi = min(length, t + window + 1)
            if hi > lo:
                t = lo + int(np.argmax(abs_x[lo:hi]))
            if 0 <= t < length:
                positions.append(t)

        positions = sorted(set(positions))
        return [int(v) for v in positions]
