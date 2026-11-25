"""automation 核心 API。

分段三次 Hermite 动画曲线：

- 切线模式：``"auto"``（Fritsch–Carlson 单调保形，端点取 0）、
  ``"flat"``（恒 0）、``"linear"``（相邻割线平均，端点取单侧割线）。
- 曲线 C1 连续（节点处左右导数按构造一致，数值误差 ≤1e-6）。
- 越界按端点切线线性外推，与解析式 ``v_end + m_end * (t - t_end)`` 一致。
- ``time_at(value)`` 提供 value→t 反解：曲线单调时牛顿迭代
  （割线/二分兜底），误差 ≤1e-6；非单调曲线抛出 ``ValueError``。
- 构造后全部系数预计算完毕且不再修改，求值无共享可变状态，并发安全。
"""

from bisect import bisect_right

__all__ = ["Automation"]

_TANGENT_MODES = ("auto", "flat", "linear")


def _is_finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and x == x and x not in (float("inf"), float("-inf"))


class Automation:
    """分段三次 Hermite 自动化曲线。

    points:   [(t, v), ...]，时间严格递增，至少 2 个节点，均为有限数。
    tangents: "auto" / "flat" / "linear" 之一（全部节点统一），
              或与节点等长的序列（逐节点指定）。
    """

    def __init__(self, points, tangents="auto"):
        pts = list(points)
        if len(pts) < 2:
            raise ValueError("need at least 2 points")

        ts = []
        vs = []
        for p in pts:
            t, v = p
            if not _is_finite_number(t) or not _is_finite_number(v):
                raise ValueError("point coordinates must be finite numbers")
            ts.append(float(t))
            vs.append(float(v))
        for i in range(len(ts) - 1):
            if not ts[i] < ts[i + 1]:
                raise ValueError("point times must be strictly increasing")

        n = len(ts)
        if isinstance(tangents, str):
            modes = [tangents] * n
        else:
            modes = list(tangents)
            if len(modes) != n:
                raise ValueError("tangents must be a mode string or one mode per point")
        for m in modes:
            if m not in _TANGENT_MODES:
                raise ValueError("unknown tangent mode: %r" % (m,))

        # 节点切线（一阶导数，dv/dt）。
        slopes = [0.0] * n
        secants = [(vs[i + 1] - vs[i]) / (ts[i + 1] - ts[i])
                   for i in range(n - 1)]
        for i in range(n):
            mode = modes[i]
            if mode == "flat":
                slopes[i] = 0.0
            elif mode == "linear":
                if i == 0:
                    slopes[i] = secants[0]
                elif i == n - 1:
                    slopes[i] = secants[-1]
                else:
                    slopes[i] = 0.5 * (secants[i - 1] + secants[i])
            else:  # auto：Fritsch–Carlson 单调保形，端点取 0
                if i == 0 or i == n - 1:
                    slopes[i] = 0.0
                else:
                    s0, s1 = secants[i - 1], secants[i]
                    if s0 == 0.0 or s1 == 0.0 or (s0 < 0.0) != (s1 < 0.0):
                        slopes[i] = 0.0
                    else:
                        h0 = ts[i] - ts[i - 1]
                        h1 = ts[i + 1] - ts[i]
                        w1 = 2.0 * h1 + h0
                        w2 = h1 + 2.0 * h0
                        slopes[i] = (w1 + w2) / (w1 / s0 + w2 / s1)

        # 每段预计算三次多项式系数：f(u) = c0 + c1 u + c2 u^2 + c3 u^3，
        # u = t - ts[i] ∈ [0, h]。
        coeffs = []
        for i in range(n - 1):
            h = ts[i + 1] - ts[i]
            s = secants[i]
            m0, m1 = slopes[i], slopes[i + 1]
            c0 = vs[i]
            c1 = m0
            c2 = (3.0 * s - 2.0 * m0 - m1) / h
            c3 = (m0 + m1 - 2.0 * s) / (h * h)
            coeffs.append((c0, c1, c2, c3))

        self._ts = tuple(ts)
        self._vs = tuple(vs)
        self._slopes = tuple(slopes)
        self._coeffs = tuple(coeffs)

    # ------------------------------------------------------------------ #
    # 求值
    # ------------------------------------------------------------------ #

    def _check_t(self, t):
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")
        t = float(t)
        if t != t or t in (float("inf"), float("-inf")):
            raise ValueError("t must be finite")
        return t

    def value_at(self, t):
        """曲线在 t 处的值；越界按端点切线线性外推。"""
        t = self._check_t(t)
        ts = self._ts
        if t <= ts[0]:
            return self._vs[0] + self._slopes[0] * (t - ts[0])
        if t >= ts[-1]:
            return self._vs[-1] + self._slopes[-1] * (t - ts[-1])
        i = bisect_right(ts, t) - 1
        c0, c1, c2, c3 = self._coeffs[i]
        u = t - ts[i]
        return ((c3 * u + c2) * u + c1) * u + c0

    def slope_at(self, t):
        """曲线在 t 处的一阶导数；越界为端点切线（外推为线性）。"""
        t = self._check_t(t)
        ts = self._ts
        if t <= ts[0]:
            return self._slopes[0]
        if t >= ts[-1]:
            return self._slopes[-1]
        i = bisect_right(ts, t) - 1
        c0, c1, c2, c3 = self._coeffs[i]
        u = t - ts[i]
        return (3.0 * c3 * u + 2.0 * c2) * u + c1

    # ------------------------------------------------------------------ #
    # 反解 value -> t
    # ------------------------------------------------------------------ #

    def _segment_monotonic(self, i):
        """第 i 段导数在 (0, h) 内是否不变号。"""
        c0, c1, c2, c3 = self._coeffs[i]
        h = self._ts[i + 1] - self._ts[i]
        a, b, c = 3.0 * c3, 2.0 * c2, c1  # 导数 a u^2 + b u + c
        eps = 1e-12 * h
        if a == 0.0:
            if b == 0.0:
                return True
            r = -c / b
            return not (eps < r < h - eps)
        disc = b * b - 4.0 * a * c
        if disc <= 0.0:
            return True  # 无实根或重根：不变号
        sq = disc ** 0.5
        r1 = (-b - sq) / (2.0 * a)
        r2 = (-b + sq) / (2.0 * a)
        for r in (r1, r2):
            if eps < r < h - eps:
                return False
        return True

    def _check_invertible(self):
        """曲线整体严格单调（可反解）时返回方向 ±1，否则抛 ValueError。"""
        vs = self._vs
        d = 0.0
        for i in range(len(vs) - 1):
            dv = vs[i + 1] - vs[i]
            if dv == 0.0:
                raise ValueError("curve is not strictly monotonic; inverse undefined")
            s = 1.0 if dv > 0.0 else -1.0
            if d == 0.0:
                d = s
            elif s != d:
                raise ValueError("curve is not monotonic; inverse undefined")
        for i in range(len(self._coeffs)):
            if not self._segment_monotonic(i):
                raise ValueError("curve is not monotonic; inverse undefined")
        return d

    def time_at(self, value, tol=1e-6):
        """反解：返回使 value_at(t) == value 的 t（误差 ≤ tol）。

        曲线非单调（值与 t 不一一对应）时抛出 ValueError。
        value 越出节点值域时按端点切线外推求解；端点切线为 0 无解时
        抛出 ValueError。
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("value must be a real number")
        y = float(value)
        if y != y or y in (float("inf"), float("-inf")):
            raise ValueError("value must be finite")

        d = self._check_invertible()
        ts, vs = self._ts, self._vs
        m0, mn = self._slopes[0], self._slopes[-1]

        lo_v, hi_v = (vs[0], vs[-1]) if d > 0 else (vs[-1], vs[0])
        if y <= lo_v:
            if y == lo_v:
                return ts[0] if d > 0 else ts[-1]
            m = m0 if d > 0 else mn
            if m == 0.0:
                raise ValueError("value out of range and endpoint tangent is zero")
            return (ts[0] if d > 0 else ts[-1]) + (y - lo_v) / m
        if y >= hi_v:
            if y == hi_v:
                return ts[-1] if d > 0 else ts[0]
            m = mn if d > 0 else m0
            if m == 0.0:
                raise ValueError("value out of range and endpoint tangent is zero")
            return (ts[-1] if d > 0 else ts[0]) + (y - hi_v) / m

        # 节点值严格单调：二分定位 y 所在段。
        key = y if d > 0 else -y
        vals = vs if d > 0 else tuple(-v for v in vs)
        i = bisect_right(vals, key) - 1
        if i < 0:
            i = 0
        if i > len(self._coeffs) - 1:
            i = len(self._coeffs) - 1

        a, b = ts[i], ts[i + 1]
        fa = self.value_at(a) - y
        t = 0.5 * (a + b)
        for _ in range(100):
            ft = self.value_at(t) - y
            if abs(ft) <= 1e-12:
                break
            # 牛顿步，越出括号则退化为二分。
            dt = self.slope_at(t)
            if dt != 0.0:
                tn = t - ft / dt
            else:
                tn = t
            if not (a < tn < b):
                tn = 0.5 * (a + b)
            if fa * ft <= 0.0:
                b = t
            else:
                a, fa = t, ft
            if tn == t:
                # 步长无进展：对已收缩的括号取中点继续前进。
                tn = 0.5 * (a + b)
                if tn == t or tn == a or tn == b:
                    break  # 括号已耗尽（达到浮点精度极限）
            t = tn
        # 最终误差校验（|f(t)-y| 与区间宽度双重保障）。
        if abs(self.value_at(t) - y) > tol and abs(b - a) > tol:
            raise ValueError("inverse solve failed to converge")
        return t

    # 常用别名。
    t_at = time_at
    inverse = time_at
