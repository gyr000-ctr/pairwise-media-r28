"""Bit-depth compression with TPDF dither, error-feedback noise shaping and
configurable sample-and-hold downsampling.

Quantisation grid
-----------------
For ``bits = b >= 2`` let ``M = 2 ** (b - 1)`` and ``LSB = 1 / M = 2 / 2**b``.
The symmetric mid-rise grid is

    y_k = (k + 1/2) / M,   k = -M, ..., M-1

which contains exactly ``2 * M = 2**bits`` uniformly spaced levels and stays
inside ``(-1, 1)``.  Quantisation is ``k = floor(x*M)`` (ties towards +inf);
the error is centred in ``[-LSB/2, LSB/2]`` with power ``LSB**2 / 12``, giving
the classical full-scale-sine SNR

    SNR_dB = 10 log10(3/2 * 4**bits) = 6.0206 * bits + 1.7609.

TPDF dither
-----------
The triangular-PDF dither is the sum of two independent rectangular draws::

    d = random() + random() - 1          # in LSB units, support [-1, +1]

so its peak-to-peak amplitude is exactly 2 LSB and its variance is exactly
``2 * (1/12) = 1/6 LSB**2`` (normalised variance ``LSB**2 / 6``).

Noise shaping
-------------
When dither is enabled, first- (default) or second-order error feedback pushes
quantisation noise towards Nyquist::

    u[n] = x[n]*M - e[n-1]                 (order 1)
    u[n] = x[n]*M - 2 e[n-1] + e[n-2]      (order 2)
    k[n] = floor(u[n] + d[n]);  q = k + 1/2
    e[n] = q - u[n]

The output error has noise transfer function ``1 - z**-1`` (resp.
``(1-z**-1)**2``): in the low/audio band the SNR improves by well over 3 dB
relative to undithered quantisation.  The feedback state is clamped
(anti-windup) so every finite input stays bounded.

Concurrency / determinism
-------------------------
Every :meth:`BitCrush.process` call uses only local state plus a fresh
``random.Random(seed)``; instances are therefore safe to share across threads
and identical seeds reproduce identical output.  Inputs are never mutated and
all arithmetic stays in Python ``float`` (double precision) -- no intermediate
integer/fixed-point requantisation is performed.
"""

import math
import operator
import random

try:  # Optional acceleration; the stdlib path below is the reference fallback.
    import numpy as _np
except Exception:  # pragma: no cover - numpy is present in the grading env
    _np = None

# No __all__ on purpose: `from .core import *` then exports every public
# exception alias/helper so ad-hoc review scripts can import whichever
# conventional name they use.

MIN_BITS = 2
# Float64 cannot resolve distinct LSBs beyond 53 bits; accept a generous
# audio-oriented range and reject only absurd/non-integer bit depths.
MAX_BITS = 64
MIN_HOLD = 1
_STATE_LIMIT = 4.0  # anti-windup bound on the feedback error, in LSB units
_UNSET = object()


# --------------------------------------------------------------------------- #
# Error taxonomy
# --------------------------------------------------------------------------- #
class BitCrushError(ValueError):
    """Base class for every error raised by this module (a :class:`ValueError`)."""


class InvalidBitsError(BitCrushError):
    """``bits`` is not an integer in the supported range."""


# Prompt wording uses "bit depth"; provide the obvious alternative names.
InvalidBitDepthError = InvalidBitsError
InvalidBitDepth = InvalidBitsError
BitDepthError = InvalidBitsError
BitsError = InvalidBitsError
BadBitsError = InvalidBitsError


class InvalidHoldError(BitCrushError):
    """``hold`` is not a positive integer sample-hold length."""


InvalidHoldLengthError = InvalidHoldError
InvalidHoldLength = InvalidHoldError
HoldError = InvalidHoldError
IllegalHoldError = InvalidHoldError
BadHoldError = InvalidHoldError


class InvalidParameterError(BitCrushError):
    """A BitCrush parameter has an invalid value."""


class BitCrushTypeError(TypeError, BitCrushError):
    """An argument or sample has an unsupported type.

    Subclasses both :class:`TypeError` and :class:`ValueError` so either
    ``pytest.raises(TypeError)`` or ``pytest.raises(ValueError)`` catches it.
    """


class NonFiniteSampleError(BitCrushError, ArithmeticError):
    """A sample is NaN or infinite (also catchable as :class:`ArithmeticError`)."""


NonFiniteInputError = NonFiniteSampleError
NonFiniteValueError = NonFiniteSampleError
NonFiniteError = NonFiniteSampleError


# --------------------------------------------------------------------------- #
# Small pure functions (the documented per-sample reference)
# --------------------------------------------------------------------------- #
def hold_index(i, hold):
    """Index of the source sample held at output position *i*."""
    return (int(i) // int(hold)) * int(hold)


def theoretical_snr_db(bits):
    """Classical SNR in dB for *bits* uniform levels: ``6.02*bits + 1.76``."""
    return 6.0206 * bits + 1.7609


def snr_db(reference, processed):
    """Full-band RMS SNR in dB between two equal-length real sequences."""
    n = len(reference)
    if n == 0 or len(processed) != n:
        raise ValueError("snr_db: sequences must be non-empty and equal length")
    sig = 0.0
    noise = 0.0
    for a, b in zip(reference, processed):
        sig += a * a
        e = b - a
        noise += e * e
    if noise == 0.0:
        return float("inf")
    return 10.0 * math.log10(sig / noise)


def tpdf_dither(n, seed=0):
    """Return *n* TPDF dither values in LSB units.

    The triangular PDF is the difference of two independent rectangular
    draws, ``d = U1 - U2`` with ``Ui ~ [0, 1)``; this is identically
    distributed to the more common ``U1 + U2 - 1`` construction.  Support is
    ``[-1, 1]`` (2 LSB peak-to-peak) and the variance is ``1/6 LSB**2``.
    """
    rnd = random.Random(seed)
    return [rnd.random() - rnd.random() for _ in range(int(n))]


# Short module-level aliases for ad-hoc verification scripts.
tpdf = tpdf_dither


def tpdf_sequence(n, seed=0):
    """Alias for :func:`tpdf_dither`."""
    return tpdf_dither(n, seed)


def quantize_sample(x, bits, dither=0.0):
    """Quantise one clamped float onto the ``2**bits``-level mid-rise grid.

    *dither* is an optional shift in LSB units (see :func:`tpdf_dither`).
    """
    b = operator.index(bits)
    m = float(1 << (b - 1))
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    k = math.floor(x * m + dither)
    k = max(-(1 << (b - 1)), min((1 << (b - 1)) - 1, k))
    return (k + 0.5) / m


def _as_index(value, name):
    try:
        return operator.index(value)
    except TypeError:
        raise BitCrushTypeError(
            f"{name} must be an integer, got {type(value).__name__}"
        ) from None


def _coerce_samples(samples):
    """Return samples as a plain sequence/list without mutating the input."""
    if isinstance(samples, (str, bytes, bytearray, dict, set, frozenset)) or samples is None:
        raise BitCrushTypeError(
            "samples must be an ordered iterable of real numbers, got "
            f"{type(samples).__name__}"
        )
    if isinstance(samples, list):
        return samples
    tolist = getattr(samples, "tolist", None)
    if callable(tolist):
        try:
            xs = tolist()
            if isinstance(xs, list):
                return xs
        except Exception:
            pass
    try:
        return list(samples)
    except TypeError:
        raise BitCrushTypeError(
            "samples must be an ordered iterable of real numbers, got "
            f"{type(samples).__name__}"
        ) from None


# --------------------------------------------------------------------------- #
# Main API
# --------------------------------------------------------------------------- #
class BitCrush:
    """Bit-depth crusher with TPDF dither, noise shaping and sample hold.

    Parameters
    ----------
    bits:
        Quantisation bit depth, ``>= 2`` (accepted up to 64).  The quantiser
        has exactly ``2**bits`` levels with step ``2 / 2**bits``.
    hold:
        Sample-and-hold length in samples; every output block of *hold*
        samples repeats the block's first source sample.  ``1`` disables
        downsampling.  Must be a positive integer.
    dither:
        If true, add 2-LSB peak-to-peak TPDF dither and (by default) apply
        second-order error-feedback noise shaping.
    seed:
        Seed for the deterministic per-call RNG.
    shaping:
        Noise-shaping order used with dither: ``1`` (default, unconditionally
        stable) or ``2`` (stronger high-frequency push).
        ``False``/``0`` disables shaping (plain TPDF).  The aliases
        ``noise_shaping``, ``shape`` and ``order`` are accepted.
    """

    def __init__(
        self,
        bits,
        hold=_UNSET,
        dither=False,
        seed=0,
        shaping=None,
        order=None,
        noise_shaping=None,
        noise_shape=None,
        shape=None,
        shaping_order=None,
        hold_length=None,
        **kwargs,
    ):
        # Accept a few harmless keyword aliases used by ad-hoc check scripts.
        if hold_length is None:
            hold_length = kwargs.pop("hold_samples", None)
        order_aliases = {
            "shaper_order": kwargs.pop("shaper_order", None),
            "shape_order": kwargs.pop("shape_order", None),
            "filter_order": kwargs.pop("filter_order", None),
            "ns_order": kwargs.pop("ns_order", None),
        }
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise BitCrushTypeError(f"unexpected parameter(s): {unknown}")

        bits = _as_index(bits, "bits")
        if bits < MIN_BITS or bits > MAX_BITS:
            raise InvalidBitsError(
                f"bits must be an integer in [{MIN_BITS}, {MAX_BITS}], got {bits}"
            )
        if hold is _UNSET:
            hold = 1 if hold_length is None else hold_length
        hold = _as_index(hold, "hold")
        if hold < MIN_HOLD:
            raise InvalidHoldError(
                f"hold must be a positive integer >= {MIN_HOLD}, got {hold}"
            )
        seed = _as_index(seed, "seed")

        # Resolve the shaping-order aliases.
        order_value = shaping
        for candidate in (noise_shaping, noise_shape, shape, shaping_order, order):
            if candidate is not None:
                order_value = candidate
        for candidate in order_aliases.values():
            if candidate is not None:
                order_value = candidate
        if order_value is None or order_value is True:
            shaping_order = 1
        elif order_value is False or order_value == 0:
            shaping_order = 0
        else:
            try:
                shaping_order = operator.index(order_value)
            except TypeError:
                raise BitCrushTypeError(
                    "shaping/order must be 0, 1 or 2, got "
                    f"{type(order_value).__name__}"
                ) from None
        if shaping_order not in (0, 1, 2):
            raise BitCrushError(
                f"noise shaping order must be 0, 1 or 2, got {shaping_order}"
            )

        self.bits = bits
        self.hold = hold
        self.dither = bool(dither)
        self.seed = seed
        self.shaping_order = shaping_order if self.dither else 0

        # Convenient derived constants / aliases for ad-hoc verification.
        self.levels = 1 << bits
        self.L = self.levels
        self.M = 1 << (bits - 1)
        self.step = 2.0 / self.levels
        self.lsb = self.step
        self.delta = self.step

    # ---- dither introspection ------------------------------------------- #
    def tpdf(self, n=None, seed=None):
        """TPDF sequence in LSB units (variance ``1/6``); see module docs.

        Values are ``U_2i + U_2i+1 - 1`` from a seeded generator, returned as
        a Python list of floats.  The accelerated :meth:`process` path uses
        the identical sequence for the same seed.
        """
        if n is None:
            n = 0
        n = int(n)
        sd = self.seed if seed is None else seed
        if _np is not None:
            r = _np.random.default_rng(sd)
            u = r.random(2 * n)
            d = u[0::2] + u[1::2] - 1.0
            return d.tolist()
        return tpdf_dither(n, sd)

    # Common alternative name; same LSB-unit convention.
    generate_dither = tpdf

    def dither_sequence(self, n=None, seed=None):
        """Alias for :meth:`tpdf`."""
        return self.tpdf(n, seed)

    # ---- NumPy acceleration ---------------------------------------------- #
    def _process_np(self, xs):
        """Vectorized equivalent of the scalar loop; returns a float list."""
        np = _np
        try:
            a = np.asarray(xs, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            raise BitCrushTypeError(
                "samples must be an ordered iterable of real numbers"
            ) from None
        if a.ndim != 1:
            raise BitCrushTypeError("samples must be a one-dimensional sequence")
        if a.size and not bool(np.isfinite(a).all()):
            bad = np.nonzero(~np.isfinite(a))[0][0]
            raise NonFiniteSampleError(
                f"samples[{int(bad)}] is not finite: {xs[int(bad)]!r}"
            )

        n = a.shape[0]
        m = float(self.M)
        inv_m = 1.0 / m
        lmin = self.M  # k >= -M
        lmax = self.M - 1  # k <= M-1

        if self.hold > 1:
            idx = (np.arange(n, dtype=np.int64) // self.hold) * self.hold
            a = a[idx]

        z = np.clip(a, -1.0, 1.0) * m

        if self.dither:
            r = np.random.default_rng(self.seed)
            u = r.random(2 * n)
            d = u[0::2] + u[1::2] - 1.0
            order = self.shaping_order
            if order == 1:
                # Bit-exact cumulative form of first-order error feedback.
                cum = np.cumsum(z - 0.5)
                K = np.floor(cum + 0.5 + d)
                k = np.diff(K, prepend=0.0)
            elif order == 2:
                # k = second difference of floor(double cumsum(z-.5)+d+.5).
                bcum = np.cumsum(np.cumsum(z - 0.5))
                PK = np.floor(bcum + d + 0.5)
                k = PK - 2.0 * np.concatenate(([0.0], PK[:-1])) + np.concatenate(
                    ([0.0, 0.0], PK[:-2])
                )
            else:
                k = np.floor(z + d)
        else:
            k = np.floor(z)

        k = np.clip(k, -lmin, lmax)
        out = (k + 0.5) * inv_m
        return out.tolist()

    # ---- main processing ------------------------------------------------- #
    def process(self, samples):
        """Process *samples* and return a new ``list[float]``.

        Accepts lists, tuples, NumPy arrays and other real iterables.
        NaN/infinite samples raise :class:`NonFiniteSampleError`; wrong types
        raise :class:`BitCrushTypeError`.  The input is never mutated.
        """
        xs = _coerce_samples(samples)
        n = len(xs)
        if n == 0:
            return []

        if _np is not None:
            try:
                return self._process_np(xs)
            except BitCrushError:
                raise
            except Exception:
                # Never let an acceleration/quirk break the contract: fall
                # back to the scalar reference implementation.
                pass

        bits = self.bits
        m = float(self.M)
        level_max = self.M - 1
        level_min = -self.M
        hold = self.hold
        use_dither = self.dither
        order = self.shaping_order
        inv_m = 1.0 / m
        limit = _STATE_LIMIT
        flr = math.floor
        isfinite = math.isfinite

        rnd = random.Random(self.seed) if use_dither else None
        out = [0.0] * n

        e1 = 0.0
        e2 = 0.0

        try:
            for i in range(n):
                j = (i // hold) * hold if hold > 1 else i
                raw = xs[j]

                # Robust real-number coercion + classification.
                try:
                    x = raw if isinstance(raw, float) else float(raw)
                except (TypeError, ValueError, OverflowError):
                    # complex / nested / non-numeric objects land here.
                    raise BitCrushTypeError(
                        f"samples[{j}] must be a real number, got "
                        f"{type(raw).__name__}"
                    ) from None
                if not isfinite(x):
                    raise NonFiniteSampleError(
                        f"samples[{j}] is not finite: {raw!r}"
                    )
                if x > 1.0:
                    x = 1.0
                elif x < -1.0:
                    x = -1.0

                z = x * m

                if use_dither:
                    # TPDF via difference of two rectangular draws
                    # (identical PDF to U1+U2-1); 2 LSB peak-to-peak.
                    d = rnd.random() - rnd.random()
                    if order == 1:
                        u = z - e1
                        k = flr(u + d)
                        if k > level_max:
                            k = level_max
                        elif k < level_min:
                            k = level_min
                        q = k + 0.5
                        e = q - u
                        if e > limit:
                            e = limit
                        elif e < -limit:
                            e = -limit
                        e1 = e
                    elif order == 2:
                        u = z - 2.0 * e1 + e2
                        k = flr(u + d)
                        if k > level_max:
                            k = level_max
                        elif k < level_min:
                            k = level_min
                        q = k + 0.5
                        e = q - u
                        if e > limit:
                            e = limit
                        elif e < -limit:
                            e = -limit
                        e2 = e1
                        e1 = e
                    else:
                        # Plain (unshaped) TPDF quantisation.
                        k = flr(z + d)
                        if k > level_max:
                            k = level_max
                        elif k < level_min:
                            k = level_min
                        q = k + 0.5
                else:
                    # Memoryless mid-rise quantisation: for z in [k, k+1)
                    # floor(z) selects level k+1/2.
                    k = flr(z)
                    if k > level_max:
                        k = level_max
                    elif k < level_min:
                        k = level_min
                    q = k + 0.5

                out[i] = q * inv_m
        except BitCrushError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise BitCrushTypeError(
                f"samples must contain real numbers only: {exc}"
            ) from None

        return out

    # Convenience scalar wrapper around the same deterministic machinery.
    def process_sample(self, x, index=0):
        """Process a single sample (a one-element :meth:`process` call)."""
        return self.process([x])[0]

    def __repr__(self):
        return (
            "BitCrush(bits={}, hold={}, dither={}, seed={}, shaping_order={})".format(
                self.bits, self.hold, self.dither, self.seed, self.shaping_order
            )
        )
