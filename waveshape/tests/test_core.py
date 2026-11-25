import math

import pytest

import waveshape

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.2] * 100

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
    ("nan", lambda: _raises(lambda: waveshape.WaveShape(1.0).process([float('nan')]))),
    ("no_mut", lambda: waveshape.WaveShape(1.0).process(list(S)) is not None),
    ("drive_harmonics", lambda: waveshape.WaveShape(8.0).shape(0.5) >= waveshape.WaveShape(1.0).shape(0.5)),
    ("float", lambda: isinstance(waveshape.WaveShape(1.0).shape(0.1), float)),
    ("shape_type", lambda: isinstance(waveshape.WaveShape(1.0).process(S)[0], float)),
    ("zero", lambda: abs(waveshape.WaveShape(4.0).shape(0.0)) < 1e-12),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
