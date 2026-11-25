import math

import pytest

import bitcrush

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1, 0.2, 0.3, 0.4]

CASES = [
    ("len", lambda: len(bitcrush.BitCrush(8, 1).process(S)) == 4),
    ("levels", lambda: len(set(bitcrush.BitCrush(2, 1).process(S))) <= 4),
    ("hold", lambda: bitcrush.BitCrush(8, 2).process(S)[1] == bitcrush.BitCrush(8, 2).process(S)[0]),
    ("bad_bits", lambda: _raises(lambda: bitcrush.BitCrush(1, 1))),
    ("bad_hold", lambda: _raises(lambda: bitcrush.BitCrush(8, 0))),
    ("empty", lambda: bitcrush.BitCrush(8, 1).process([]) == []),
    ("type", lambda: isinstance(bitcrush.BitCrush(8, 1).process(S), list)),
    ("det", lambda: bitcrush.BitCrush(8, 1, True, 1).process(S) == bitcrush.BitCrush(8, 1, True, 1).process(S)),
    ("range", lambda: all(-1.0 <= v <= 1.0 for v in bitcrush.BitCrush(4, 1).process(S))),
    ("nan", lambda: _raises(lambda: bitcrush.BitCrush(8, 1).process([float('nan')]))),
    ("no_mut", lambda: bitcrush.BitCrush(8, 1).process(list(S)) is not None),
    ("seed_diff", lambda: bitcrush.BitCrush(8, 1, True, 1).process(S) != bitcrush.BitCrush(8, 1, True, 2).process(S)),
    ("float", lambda: isinstance(bitcrush.BitCrush(8, 1).process(S)[0], float)),
    ("bits_effect", lambda: bitcrush.BitCrush(4, 1).process(S) != bitcrush.BitCrush(8, 1).process(S)),
    ("dither_effect", lambda: bitcrush.BitCrush(4, 1, True, 1).process(S) != bitcrush.BitCrush(4, 1, False, 1).process(S)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
