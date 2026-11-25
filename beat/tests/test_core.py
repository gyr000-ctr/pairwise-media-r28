import math

import pytest

import beat

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1] * 1000

CASES = [
    ("tempo_range", lambda: beat.Beat(48000, (60, 180)).tempo(S) >= 0.0),
    ("bad_range", lambda: _raises(lambda: beat.Beat(48000, (180, 60)))),
    ("bad_rate", lambda: _raises(lambda: beat.Beat(0, (60, 180)))),
    ("beats_type", lambda: isinstance(beat.Beat(48000, (60, 180)).beats(S), list)),
    ("empty", lambda: beat.Beat(48000, (60, 180)).tempo([]) == 0.0),
    ("type", lambda: isinstance(beat.Beat(48000, (60, 180)).tempo(S), float)),
    ("det", lambda: beat.Beat(48000, (60, 180)).tempo(S) == beat.Beat(48000, (60, 180)).tempo(S)),
    ("nan", lambda: _raises(lambda: beat.Beat(48000, (60, 180)).tempo([float('nan')] * 100))),
    ("no_mut", lambda: beat.Beat(48000, (60, 180)).tempo(list(S)) is not None),
    ("beats_sorted", lambda: all(a <= b for a, b in zip(beat.Beat(48000, (60, 180)).beats(S), beat.Beat(48000, (60, 180)).beats(S)[1:]))),
    ("beats_int", lambda: all(isinstance(v, int) for v in beat.Beat(48000, (60, 180)).beats(S))),
    ("silence", lambda: beat.Beat(48000, (60, 180)).tempo([0.0] * 1000) == 0.0),
    ("float", lambda: isinstance(beat.Beat(48000, (60, 180)).tempo(S), float)),
    ("range_bpm", lambda: 0.0 <= beat.Beat(48000, (60, 180)).tempo(S) <= 180.0),
    ("beats_bounds", lambda: all(0 <= v < 1000 for v in beat.Beat(48000, (60, 180)).beats(S))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
