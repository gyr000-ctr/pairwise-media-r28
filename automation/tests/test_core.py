import math

import pytest

import automation

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

P = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]

CASES = [
    ("endpoints", lambda: abs(automation.Automation(P).value_at(0.0)) < 1e-9),
    ("mid", lambda: 0.0 <= automation.Automation(P).value_at(1.0) <= 1.0),
    ("clamp_low", lambda: automation.Automation(P).value_at(-5.0) == 0.0),
    ("clamp_high", lambda: automation.Automation(P).value_at(99.0) == 0.0),
    ("bad_points", lambda: _raises(lambda: automation.Automation([(1.0, 0.0), (0.0, 1.0)]))),
    ("too_few", lambda: _raises(lambda: automation.Automation([(0.0, 0.0)]))),
    ("slope", lambda: isinstance(automation.Automation(P).slope_at(0.5), float)),
    ("type", lambda: isinstance(automation.Automation(P).value_at(0.5), float)),
    ("det", lambda: automation.Automation(P).value_at(0.5) == automation.Automation(P).value_at(0.5)),
    ("nan", lambda: _raises(lambda: automation.Automation(P).value_at(float('nan')))),
    ("no_mut", lambda: automation.Automation(list(P)).value_at(0.5) is not None),
    ("continuity", lambda: abs(automation.Automation(P).value_at(0.9999) - automation.Automation(P).value_at(1.0)) < 1e-3),
    ("slope_type", lambda: isinstance(automation.Automation(P).slope_at(1.0), float)),
    ("flat", lambda: automation.Automation([(0.0, 0.5), (1.0, 0.5)]).value_at(0.5) == 0.5),
    ("three_nodes", lambda: isinstance(automation.Automation(P).value_at(1.5), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
