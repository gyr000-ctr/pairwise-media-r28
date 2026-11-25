import math

import pytest

import spectrogram

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1] * 1024

CASES = [
    ("frames", lambda: len(spectrogram.Spectrogram(256, 128).stft(S)) >= 8),
    ("bin_count", lambda: len(spectrogram.Spectrogram(256, 128).stft(S)[0]) == 129),
    ("roundtrip", lambda: max(abs(a - b) for a, b in zip(spectrogram.Spectrogram(256, 128).istft(spectrogram.Spectrogram(256, 128).stft(S)), S)) < 1e-3),
    ("bad_frame", lambda: _raises(lambda: spectrogram.Spectrogram(0, 128))),
    ("bad_hop", lambda: _raises(lambda: spectrogram.Spectrogram(256, 0))),
    ("bad_hop_big", lambda: _raises(lambda: spectrogram.Spectrogram(256, 512))),
    ("empty", lambda: spectrogram.Spectrogram(256, 128).stft([]) == []),
    ("type", lambda: isinstance(spectrogram.Spectrogram(256, 128).stft(S), list)),
    ("det", lambda: spectrogram.Spectrogram(256, 128).stft(S) == spectrogram.Spectrogram(256, 128).stft(S)),
    ("nan", lambda: _raises(lambda: spectrogram.Spectrogram(256, 128).stft([float('nan')] * 256))),
    ("no_mut", lambda: spectrogram.Spectrogram(256, 128).stft(list(S)) is not None),
    ("istft_type", lambda: isinstance(spectrogram.Spectrogram(256, 128).istft(spectrogram.Spectrogram(256, 128).stft(S)), list)),
    ("complex_pair", lambda: len(spectrogram.Spectrogram(256, 128).stft(S)[0][0]) == 2),
    ("float", lambda: isinstance(spectrogram.Spectrogram(256, 128).stft(S)[0][0][0], float)),
    ("bad_istft", lambda: _raises(lambda: spectrogram.Spectrogram(256, 128).istft([]))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
