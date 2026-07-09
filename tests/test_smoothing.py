from __future__ import annotations

import numpy as np

from clockd.services.pipeline import _smooth_positions
from clockd.utils.units import mph_to_ms


def test_smooth_positions_identity():
    """Window of 1 returns original points."""
    pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float64)
    result = _smooth_positions(pts, window=1)
    np.testing.assert_array_equal(result, pts)


def test_smooth_positions_reduces_noise():
    """Smoothing should reduce jitter in noisy positions."""
    # Straight line with noise
    clean = np.array([[i, 0.0] for i in range(20)])
    noise = clean.copy()
    noise[:, 1] = np.random.default_rng(42).normal(0, 2, 20)

    smoothed = _smooth_positions(noise, window=5)
    # Smoothed y values should have lower variance than raw
    assert np.std(smoothed[:, 1]) < np.std(noise[:, 1])


def test_smooth_positions_short_array():
    """If array is shorter than window, return original."""
    pts = np.array([[0, 0], [1, 1]], dtype=np.float64)
    result = _smooth_positions(pts, window=5)
    np.testing.assert_array_equal(result, pts)


def test_mph_to_ms():
    assert abs(mph_to_ms(60) - 26.8224) < 0.01
    assert abs(mph_to_ms(0)) < 0.001
