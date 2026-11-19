"""Rate of change and fundamental of samples over whole cycles."""

import math

import numpy as np
import pytest

from polecraft.studies._cycle import derivative, fundamental


def test_derivative_of_a_sum_of_harmonics_is_exact() -> None:
    period = 0.3
    x = period * np.arange(16) / 16
    w = 2 * math.pi / period
    values = 0.5 + 2.0 * np.cos(w * x - 0.4) + 0.3 * np.sin(3 * w * x) - 0.1 * np.cos(7 * w * x)
    slope = -2.0 * w * np.sin(w * x - 0.4) + 0.9 * w * np.cos(3 * w * x) + 0.7 * w * np.sin(7 * w * x)
    assert np.allclose(derivative(values, period), slope, rtol=0, atol=1e-12 * w)


def test_derivative_gives_zero_for_the_harmonic_that_cannot_be_told_apart() -> None:
    # At exactly half the sampling rate, cos samples as +-1 and sin as 0: the
    # slope there is unknown.
    values = np.cos(math.pi * np.arange(8))
    assert np.allclose(derivative(values, 1.0), 0.0, atol=1e-12)


def test_derivative_keeps_the_highest_harmonic_of_an_odd_count() -> None:
    # Nine samples hold harmonics up to the fourth exactly.
    x = np.arange(9) / 9
    values = np.sin(2 * math.pi * 4 * x) + 0.5 * np.cos(2 * math.pi * x)
    slope = 8 * math.pi * np.cos(2 * math.pi * 4 * x) - math.pi * np.sin(2 * math.pi * x)
    assert np.allclose(derivative(values, 1.0), slope, rtol=0, atol=1e-12)


def test_fundamental_is_the_peak_of_the_first_harmonic_of_the_cycles() -> None:
    x = 2 * math.pi * np.arange(36) / 12  # three cycles
    values = 0.2 + 1.5 * np.cos(x - 1.0) + 0.4 * np.sin(2 * x) + 0.3 * np.cos(x / 3)
    assert fundamental(values, cycles=3) == pytest.approx(1.5, rel=1e-12)
    # Read as one cycle, the window-long part is the fundamental instead.
    assert fundamental(values) == pytest.approx(0.3, rel=1e-12)
