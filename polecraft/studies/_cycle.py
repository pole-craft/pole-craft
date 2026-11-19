"""Samples taken evenly over whole cycles of a periodic quantity."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def derivative(values: ArrayLike, period: float) -> NDArray[np.float64]:
    """Rate of change of samples taken evenly over one period of a periodic function.

    Each harmonic n of the samples is turned into its derivative, i n (2 pi /
    period) times itself, which is exact for every harmonic below half the
    sampling rate. With an even number of samples, a harmonic at exactly half
    the sampling rate has no known slope and comes out as zero.
    """
    values = np.asarray(values, dtype=float)
    count = len(values)
    waves = np.fft.rfft(values)
    factor = 1j * np.arange(len(waves)) * (2.0 * math.pi / period)
    return np.fft.irfft(waves * factor, count)


def fundamental(values: ArrayLike, cycles: int = 1) -> float:
    """Peak amplitude of the fundamental of samples over ``cycles`` whole cycles."""
    values = np.asarray(values, dtype=float)
    return float(abs(np.fft.rfft(values)[cycles]) / (len(values) / 2))
