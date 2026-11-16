"""Magnetic material models for the field solver.

The solver works with the reluctivity nu = H / B (metres per henry), the
inverse of the permeability, as a function of the flux density magnitude B
(tesla). Air, copper and magnets have a constant reluctivity. Steel follows a
BH curve, and the Newton iteration also needs d(nu)/dB.

A magnet's remanence is a source term for the solver. ``Magnet`` carries it
together with the recoil permeability, which is what the field sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import resources
from typing import Protocol

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

# Permeability of free space in H/m. The classical value, as FEMM uses; the
# measured value differs from it by less than one part in a billion.
MU0 = 4e-7 * math.pi


class Material(Protocol):
    """What the solver needs from a material."""

    @property
    def linear(self) -> bool:
        """True if the reluctivity does not depend on the flux density."""

    def reluctivity(self, b: ArrayLike) -> NDArray[np.float64]:
        """Reluctivity H / B in m/H at flux density magnitudes ``b`` in tesla."""

    def reluctivity_derivative(self, b: ArrayLike) -> NDArray[np.float64]:
        """d(nu)/dB in m/(H T) at flux density magnitudes ``b`` in tesla."""


@dataclass(frozen=True)
class LinearMaterial:
    """A material with constant permeability: air, copper, a magnet's recoil line."""

    relative_permeability: float = 1.0

    def __post_init__(self) -> None:
        mu_r = self.relative_permeability
        if not (math.isfinite(mu_r) and mu_r > 0.0):
            raise ValueError(f"relative permeability must be a positive number, not {mu_r}")

    @property
    def linear(self) -> bool:
        return True

    def reluctivity(self, b: ArrayLike) -> NDArray[np.float64]:
        """1 / (mu_r mu0) in m/H, in the shape of ``b``."""
        return np.full(np.shape(b), 1.0 / (self.relative_permeability * MU0))

    def reluctivity_derivative(self, b: ArrayLike) -> NDArray[np.float64]:
        """Zero, in the shape of ``b``."""
        return np.zeros(np.shape(b))


AIR = LinearMaterial(1.0)
COPPER = LinearMaterial(1.0)


@dataclass(frozen=True)
class Magnet:
    """A permanent magnet on its recoil line: B = mu_r mu0 H + Br."""

    remanence: float  # Br in tesla
    relative_permeability: float = 1.05  # recoil permeability

    def __post_init__(self) -> None:
        if not (math.isfinite(self.remanence) and self.remanence >= 0.0):
            raise ValueError(f"remanence must be zero or more, in tesla, not {self.remanence}")
        LinearMaterial(self.relative_permeability)  # checks the permeability

    @property
    def linear(self) -> bool:
        return True

    def reluctivity(self, b: ArrayLike) -> NDArray[np.float64]:
        """1 / (mu_r mu0) in m/H, in the shape of ``b``."""
        return np.full(np.shape(b), 1.0 / (self.relative_permeability * MU0))

    def reluctivity_derivative(self, b: ArrayLike) -> NDArray[np.float64]:
        """Zero, in the shape of ``b``."""
        return np.zeros(np.shape(b))


@dataclass(frozen=True)
class FrohlichSteel:
    """A saturating material given by a formula, not by measured data.

    B = mu0 H + Js H / (a + H): the polarisation rises towards ``saturation``
    (Js, in tesla) and is half of it at ``half_field`` (a, in A/m). The curve
    starts with permeability mu0 + Js / a and ends with the slope of air. It
    is for tests and placeholders with a known answer, not a steel grade.
    """

    saturation: float  # Js in tesla
    half_field: float  # a in A/m

    def __post_init__(self) -> None:
        for name, value in (("saturation", self.saturation), ("half_field", self.half_field)):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be a positive number, not {value}")

    @property
    def linear(self) -> bool:
        return False

    def flux_density(self, h: ArrayLike) -> NDArray[np.float64]:
        """Flux density B in tesla at field strengths ``h`` in A/m."""
        h = np.asarray(h, dtype=float)
        return MU0 * h + self.saturation * h / (self.half_field + h)

    def _terms(self, b: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        """B as an array, and p and the root of the quadratic for H.

        The law is the quadratic mu0 H^2 + p H - a B = 0 with
        p = mu0 a + Js - B, so H = (root - p) / (2 mu0),
        root = sqrt(p^2 + 4 mu0 a B). While p is positive that difference
        cancels badly, and the equal form H = 2 a B / (p + root) is used.
        """
        b = np.asarray(b, dtype=float)
        p = MU0 * self.half_field + self.saturation - b
        return b, p, np.sqrt(p * p + 4.0 * MU0 * self.half_field * b)

    def field_strength(self, b: ArrayLike) -> NDArray[np.float64]:
        """Field strength H in A/m at flux density magnitudes ``b`` in tesla."""
        b, p, root = self._terms(b)
        return np.where(p > 0.0, 2.0 * self.half_field * b / (p + root), (root - p) / (2.0 * MU0))

    def reluctivity(self, b: ArrayLike) -> NDArray[np.float64]:
        """Reluctivity H / B in m/H at flux density magnitudes ``b`` in tesla."""
        b, p, root = self._terms(b)
        beyond = p <= 0.0  # only above Js, so b is not zero there
        high = np.divide(root - p, 2.0 * MU0 * b, out=np.zeros_like(root), where=beyond)
        return np.where(beyond, high, 2.0 * self.half_field / (p + root))

    def reluctivity_derivative(self, b: ArrayLike) -> NDArray[np.float64]:
        """d(nu)/dB in m/(H T) at flux density magnitudes ``b`` in tesla."""
        b, p, root = self._terms(b)
        a = self.half_field
        beyond = p <= 0.0
        # Below Js: the derivative of 2 a / (p + root), which has no
        # cancellation near B = 0.
        low = 2.0 * a * (root + p - 2.0 * MU0 * a) / (root * (p + root) ** 2)
        # Above Js: (dH/dB - nu) / B, with dB/dH = mu0 + Js a / (a + H)^2.
        h = (root - p) / (2.0 * MU0)
        slope = 1.0 / (MU0 + self.saturation * a / (a + h) ** 2)
        nu = np.divide(h, b, out=np.zeros_like(root), where=beyond)
        high = np.divide(slope - nu, b, out=np.zeros_like(root), where=beyond)
        return np.where(beyond, high, low)


class BHCurve:
    """A soft magnetic material described by a table of B against H.

    The table starts at B = 0, H = 0 and both columns increase. Between the
    table points H(B) is a monotone cubic (PCHIP), so the curve passes
    through every point, never turns back, and has a continuous slope, which
    the Newton iteration needs to converge smoothly. Beyond the last point
    the curve continues as a straight line with the slope it has there.

    That straight line is a stand-in, not physics: real steel ends up with
    the slope of air (1 / mu0). Use a table that reaches past the flux
    densities in the machine.
    """

    def __init__(self, b: ArrayLike, h: ArrayLike) -> None:
        """Build the curve from flux densities ``b`` in tesla and field strengths ``h`` in A/m."""
        self.b = np.array(b, dtype=float)
        self.h = np.array(h, dtype=float)
        _check_table(self.b, self.h)

        width = np.diff(self.b)
        secant = np.diff(self.h) / width
        slope = PchipInterpolator(self.b, self.h)(self.b, 1)
        # PCHIP sets an end slope to zero when the neighbouring intervals
        # disagree strongly. Zero at B = 0 would mean zero reluctivity, and
        # zero at the last point a flat line beyond it. The secant of the end
        # interval is a safe floor that keeps the cubic monotone.
        slope[0] = max(slope[0], secant[0])
        slope[-1] = max(slope[-1], secant[-1])

        # Per interval, H = h0 + t (d + t (c2 + t c3)) with t = B - b0. The
        # last entry is the straight line beyond the table.
        self._d = slope
        self._c2 = np.append((3.0 * secant - 2.0 * slope[:-1] - slope[1:]) / width, 0.0)
        self._c3 = np.append((slope[:-1] + slope[1:] - 2.0 * secant) / width**2, 0.0)

    @property
    def linear(self) -> bool:
        return False

    def _locate(self, b: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """The input as an array, and the interval each value falls in."""
        b = np.asarray(b, dtype=float)
        interval = np.searchsorted(self.b, b, side="right") - 1
        return b, np.clip(interval, 0, self.b.size - 1)

    def _h_and_slope(
        self, b: NDArray[np.float64], i: NDArray[np.intp]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        t = b - self.b[i]
        d, c2, c3 = self._d[i], self._c2[i], self._c3[i]
        return self.h[i] + t * (d + t * (c2 + t * c3)), d + t * (2.0 * c2 + 3.0 * c3 * t)

    def field_strength(self, b: ArrayLike) -> NDArray[np.float64]:
        """Field strength H in A/m at flux density magnitudes ``b`` in tesla."""
        b, i = self._locate(b)
        return self._h_and_slope(b, i)[0]

    def reluctivity(self, b: ArrayLike) -> NDArray[np.float64]:
        """Reluctivity H / B in m/H at flux density magnitudes ``b`` in tesla."""
        b, i = self._locate(b)
        return self._nu(b, i, self._h_and_slope(b, i)[0])

    def reluctivity_derivative(self, b: ArrayLike) -> NDArray[np.float64]:
        """d(nu)/dB in m/(H T) at flux density magnitudes ``b`` in tesla."""
        b, i = self._locate(b)
        h, slope = self._h_and_slope(b, i)
        first = i == 0
        # nu = H / B, so d(nu)/dB = (dH/dB - nu) / B.
        rest = np.divide(slope - self._nu(b, i, h), b, out=np.zeros_like(b), where=~first)
        return np.where(first, self._c2[0] + 2.0 * self._c3[0] * b, rest)

    def _nu(
        self, b: NDArray[np.float64], i: NDArray[np.intp], h: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        first = i == 0
        # In the first interval H has no constant term, so H / B is a
        # polynomial and needs no division: it stays exact down to B = 0.
        rest = np.divide(h, b, out=np.zeros_like(b), where=~first)
        return np.where(first, self._d[0] + b * (self._c2[0] + b * self._c3[0]), rest)


def _check_table(b: NDArray[np.float64], h: NDArray[np.float64]) -> None:
    """Raise ValueError if the table cannot describe a BH curve."""
    if b.ndim != 1 or h.ndim != 1:
        raise ValueError("BH table: B and H must be one-dimensional lists")
    if b.size != h.size:
        raise ValueError(
            f"BH table: B and H must have the same number of values, not {b.size} and {h.size}"
        )
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(h))):
        raise ValueError("BH table: every value must be a finite number")
    if b.size < 2:
        raise ValueError("BH table: at least two rows are needed")
    if b[0] != 0.0 or h[0] != 0.0:
        raise ValueError(f"BH table: must start at B = 0, H = 0, not B = {b[0]:g}, H = {h[0]:g}")
    for name, column in (("flux density", b), ("field strength", h)):
        falls = np.flatnonzero(np.diff(column) <= 0.0)
        if falls.size:
            row = int(falls[0]) + 2
            raise ValueError(
                f"BH table: {name} must increase from row to row; "
                f"row {row} ({column[row - 1]:g}) is not above row {row - 1} ({column[row - 2]:g})"
            )


def material_names() -> list[str]:
    """Names of the materials in the library shipped with the package."""
    folder = resources.files("polecraft.data").joinpath("materials")
    return sorted(
        entry.name[: -len(".yaml")] for entry in folder.iterdir() if entry.name.endswith(".yaml")
    )


def load_material(name: str) -> BHCurve:
    """Load a material from the library by name.

    Only steels given by a BH table are in the library so far. Raises
    ValueError for a name that is not there.
    """
    names = material_names()
    if name not in names:
        raise ValueError(f"unknown material '{name}'; the library has: {', '.join(names)}")
    entry = resources.files("polecraft.data").joinpath("materials", f"{name}.yaml")
    data = yaml.safe_load(entry.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("kind") != "steel":
        raise ValueError(f"material file {name}.yaml: only 'kind: steel' is supported so far")
    try:
        table = np.array(data["bh_curve"], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"material file {name}.yaml: 'bh_curve' must be a list of [B, H] rows"
        ) from None
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError(f"material file {name}.yaml: 'bh_curve' must be a list of [B, H] rows")
    return BHCurve(table[:, 0], table[:, 1])
