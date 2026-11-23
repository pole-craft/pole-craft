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
from scipy.optimize import nnls

# Permeability of free space in H/m. The classical value, as FEMM uses; the
# measured value differs from it by less than one part in a billion.
MU0 = 4e-7 * math.pi

# Temperature in degrees Celsius at which library data for magnets is given.
REFERENCE_TEMPERATURE = 20.0


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
    """A permanent magnet on its recoil line: B = mu_r mu0 H + Br.

    The numbers are those at ``temperature`` (degrees Celsius, 20 for a
    library entry). The two coefficients, as fractions per kelvin about 20 C
    (-0.0012 is -0.12 %/K), say how the remanence and the coercivity change
    with temperature; ``at_temperature`` applies them. They are None when
    the entry has no temperature data.
    """

    remanence: float  # Br in tesla
    relative_permeability: float = 1.05  # recoil permeability
    remanence_coefficient: float | None = None  # per kelvin
    coercivity_coefficient: float | None = None  # per kelvin
    temperature: float = REFERENCE_TEMPERATURE  # degrees Celsius

    def __post_init__(self) -> None:
        if not (math.isfinite(self.remanence) and self.remanence >= 0.0):
            raise ValueError(f"remanence must be zero or more, in tesla, not {self.remanence}")
        LinearMaterial(self.relative_permeability)  # checks the permeability
        for name in ("remanence_coefficient", "coercivity_coefficient"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, not {value}")
        if not math.isfinite(self.temperature):
            raise ValueError(f"temperature must be a finite number, not {self.temperature}")

    @property
    def coercivity(self) -> float:
        """Normal coercivity Hcb in A/m: where the recoil line crosses B = 0."""
        return self.remanence / (self.relative_permeability * MU0)

    def at_temperature(self, temperature: float) -> Magnet:
        """The magnet at ``temperature`` in degrees Celsius, from the 20 C entry.

        Br and Hcb each change linearly with their coefficient. The recoil
        permeability follows from them, mu_r = Br / (mu0 Hcb), so the recoil
        line stays consistent. The linear law is an approximation that holds
        over the range a datasheet gives, not far beyond it. Raises
        ValueError for a magnet with no temperature data, one that is not at
        20 C, or a temperature at which Br or Hcb would reach zero.
        """
        if self.remanence_coefficient is None or self.coercivity_coefficient is None:
            raise ValueError("this magnet has no temperature coefficients")
        if self.temperature != REFERENCE_TEMPERATURE:
            raise ValueError(
                f"start from the {REFERENCE_TEMPERATURE:g} C magnet; "
                f"this one is already at {self.temperature:g} C"
            )
        if not math.isfinite(temperature):
            raise ValueError(f"temperature must be a finite number, not {temperature}")
        rise = temperature - REFERENCE_TEMPERATURE
        remanence = self.remanence * (1.0 + self.remanence_coefficient * rise)
        coercivity = self.coercivity * (1.0 + self.coercivity_coefficient * rise)
        if remanence <= 0.0 or coercivity <= 0.0:
            raise ValueError(
                f"at {temperature:g} C the linear temperature law gives a remanence or "
                f"coercivity of zero or less; it does not hold that far from 20 C"
            )
        return Magnet(
            remanence=remanence,
            relative_permeability=remanence / (MU0 * coercivity),
            remanence_coefficient=self.remanence_coefficient,
            coercivity_coefficient=self.coercivity_coefficient,
            temperature=temperature,
        )

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
    the Newton iteration needs to converge smoothly.

    Beyond the last point (b_n, h_n) the polarisation J = B - mu0 H levels
    off at ``saturation`` and the slope of the curve tends to that of air:

        H = h_n + u / mu0 - (dJ / mu0) (1 - exp(-u / L)),   u = B - b_n,

    with dJ = saturation - J_n, and L chosen so that the slope at the last
    point is the one the table has there. When no saturation is given the
    tail adds dJ = (b_n - b_n-1) (1 - mu0 s_n), the last table step scaled
    by how far the end slope s_n = dH/dB is from that of air. That is a
    rule, not a measurement: for a table ending at 1.8 T it levels off near
    1.9 T, likely below the real saturation, so beyond the table the steel
    is taken as more saturated than it is. A table whose end is already as
    soft as air continues as a straight line.
    """

    def __init__(self, b: ArrayLike, h: ArrayLike, saturation: float | None = None) -> None:
        """Build the curve from flux densities ``b`` in tesla and field strengths ``h`` in A/m.

        ``saturation`` is the polarisation in tesla the steel tends to
        beyond the table; it must be above the polarisation at the last row.
        """
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
        self._tail(slope[-1], width[-1], saturation)

    def _tail(self, end_slope: float, last_step: float, saturation: float | None) -> None:
        """Set the law beyond the table; see the class docstring."""
        end_polarisation = self.b[-1] - MU0 * self.h[-1]
        softness = 1.0 - MU0 * end_slope  # 0 when the end is as soft as air
        if saturation is not None and not (
            math.isfinite(saturation) and saturation > end_polarisation
        ):
            raise ValueError(
                f"saturation polarisation must be above the polarisation at the last row "
                f"({end_polarisation:.4g} T), not {saturation}"
            )
        if softness <= 1e-9:
            self.saturation: float | None = None
            self._rise = self._length = 0.0
            return
        rise = saturation - end_polarisation if saturation is not None else last_step * softness
        self.saturation = end_polarisation + rise
        self._rise = rise / MU0  # dJ / mu0 in A/m
        self._length = rise / softness  # L in tesla

    def stacked(self, factor: float) -> BHCurve:
        """The curve of a lamination stack with this stacking factor (0 < factor <= 1).

        Across the laminations the steel and the gaps between them carry the
        same H side by side, so B = factor B_steel + (1 - factor) mu0 H, and
        the polarisation, saturation included, scales by the factor.
        """
        if not (math.isfinite(factor) and 0.0 < factor <= 1.0):
            raise ValueError(f"stacking factor must be above 0 and at most 1, not {factor}")
        saturation = None if self.saturation is None else factor * self.saturation
        return BHCurve(factor * self.b + (1.0 - factor) * MU0 * self.h, self.h, saturation)

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
        h = self.h[i] + t * (d + t * (c2 + t * c3))
        slope = d + t * (2.0 * c2 + 3.0 * c3 * t)
        if self._length == 0.0:
            return h, slope
        beyond = i == self.b.size - 1
        u = np.where(beyond, t, 0.0) / self._length
        decay = np.exp(-u)
        tail_h = self.h[-1] + t / MU0 + self._rise * np.expm1(-u)
        tail_slope = 1.0 / MU0 - self._rise / self._length * decay
        return np.where(beyond, tail_h, h), np.where(beyond, tail_slope, slope)

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


_KINDS = ("steel", "magnet", "conductor")
_UNITS = {
    "remanence_T": "tesla",
    "relative_permeability": "relative to mu0",
    "resistivity_ohm_m": "ohm metre at 20 C",
    "temperature_coefficient_per_K": "per kelvin",
    "density_kg_m3": "kilogram per cubic metre",
    "thickness_mm": "millimetre",
    "saturation_polarisation_T": "tesla",
    "loss_step_T": "tesla",
    "remanence_temp_coefficient_per_K": "fraction per kelvin, about 20 C",
    "coercivity_temp_coefficient_per_K": "fraction per kelvin, about 20 C",
}
_COEFFICIENTS = {
    "remanence_temp_coefficient_per_K": "remanence_coefficient",
    "coercivity_temp_coefficient_per_K": "coercivity_coefficient",
}


@dataclass(frozen=True)
class Conductor:
    """A winding conductor: resistivity at 20 C, how it rises with temperature, density."""

    resistivity: float  # ohm metre at 20 C
    temperature_coefficient: float  # per kelvin, linear about 20 C
    density: float  # kg/m^3

    def __post_init__(self) -> None:
        for name in ("resistivity", "temperature_coefficient", "density"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be a positive number, not {value}")

    def resistivity_at(self, temperature: float) -> float:
        """Resistivity in ohm metres at ``temperature`` in degrees Celsius."""
        return self.resistivity * (1.0 + self.temperature_coefficient * (temperature - 20.0))


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


def _read_entry(name: str) -> dict:
    """Read and check the fields every library file has."""
    names = material_names()
    if name not in names:
        raise ValueError(f"unknown material '{name}'; the library has: {', '.join(names)}")
    entry = resources.files("polecraft.data").joinpath("materials", f"{name}.yaml")
    data = yaml.safe_load(entry.read_text(encoding="utf-8"))
    where = f"material file {name}.yaml"
    if not isinstance(data, dict):
        raise ValueError(f"{where}: must be a mapping of fields")
    kind = data.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"{where}: 'kind' must be one of {', '.join(_KINDS)}, not {kind!r}")
    if data.get("name") != name:
        raise ValueError(f"{where}: 'name' must be '{name}', not {data.get('name')!r}")
    for field in ("description", "source"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f"{where}: '{field}' must be a non-empty text")
    return data


def _number(data: dict, field: str, name: str, *, positive: bool = True) -> float:
    """A finite number field, positive unless ``positive`` is False (then zero or more)."""
    value = data.get(field)
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if not ok or (value <= 0.0 if positive else value < 0.0):
        unit = f" ({_UNITS[field]})" if field in _UNITS else ""
        need = "a positive number" if positive else "zero or more"
        raise ValueError(f"material file {name}.yaml: '{field}'{unit} must be {need}, not {value!r}")
    return float(value)


def _coefficient(data: dict, field: str, name: str) -> float:
    """A temperature coefficient: any finite number, normally negative for a magnet."""
    value = data.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(
            f"material file {name}.yaml: '{field}' ({_UNITS[field]}) must be a number, "
            f"not {value!r}"
        )
    return float(value)


def load_material(name: str) -> BHCurve | Magnet | Conductor:
    """Load a material from the library by name.

    Returns a ``BHCurve`` for a steel, a ``Magnet`` for a magnet grade and a
    ``Conductor`` for a conductor. Raises ValueError for a name that is not
    in the library or a file with a missing or impossible field.
    """
    data = _read_entry(name)
    kind = data["kind"]
    if kind == "magnet":
        coefficients = {}
        given = [f for f in _COEFFICIENTS if f in data]
        if given and len(given) != len(_COEFFICIENTS):
            raise ValueError(
                f"material file {name}.yaml: give both {' and '.join(_COEFFICIENTS)}, or neither"
            )
        for field, argument in _COEFFICIENTS.items():
            if field in data:
                coefficients[argument] = _coefficient(data, field, name)
        return Magnet(
            remanence=_number(data, "remanence_T", name, positive=False),
            relative_permeability=_number(data, "relative_permeability", name),
            **coefficients,
        )
    if kind == "conductor":
        return Conductor(
            resistivity=_number(data, "resistivity_ohm_m", name),
            temperature_coefficient=_number(data, "temperature_coefficient_per_K", name),
            density=_number(data, "density_kg_m3", name),
        )
    return _steel(data, name)


def _steel(data: dict, name: str) -> BHCurve:
    """The BH curve of a steel entry, from a [B, H] or a [J, H] table."""
    where = f"material file {name}.yaml"
    given = [field for field in ("bh_curve", "jh_curve") if field in data]
    if len(given) != 1:
        raise ValueError(f"{where}: give one of 'bh_curve' and 'jh_curve'")
    field = given[0]
    rows = "[B, H]" if field == "bh_curve" else "[J, H]"
    try:
        table = np.array(data[field], dtype=float)
    except (TypeError, ValueError):
        table = np.empty(0)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError(f"{where}: '{field}' must be a list of {rows} rows")
    first, h = table[:, 0], table[:, 1]
    # A datasheet gives polarisation J; the solver needs B = J + mu0 H.
    b = first + MU0 * h if field == "jh_curve" else first
    if b.size and b[0] > 0.0 and h[0] > 0.0:
        # Data usually start above zero; the curve starts at the origin.
        b, h = np.insert(b, 0, 0.0), np.insert(h, 0, 0.0)
    saturation = None
    if "saturation_polarisation_T" in data:
        saturation = _number(data, "saturation_polarisation_T", name)
    try:
        return BHCurve(b, h, saturation)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


@dataclass(frozen=True)
class CoreLossCoefficients:
    """Coefficients of p = kh f B^beta + kc f^2 B^2 + ke f^1.5 B^1.5 in W/kg.

    The three terms are hysteresis, classical eddy current and excess loss
    (Bertotti), for a sinusoidal flux density of peak B in tesla at f in Hz.
    """

    hysteresis: float  # kh, W/kg per (Hz T^beta)
    eddy: float  # kc, W/kg per (Hz T)^2
    excess: float  # ke, W/kg per (Hz T)^1.5
    hysteresis_exponent: float = 2.0  # beta

    def loss(self, frequency: ArrayLike, peak: ArrayLike) -> NDArray[np.float64]:
        """Specific loss in W/kg at ``frequency`` in Hz and peak flux density ``peak`` in T."""
        f = np.asarray(frequency, dtype=float)
        b = np.asarray(peak, dtype=float)
        return (
            self.hysteresis * f * b**self.hysteresis_exponent
            + self.eddy * (f * b) ** 2
            + self.excess * (f * b) ** 1.5
        )


@dataclass(frozen=True)
class CoreLossData:
    """Measured specific loss of a steel grade, and the sheet it was measured on.

    ``tables`` maps a frequency in Hz to (peak polarisation in T, loss in W/kg).
    """

    thickness: float  # m
    density: float  # kg/m^3
    resistivity: float  # ohm m
    tables: dict[float, tuple[NDArray[np.float64], NDArray[np.float64]]]

    def eddy_coefficient(self) -> float:
        """Classical eddy-current kc = pi^2 d^2 / (6 rho density) for a sheet of thickness d."""
        return math.pi**2 * self.thickness**2 / (6.0 * self.resistivity * self.density)

    def fit(
        self,
        max_frequency: float = 1000.0,
        min_polarisation: float = 0.3,
        hysteresis_exponent: float = 2.0,
    ) -> CoreLossCoefficients:
        """Fit kh and ke to the tables, with kc from the sheet's thickness and resistivity.

        Least squares on the relative error, with neither coefficient below
        zero. The tables give loss against peak polarisation J; below 1.8 T
        J and B differ by less than 1 %, and the fit treats them as equal.
        """
        f, b, p = self.points(max_frequency, min_polarisation)
        if f.size < 2:
            raise ValueError("too few loss data points in the chosen range to fit")
        kc = self.eddy_coefficient()
        basis = np.column_stack([f * b**hysteresis_exponent, (f * b) ** 1.5]) / p[:, None]
        (kh, ke), _ = nnls(basis, 1.0 - kc * (f * b) ** 2 / p)
        return CoreLossCoefficients(float(kh), kc, float(ke), hysteresis_exponent)

    def points(
        self, max_frequency: float = math.inf, min_polarisation: float = 0.0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Frequency, polarisation and loss of every table point in the range, as flat arrays."""
        chosen = [(f, j, p) for f, (j, p) in self.tables.items() if f <= max_frequency]
        f = np.concatenate([np.full(j.size, freq) for freq, j, _ in chosen] or [np.empty(0)])
        j = np.concatenate([j for _, j, _ in chosen] or [np.empty(0)])
        p = np.concatenate([p for _, _, p in chosen] or [np.empty(0)])
        keep = j >= min_polarisation - 1e-12
        return f[keep], j[keep], p[keep]


def load_core_loss(name: str) -> CoreLossData:
    """The loss tables of a steel in the library, with its thickness, density and resistivity.

    Raises ValueError for a name not in the library, a material that is not
    a steel, or a steel with no loss data.
    """
    data = _read_entry(name)
    where = f"material file {name}.yaml"
    if data["kind"] != "steel" or "loss_W_per_kg" not in data:
        raise ValueError(f"{where}: no loss data ('loss_W_per_kg') for this material")
    raw = data["loss_W_per_kg"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{where}: 'loss_W_per_kg' must map frequencies in Hz to lists of losses")
    tables = {}
    step = _number(data, "loss_step_T", name) if "loss_step_T" in data else 0.1
    for frequency, losses in raw.items():
        ok = isinstance(frequency, (int, float)) and not isinstance(frequency, bool)
        try:
            loss = np.array(losses, dtype=float)
        except (TypeError, ValueError):
            loss = np.empty(0)
        if not ok or frequency <= 0 or loss.ndim != 1 or loss.size == 0:
            raise ValueError(f"{where}: 'loss_W_per_kg' must map frequencies in Hz to lists of losses")
        if not (np.all(np.isfinite(loss)) and np.all(loss > 0.0) and np.all(np.diff(loss) > 0.0)):
            raise ValueError(
                f"{where}: losses at {frequency} Hz must be positive and rise with polarisation"
            )
        tables[float(frequency)] = (step * np.arange(1, loss.size + 1), loss)
    return CoreLossData(
        thickness=_number(data, "thickness_mm", name) * 1e-3,
        density=_number(data, "density_kg_m3", name),
        resistivity=_number(data, "resistivity_ohm_m", name),
        tables=tables,
    )
