"""Mechanical metrics: mass, rotor inertia, torque density and motor constant.

Everything here follows from the geometry, the winding and the densities of
the materials; no field is solved.

- Masses are those of the active parts: stator steel, rotor steel, magnets
  and the copper of the winding, end turns included. The shaft, bearings,
  housing, insulation, glue and the varnish are not counted, so the motor's
  real mass is higher, by an amount that depends on the build.
- The rotor inertia is that of the turning part, its steel and magnets, about
  the axis. A QDD motor's rotor is usually the larger part of what the
  gearbox sees, but anything fixed to it (a shaft, a hub, an encoder disc)
  adds to it and is not counted.
- Steel is taken as a stack of laminations with the design's stacking factor.
  Magnets run the stack length plus the design's overhang at each end.
- The motor constant is the torque a motor gives per square root of the
  copper loss, Km = T / sqrt(P) in N m / sqrt(W). Two windings of the same
  motor with the same copper in the slot have the same Km, however the turns
  and wire size are chosen: it measures the motor, not its winding.

Densities in kg/m3 come from the material library where it has them (steel
with loss data, the conductor), and are otherwise given by the caller, for
example ``densities={"N42SH": ...}`` from the magnet's datasheet. None is
assumed: the library's placeholder steel and magnet have no density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np

from polecraft.core.geometry import Region
from polecraft.core.winding import PhaseWinding, phase_winding, wire_area

if TYPE_CHECKING:
    from polecraft.core.materials import Conductor
    from polecraft.machines.base import Machine

# Chord tolerance, in metres, for the outline of a region with arcs.
OUTLINE_TOLERANCE = 1e-9


def polar_moment_of_area(region: Region) -> float:
    """Integral of r^2 over a region's area about the machine axis, in m^4.

    From the outline's vertices (arcs cut into short chords) by the
    polygon formula for the sum of the second moments about x and y.
    """
    points = region.polygon(OUTLINE_TOLERANCE)
    x0, y0 = points[:, 0], points[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    cross = x0 * y1 - x1 * y0
    total = np.sum(cross * (x0**2 + x0 * x1 + x1**2 + y0**2 + y0 * y1 + y1**2)) / 12.0
    return float(total)


@dataclass(frozen=True)
class Mechanical:
    """Masses, rotor inertia and size of a machine's active parts."""

    stator_steel: float  # kg
    rotor_steel: float  # kg
    magnets: float  # kg
    copper: float  # kg, all three phases with the end turns
    rotor_inertia: float  # kg m^2, rotor steel and magnets about the axis
    outer_radius: float  # m
    stack_length: float  # m

    @property
    def mass(self) -> float:
        """Mass of the active parts, kg."""
        return self.stator_steel + self.rotor_steel + self.magnets + self.copper

    @property
    def rotor_mass(self) -> float:
        """Mass of the turning part, kg."""
        return self.rotor_steel + self.magnets

    @property
    def volume(self) -> float:
        """Volume of the cylinder the stack fills, in m^3: pi R^2 L, end turns not included."""
        return math.pi * self.outer_radius**2 * self.stack_length

    def torque_density(self, torque: float) -> float:
        """Torque per kilogram of active parts, in N m/kg."""
        return _positive(torque, "torque") / self.mass

    def torque_per_volume(self, torque: float) -> float:
        """Torque per cubic metre of the stack's cylinder, in N m/m^3."""
        return _positive(torque, "torque") / self.volume


def _positive(value: float, name: str) -> float:
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be a positive number, not {value}")
    return float(value)


def _steel_density(grade: str, densities: Mapping[str, float]) -> float:
    if grade in densities:
        return _positive(densities[grade], f"density of '{grade}'")
    from polecraft.core.materials import load_core_loss

    try:
        return load_core_loss(grade).density
    except ValueError:
        raise ValueError(
            f"no density for the steel '{grade}': the library has none for it, so give "
            f"densities={{'{grade}': ...}} in kg/m3"
        ) from None


def mechanical(
    machine: Machine,
    densities: Mapping[str, float] | None = None,
    conductor: Conductor | None = None,
) -> Mechanical:
    """Mass, rotor inertia and size of a machine's active parts.

    ``densities`` maps a material grade of the design (the steel, the
    magnet) to its density in kg/m3. A steel is looked up in the library
    when it is not given; the magnet always has to be, and a grade given
    overrides the library's. ``conductor`` is as for ``phase_winding``.

    Raises ValueError for a grade without a density, a density that is not
    positive, or a machine that does not describe its coils.
    """
    densities = dict(densities or {})
    design = machine.design
    geometry = machine.build_geometry(full=True)
    stack = geometry.stack_length
    factor = design.steel.stacking_factor
    magnet_length = stack + 2.0 * design.axial.magnet_overhang_mm * 1e-3
    magnet_density = None
    totals = {"stator": 0.0, "rotor": 0.0, "magnet": 0.0}
    inertia = 0.0
    for turning, regions in ((False, geometry.stator), (True, geometry.rotor)):
        for region in regions:
            if region.kind == "steel":
                length, density = stack * factor, _steel_density(region.material, densities)
                totals["rotor" if turning else "stator"] += region.area() * length * density
            elif region.kind == "magnet":
                if magnet_density is None:
                    if region.material not in densities:
                        raise ValueError(
                            f"no density for the magnet '{region.material}': give "
                            f"densities={{'{region.material}': ...}} in kg/m3 from its datasheet"
                        )
                    magnet_density = _positive(
                        densities[region.material], f"density of '{region.material}'"
                    )
                length, density = magnet_length, magnet_density
                totals["magnet"] += region.area() * length * density
            else:
                continue
            if turning:
                inertia += polar_moment_of_area(region) * length * density
    winding = phase_winding(machine, conductor)
    return Mechanical(
        stator_steel=totals["stator"],
        rotor_steel=totals["rotor"],
        magnets=totals["magnet"],
        copper=_copper_mass(winding),
        rotor_inertia=inertia,
        outer_radius=geometry.outer_radius,
        stack_length=stack,
    )


def _copper_mass(winding: PhaseWinding) -> float:
    """Mass of the wire of all three phases in kg: length times area times density."""
    area = winding.strands * wire_area(winding.wire_diameter)
    return 3.0 * winding.wire_length * area * winding.conductor.density


def motor_constant(torque_constant: float, winding: PhaseWinding) -> float:
    """Motor constant Km = T / sqrt(P) in N m / sqrt(W), at the winding's temperature.

    ``torque_constant`` is the torque per ampere of peak phase current, in
    N m/A, as the loaded-torque study reports it. At that current I the
    copper loss of the three phases is 1.5 R I^2, so Km = kt / sqrt(1.5 R).
    Raises ValueError for a torque constant or resistance that is not
    positive.
    """
    kt = _positive(torque_constant, "torque constant")
    return kt / math.sqrt(1.5 * _positive(winding.resistance(), "resistance"))
