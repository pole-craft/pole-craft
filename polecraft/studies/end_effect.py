"""End effect: how much of the magnet's flux a stack of finite length delivers.

The 2D model is a slice of a machine of infinite length: every millimetre of
stack carries the flux of the middle. At the two ends of a real stack the
field is different. Some of the magnet's flux curls round the end of the
magnet back to its own yoke and never reaches the stator, and some fringes
from the magnet into the end face of the stator. In a long machine these
are a small part of the whole; a QDD motor is short and wide, so they
matter.

This study estimates the net effect with a second, separate 2D model: a cut
through the middle of a pole in the plane that holds the radius and the
axis, from the middle of the stack to well past its end. In that plane:

- The stator and rotor steel are taken as ideal iron (infinite
  permeability), so each is one magnetic equipotential. Over a pair of
  poles they are at the same potential, so both are held at zero, and the
  magnet drives its flux across the gap between them.
- The field is solved for the magnetic scalar potential (B = -mu grad(phi)
  + Br), on a grid of triangles graded towards the corners.
- The flux that enters the stator, per unit of width round the gap, is
  compared with what a stack of infinite length gives over the same length,
  Br lm / (lm + mu_r g) per metre. Their ratio is the end factor.

Far from the end the cut sees exactly the 1D magnetic circuit, so the
factor tends to 1 as the stack grows, as 1 + 2 d / L with d the length
gained (or lost, if negative) at each end.

What the factor is, and is not:

- An approximation. The cut is taken at the middle of a pole and treated
  as flat (radius much larger than the gap and the magnet); the leakage at
  the ends between neighbouring poles, round the circumference, is left out,
  as is the slotting of the stator face. The steel does not saturate.
- It scales what the magnet's flux makes: the magnet flux linkage, the
  back-EMF and, while the steel is not saturated, the torque. It does not
  touch the end-turn inductance, which ``polecraft.core.winding`` estimates
  separately.
- The rotor yoke is taken to run as far as the magnet, overhang included.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from polecraft.core.solver import assemble_stiffness, shape_gradients, solve_linear

if TYPE_CHECKING:
    from polecraft.machines.base import Machine

# Smallest grid step as a part of the gap, the rate the steps grow at away
# from a corner, and how far the model reaches past the end, in gap plus
# magnet lengths.
FINE = 0.1
GROWTH = 1.15
REACH = 30.0


@dataclass(frozen=True)
class EndEffect:
    """Flux of a finite stack against that of the 2D model over the same length."""

    stack_length: float  # m
    gap: float  # m
    magnet_thickness: float  # m
    magnet_overhang: float  # m, at each end
    remanence: float  # T
    relative_permeability: float
    flux: float  # Wb per metre of width, into the stator over half the stack

    @property
    def flux_2d(self) -> float:
        """What the 2D model gives over half the stack, Wb per metre of width."""
        middle = self.remanence * self.magnet_thickness / (
            self.magnet_thickness + self.relative_permeability * self.gap
        )
        return middle * 0.5 * self.stack_length

    @property
    def factor(self) -> float:
        """The end factor: flux of the finite stack over that of the 2D model."""
        return self.flux / self.flux_2d

    @property
    def extension(self) -> float:
        """Length in m gained at each end (lost, if negative): factor = 1 + 2 d / L."""
        return 0.5 * (self.factor - 1.0) * self.stack_length

    def corrected(self, value: float) -> float:
        """A 2D result made by the magnet's flux (EMF, flux linkage, torque) times the factor."""
        return value * self.factor


def _graded(points: list[float], fine: float, coarse: float) -> NDArray[np.float64]:
    """Grid lines through every given point, ``fine`` apart there, growing between."""
    points = sorted(set(points))
    lines = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        span = end - start
        # Steps grow from both ends towards the middle.
        steps, size = [], fine
        while 2.0 * (sum(steps) + size) <= span:
            steps.append(size)
            size = min(size * GROWTH, coarse)
        middle = span - 2.0 * sum(steps)
        pieces = max(1, math.ceil(middle / coarse))
        sizes = steps + [middle / pieces] * pieces + steps[::-1]
        position = start
        for size in sizes[:-1]:
            position += size
            lines.append(position)
        lines.append(end)
    return np.array(lines)


def end_factor(
    stack_length: float,
    gap: float,
    magnet_thickness: float,
    remanence: float = 1.0,
    relative_permeability: float = 1.05,
    magnet_overhang: float = 0.0,
) -> EndEffect:
    """End factor of a surface-magnet machine from its lengths, in metres.

    ``gap`` is the air gap, ``magnet_thickness`` the magnet's length along
    its magnetisation, ``magnet_overhang`` how far the magnet runs past the
    stack at each end. The factor does not depend on ``remanence``; it sets
    the size of ``flux``.

    Raises ValueError for lengths that are not positive (the overhang may
    be 0), or a permeability that is not positive.
    """
    for name, value in (
        ("stack length", stack_length),
        ("gap", gap),
        ("magnet thickness", magnet_thickness),
        ("remanence", remanence),
        ("relative permeability", relative_permeability),
    ):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be a positive number, not {value}")
    if not (math.isfinite(magnet_overhang) and magnet_overhang >= 0.0):
        raise ValueError(f"magnet overhang must be 0 or more, not {magnet_overhang}")
    g, lm = gap, magnet_thickness
    half, magnet_end = 0.5 * stack_length, 0.5 * stack_length + magnet_overhang
    reach = REACH * (g + lm)
    fine, coarse = FINE * g, 0.5 * (g + lm)
    # x is the radius, from the stator face outwards; y is along the axis,
    # from the middle of the stack. The stator lies below x = 0 up to the
    # end of the stack, the rotor yoke beyond x = g + lm up to the end of
    # the magnet.
    x = _graded([-reach, 0.0, g, g + lm, g + lm + reach], fine, coarse)
    y = _graded([0.0, half, magnet_end, magnet_end + reach], fine, coarse)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    nx, ny = len(x), len(y)
    corner = (np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]).ravel()
    triangles = np.concatenate(
        [
            np.column_stack([corner, corner + ny, corner + ny + 1]),
            np.column_stack([corner, corner + ny + 1, corner + 1]),
        ]
    )
    centre = nodes[triangles].mean(axis=1)
    in_magnet = (centre[:, 0] > g) & (centre[:, 0] < g + lm) & (centre[:, 1] < magnet_end)
    in_stator = (centre[:, 0] < 0.0) & (centre[:, 1] < half)
    in_rotor = (centre[:, 0] > g + lm) & (centre[:, 1] < magnet_end)
    # Relative permeability as the coefficient; potentials then come out in
    # amperes when the remanence is divided by mu0.
    coefficient = np.where(in_magnet, relative_permeability, 1.0)
    stiffness = assemble_stiffness(nodes, triangles, coefficient)
    gradients, area = shape_gradients(nodes, triangles)
    mu0 = 4e-7 * math.pi
    source = np.zeros((len(triangles), 2))
    source[in_magnet, 0] = remanence / mu0  # magnetised along the radius
    local = area[:, None] * np.einsum("ed,eid->ei", source, gradients)
    load = np.bincount(triangles.ravel(), weights=local.ravel(), minlength=len(nodes))
    # Ideal iron: every node on or in the steel is at potential zero.
    stator_nodes = np.unique(triangles[in_stator])
    rotor_nodes = np.unique(triangles[in_rotor])
    fixed = np.union1d(stator_nodes, rotor_nodes)
    potential = solve_linear(stiffness, load, fixed, 0.0)
    # The flux leaving the free region into the stator is the out-of-balance
    # of the stator's nodes: B.n integrated over its surface.
    reaction = stiffness @ potential - load
    flux = mu0 * float(reaction[stator_nodes].sum())
    return EndEffect(
        stack_length=float(stack_length),
        gap=float(g),
        magnet_thickness=float(lm),
        magnet_overhang=float(magnet_overhang),
        remanence=float(remanence),
        relative_permeability=float(relative_permeability),
        flux=flux,
    )


def end_effect(machine: Machine, relative_permeability: float = 1.05) -> EndEffect:
    """End factor of a machine's design, from its gap, magnet and stack lengths.

    ``relative_permeability`` is the magnet's recoil permeability; the
    factor depends on it only a little.
    """
    design = machine.design
    mm = 1e-3
    return end_factor(
        stack_length=design.axial.stack_length_mm * mm,
        gap=design.geometry.air_gap_mm * mm,
        magnet_thickness=design.geometry.magnet.thickness_mm * mm,
        relative_permeability=relative_permeability,
        magnet_overhang=design.axial.magnet_overhang_mm * mm,
    )
