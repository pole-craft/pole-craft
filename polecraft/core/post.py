"""Results read off a solved field: flux linkage and torque.

The flux density in every triangle comes with the solution itself. This
module turns a problem and its solution into the quantities a machine is
judged by. Like the solver, it knows nothing about machine types: it works
from the windings and the air gap that the problem describes.

Results are for the whole machine: the meshed part, times the depth of the
model, times the number of times that part repeats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from polecraft.core.materials import MU0
from polecraft.core.solver import shape_gradients

if TYPE_CHECKING:
    from polecraft.core.problem import Problem
    from polecraft.core.solver import Solution


def linkage_per_metre(
    nodes: ArrayLike, triangles: ArrayLike, potential: ArrayLike, turn_density: ArrayLike
) -> float:
    """Flux linked by a winding, in Wb per metre of depth.

    A conductor at a point links the potential A there: the flux between it
    and wherever A is zero, per metre. A winding links the sum over its
    turns, which is the integral of A times the turns per square metre.
    ``turn_density`` is that density in every triangle, positive where a
    positive current flows out of the plane and negative where it returns,
    so a coil with both sides in the mesh links the flux between them.
    """
    triangles = np.asarray(triangles)
    _, area = shape_gradients(nodes, triangles)
    potential = np.asarray(potential, dtype=float)
    turn_density = np.asarray(turn_density, dtype=float)
    if potential.shape != (len(np.asarray(nodes)),):
        raise ValueError(f"potential must have one value per node, not shape {potential.shape}")
    if turn_density.shape != (len(triangles),):
        raise ValueError(
            f"turn density must have one value per triangle, not shape {turn_density.shape}"
        )
    return float((turn_density * area * potential[triangles].mean(axis=1)).sum())


def ring_torque_per_metre(
    nodes: ArrayLike,
    triangles: ArrayLike,
    flux_density: ArrayLike,
    ring: ArrayLike,
    inner_radius: float,
    outer_radius: float,
) -> float:
    """Torque on whatever a ring of air encloses, in N m per metre of depth.

    The Maxwell stress on a circle of radius r in air turns what is inside it
    with a torque of r^2 / mu0 times the integral of B_r B_t round the
    circle, counter-clockwise positive. Every circle in the gap gives the
    same torque, so the average over the thickness of a ring is used: the
    integral of r B_r B_t over the ring's area, divided by mu0 and by its
    thickness. That uses every triangle of the ring and is much less
    sensitive to the mesh than a single circle.

    ``ring`` lists the triangles that fill the ring between the two radii,
    over the full circle. For part of a circle the result is that part's
    share. ``flux_density`` is (Bx, By) in every triangle of the mesh.
    """
    nodes = np.asarray(nodes, dtype=float)
    triangles = np.asarray(triangles)
    ring = np.asarray(ring)
    if not np.issubdtype(ring.dtype, np.integer):
        # A true/false mask would be read as triangles 0 and 1.
        raise ValueError("the ring must be given as triangle numbers")
    if not (np.isfinite(inner_radius) and np.isfinite(outer_radius)) or not (
        0.0 < inner_radius < outer_radius
    ):
        raise ValueError(
            "ring radii must be positive, the inner one smaller: "
            f"got {inner_radius} and {outer_radius}"
        )
    flux_density = np.asarray(flux_density, dtype=float)
    if flux_density.shape != (len(triangles), 2):
        raise ValueError(
            f"flux density must be (Bx, By) for every triangle, not shape {flux_density.shape}"
        )
    corners = triangles[ring]
    _, area = shape_gradients(nodes, corners)
    centre = nodes[corners].mean(axis=1)
    b = flux_density[ring]
    # r B_r is B . centre, and r B_t is centre x B; their product over r is
    # r B_r B_t at the centre of the triangle.
    r_radial = (b * centre).sum(axis=1)
    r_around = centre[:, 0] * b[:, 1] - centre[:, 1] * b[:, 0]
    radius = np.hypot(centre[:, 0], centre[:, 1])
    total = (r_radial * r_around / radius * area).sum()
    return float(total / (MU0 * (outer_radius - inner_radius)))


def _check_pair(problem: Problem, solution: Solution) -> None:
    if solution.potential.shape != (len(problem.nodes),) or solution.flux_density.shape != (
        len(problem.triangles),
        2,
    ):
        raise ValueError("the solution does not belong to this problem: the sizes differ")


def flux_linkage(problem: Problem, solution: Solution) -> dict[str, float]:
    """Flux linked by each winding of the whole machine, in webers.

    One value per winding of the problem, by name. For a machine these are
    the phases, each with all its coils in series.
    """
    _check_pair(problem, solution)
    scale = problem.depth * problem.repeats
    return {
        name: scale
        * linkage_per_metre(problem.nodes, problem.triangles, solution.potential, turn_density)
        for name, turn_density in problem.windings.items()
    }


def torque(problem: Problem, solution: Solution) -> float:
    """Torque on the turning part of the whole machine, in N m.

    Counter-clockwise positive, from the Maxwell stress in the problem's air
    gap. Raises ValueError if the problem has no air gap marked.
    """
    _check_pair(problem, solution)
    gap = problem.gap
    if gap is None:
        raise ValueError("the problem has no air gap marked, so there is no torque to take")
    on_inside = ring_torque_per_metre(
        problem.nodes,
        problem.triangles,
        solution.flux_density,
        gap.triangles,
        gap.inner_radius,
        gap.outer_radius,
    )
    # What turns the inside one way turns the outside the other.
    on_rotor = -on_inside if gap.rotor_outside else on_inside
    return problem.depth * problem.repeats * on_rotor
