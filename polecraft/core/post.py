"""Results read off a solved field: flux linkage, torque and core loss.

The flux density in every triangle comes with the solution itself. This
module turns a problem and its solution into the quantities a machine is
judged by. Like the solver, it knows nothing about machine types: it works
from the windings and the air gap that the problem describes.

Results are for the whole machine: the meshed part, times the depth of the
model, times the number of times that part repeats.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from polecraft.core.materials import MU0
from polecraft.core.solver import shape_gradients

if TYPE_CHECKING:
    from polecraft.core.materials import CoreLossCoefficients
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


def gap_flux_density(
    problem: Problem, solution: Solution, angles: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Flux density in the middle of the air gap, in tesla, at given angles.

    Returns the radial part (positive outwards) and the tangential part
    (positive counter-clockwise) at each angle, in radians from the x axis,
    on the circle halfway between the two radii of the problem's gap. The
    values are those of the gap triangle each point lies in, so they step
    from triangle to triangle.

    An angle outside the meshed part is answered from the part that is
    meshed: the field repeats, with its sign reversed from one part to the
    next when the coupling is anti-periodic.
    """
    _check_pair(problem, solution)
    gap = problem.gap
    if gap is None:
        raise ValueError("the problem has no air gap marked, so there is no gap field to read")
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 1 or not np.all(np.isfinite(angles)):
        raise ValueError("angles must be a list of finite numbers, in radians")
    radius = 0.5 * (gap.inner_radius + gap.outer_radius)
    corners = problem.nodes[problem.triangles[gap.triangles]]
    origin = corners[:, 0]
    # Each row solves for the two weights of a point on the triangle's sides.
    sides = np.stack([corners[:, 1] - origin, corners[:, 2] - origin], axis=2)
    inverse = np.linalg.inv(sides)
    turn = 2.0 * np.pi
    span = turn / problem.repeats
    radial = np.zeros(angles.shape)
    around = np.zeros(angles.shape)
    missing = np.ones(angles.shape, dtype=bool)
    slack = 1e-9  # a point on a shared side belongs to either triangle
    # The angles each triangle spans, measured from its own middle so that
    # the jump of the angle at pi does not split it. A point is only tried
    # against the triangles whose span holds its angle.
    middle = np.arctan2(corners[:, :, 1].sum(axis=1), corners[:, :, 0].sum(axis=1))
    offset = (np.arctan2(corners[:, :, 1], corners[:, :, 0]) - middle[:, None] + np.pi) % turn - np.pi
    low = (middle + offset.min(axis=1) - slack) % turn
    high = low + (offset.max(axis=1) - offset.min(axis=1)) + 2.0 * slack
    for part in range(problem.repeats):
        todo = np.flatnonzero(missing)
        if todo.size == 0:
            break
        # Turned back by whole parts until it falls in the meshed one. The
        # radial and tangential parts of the field turn with the point.
        turned = angles[todo] - part * span
        reduced = turned % turn
        order = np.argsort(reduced, kind="stable")
        ordered = reduced[order]
        pairs_triangle, pairs_point = [], []
        for shift in (0.0, turn):  # a span may run past the end of the turn
            first = np.searchsorted(ordered, low - shift, side="left")
            count = np.searchsorted(ordered, high - shift, side="right") - first
            within = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
            pairs_triangle.append(np.repeat(np.arange(len(low)), count))
            pairs_point.append(order[np.repeat(first, count) + within])
        triangle, point = np.concatenate(pairs_triangle), np.concatenate(pairs_point)
        position = radius * np.column_stack([np.cos(turned[point]), np.sin(turned[point])])
        weights = np.einsum("kij,kj->ki", inverse[triangle], position - origin[triangle])
        inside = (weights >= -slack).all(axis=1) & (weights.sum(axis=1) <= 1.0 + slack)
        triangle, point = triangle[inside], point[inside]
        # One triangle for each point: the lowest-numbered of those that hold it.
        by_point = np.lexsort((triangle, point))
        point, firsts = np.unique(point[by_point], return_index=True)
        b = solution.flux_density[gap.triangles[triangle[by_point][firsts]]]
        if problem.antiperiodic and part % 2:
            b = -b
        cos, sin = np.cos(turned[point]), np.sin(turned[point])
        radial[todo[point]] = b[:, 0] * cos + b[:, 1] * sin
        around[todo[point]] = b[:, 1] * cos - b[:, 0] * sin
        missing[todo[point]] = False
    if missing.any():
        raise ValueError("the gap triangles do not cover the middle of the gap all the way round")
    return radial, around


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


def material_flux_density(
    steps: Sequence[tuple[Problem, Solution]], triangles: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Flux density in some triangles over a series of solves, as the material sees it.

    Steel that turns with a rotor sees the field in its own frame, not in
    the fixed one the solver works in. Each triangle's field is turned back
    by the angle its first side has turned since the first solve, so a
    field that only follows the triangle round comes out constant.

    ``steps`` are (problem, solution) pairs in which the listed triangles
    keep their numbers and corner nodes. Returns the field, shape (steps,
    triangles, 2) in tesla, and the angle each triangle has turned at each
    step, shape (steps, triangles), in radians.
    """
    triangles = np.asarray(triangles)
    if triangles.ndim != 1 or (triangles.size and not np.issubdtype(triangles.dtype, np.integer)):
        raise ValueError("the triangles must be given as a list of triangle numbers")
    if len(steps) == 0:
        raise ValueError("give at least one solved step")
    field = np.empty((len(steps), triangles.size, 2))
    turn = np.empty((len(steps), triangles.size))
    corners = None
    for number, (problem, solution) in enumerate(steps):
        _check_pair(problem, solution)
        here = problem.triangles[triangles]
        if corners is None:
            corners = here
        elif not np.array_equal(here, corners):
            raise ValueError(f"the triangles at step {number} are not those of the first step")
        side = problem.nodes[here[:, 1]] - problem.nodes[here[:, 0]]
        turn[number] = np.arctan2(side[:, 1], side[:, 0])
        field[number] = solution.flux_density[triangles]
    turn = np.unwrap(turn - turn[0], axis=0)
    cos, sin = np.cos(turn), np.sin(turn)
    bx, by = field[..., 0], field[..., 1]
    return np.stack([cos * bx + sin * by, cos * by - sin * bx], axis=2), turn


# Mean of |cos|^1.5 over a turn: it ties the mean of |dB/dt|^1.5 of a sine
# wave to its peak, so that the excess term gives ke (f B)^1.5 for one.
_EXCESS_MEAN = math.gamma(1.25) / (math.sqrt(math.pi) * math.gamma(1.75))


def core_loss_density(
    waveform: ArrayLike,
    duration: float,
    coefficients: CoreLossCoefficients,
    periodic: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Hysteresis, eddy-current and excess loss in W/kg from flux density waveforms.

    ``waveform`` is (Bx, By) in tesla in the material's own frame, shape
    (samples, places, 2), and ``duration`` the time in seconds it covers.
    The three results have one value per place. For a sine wave of peak B
    along any direction, at f = 1 / duration, they are the three terms of
    the coefficients' formula: kh f B^beta, kc (f B)^2 and ke (f B)^1.5.
    For other waveforms:

    - Eddy-current loss is kc / (2 pi^2) times the mean of |dB/dt|^2, which
      is what the sheet formula behind kc says for any waveform slow enough
      for the field to fill the sheet (no skin effect).
    - Excess loss is the mean of |dB/dt|^1.5, scaled to agree for a sine.
    - Hysteresis loss is kh f (Ba^beta + Bb^beta): Ba is half the
      peak-to-peak swing along the direction in which the field varies
      most, Bb the same at right angles to it. A field of fixed size that
      turns round once is so counted as two alternating fields, the usual
      approximation; the real loss in a turning field differs, and falls
      near saturation. Smaller loops within the cycle are not counted, and
      a steady field under the swing is taken to change nothing.

    With ``periodic`` the samples are evenly spaced over one whole period,
    the last one step short of its end, and the rates of change are exact
    for every harmonic below half the number of samples. Without it the
    samples run from the start to the end of the time, both included, and
    need not close; the rates are differences between neighbours, which
    lose a little of the fast harmonics (a factor sin(x)^2 / x^2 in the eddy
    loss, x = pi n / intervals for harmonic n).

    A constant field gives no loss.
    """
    b = np.asarray(waveform, dtype=float)
    if b.ndim != 3 or b.shape[2] != 2:
        raise ValueError(f"waveform must have shape (samples, places, 2), not {b.shape}")
    if not (math.isfinite(duration) and duration > 0.0):
        raise ValueError(f"duration must be a positive time in seconds, not {duration}")
    samples = len(b)
    if samples < (4 if periodic else 2):
        raise ValueError(f"too few samples ({samples}) to take a rate of change from")
    if periodic:
        waves = np.fft.rfft(b, axis=0)
        factor = 1j * np.arange(len(waves)) * (2.0 * math.pi / duration)
        if samples % 2 == 0:
            factor[-1] = 0.0  # at half the sampling rate the slope is not known
        rate = np.fft.irfft(waves * factor[:, None, None], samples, axis=0)
    else:
        rate = np.diff(b, axis=0) * ((samples - 1) / duration)
    frequency = 1.0 / duration
    squared = (rate**2).sum(axis=2)
    eddy = coefficients.eddy / (2.0 * math.pi**2) * squared.mean(axis=0)
    excess = (
        coefficients.excess / ((2.0 * math.pi) ** 1.5 * _EXCESS_MEAN) * (squared**0.75).mean(axis=0)
    )
    # The direction of greatest variation is the main axis of the scatter of
    # the samples about their mean.
    centred = b - b.mean(axis=0)
    x, y = centred[..., 0], centred[..., 1]
    angle = 0.5 * np.arctan2(2.0 * (x * y).sum(axis=0), (x * x).sum(axis=0) - (y * y).sum(axis=0))
    cos, sin = np.cos(angle), np.sin(angle)
    along = 0.5 * np.ptp(cos * x + sin * y, axis=0)
    across = 0.5 * np.ptp(cos * y - sin * x, axis=0)
    beta = coefficients.hysteresis_exponent
    hysteresis = coefficients.hysteresis * frequency * (along**beta + across**beta)
    return hysteresis, eddy, excess


def magnet_field(
    problem: Problem, solution: Solution
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
    """Field along the magnetisation in every magnetised triangle.

    Returns the numbers of the triangles with a remanence, the field H
    along their magnetisation in A/m and the flux density B along it in
    tesla. On the recoil line of a magnet B = mu_r mu0 H + Br, so
    H = (B - Br) / (mu_r mu0); H is negative where it opposes the
    magnetisation, which is where a magnet can be demagnetised. The
    magnetisation's direction is that of the triangle's remanence vector.
    """
    _check_pair(problem, solution)
    remanence = problem.remanence
    size = np.hypot(remanence[:, 0], remanence[:, 1])
    triangles = np.flatnonzero(size > 0.0)
    direction = remanence[triangles] / size[triangles, None]
    along_b = (solution.flux_density[triangles] * direction).sum(axis=1)
    # A magnetised triangle has a linear material: one reluctivity each.
    reluctivity = np.array(
        [float(model.reluctivity(np.zeros(1))[0]) for model in problem.materials]
    )[problem.triangle_material[triangles]]
    along_h = reluctivity * (along_b - size[triangles])
    return triangles, along_h, along_b
