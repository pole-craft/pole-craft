"""Turning the rotor without meshing again.

The stator and the rotor keep the triangles they were meshed with. The rotor
nodes are turned as a rigid body, and only the band, the single layer of
triangles between two rings of equally spaced nodes, is joined again: each
stator ring node to the rotor ring node now nearest to facing it.

At an angle that is not a whole number of node spacings, the nodes of the
rotor ring stop at the nearest whole spacing, so the band keeps its shape,
and the rotor's own layer of air next to the ring takes up the rest, half a
spacing at most. Torque is taken in the band, and triangles that lean with
the angle would spoil it: their error changes with every spacing turned,
and the torque sees the rate of that change. So the lean is put where no
torque is taken.

The field repeats when the rotor has turned by the meshed angle, or by twice
that if the mesh is antiperiodic. Whole repeats are taken off the angle
first, so the rotor slice never moves more than one slice from the stator
slice, and each meshed node of the rotor keeps the potential it would have
had without that. The potential at the meshed nodes, which come first, is
then a fair first guess at the next angle; the copies of rotor nodes that
follow them change in number with the angle. ``solve_angles`` steps through
a list of angles that way.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from polecraft.core.mesh import Mesh, join_band
from polecraft.core.problem import Problem
from polecraft.core.solver import Solution, solve
from polecraft.timing import Timings


def band_steps(mesh: Mesh, angle: float) -> tuple[int, float]:
    """How the band is joined for a rotor angle: steps, and the turn of the nodes.

    ``steps`` is the whole number of node spacings the rotor ring is moved on
    against the stator ring, and the second value the angle in radians the
    rotor's nodes are turned by. Both are for ``angle`` less whole repeats of the
    field, chosen so that the rotor slice stays within one slice of the
    stator slice.
    """
    band = mesh.band
    if band is None:
        raise ValueError("the mesh has no band, so its rotor cannot be turned")
    if not math.isfinite(angle):
        raise ValueError(f"the rotor angle must be a finite number of radians, not {angle}")
    divisions = band.divisions
    spacing = band.span / divisions
    # Node spacings in one repeat of the field.
    period = 2 * divisions if mesh.antiperiodic and not band.full else divisions
    # Taking whole repeats off first keeps the rounding of a large angle
    # out of what follows.
    angle = math.remainder(angle, period * spacing)
    nearest = round(angle / spacing)
    part = min(max(angle - nearest * spacing, -0.5 * spacing), 0.5 * spacing)
    # Whole repeats more or less, so that the rotor ring place facing the
    # first stator ring node is within half a repeat of where it was meshed.
    repeats = -((period - 2 * (nearest - band.facing)) // (2 * period))
    steps = nearest - repeats * period
    return steps, steps * spacing + part


def step_angle(mesh: Mesh) -> float:
    """The angle between neighbouring nodes of the band rings, in radians.

    Turning the rotor by a whole number of these moves every node of the
    rotor's ring onto the place of another, and no triangle changes shape.
    """
    if mesh.band is None:
        raise ValueError("the mesh has no band, so its rotor cannot be turned")
    return mesh.band.span / mesh.band.divisions


def rotate(mesh: Mesh, angle: float) -> Mesh:
    """The mesh with its rotor turned to an angle.

    ``angle`` is the rotor's mechanical angle in radians, counter-clockwise
    from the position the mesh was made in; it is not added to an angle the
    mesh has been turned to before. Any angle will do.

    The meshed nodes and triangles keep their numbers, and the stator and
    rotor triangles are the ones Gmsh made: nothing is meshed again. Only
    the band triangles, the copies of rotor nodes the band needs in a slice,
    and the periodic pairs of those copies are made anew; they are the last
    entries of their arrays, and the number of copies changes with the angle.

    The band triangles have the same shape at every angle. Between whole
    node spacings (``step_angle``) the rotor's triangles next to the band
    lean by up to half a spacing, which adds mesh noise to the torque; at
    whole spacings no triangle changes shape at all. The noise is small
    against the torque of a loaded machine but not against its cogging
    torque, which needs a fine mesh in any case.

    Raises ValueError if the mesh has no band or the angle is not finite.
    """
    steps, turn = band_steps(mesh, angle)
    return dataclasses.replace(join_band(mesh, steps, turn), rotor_angle=float(angle))


@dataclasses.dataclass(frozen=True, eq=False)
class Step:
    """The field at one rotor angle."""

    angle: float  # rotor angle, mechanical radians
    problem: Problem
    solution: Solution


@dataclasses.dataclass(frozen=True, eq=False)
class RotorRun:
    """The fields at a series of rotor angles, and the time they took."""

    steps: tuple[Step, ...]
    timings: Timings
    solver: str

    @property
    def angles(self) -> NDArray[np.float64]:
        """The rotor angles in mechanical radians, in the order solved."""
        return np.array([step.angle for step in self.steps])

    @property
    def iterations(self) -> tuple[int, ...]:
        """Newton iterations taken at each angle."""
        return tuple(step.solution.iterations for step in self.steps)

    def timing_report(self) -> str:
        """Time spent in each part of the run, added over all angles, as text."""
        heading = (
            f"Timing (solver: {self.solver}, {len(self.steps)} rotor angles, "
            f"{sum(self.iterations)} Newton iterations)"
        )
        return "\n".join([heading] + self.timings.report().splitlines()[1:])


def _start_from(previous: Solution, mesh: Mesh) -> NDArray[np.float64]:
    """A start for a solve on ``mesh`` from the solution at another angle.

    The meshed nodes come first and keep their numbers at every angle, so
    their potentials carry over. The copies after them differ in number;
    they are left at zero, and the solver sets each from its original.
    """
    meshed = len(mesh.band.rest)
    start = np.zeros(len(mesh.nodes))
    start[:meshed] = previous.potential[:meshed]
    return start


def solve_angles(
    mesh: Mesh,
    angles: ArrayLike | Sequence[float],
    problem_at: Callable[[Mesh], Problem],
    warm: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 30,
    solver: str | None = None,
) -> RotorRun:
    """Solve the field at each of a series of rotor angles, in order.

    ``angles`` are mechanical radians, as in ``rotate``. At each the mesh is
    turned and ``problem_at`` is given the turned mesh and returns the
    problem on it, for example
    ``lambda turned: build_problem(machine, turned, materials, current)``,
    which has the phase currents follow the rotor.

    With ``warm`` each solve starts from the field at the angle before,
    which saves Newton iterations when the steel saturates; the results are
    the same as from a cold start (zero), to within the tolerance. Neighbouring
    angles close together help most. ``tolerance``, ``max_iterations`` and
    ``solver`` are passed to ``solve`` for every angle.

    Raises ValueError if there are no angles, or if a problem is not on the
    turned mesh; ConvergenceError from ``solve`` as there.
    """
    angles = np.asarray(angles, dtype=float).ravel()
    if angles.size == 0:
        raise ValueError("give at least one rotor angle")
    timings = Timings()
    steps: list[Step] = []
    for angle in angles:
        started = time.perf_counter()
        turned = rotate(mesh, float(angle))
        built = time.perf_counter()
        problem = problem_at(turned)
        timings.add("turn rotor", built - started)
        timings.add("build problem", time.perf_counter() - built)
        nodes = np.asarray(problem.nodes)
        if nodes.shape != turned.nodes.shape:
            raise ValueError(
                f"the problem at {angle} rad has {len(nodes)} nodes, "
                f"the turned mesh {len(turned.nodes)}: build it on the turned mesh"
            )
        # A whole machine has the same number of nodes at every angle, so a
        # problem left on the unturned mesh shows only in where they are.
        if not np.array_equal(nodes, turned.nodes):
            raise ValueError(
                f"the nodes of the problem at {angle} rad are not where the "
                "turned mesh has them: build it on the turned mesh"
            )
        start = _start_from(steps[-1].solution, turned) if warm and steps else None
        solution = solve(
            problem, start, tolerance=tolerance, max_iterations=max_iterations, solver=solver
        )
        for name, seconds in solution.timings.stages:
            timings.add(name, seconds)
        steps.append(Step(float(angle), problem, solution))
    return RotorRun(steps=tuple(steps), timings=timings, solver=steps[-1].solution.solver)
