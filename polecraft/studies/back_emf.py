"""Back-EMF: the voltage the magnets induce in each phase as the rotor turns.

With no current, the flux linked by each phase is solved at evenly spaced
rotor angles over whole electrical cycles. The back-EMF is its rate of
change in time, e = d(psi)/dt = speed * d(psi)/d(angle), with the motor
convention v = R i + d(psi)/dt: current in step with e gives motoring torque.

The field does not depend on speed (no eddy currents in a 2D magnetostatic
solve), so the EMF is exactly proportional to it. The values are those of
the 2D model times the stack length, with no correction for the ends of a
short machine.

The rate of change is taken from the harmonics of the samples (they cover
whole cycles), which is exact for every harmonic below half the number of
samples a cycle. Harmonics above that alias onto lower ones; 48 samples a
cycle are enough for the waveforms of surface-magnet machines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from polecraft.core.motion import RotorRun, solve_angles
from polecraft.core.post import flux_linkage
from polecraft.core.problem import build_problem
from polecraft.studies._cycle import derivative, fundamental

if TYPE_CHECKING:
    from polecraft.core.materials import Material
    from polecraft.core.mesh import Mesh
    from polecraft.machines.base import Machine


@dataclass(frozen=True, eq=False)
class BackEmf:
    """Phase flux linkage and back-EMF against rotor angle, at one speed."""

    angles: NDArray[np.float64]  # rotor angle, mechanical radians
    speed: float  # mechanical rad/s
    pole_pairs: int
    cycles: int  # whole electrical cycles the angles cover
    linkage: dict[str, NDArray[np.float64]]  # Wb, each phase with all its coils in series
    emf: dict[str, NDArray[np.float64]]  # V, phase back-EMF at each angle
    run: RotorRun  # the solves, with the problem and field at each angle

    @property
    def times(self) -> NDArray[np.float64]:
        """Time at each angle in seconds, from angle 0 at this speed."""
        return self.angles / self.speed

    @property
    def frequency(self) -> float:
        """Electrical frequency in Hz: pole pairs times turns per second."""
        return self.pole_pairs * abs(self.speed) / (2.0 * math.pi)

    @property
    def amplitude(self) -> dict[str, float]:
        """Peak of the fundamental of each phase's back-EMF, in volts."""
        return {phase: fundamental(values, self.cycles) for phase, values in self.emf.items()}

    @property
    def constant(self) -> float:
        """Back-EMF constant in V s/rad: phase fundamental peak per mechanical rad/s.

        The mean over the phases of pole pairs times the fundamental of the
        linkage, so it does not depend on the speed the study was run at.
        """
        values = [fundamental(linked, self.cycles) for linked in self.linkage.values()]
        return self.pole_pairs * float(np.mean(values))

    def timing_report(self) -> str:
        """Time spent over all the solves, as text."""
        return self.run.timing_report()


def back_emf(
    machine: Machine,
    mesh: Mesh,
    materials: Mapping[str, Material],
    speed: float,
    points: int = 48,
    cycles: int = 1,
    solver: str | None = None,
) -> BackEmf:
    """Phase flux linkage and back-EMF of a machine over whole electrical cycles.

    ``speed`` is the rotor's mechanical speed in rad/s (rpm times pi / 30),
    counter-clockwise positive. The field is solved with no current at
    ``points`` rotor angles in each electrical cycle, evenly spaced from the
    angle the mesh was made at, each solve starting from the one before.
    Angles at whole node spacings of the band (see ``step_angle``) avoid
    its small mesh noise. ``materials`` are as for ``build_problem``;
    ``solver`` is as for ``solve``.

    Raises ValueError if ``points`` is under 4, ``cycles`` under 1 or
    ``speed`` zero or not finite.
    """
    if points < 4:
        raise ValueError(f"points must be at least 4 a cycle, not {points}")
    if cycles < 1:
        raise ValueError(f"cycles must be at least 1, not {cycles}")
    if not math.isfinite(speed) or speed == 0.0:
        raise ValueError(f"speed must be a finite number of rad/s other than 0, not {speed}")
    layout = machine.symmetry()
    pole_pairs = layout.poles * layout.periods // 2
    cycle = 2.0 * math.pi / pole_pairs  # mechanical radians
    angles = cycle * np.arange(points * cycles) / points
    run = solve_angles(
        mesh, angles, lambda turned: build_problem(machine, turned, materials), solver=solver
    )
    linked = [flux_linkage(step.problem, step.solution) for step in run.steps]
    linkage = {phase: np.array([values[phase] for values in linked]) for phase in linked[0]}
    emf = {
        phase: speed * derivative(values, cycle * cycles) for phase, values in linkage.items()
    }
    return BackEmf(
        angles=angles,
        speed=float(speed),
        pole_pairs=pole_pairs,
        cycles=cycles,
        linkage=linkage,
        emf=emf,
        run=run,
    )
