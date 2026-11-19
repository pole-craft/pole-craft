"""Loaded torque: torque over an electrical cycle with phase currents locked to the rotor.

At each rotor angle the machine's ``excitation`` gives three sinusoidal
phase currents that turn with the rotor, at a fixed advance angle ahead of
the back-EMF. The mean of the torque over the cycle is the torque the
motor gives, its peak-to-peak variation the ripple, and mean torque per
ampere the torque constant.

The torque at each angle is taken from the stress in the air gap. The mean
is also found a second, independent way, from the phase flux linkages:
torque is sum(i dpsi/dangle) less the rate of change of the stored magnetic
energy, and over a whole cycle the stored energy returns to where it
started, so the mean of sum(i dpsi/dangle) is the mean torque, with the
steel saturated or not. The two agree to a few tenths of a percent and come
closer with a finer mesh.

Values are those of the 2D model times the stack length, with no correction
for the ends of a short machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from polecraft.core.motion import RotorRun, solve_angles
from polecraft.core.post import flux_linkage, torque
from polecraft.core.problem import build_problem
from polecraft.studies._cycle import derivative

if TYPE_CHECKING:
    from polecraft.core.materials import Material
    from polecraft.core.mesh import Mesh
    from polecraft.machines.base import Machine


@dataclass(frozen=True, eq=False)
class LoadedTorque:
    """Torque against rotor angle over one electrical cycle, at one current."""

    angles: NDArray[np.float64]  # rotor angle, mechanical radians
    current: float  # peak phase current, A
    advance: float  # electrical radians ahead of the back-EMF
    pole_pairs: int
    torque: NDArray[np.float64]  # N m, whole machine, on the rotor, from the gap stress
    currents: dict[str, NDArray[np.float64]]  # A, each phase at each angle
    linkage: dict[str, NDArray[np.float64]]  # Wb, each phase at each angle
    run: RotorRun  # the solves, with the problem and field at each angle

    @property
    def mean(self) -> float:
        """Mean torque over the cycle, in N m."""
        return float(np.mean(self.torque))

    @property
    def peak_to_peak(self) -> float:
        """Largest less smallest torque over the cycle, in N m."""
        return float(np.ptp(self.torque))

    @property
    def ripple(self) -> float:
        """Torque ripple: peak-to-peak over the size of the mean (0.05 is 5 %).

        Not a useful measure where the mean is near zero, with the current
        nearly at right angles to the back-EMF; infinite where it is zero.
        """
        size = abs(self.mean)
        return self.peak_to_peak / size if size > 0.0 else math.inf

    @property
    def constant(self) -> float:
        """Torque constant: mean torque per ampere of peak phase current, N m/A."""
        return self.mean / self.current

    @property
    def mean_from_linkage(self) -> float:
        """Mean torque in N m found from the phase flux linkages instead of the gap stress."""
        cycle = 2.0 * math.pi / self.pole_pairs
        power = sum(
            self.currents[phase] * derivative(self.linkage[phase], cycle) for phase in self.linkage
        )
        return float(np.mean(power))

    def timing_report(self) -> str:
        """Time spent over all the solves, as text."""
        return self.run.timing_report()


def loaded_torque(
    machine: Machine,
    mesh: Mesh,
    materials: Mapping[str, Material],
    current: float,
    advance: float = 0.0,
    points: int = 48,
    solver: str | None = None,
) -> LoadedTorque:
    """Torque of a machine over one electrical cycle with its phase currents.

    ``current`` is the peak phase current in amperes and ``advance`` its
    angle ahead of the back-EMF in electrical radians, both as in
    ``Machine.excitation``; advance 0 gives the most torque per ampere in a
    surface-magnet machine. The field is solved at ``points`` rotor angles
    evenly spaced over the cycle from the angle the mesh was made at, each
    solve starting from the one before. Angles at whole node spacings of the
    band (see ``step_angle``) avoid its small mesh noise. ``materials`` are
    as for ``build_problem``; ``solver`` is as for ``solve``.

    Raises ValueError if ``current`` is zero or not finite (the cogging study
    is for no current), ``advance`` is not finite, or ``points`` is under 4.
    """
    if not math.isfinite(current) or current == 0.0:
        raise ValueError(f"current must be a finite number of amperes other than 0, not {current}")
    if not math.isfinite(advance):
        raise ValueError(f"advance must be a finite angle in radians, not {advance}")
    if points < 4:
        raise ValueError(f"points must be at least 4 a cycle, not {points}")
    layout = machine.symmetry()
    pole_pairs = layout.poles * layout.periods // 2
    angles = 2.0 * math.pi / pole_pairs * np.arange(points) / points
    run = solve_angles(
        mesh,
        angles,
        lambda turned: build_problem(machine, turned, materials, current=current, advance=advance),
        solver=solver,
    )
    linked = [flux_linkage(step.problem, step.solution) for step in run.steps]
    phases = list(linked[0])
    driven = [machine.excitation(float(angle), current, advance) for angle in angles]
    return LoadedTorque(
        angles=angles,
        current=float(current),
        advance=float(advance),
        pole_pairs=pole_pairs,
        torque=np.array([torque(step.problem, step.solution) for step in run.steps]),
        currents={phase: np.array([values[phase] for values in driven]) for phase in phases},
        linkage={phase: np.array([values[phase] for values in linked]) for phase in phases},
        run=run,
    )
