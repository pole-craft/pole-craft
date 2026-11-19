"""Cogging torque: the torque of the magnets alone as the rotor turns.

With no current, torque is minus the rate of change of the stored magnetic
energy with rotor angle. The energy repeats whenever the rotor has turned so
that magnets again face the slots as before, every 360 / LCM(slots, poles)
mechanical degrees, so cogging torque has that period and averages to zero
over it.

Cogging torque is small against the torque of a loaded machine, and so is
hard to resolve. It needs node spacings around the air gap much finer than
the period: with fewer than ``FEW_SPACINGS`` a period the waveform is mostly
mesh noise, and a warning says so. Even above that, check the result against
a finer mesh (``build_mesh(..., size=...)``). Where the cogging torque is a
thousandth of the rated torque or less, as on a motor with many slots and
poles, its period and mean come out right but its size may not converge: on
the 36-slot 42-pole example it moved between 1.8 and 2.4 mN m over mesh
sizes 0.5 to 0.2, without settling. Take such a value as an order of
magnitude.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from polecraft.core.motion import RotorRun, solve_angles, step_angle
from polecraft.core.post import torque
from polecraft.core.problem import build_problem
from polecraft.core.winding import symmetry

if TYPE_CHECKING:
    from polecraft.core.materials import Material
    from polecraft.core.mesh import Mesh
    from polecraft.machines.base import Machine

# Below this many node spacings of the band in a cogging period the computed
# waveform is mostly mesh noise (measured on the example designs).
FEW_SPACINGS = 10


class CoarseMeshWarning(UserWarning):
    """The mesh is too coarse around the air gap for the cogging period."""


def cogging_period(slots: int, poles: int) -> float:
    """The cogging period, 2 pi / LCM(slots, poles), in mechanical radians."""
    symmetry(slots, poles)  # checks the counts
    return 2.0 * math.pi / math.lcm(slots, poles)


@dataclass(frozen=True, eq=False)
class Cogging:
    """Cogging torque against rotor angle."""

    angles: NDArray[np.float64]  # rotor angle, mechanical radians
    torque: NDArray[np.float64]  # N m, whole machine, on the rotor
    period: float  # cogging period, mechanical radians
    spacings_per_period: float  # node spacings of the band in one period
    run: RotorRun  # the solves, with the problem and field at each angle

    @property
    def peak_to_peak(self) -> float:
        """Largest less smallest torque, in N m."""
        return float(np.ptp(self.torque))

    @property
    def mean(self) -> float:
        """Mean torque in N m; near zero over whole periods."""
        return float(np.mean(self.torque))

    def timing_report(self) -> str:
        """Time spent over all the solves, as text."""
        return self.run.timing_report()


def cogging_torque(
    machine: Machine,
    mesh: Mesh,
    materials: Mapping[str, Material],
    points: int = 24,
    periods: int = 1,
    solver: str | None = None,
) -> Cogging:
    """Cogging torque of a machine over whole cogging periods.

    The field is solved with no current at ``points`` rotor angles in each
    period, evenly spaced from the angle the mesh was made at, the last one
    a step short of the end. Each solve starts from the one before.
    ``materials`` are as for ``build_problem``; ``solver`` is as for ``solve``.

    Warns with ``CoarseMeshWarning`` if the band has fewer than
    ``FEW_SPACINGS`` node spacings in a period. Raises ValueError if
    ``points`` is under 4 or ``periods`` under 1.
    """
    if points < 4:
        raise ValueError(f"points must be at least 4 a period, not {points}")
    if periods < 1:
        raise ValueError(f"periods must be at least 1, not {periods}")
    layout = machine.symmetry()
    period = cogging_period(layout.slots * layout.periods, layout.poles * layout.periods)
    spacings = period / step_angle(mesh)
    if spacings < FEW_SPACINGS - 1e-9:
        warnings.warn(
            f"the mesh has only {spacings:.3g} node spacings around the air gap in a "
            f"cogging period; with fewer than {FEW_SPACINGS} the cogging torque is mostly "
            "mesh noise. Use a finer mesh (build_mesh size).",
            CoarseMeshWarning,
            stacklevel=2,
        )
    angles = period * np.arange(points * periods) / points
    run = solve_angles(
        mesh, angles, lambda turned: build_problem(machine, turned, materials), solver=solver
    )
    return Cogging(
        angles=angles,
        torque=np.array([torque(step.problem, step.solution) for step in run.steps]),
        period=period,
        spacings_per_period=spacings,
        run=run,
    )
