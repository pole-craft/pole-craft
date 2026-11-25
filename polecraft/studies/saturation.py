"""Torque against current: how saturation bends the torque at high current.

At each peak phase current the mean torque over one electrical cycle is
found as in ``polecraft.studies.loaded_torque``, at a fixed advance angle.
While the steel is far from saturation the torque rises in proportion to
the current, and the torque constant (torque per ampere) is that of the
magnet's flux, 1.5 times the back-EMF constant at advance 0. As the
current grows, its own field saturates the teeth, the flux that crosses
the gap no longer grows with it, and the torque constant falls. A QDD
motor is often driven several times over its continuous current for short
peaks, so this fall decides the peak torque.

Values are those of the 2D model times the stack length, with no correction
for the ends of a short machine. The advance is held fixed; with
saturation, the angle that gives the most torque per ampere moves a little
past 0 at high current (see ``polecraft.studies.flux_map`` to find it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from polecraft.studies.loaded_torque import LoadedTorque, loaded_torque
from polecraft.timing import Timings

if TYPE_CHECKING:
    from polecraft.core.materials import Material
    from polecraft.core.mesh import Mesh
    from polecraft.machines.base import Machine


@dataclass(frozen=True, eq=False)
class TorqueAgainstCurrent:
    """Mean torque at a series of peak phase currents, at one advance angle."""

    currents: NDArray[np.float64]  # A, peak phase, rising
    advance: float  # electrical radians ahead of the back-EMF
    torque: NDArray[np.float64]  # N m, mean over the cycle
    peak_to_peak: NDArray[np.float64]  # N m, torque ripple at each current
    runs: tuple[LoadedTorque, ...]  # the loaded-torque study at each current

    @property
    def constant(self) -> NDArray[np.float64]:
        """Torque constant at each current: mean torque per ampere, N m/A."""
        return self.torque / self.currents

    @property
    def incremental_constant(self) -> NDArray[np.float64]:
        """Torque gained per extra ampere between neighbouring currents, N m/A.

        One value per step, between currents n and n + 1.
        """
        return np.diff(self.torque) / np.diff(self.currents)

    @property
    def fall(self) -> NDArray[np.float64]:
        """How far the torque constant has fallen from that at the lowest current.

        0 where it is the same, 0.2 where it is 20 % lower.
        """
        return 1.0 - self.constant / self.constant[0]

    def timing_report(self) -> str:
        """Time spent over all the solves, as text."""
        timings = Timings()
        steps = 0
        for run in self.runs:
            steps += len(run.run.steps)
            for name, seconds in run.run.timings.stages:
                timings.add(name, seconds)
        heading = f"Timing ({len(self.runs)} currents, {steps} rotor angles)"
        return "\n".join([heading] + timings.report().splitlines()[1:])


def torque_against_current(
    machine: Machine,
    mesh: Mesh,
    materials: Mapping[str, Material],
    currents: ArrayLike,
    advance: float = 0.0,
    points: int = 24,
    solver: str | None = None,
) -> TorqueAgainstCurrent:
    """Mean torque of a machine at each of a series of peak phase currents.

    ``currents`` are peak phase currents in amperes, above 0 and rising, up
    to the drive's peak; ``advance`` is the current's angle ahead of the
    back-EMF in electrical radians, as in ``Machine.excitation``. Each
    current is solved at ``points`` rotor angles of one electrical cycle;
    the torque of the example motors has a strong 12th harmonic, which
    fewer than 24 points take into the mean. ``materials`` are as for
    ``build_problem``, ``solver`` as for ``solve``.

    Raises ValueError for currents that are not finite, above 0 and rising,
    or as ``loaded_torque`` does.
    """
    values = np.asarray(currents, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("currents must be a list of at least one current")
    if not (np.all(np.isfinite(values)) and np.all(values > 0.0)):
        raise ValueError("currents must be finite numbers of amperes above 0")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("currents must rise from one to the next")
    if not math.isfinite(advance):
        raise ValueError(f"advance must be a finite angle in radians, not {advance}")
    runs = tuple(
        loaded_torque(machine, mesh, materials, float(i), advance, points, solver) for i in values
    )
    return TorqueAgainstCurrent(
        currents=values,
        advance=float(advance),
        torque=np.array([run.mean for run in runs]),
        peak_to_peak=np.array([run.peak_to_peak for run in runs]),
        runs=runs,
    )
