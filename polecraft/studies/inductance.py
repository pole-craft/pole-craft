"""d- and q-axis inductance from the change in flux linkage that a current makes.

The d axis is the magnet's axis and the q axis lies 90 electrical degrees
ahead of it, so that q current is the current that makes torque. Both are
taken from the machine's ``excitation``, which knows where its phase axes
are: the currents it gives at advance -90 degrees point along +d, those at
advance 0 along +q. Phase values (currents or flux linkages) become d and q
values by projecting them on those two patterns, which is the usual
amplitude-invariant Park transform: three phase currents of peak I at
advance a give i_d = -I sin(a) and i_q = I cos(a).

The field is solved three times over the rotor angles of one electrical
cycle: with no current, with current I on the -d axis and with current I on
the +q axis. At each angle

    Ld = (psi_d with d current - psi_d with none) / i_d
    Lq = (psi_q with q current - psi_q with none) / i_q

and the results are their means over the angles. Subtracting the no-current
linkage at the same angle, on the same mesh, takes out the magnet's flux and
most of the mesh's own error.

What these numbers are, and are not:

- They are apparent (secant) inductances, taken from no current to I along
  one axis at a time. With saturating steel they depend on I and, for the d
  axis, on its sign; they are not the incremental inductance a current
  controller sees at a working point. The d current is negative (it opposes
  the magnet), as when weakening the field at speed.
- A surface-magnet rotor looks the same along d and q, so Ld and Lq are close
  while the steel is far from saturation. Under load they part: q current
  saturates the teeth that the magnet flux already loads, which lowers Lq,
  and lowers the magnet's own d-axis linkage (``d_under_q``). Using the
  no-load magnet flux with these inductances overstates the torque of a
  heavily loaded machine; use the loaded flux linkages for that.
- They are per phase, with all coils of a phase in series, in the 2D model
  times the stack length. The end turns are added from ``end_winding``, an
  estimate good to some tens of percent (see
  ``polecraft.core.winding.end_turn_inductance``).
- An inductance is a difference of two linkages and settles more slowly with
  the mesh than torque does: on the reference motor mesh size 1 gives values
  about 1 % below those of a fine mesh.
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
from polecraft.core.winding import EndWinding, end_winding_inductance

if TYPE_CHECKING:
    from polecraft.core.materials import Material
    from polecraft.core.mesh import Mesh
    from polecraft.machines.base import Machine

# Advance angles, in electrical radians, of current along +d and +q.
D_AXIS, Q_AXIS = -0.5 * math.pi, 0.0
# How far the excitation's patterns may be from a balanced sinusoidal set.
BALANCE_TOLERANCE = 1e-9


def dq_components(
    machine: Machine, angle: float, values: Mapping[str, float]
) -> tuple[float, float]:
    """The d and q parts of phase values (currents in A, or linkages in Wb) at a rotor angle.

    ``angle`` is the rotor's mechanical angle in radians. The axes come from
    ``machine.excitation``. Raises ValueError if the phases of ``values`` are
    not those of the excitation, or if the excitation is not a balanced set
    of sinusoids, for which this transform would not hold.
    """
    along_d = machine.excitation(angle, 1.0, D_AXIS)
    along_q = machine.excitation(angle, 1.0, Q_AXIS)
    phases = sorted(along_d)
    if sorted(values) != phases:
        raise ValueError(
            f"values must be given for the phases {', '.join(phases)}, "
            f"not {', '.join(sorted(values))}"
        )
    d = np.array([along_d[phase] for phase in phases])
    q = np.array([along_q[phase] for phase in phases])
    half = len(phases) / 2.0
    if not (
        abs(d @ q) < BALANCE_TOLERANCE
        and abs(d @ d - half) < BALANCE_TOLERANCE
        and abs(q @ q - half) < BALANCE_TOLERANCE
    ):
        raise ValueError("the machine's excitation is not a balanced set of sinusoids")
    x = np.array([values[phase] for phase in phases], dtype=float)
    return float(x @ d / half), float(x @ q / half)


@dataclass(frozen=True, eq=False)
class Inductance:
    """d- and q-axis inductance of one phase over the rotor angles of one cycle.

    Inductances are in H and flux linkages in Wb, for one phase with all its
    coils in series, at each rotor angle in ``angles``.
    """

    angles: NDArray[np.float64]  # rotor angle, mechanical radians
    current: float  # peak phase current of the test, A: i_d = -current, then i_q = +current
    pole_pairs: int
    magnet_linkage: NDArray[np.float64]  # psi_d with no current
    d_axis_2d: NDArray[np.float64]  # Ld at each angle, 2D model times the stack length
    q_axis_2d: NDArray[np.float64]  # Lq at each angle, 2D model times the stack length
    d_under_q: NDArray[np.float64]  # psi_d with the q current: the magnet's flux under load
    q_under_d: NDArray[np.float64]  # psi_q with the d current
    end_winding: EndWinding | None  # None if the machine does not describe its coils
    runs: dict[str, RotorRun]  # the solves: "none", "d" and "q"

    @property
    def d_2d(self) -> float:
        """Mean Ld over the angles, from the 2D model alone, H."""
        return float(np.mean(self.d_axis_2d))

    @property
    def q_2d(self) -> float:
        """Mean Lq over the angles, from the 2D model alone, H."""
        return float(np.mean(self.q_axis_2d))

    @property
    def d(self) -> float:
        """Ld with the end turns' estimate added, H."""
        return self.d_2d + self._ends()

    @property
    def q(self) -> float:
        """Lq with the end turns' estimate added, H."""
        return self.q_2d + self._ends()

    @property
    def magnet_flux(self) -> float:
        """Mean magnet flux linkage of a phase with no current, Wb."""
        return float(np.mean(self.magnet_linkage))

    @property
    def saliency(self) -> float:
        """Lq over Ld in the 2D model; 1 for a rotor that looks the same both ways."""
        return self.q_2d / self.d_2d

    def _ends(self) -> float:
        # A self-inductance that every phase has alike, with no mutual part,
        # adds to d and q equally.
        return 0.0 if self.end_winding is None else self.end_winding.estimate

    def timing_report(self) -> str:
        """Time spent over the three sets of solves, as text."""
        return "\n".join(f"{name}:\n{run.timing_report()}" for name, run in self.runs.items())


def inductance(
    machine: Machine,
    mesh: Mesh,
    materials: Mapping[str, Material],
    current: float,
    points: int = 12,
    solver: str | None = None,
) -> Inductance:
    """d- and q-axis inductance of a machine at a test current.

    ``current`` is the peak phase current in amperes, as in
    ``Machine.excitation``. The field is solved with no current, with
    ``current`` along -d and with ``current`` along +q, at ``points`` rotor
    angles evenly spread over one electrical cycle, starting from the angle
    the mesh is turned to (``mesh.rotor_angle``; 0 for a mesh as built);
    points=1 gives the values at that angle alone. The dq values of
    a balanced machine vary with angle at multiples of six times the
    electrical frequency, so the mean of 12 points is exact for all but the
    12th harmonic and above. ``materials`` are as for ``build_problem``;
    ``solver`` is as for ``solve``.

    Raises ValueError if ``current`` is not a finite number above 0,
    ``points`` is under 1, or the coil's sizes cannot give an end-turn
    estimate; all of these are checked before any field is solved.
    """
    if not (math.isfinite(current) and current > 0.0):
        raise ValueError(f"current must be a finite number of amperes above 0, not {current}")
    if points < 1:
        raise ValueError(f"points must be at least 1, not {points}")
    # Before the solves, so that a coil it cannot handle costs no field solve.
    try:
        ends: EndWinding | None = end_winding_inductance(machine)
    except NotImplementedError:
        ends = None
    layout = machine.symmetry()
    pole_pairs = layout.poles * layout.periods // 2
    angles = mesh.rotor_angle + 2.0 * math.pi / pole_pairs * np.arange(points) / points
    tests = {"none": (0.0, Q_AXIS), "d": (current, -D_AXIS), "q": (current, Q_AXIS)}
    runs: dict[str, RotorRun] = {}
    linkage: dict[str, list[tuple[float, float]]] = {}
    applied: dict[str, list[tuple[float, float]]] = {}
    for name, (size, advance) in tests.items():
        run = solve_angles(
            mesh,
            angles,
            lambda turned, size=size, advance=advance: build_problem(
                machine, turned, materials, current=size, advance=advance
            ),
            solver=solver,
        )
        runs[name] = run
        linkage[name] = [
            dq_components(machine, float(angle), flux_linkage(step.problem, step.solution))
            for angle, step in zip(angles, run.steps)
        ]
        applied[name] = [
            dq_components(machine, float(angle), machine.excitation(float(angle), size, advance))
            for angle in angles
        ]
    none = np.array(linkage["none"])
    by_d = np.array(linkage["d"])
    by_q = np.array(linkage["q"])
    i_d = np.array(applied["d"])[:, 0]
    i_q = np.array(applied["q"])[:, 1]
    return Inductance(
        angles=angles,
        current=float(current),
        pole_pairs=pole_pairs,
        magnet_linkage=none[:, 0],
        d_axis_2d=(by_d[:, 0] - none[:, 0]) / i_d,
        q_axis_2d=(by_q[:, 1] - none[:, 1]) / i_q,
        d_under_q=by_q[:, 0],
        q_under_d=by_d[:, 1],
        end_winding=ends,
        runs=runs,
    )
