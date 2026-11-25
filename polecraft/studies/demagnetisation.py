"""Demagnetisation: the magnets' working point against the knee of their curve.

A sintered NdFeB magnet works on its straight recoil line, B = mu_r mu0 H +
Br, as long as the field opposing its magnetisation stays below the knee
of its demagnetisation curve, Hk. Beyond the knee the polarisation falls,
and when the field goes away the magnet comes back on a lower recoil line:
part of its flux is lost for good. Hk follows the intrinsic coercivity Hcj,
which falls quickly with temperature, so a magnet that is safe cold can be
damaged hot by a current it carried before.

The study solves the field over one electrical cycle with the magnet at its
temperature (its remanence lowered by its coefficient) and the phase
currents locked to the rotor. In every magnet triangle it finds the most
demagnetising field along the magnetisation over the cycle, and compares it
with Hk at that temperature. The magnet's own field already opposes it
(more so near the gap and at its edges); current along -d (advance +90
degrees, as in field weakening or a short circuit) adds the most.

What the check is, and is not:

- The solve stays on the recoil line. Where a triangle passes the knee the
  field there is no longer right, and nor is anything after it; the check
  says that the knee is reached, not how much flux is lost.
- The field is constant over a first-order triangle. At the sharp corners
  of a magnet the field is highest and the mesh decides how high, so the
  worst single triangle depends on the mesh. ``share_beyond`` gives how much
  of the magnet passes the knee, which settles with a finer mesh.
- Hk / Hcj is taken as the same at every temperature, and Hcj as linear in
  temperature; both are approximations of a datasheet's curves.
- 2D: the ends of the magnets, where the field differs, are not modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from polecraft.core.materials import MU0, REFERENCE_TEMPERATURE, Magnet
from polecraft.core.motion import RotorRun, solve_angles
from polecraft.core.post import magnet_field
from polecraft.core.problem import build_problem
from polecraft.core.solver import shape_gradients

if TYPE_CHECKING:
    from polecraft.core.materials import Material
    from polecraft.core.mesh import Mesh
    from polecraft.machines.base import Machine


@dataclass(frozen=True, eq=False)
class Demagnetisation:
    """The most demagnetising field in each magnet triangle over a cycle, against Hk.

    Arrays have one entry per magnet triangle, in the order of
    ``triangles``. Fields are in A/m, counted positive when they oppose the
    magnetisation.
    """

    current: float  # A, peak phase
    advance: float  # electrical radians ahead of the back-EMF
    temperature: float  # degrees Celsius of the magnets
    remanence: float  # T, Br at that temperature
    knee: float  # A/m, Hk at that temperature
    triangles: NDArray[np.intp]  # numbers of the magnet triangles in the mesh
    area: NDArray[np.float64]  # m^2, of each triangle
    field: NDArray[np.float64]  # A/m, most demagnetising field over the cycle
    flux_density: NDArray[np.float64]  # T, lowest flux density along the magnetisation
    run: RotorRun  # the solves, with the problem and field at each angle

    @property
    def worst(self) -> float:
        """Most demagnetising field anywhere in the magnets over the cycle, A/m."""
        return float(self.field.max())

    @property
    def margin(self) -> float:
        """How far the worst field is from the knee, as a part of Hk; below 0 past it."""
        return 1.0 - self.worst / self.knee

    @property
    def beyond(self) -> NDArray[np.bool_]:
        """Whether each triangle passes the knee at some angle of the cycle."""
        return self.field > self.knee

    @property
    def share_beyond(self) -> float:
        """Part of the magnet area that passes the knee: 0.01 is 1 %."""
        return float(self.area[self.beyond].sum() / self.area.sum())

    @property
    def flagged(self) -> bool:
        """Whether any part of the magnets passes the knee."""
        return bool(self.beyond.any())

    @property
    def knee_flux_density(self) -> float:
        """Flux density along the magnetisation at the knee, on the recoil line, T.

        Br - mu_r mu0 Hk. Below 0 when the knee lies in the third quadrant,
        as it does for a cold NdFeB magnet.
        """
        return float(self.remanence - self._permeability * MU0 * self.knee)

    @property
    def _permeability(self) -> float:
        problem = self.run.steps[0].problem
        model = problem.materials[problem.triangle_material[self.triangles[0]]]
        return float(model.relative_permeability)

    def timing_report(self) -> str:
        """Time spent over all the solves, as text."""
        return self.run.timing_report()


def demagnetisation(
    machine: Machine,
    mesh: Mesh,
    materials: Mapping[str, Material],
    current: float,
    advance: float = 0.0,
    temperature: float | None = None,
    knee_ratio: float | None = None,
    points: int = 24,
    solver: str | None = None,
) -> Demagnetisation:
    """Check a machine's magnets against demagnetisation at a current and temperature.

    ``current`` is the peak phase current in amperes and ``advance`` its
    angle ahead of the back-EMF in electrical radians, as in
    ``Machine.excitation``: advance pi/2 puts it all along -d, which
    demagnetises most. ``materials`` are as for ``build_problem``, with the
    design's magnet grade given at 20 C (a library entry) with its
    remanence coefficient, its intrinsic coercivity and the coefficient of
    that; the study takes it to ``temperature`` (degrees Celsius, the
    design's magnet temperature unless given). ``knee_ratio`` is Hk / Hcj
    from the grade's datasheet, unless the grade carries it. The field is
    solved at ``points`` rotor angles of one electrical cycle.

    Raises ValueError for a current or advance that is not finite, points
    under 1, a magnet grade with no model, not a ``Magnet``, not at 20 C, or
    without the data the check needs.
    """
    if not (math.isfinite(current) and math.isfinite(advance)):
        raise ValueError("current and advance must be finite numbers")
    if points < 1:
        raise ValueError(f"points must be at least 1, not {points}")
    grade = machine.design.magnet.material
    magnet = materials.get(grade)
    if not isinstance(magnet, Magnet):
        raise ValueError(f"the magnet grade '{grade}' needs a Magnet model in materials")
    if magnet.temperature != REFERENCE_TEMPERATURE:
        raise ValueError(
            f"give the magnet at {REFERENCE_TEMPERATURE:g} C, as in the library; "
            f"the study takes it to its temperature"
        )
    if temperature is None:
        temperature = machine.design.magnet.temperature_C
    knee = magnet.knee_field(temperature, knee_ratio)
    hot = magnet if temperature == REFERENCE_TEMPERATURE else magnet.at_temperature(temperature)
    warm = {**materials, grade: hot}
    layout = machine.symmetry()
    pole_pairs = layout.poles * layout.periods // 2
    angles = mesh.rotor_angle + 2.0 * math.pi / pole_pairs * np.arange(points) / points
    run = solve_angles(
        mesh,
        angles,
        lambda turned: build_problem(machine, turned, warm, current=current, advance=advance),
        solver=solver,
    )
    found = [magnet_field(step.problem, step.solution) for step in run.steps]
    triangles = found[0][0]
    if triangles.size == 0:
        raise ValueError("the machine has no magnetised triangles to check")
    # The magnets turn with the rotor; their triangles keep their numbers.
    field = -np.min([along_h for _, along_h, _ in found], axis=0)
    flux_density = np.min([along_b for _, _, along_b in found], axis=0)
    first = run.steps[0].problem
    _, area = shape_gradients(first.nodes, first.triangles[triangles])
    return Demagnetisation(
        current=float(current),
        advance=float(advance),
        temperature=float(temperature),
        remanence=hot.remanence,
        knee=knee,
        triangles=triangles,
        area=area,
        field=field,
        flux_density=flux_density,
        run=run,
    )
