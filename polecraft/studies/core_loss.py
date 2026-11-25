"""Core loss: the loss in the steel from how the field in it changes over a cycle.

The field is solved at evenly spaced rotor angles over one electrical cycle,
with the phase currents following the rotor. Every steel triangle then has a
waveform of flux density, and its loss per kilogram is worked out from that
waveform with the steel's loss coefficients (see
``polecraft.core.post.core_loss_density``): hysteresis from the swing of the
field, eddy-current and excess loss from its rate of change. Each triangle's
loss is weighted by the mass of steel it stands for.

The stator's steel sees a field that repeats every electrical cycle. Steel
that turns with the rotor is followed in its own frame, where a steady
magnet field is constant and only the slots and the stator currents passing
by make loss. Its field does not, in general, repeat after one electrical
cycle (in a machine with a fractional number of slots per pole it repeats
after a different time), so its rates of change are taken as differences
between neighbouring angles. Summed over the rotor that still gives the
right eddy-current and excess loss, because the steel one pole pair further
on sees the same waveform one electrical cycle later, and together the
poles cover the whole of it. The rotor's hysteresis loss is taken from the
swing seen within the cycle solved, which is an estimate.

The field does not depend on speed in this model, so the loss at any other
speed follows from one run: hysteresis in proportion to the speed, eddy
current to its square, excess to its power 1.5 (``CoreLoss.at_speed``).

What the numbers are, and are not:

- The coefficients are fitted to losses measured on sheet in a sine-wave,
  alternating field. Stamping, stress from a press fit and turning fields
  raise the loss of a built machine above that, commonly by a few tens of
  percent, and no factor for it is applied.
- All steel is taken as laminated from the same sheet. A solid rotor yoke
  carries eddy currents this model does not have, and would lose more.
- Eddy currents in the magnets and in the winding are not included.
- The field is that of the solid 2D section. The mass is the section times
  the stack length times the stacking factor times the density; the flux
  density in the sheets is not raised for the stacking factor.
- No reaction of the eddy currents on the field: right while the sheet is
  thin against the skin depth, which the fitted frequency range reflects.
"""

from __future__ import annotations

import dataclasses
import math
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from polecraft.core.materials import CoreLossCoefficients, CoreLossData, LossRangeWarning
from polecraft.core.motion import RotorRun, solve_angles
from polecraft.core.post import core_loss_density, material_flux_density
from polecraft.core.problem import build_problem
from polecraft.core.solver import shape_gradients

if TYPE_CHECKING:
    from polecraft.core.materials import Material
    from polecraft.core.mesh import Mesh
    from polecraft.machines.base import Machine

# A triangle that has turned by more than this, in radians, is on the rotor.
TURNED = 1e-9


@dataclass(frozen=True, eq=False)
class CoreLoss:
    """Core loss of a machine at one speed and current, triangle by triangle.

    The arrays have one entry per steel triangle, in the order of
    ``triangles``. Losses are in watts for the whole machine.
    """

    speed: float  # mechanical rad/s
    pole_pairs: int
    current: float  # peak phase current, A
    advance: float  # electrical radians ahead of the back-EMF
    triangles: NDArray[np.intp]  # numbers of the steel triangles in the mesh
    mass: NDArray[np.float64]  # kg of steel each stands for in the whole machine
    turning: NDArray[np.bool_]  # whether the triangle turns with the rotor
    peak: NDArray[np.float64]  # T, largest flux density over the cycle
    beyond_fit: NDArray[np.bool_]  # peak above the range the coefficients were fitted over
    hysteresis_density: NDArray[np.float64]  # W/kg
    eddy_density: NDArray[np.float64]  # W/kg
    excess_density: NDArray[np.float64]  # W/kg
    fitted_frequencies: tuple[float, float] | None  # Hz, range common to the coefficients
    run: RotorRun  # the solves, with the problem and field at each angle

    @property
    def frequency(self) -> float:
        """Electrical frequency in Hz: pole pairs times turns per second."""
        return self.pole_pairs * abs(self.speed) / (2.0 * math.pi)

    @property
    def density(self) -> NDArray[np.float64]:
        """Loss in each steel triangle, in W/kg."""
        return self.hysteresis_density + self.eddy_density + self.excess_density

    @property
    def hysteresis(self) -> float:
        """Hysteresis loss in watts."""
        return float(self.mass @ self.hysteresis_density)

    @property
    def eddy(self) -> float:
        """Eddy-current loss in the laminations, in watts."""
        return float(self.mass @ self.eddy_density)

    @property
    def excess(self) -> float:
        """Excess loss in watts."""
        return float(self.mass @ self.excess_density)

    @property
    def total(self) -> float:
        """Core loss of the whole machine in watts."""
        return self.hysteresis + self.eddy + self.excess

    @property
    def stator(self) -> float:
        """Loss in the steel that stands still, in watts."""
        return float(self.mass[~self.turning] @ self.density[~self.turning])

    @property
    def rotor(self) -> float:
        """Loss in the steel that turns with the rotor, in watts."""
        return float(self.mass[self.turning] @ self.density[self.turning])

    @property
    def share_beyond_fit(self) -> float:
        """Part of the loss that comes from steel driven above the fitted flux density."""
        total = self.total
        if total == 0.0:
            return 0.0
        return float(self.mass[self.beyond_fit] @ self.density[self.beyond_fit]) / total

    def at_speed(self, speed: float) -> CoreLoss:
        """The loss at another speed in mechanical rad/s, from the same fields.

        Hysteresis loss goes with the speed, eddy-current loss with its
        square and excess loss with its power 1.5. Warns, like the study,
        where the frequency leaves the range the coefficients were fitted
        over.
        """
        _check_speed(speed)
        ratio = abs(speed / self.speed)
        scaled = dataclasses.replace(
            self,
            speed=float(speed),
            hysteresis_density=self.hysteresis_density * ratio,
            eddy_density=self.eddy_density * ratio**2,
            excess_density=self.excess_density * ratio**1.5,
        )
        scaled._warn_outside_fit()
        return scaled

    def _warn_outside_fit(self) -> None:
        if self.fitted_frequencies is None:
            return
        low, high = self.fitted_frequencies
        margin = 1e-9
        if not low * (1.0 - margin) <= self.frequency <= high * (1.0 + margin):
            warnings.warn(
                f"core loss at {self.frequency:g} Hz, outside the {low:g} to {high:g} Hz the "
                "loss coefficients were fitted over; the value is an extrapolation",
                LossRangeWarning,
                stacklevel=3,
            )

    def timing_report(self) -> str:
        """Time spent over all the solves, as text."""
        return self.run.timing_report()


def _check_speed(speed: float) -> None:
    if not math.isfinite(speed) or speed == 0.0:
        raise ValueError(f"speed must be a finite number of rad/s other than 0, not {speed}")


def core_loss(
    machine: Machine,
    mesh: Mesh,
    materials: Mapping[str, Material],
    losses: Mapping[str, CoreLossData],
    speed: float,
    current: float = 0.0,
    advance: float = 0.0,
    points: int = 48,
    stacking_factor: float | None = None,
    coefficients: Mapping[str, CoreLossCoefficients] | None = None,
    solver: str | None = None,
) -> CoreLoss:
    """Core loss of a machine over one electrical cycle.

    ``speed`` is the rotor's mechanical speed in rad/s (rpm times pi / 30).
    ``current`` is the peak phase current in amperes and ``advance`` its
    angle ahead of the back-EMF in electrical radians, as in
    ``Machine.excitation``; with no current the result is the no-load loss.
    ``materials`` are as for ``build_problem``. ``losses`` gives the loss
    data of each steel by the same grade names (``load_core_loss``); a
    grade left out is taken to have no loss. The coefficients are fitted to
    the data with ``CoreLossData.fit`` unless ``coefficients`` gives them
    for a grade.

    The field is solved at ``points`` rotor angles in the cycle and once
    more at its end, each solve starting from the one before.
    ``stacking_factor`` is the part of the stack length that is steel; left
    out, it is the design's. ``solver`` is as for ``solve``.

    Warns with ``LossRangeWarning`` where the electrical frequency is
    outside the range the coefficients were fitted over.

    Raises ValueError if ``speed`` is zero or not finite, ``points`` is
    under 8, a grade in ``losses`` has no model in ``materials`` or is not
    in the machine, or two grades share one model but not their loss data.
    """
    _check_speed(speed)
    if not (math.isfinite(current) and math.isfinite(advance)):
        raise ValueError("current and advance must be finite numbers")
    if points < 8:
        raise ValueError(f"points must be at least 8 a cycle, not {points}")
    if stacking_factor is None:
        stacking_factor = machine.design.steel.stacking_factor
    if not (math.isfinite(stacking_factor) and 0.0 < stacking_factor <= 1.0):
        raise ValueError(f"stacking factor must be above 0 and at most 1, not {stacking_factor}")
    layout = machine.symmetry()
    pole_pairs = layout.poles * layout.periods // 2
    cycle = 2.0 * math.pi / pole_pairs  # mechanical radians
    duration = cycle / abs(speed)  # seconds
    angles = cycle * np.arange(points + 1) / points
    run = solve_angles(
        mesh,
        angles,
        lambda turned: build_problem(machine, turned, materials, current=current, advance=advance),
        solver=solver,
    )
    started = time.perf_counter()
    first = run.steps[0].problem
    steps = [(step.problem, step.solution) for step in run.steps]
    _, area = shape_gradients(first.nodes, first.triangles)

    parts = []
    used: dict[int, CoreLossData] = {}
    fitted: list[tuple[float, float]] = []
    for name, data in losses.items():
        if name not in materials:
            raise ValueError(f"no material model for '{name}', which has loss data")
        found = [n for n, model in enumerate(first.materials) if model is materials[name]]
        if not found:
            raise ValueError(f"'{name}' has loss data but is not a material of this machine")
        if found[0] in used:
            if used[found[0]] is not data:
                raise ValueError(
                    f"'{name}' shares its material model with another grade but not its loss data"
                )
            continue
        used[found[0]] = data
        fit = (coefficients or {}).get(name) or data.fit()
        if fit.frequency_range is not None:
            fitted.append(fit.frequency_range)
        triangles = np.flatnonzero(first.triangle_material == found[0])
        field, turn = material_flux_density(steps, triangles)
        turning = np.abs(turn).max(axis=0) > TURNED
        terms = np.zeros((3, triangles.size))
        # Steel that stands still closes its cycle: the last angle repeats the first.
        terms[:, ~turning] = core_loss_density(field[:-1, ~turning], duration, fit)
        terms[:, turning] = core_loss_density(field[:, turning], duration, fit, periodic=False)
        peak = np.hypot(field[..., 0], field[..., 1]).max(axis=0)
        limit = math.inf if fit.max_polarisation is None else fit.max_polarisation
        mass = area[triangles] * first.depth * first.repeats * stacking_factor * data.density
        parts.append((triangles, mass, turning, peak, peak > limit, *terms))
    if not parts:
        raise ValueError("give the loss data of at least one steel")
    joined = [np.concatenate(column) for column in zip(*parts)]
    order = np.argsort(joined[0])
    triangles, mass, turning, peak, beyond, hysteresis, eddy, excess = (
        column[order] for column in joined
    )
    run.timings.add("core loss", time.perf_counter() - started)
    result = CoreLoss(
        speed=float(speed),
        pole_pairs=pole_pairs,
        current=float(current),
        advance=float(advance),
        triangles=triangles,
        mass=mass,
        turning=turning,
        peak=peak,
        beyond_fit=beyond,
        hysteresis_density=hysteresis,
        eddy_density=eddy,
        excess_density=excess,
        fitted_frequencies=(
            (max(low for low, _ in fitted), min(high for _, high in fitted)) if fitted else None
        ),
        run=run,
    )
    result._warn_outside_fit()
    return result
