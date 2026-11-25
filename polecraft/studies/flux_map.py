"""Flux and torque maps over d- and q-axis current.

The field is solved once at each point of a grid of d and q currents, over
the rotor angles of one electrical cycle, and the means of the d and q flux
linkages and of the torque are kept. Any operating point inside the grid is
then interpolated, not solved again; that is what makes torque-speed curves
and efficiency maps fast.

Currents and flux linkages are amplitude-invariant d and q values, as in
``polecraft.studies.inductance``: three phase currents of peak I at advance
a are i_d = -I sin(a), i_q = I cos(a). In these terms

    torque = 1.5 p (psi_d i_q - psi_q i_d)
    v_d = R i_d - w psi_q,  v_q = R i_q + w psi_d  (w electrical rad/s)

The torque is kept twice: from the stress in the air gap (``torque``) and
from the formula above with the mean linkages (``torque_from_linkage``).
Over a whole cycle the two are the same mean torque, so their agreement
checks the solves; they differ by the mesh's error and by harmonics that
the angles sampled do not resolve.

What the maps are, and are not:

- Means over the cycle. The torque ripple and the ripple of the linkages
  are not kept. The angles are evenly spaced, so a harmonic of the torque
  at a multiple of ``points`` times the electrical frequency shows in the
  mean as if it were steady. The example motors have a strong 12th
  harmonic: with 12 points the mean torque of the 12-slot, 10-pole motor
  is up to 1 % off at 40 A, with 24 points 0.1 %. Hence 24 by default.
- The 2D model times the stack length: no end-turn inductance and no
  correction for the ends of a short machine. ``Inductance.end_winding``
  gives the end-turn part; it adds L_end i_d to psi_d and L_end i_q to
  psi_q and leaves the torque alone.
- Inside the grid the values are bicubic splines through the solved points
  (lower order with fewer than four points on an axis). Outside it there is
  no value: the steel's saturation cannot be guessed by extrapolation.

Given the steel's loss data, the map also holds the core loss at each grid
point (see ``polecraft.studies.core_loss``), from the same solves and one
more at the end of the cycle. The field does not depend on speed, so the
loss is kept as three numbers, h, e and x, that give it at any mechanical
speed w as h w + e w^2 + x w^1.5 (hysteresis, eddy current, excess).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RectBivariateSpline

from polecraft.core.motion import solve_angles
from polecraft.core.post import flux_linkage, torque
from polecraft.core.problem import build_problem
from polecraft.studies.core_loss import core_loss_of_run
from polecraft.studies.inductance import dq_components
from polecraft.timing import Timings

if TYPE_CHECKING:
    from polecraft.core.materials import CoreLossCoefficients, CoreLossData, Material
    from polecraft.core.mesh import Mesh
    from polecraft.machines.base import Machine

# How far past the edge of the grid, as a part of its span, still counts as on it.
EDGE = 1e-9


def current_and_advance(i_d: float, i_q: float) -> tuple[float, float]:
    """Peak phase current in A and advance in electrical radians for d and q currents."""
    size = math.hypot(i_d, i_q)
    # Adding 0 turns -0.0 into 0.0, so that negative q current is at +pi, not -pi.
    return size, (math.atan2(0.0 - i_d, i_q) if size > 0.0 else 0.0)


@dataclass(frozen=True, eq=False)
class OperatingPoint:
    """Cycle-mean flux linkages and torque at one pair of d and q currents."""

    d_current: float  # A
    q_current: float  # A
    d_linkage: float  # Wb
    q_linkage: float  # Wb
    torque: float  # N m, from the air-gap stress
    # Core loss of the whole machine at 1 mechanical rad/s, by part, in W;
    # None without loss data. At speed w: hysteresis w + eddy w^2 + excess w^1.5.
    hysteresis: float | None = None
    eddy: float | None = None
    excess: float | None = None
    # Frequency range in Hz the loss coefficients were fitted over, if any.
    fitted_frequencies: tuple[float, float] | None = None


def operating_point(
    machine: Machine,
    mesh: Mesh,
    materials: Mapping[str, Material],
    d_current: float,
    q_current: float,
    points: int = 24,
    solver: str | None = None,
    timings: Timings | None = None,
    losses: Mapping[str, CoreLossData] | None = None,
    coefficients: Mapping[str, CoreLossCoefficients] | None = None,
    stacking_factor: float | None = None,
) -> OperatingPoint:
    """Solve one operating point: means over ``points`` rotor angles of a cycle.

    The angles are evenly spaced from the mesh's rotor angle, each solve
    starting from the one before. ``materials`` are as for
    ``build_problem``, ``solver`` as for ``solve``; the time taken is added
    to ``timings`` if one is given.

    With ``losses`` (and optionally ``coefficients`` and
    ``stacking_factor``, all as for ``core_loss``) the core loss is found
    too, which takes one more solve at the end of the cycle and at least 8
    points.
    """
    for name, value in (("d current", d_current), ("q current", q_current)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number of amperes, not {value}")
    least = 8 if losses else 1
    if points < least:
        raise ValueError(f"points must be at least {least}, not {points}")
    size, advance = current_and_advance(d_current, q_current)
    layout = machine.symmetry()
    pole_pairs = layout.poles * layout.periods // 2
    # With losses, the end of the cycle too: the core loss needs it closed.
    solved = points + 1 if losses else points
    angles = mesh.rotor_angle + 2.0 * math.pi / pole_pairs * np.arange(solved) / points
    run = solve_angles(
        mesh,
        angles,
        lambda turned: build_problem(machine, turned, materials, current=size, advance=advance),
        solver=solver,
    )
    started = time.perf_counter()
    cycle = run.steps[:points]
    linked = np.array(
        [
            dq_components(machine, float(angle), flux_linkage(step.problem, step.solution))
            for angle, step in zip(angles, cycle)
        ]
    )
    parts: dict[str, object] = {}
    if losses:
        if stacking_factor is None:
            stacking_factor = machine.design.steel.stacking_factor
        loss = core_loss_of_run(
            run, materials, losses, 1.0, pole_pairs, stacking_factor, coefficients, size, advance
        )
        parts = {
            "hysteresis": loss.hysteresis,
            "eddy": loss.eddy,
            "excess": loss.excess,
            "fitted_frequencies": loss.fitted_frequencies,
        }
    result = OperatingPoint(
        d_current=float(d_current),
        q_current=float(q_current),
        d_linkage=float(linked[:, 0].mean()),
        q_linkage=float(linked[:, 1].mean()),
        torque=float(np.mean([torque(step.problem, step.solution) for step in cycle])),
        **parts,
    )
    if timings is not None:
        for name, seconds in run.timings.stages:
            timings.add(name, seconds)
        timings.add("results", time.perf_counter() - started)
    return result


@dataclass(frozen=True, eq=False)
class FluxMap:
    """d and q flux linkage and torque on a grid of d and q currents.

    The grids have one row per d current and one column per q current.
    """

    d_currents: NDArray[np.float64]  # A, rising
    q_currents: NDArray[np.float64]  # A, rising
    pole_pairs: int
    points: int  # rotor angles solved at each grid point
    d_linkage: NDArray[np.float64]  # Wb
    q_linkage: NDArray[np.float64]  # Wb
    torque: NDArray[np.float64]  # N m, from the air-gap stress
    timings: Timings
    # Core loss in W at 1 mechanical rad/s, by part; None without loss data.
    # At speed w the loss is hysteresis w + eddy w^2 + excess w^1.5.
    hysteresis: NDArray[np.float64] | None = None
    eddy: NDArray[np.float64] | None = None
    excess: NDArray[np.float64] | None = None
    fitted_frequencies: tuple[float, float] | None = None  # Hz, of the loss coefficients

    @property
    def has_core_loss(self) -> bool:
        """Whether the map holds the core loss."""
        return self.hysteresis is not None

    @property
    def torque_from_linkage(self) -> NDArray[np.float64]:
        """Torque on the grid from 1.5 p (psi_d i_q - psi_q i_d), N m."""
        i_d, i_q = np.meshgrid(self.d_currents, self.q_currents, indexing="ij")
        return 1.5 * self.pole_pairs * (self.d_linkage * i_q - self.q_linkage * i_d)

    def covers(self, d_current: ArrayLike, q_current: ArrayLike) -> NDArray[np.bool_]:
        """Whether each operating point lies inside the grid."""
        i_d, i_q = np.broadcast_arrays(
            np.asarray(d_current, dtype=float), np.asarray(q_current, dtype=float)
        )
        inside = np.ones(i_d.shape, dtype=bool)
        for values, axis in ((i_d, self.d_currents), (i_q, self.q_currents)):
            slack = EDGE * (axis[-1] - axis[0])
            inside &= (values >= axis[0] - slack) & (values <= axis[-1] + slack)
        return inside

    def at(
        self, d_current: ArrayLike, q_current: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """psi_d and psi_q in Wb and torque in N m at operating points in the grid.

        ``d_current`` and ``q_current`` are in amperes and may be arrays of
        the same shape (or one a single value); the results have that shape.
        Raises ValueError for a point outside the grid.
        """
        i_d, i_q = np.broadcast_arrays(
            np.asarray(d_current, dtype=float), np.asarray(q_current, dtype=float)
        )
        x, y = self._inside(i_d, i_q)
        return tuple(  # type: ignore[return-value]
            spline(x, y, grid=False).reshape(i_d.shape) for spline in self._splines()
        )

    def core_loss(
        self, d_current: ArrayLike, q_current: ArrayLike, speed: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Hysteresis, eddy-current and excess loss in W at operating points in the grid.

        ``speed`` is in mechanical rad/s; it and the currents (A) are
        broadcast together. The parts are the map's values at 1 rad/s
        scaled by the speed, its square and its power 1.5; they never fall
        below zero, which a spline between small values could. Raises
        ValueError for a point outside the grid or a map without core loss.
        """
        if not self.has_core_loss:
            raise ValueError("this map was made without loss data, so it has no core loss")
        i_d, i_q, w = np.broadcast_arrays(
            np.asarray(d_current, dtype=float),
            np.asarray(q_current, dtype=float),
            np.abs(np.asarray(speed, dtype=float)),
        )
        x, y = self._inside(i_d, i_q)
        h, e, k = (
            np.maximum(spline(x, y, grid=False).reshape(i_d.shape), 0.0)
            for spline in self._splines(losses=True)
        )
        return h * w, e * w**2, k * w**1.5

    def _inside(
        self, i_d: NDArray[np.float64], i_q: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The currents as flat arrays clipped onto the grid; ValueError if one is off it."""
        inside = self.covers(i_d, i_q)
        if not inside.all():
            where = np.flatnonzero(~inside.ravel())[0]
            raise ValueError(
                f"the operating point i_d = {i_d.ravel()[where]:g} A, "
                f"i_q = {i_q.ravel()[where]:g} A is outside the map, which covers "
                f"i_d {self.d_currents[0]:g} to {self.d_currents[-1]:g} A and "
                f"i_q {self.q_currents[0]:g} to {self.q_currents[-1]:g} A"
            )
        x = np.clip(i_d, self.d_currents[0], self.d_currents[-1]).ravel()
        y = np.clip(i_q, self.q_currents[0], self.q_currents[-1]).ravel()
        return x, y

    def _splines(self, losses: bool = False) -> tuple[RectBivariateSpline, ...]:
        key = "_cached_loss_splines" if losses else "_cached_splines"
        cached = self.__dict__.get(key)
        if cached is None:
            kx = min(3, len(self.d_currents) - 1)
            ky = min(3, len(self.q_currents) - 1)
            grids = (
                (self.hysteresis, self.eddy, self.excess)
                if losses
                else (self.d_linkage, self.q_linkage, self.torque)
            )
            cached = tuple(
                RectBivariateSpline(self.d_currents, self.q_currents, values, kx=kx, ky=ky, s=0)
                for values in grids
            )
            self.__dict__[key] = cached
        return cached

    def timing_report(self) -> str:
        """Time spent over all the solves, as text."""
        count = len(self.d_currents) * len(self.q_currents)
        angles = count * (self.points + (1 if self.has_core_loss else 0))
        heading = f"Timing ({count} operating points, {angles} rotor angles)"
        return "\n".join([heading] + self.timings.report().splitlines()[1:])


def _axis(values: ArrayLike, name: str) -> NDArray[np.float64]:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a list of at least two currents")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be finite numbers of amperes")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must rise from one to the next")
    return axis


def flux_map(
    machine: Machine,
    mesh: Mesh,
    materials: Mapping[str, Material],
    d_currents: ArrayLike,
    q_currents: ArrayLike,
    points: int = 24,
    solver: str | None = None,
    losses: Mapping[str, CoreLossData] | None = None,
    coefficients: Mapping[str, CoreLossCoefficients] | None = None,
    stacking_factor: float | None = None,
) -> FluxMap:
    """Map the flux linkages and torque of a machine over d and q current.

    ``d_currents`` and ``q_currents`` are the grid's axes in amperes,
    amplitude-invariant (a d or q current of I is a phase current of peak
    I), each rising, at least two. Field weakening needs negative d
    currents, and braking negative q currents. At each grid point the field
    is solved at ``points`` rotor angles of one electrical cycle; see
    ``operating_point``. A harmonic at a multiple of ``points`` times the
    electrical frequency is taken into the mean; see the module's notes.

    Grid points closer together give closer interpolation; where the steel
    saturates the values bend most.

    With ``losses`` the map holds the core loss too; ``losses``,
    ``coefficients`` and ``stacking_factor`` are as for ``core_loss``, and
    the coefficients are fitted once for the whole map.
    """
    d_axis = _axis(d_currents, "d currents")
    q_axis = _axis(q_currents, "q currents")
    least = 8 if losses else 1
    if points < least:
        raise ValueError(f"points must be at least {least}, not {points}")
    if losses:
        fits = dict(coefficients or {})
        for name, data in losses.items():
            if name not in fits:
                fits[name] = data.fit()
        coefficients = fits
    layout = machine.symmetry()
    timings = Timings()
    shape = (d_axis.size, q_axis.size)
    grids = {name: np.empty(shape) for name in ("d", "q", "t", "h", "e", "x")}
    fitted = None
    for row, i_d in enumerate(d_axis):
        for column, i_q in enumerate(q_axis):
            point = operating_point(
                machine,
                mesh,
                materials,
                float(i_d),
                float(i_q),
                points,
                solver,
                timings,
                losses,
                coefficients,
                stacking_factor,
            )
            grids["d"][row, column] = point.d_linkage
            grids["q"][row, column] = point.q_linkage
            grids["t"][row, column] = point.torque
            if losses:
                grids["h"][row, column] = point.hysteresis
                grids["e"][row, column] = point.eddy
                grids["x"][row, column] = point.excess
                fitted = point.fitted_frequencies
    return FluxMap(
        d_currents=d_axis,
        q_currents=q_axis,
        pole_pairs=layout.poles * layout.periods // 2,
        points=points,
        d_linkage=grids["d"],
        q_linkage=grids["q"],
        torque=grids["t"],
        timings=timings,
        hysteresis=grids["h"] if losses else None,
        eddy=grids["e"] if losses else None,
        excess=grids["x"] if losses else None,
        fitted_frequencies=fitted,
    )
