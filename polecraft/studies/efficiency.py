"""Efficiency map: efficiency over speed and shaft torque, at the current that loses least.

Each point of the map is a mechanical speed w and a torque at the shaft.
Many d and q currents give that torque; the study takes the one with the
least loss that the drive's voltage and current limits allow, as a drive
that minimises loss would. The flux linkages and core loss come from a
``FluxMap`` made with the steel's loss data (``flux_map(..., losses=...)``).

The power balance at each point, in watts, is

    input  = 1.5 (v_d i_d + v_q i_q) = copper + T_em w
    output = T w,  with T = T_em - core / w
    input  = output + copper + core

where T_em = 1.5 p (psi_d i_q - psi_q i_d) is the electromagnetic torque
from the flux linkages and copper = 1.5 R |i|^2. The core loss is taken
from the rotor as a drag torque, core / w; it is h + e w + x w^0.5 with the
map's parts at 1 rad/s, so it stays finite at standstill. The voltage is
that of the steady state with the cycle-mean linkages, as in
``polecraft.studies.torque_speed``, so the balance holds exactly.

T_em is the torque from the linkages, not the map's air-gap torque: with
it the electrical input and the losses balance exactly. The two agree to
the map's accuracy (a few tenths of a percent).

What the map is, and is not:

- Motoring only (torque and speed 0 or more). Points the limits cannot
  reach have no values: NaN.
- Losses are copper (DC resistance at the winding temperature passed in)
  and steel. Not included: friction and windage, eddy currents in the
  magnets and in the winding (skin and proximity effects), the drive's own
  loss, and the extra loss of a real built core (stamping, stress), which
  the core loss study describes. The efficiency is therefore an upper
  bound.
- The current is found in two stages: the best of a grid of advance
  angles 0.5 degree apart, then, within a grid step of it, the exact edge
  of the voltage limit (bisection) and the least loss (golden-section
  search). The second stage matters with field weakening, where the best
  current lies on the voltage limit and the grid alone overstates the loss
  by up to a few percent.
- Without core loss, zero torque above the speed at which the magnet's
  back-EMF alone reaches the voltage limit is shown as out of reach,
  though current along -d alone would hold it there. With core loss, as
  in any real map, zero shaft torque needs some q current and is found.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from polecraft.core.materials import LossRangeWarning

if TYPE_CHECKING:
    from polecraft.studies.flux_map import FluxMap

# Advance angles tried, from 0 to 90 degrees, and sizes of current, from 0
# to the limit, in the tables the search works from.
ANGLES = 181
LEVELS = 400
# Newton steps that bring the torque of the chosen current onto the one asked for.
NEWTON = 4


@dataclass(frozen=True, eq=False)
class EfficiencyMap:
    """Losses and efficiency on a grid of speed and shaft torque.

    Arrays have one row per speed and one column per torque; NaN where the
    point cannot be reached. Currents and voltages are peak phase values in
    d and q terms; powers are in watts.
    """

    speeds: NDArray[np.float64]  # mechanical rad/s
    torques: NDArray[np.float64]  # N m, the shaft torques asked for
    torque: NDArray[np.float64]  # N m, shaft torque given
    electromagnetic_torque: NDArray[np.float64]  # N m, from the linkages
    d_current: NDArray[np.float64]  # A
    q_current: NDArray[np.float64]  # A
    d_voltage: NDArray[np.float64]  # V
    q_voltage: NDArray[np.float64]  # V
    copper_loss: NDArray[np.float64]  # W
    hysteresis_loss: NDArray[np.float64]  # W
    eddy_loss: NDArray[np.float64]  # W
    excess_loss: NDArray[np.float64]  # W
    voltage_limit: float  # V, peak phase
    current_limit: float  # A, peak phase
    resistance: float  # ohm, phase
    end_inductance: float  # H

    @property
    def reached(self) -> NDArray[np.bool_]:
        """Whether each point is within the limits."""
        return np.isfinite(self.torque)

    @property
    def core_loss(self) -> NDArray[np.float64]:
        """Loss in the steel, W."""
        return self.hysteresis_loss + self.eddy_loss + self.excess_loss

    @property
    def losses(self) -> NDArray[np.float64]:
        """Copper and core loss, W."""
        return self.copper_loss + self.core_loss

    @property
    def output_power(self) -> NDArray[np.float64]:
        """Power at the shaft, torque times speed, W."""
        return self.torque * self.speeds[:, None]

    @property
    def input_power(self) -> NDArray[np.float64]:
        """Electrical power into the three phases, 1.5 (v_d i_d + v_q i_q), W."""
        return 1.5 * (self.d_voltage * self.d_current + self.q_voltage * self.q_current)

    @property
    def efficiency(self) -> NDArray[np.float64]:
        """Output over input power; 0 where nothing goes in or nothing comes out."""
        inp, out = self.input_power, self.output_power
        # Where no torque is asked for, the torque given is zero only to the
        # solver's tolerance, of either sign.
        nothing = (inp <= 0.0) | (self.torques[None, :] == 0.0) | (self.speeds[:, None] == 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(nothing, 0.0, out / inp)
        return np.where(self.reached, ratio, np.nan)

    @property
    def current(self) -> NDArray[np.float64]:
        """Peak phase current, A."""
        return np.hypot(self.d_current, self.q_current)

    @property
    def voltage(self) -> NDArray[np.float64]:
        """Peak phase voltage, V."""
        return np.hypot(self.d_voltage, self.q_voltage)


def _check(
    speeds: ArrayLike,
    torques: ArrayLike,
    voltage: float,
    current: float,
    resistance: float,
    end_inductance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    for name, value in (("voltage", voltage), ("current", current)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} limit must be a positive number, not {value}")
    for name, value in (("resistance", resistance), ("end inductance", end_inductance)):
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be a number, 0 or more, not {value}")
    axes = []
    for name, unit, values in (("speeds", "rad/s", speeds), ("torques", "N m", torques)):
        axis = np.asarray(values, dtype=float)
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError(f"{name} must be a list of at least one value")
        if not (np.all(np.isfinite(axis)) and np.all(axis >= 0.0)):
            raise ValueError(f"{name} must be finite numbers of {unit}, 0 or more")
        axes.append(axis)
    return axes[0], axes[1]


def efficiency_map(
    fmap: FluxMap,
    speeds: ArrayLike,
    torques: ArrayLike,
    voltage: float,
    current: float,
    resistance: float,
    end_inductance: float = 0.0,
) -> EfficiencyMap:
    """Losses and efficiency at each speed and shaft torque, at the least-loss current.

    ``fmap`` is the machine's flux map, made with loss data for the core
    loss (without it the core loss is taken as zero); it must cover d
    currents from -``current`` to 0 and q currents from 0 to ``current``.
    ``speeds`` are mechanical rad/s and ``torques`` shaft torques in N m,
    both 0 or more. ``voltage``, ``current``, ``resistance`` and
    ``end_inductance`` are as for ``torque_speed``.

    Warns with ``LossRangeWarning`` if an electrical frequency of the map
    is outside the range the loss coefficients were fitted over.

    Raises ValueError for limits that are not positive, a negative
    resistance or end inductance, speeds or torques that are negative or
    not finite, or a map that does not cover the currents.
    """
    speeds, torques = _check(speeds, torques, voltage, current, resistance, end_inductance)
    if not fmap.covers([-current, -current, 0.0, 0.0], [0.0, current, 0.0, current]).all():
        raise ValueError(
            f"the flux map must cover d currents from {-current:g} to 0 A and q currents "
            f"from 0 to {current:g} A for a current limit of {current:g} A"
        )
    _warn_frequencies(fmap, speeds)
    p = fmap.pole_pairs

    # Tables over advance angle (rows) and size of current (columns); they
    # do not depend on speed.
    angle = np.linspace(0.0, 0.5 * math.pi, ANGLES)[:, None]
    level = current * np.arange(LEVELS) / (LEVELS - 1)
    size = np.broadcast_to(level, (ANGLES, LEVELS))
    table = _state(fmap, size, np.broadcast_to(angle, size.shape), end_inductance)

    advance_table = np.broadcast_to(angle, size.shape)
    # Along the angle, at each size of current, the torque peaks at the
    # best angle (MTPA) and then falls; past the peak, more angle means
    # less voltage, so field weakening is searched there.
    peak = np.argmax(table["t"], axis=0)

    shape = (speeds.size, torques.size)
    found = {name: np.full(shape, np.nan) for name in ("size", "angle")}
    for n, speed in enumerate(speeds):
        w = p * speed
        drag = table["h"] + table["e"] * speed + table["x"] * math.sqrt(speed)
        shaft = table["t"] - drag
        volts = np.hypot(
            resistance * table["i_d"] - w * table["psi_q"],
            resistance * table["i_q"] + w * table["psi_d"],
        )
        loss_table = 1.5 * resistance * size**2 + speed * drag
        candidates = []
        # Each torque asked for, found two ways in the same table: along the
        # size of current at each angle, and along the angle past the peak at
        # each size. A narrow window of angles that fit the voltage at high
        # speed is wide in current, and the other way about.
        for values, start, rising in (
            ((shaft, size, advance_table, volts, loss_table), np.zeros(ANGLES, int), True),
            (tuple(v.T for v in (shaft, size, advance_table, volts, loss_table)), peak, False),
        ):
            k, before, part, ok = _crossing(values[0], torques, start, rising)
            rows = np.arange(len(start))[:, None]

            def along(table_values: NDArray[np.float64]) -> NDArray[np.float64]:
                low, high = table_values[rows, before], table_values[rows, k]
                return low + part * (high - low)

            fits = ok & (along(values[3]) <= voltage)
            candidates.append(
                (np.where(fits, along(values[4]), np.inf), along(values[1]), along(values[2]))
            )
        loss, sizes, angles = (np.concatenate(column) for column in zip(*candidates))
        best = np.argmin(loss, axis=0)
        columns = np.arange(torques.size)
        chosen = np.isfinite(loss[best, columns])
        found["size"][n] = np.where(chosen, sizes[best, columns], np.nan)
        found["angle"][n] = np.where(chosen, angles[best, columns], np.nan)

    reached = np.isfinite(found["size"])
    speed = np.broadcast_to(speeds[:, None], shape)
    asked = np.broadcast_to(torques[None, :], shape)
    point = _Points(
        fmap, speed[reached], asked[reached], voltage, current, resistance, end_inductance
    )
    size, advance, fits = point.refine(found["size"][reached], found["angle"][reached])
    # A point whose interpolated current fitted but whose exact one does not,
    # within a hair of the limits, is left out rather than shown over them.
    reached[reached] = fits
    sizes, advances = np.zeros(shape), np.zeros(shape)
    sizes[reached], advances[reached] = size[fits], advance[fits]
    state = _state(fmap, sizes, advances, end_inductance)
    w = p * speed
    shaft = state["t"] - _drag(state, speed)

    def masked(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(reached, values, np.nan)

    return EfficiencyMap(
        speeds=speeds,
        torques=torques,
        torque=masked(shaft),
        electromagnetic_torque=masked(state["t"]),
        d_current=masked(state["i_d"]),
        q_current=masked(state["i_q"]),
        d_voltage=masked(resistance * state["i_d"] - w * state["psi_q"]),
        q_voltage=masked(resistance * state["i_q"] + w * state["psi_d"]),
        copper_loss=masked(1.5 * resistance * (state["i_d"] ** 2 + state["i_q"] ** 2)),
        hysteresis_loss=masked(state["h"] * speed),
        eddy_loss=masked(state["e"] * speed**2),
        excess_loss=masked(state["x"] * speed**1.5),
        voltage_limit=float(voltage),
        current_limit=float(current),
        resistance=float(resistance),
        end_inductance=float(end_inductance),
    )


def _crossing(
    values: NDArray[np.float64],
    targets: NDArray[np.float64],
    start: NDArray[np.intp],
    rising: bool,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.bool_]]:
    """Where each line of a table first reaches each target, from its start on.

    ``values`` has one line per row; ``start`` is the first step of each
    line looked at. A rising line reaches a target at the first step at or
    above it, a falling one at the first at or below it; a falling line
    must start at or above the target. Returns, for each line and target,
    that step, the one before it (not before the start), the fraction of
    the way between them, and whether it is reached at all.
    """
    steps = np.arange(values.shape[1])
    if rising:
        hit = values[:, None, :] >= targets[None, :, None]
    else:
        hit = values[:, None, :] <= targets[None, :, None]
    hit &= (steps[None, :] >= start[:, None])[:, None, :]
    k = np.argmax(hit, axis=2)
    ok = hit.any(axis=2)
    rows = np.arange(values.shape[0])[:, None]
    if not rising:
        ok &= values[rows[:, 0], start][:, None] >= targets[None, :]
    before = np.maximum(k - 1, start[:, None])
    low, high = values[rows, before], values[rows, k]
    with np.errstate(invalid="ignore", divide="ignore"):
        part = np.where(k > before, (targets[None, :] - low) / (high - low), 0.0)
    return k, before, np.clip(np.nan_to_num(part), 0.0, 1.0), ok


class _Points:
    """Operating points, each a speed and a shaft torque, and their least-loss current."""

    def __init__(
        self,
        fmap: FluxMap,
        speed: NDArray[np.float64],
        torque: NDArray[np.float64],
        voltage: float,
        current: float,
        resistance: float,
        end_inductance: float,
    ) -> None:
        self.fmap = fmap
        self.speed = speed
        self.torque = torque
        self.voltage = voltage
        self.current = current
        self.resistance = resistance
        self.end_inductance = end_inductance
        self.tolerance = 1e-9 * (1.0 + np.abs(torque))  # N m

    def _shaft(self, size: NDArray[np.float64], angle: NDArray[np.float64]) -> tuple:
        state = _state(self.fmap, size, angle, self.end_inductance)
        return state["t"] - _drag(state, self.speed), state

    def solve(
        self, angle: NDArray[np.float64], seed: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Size of current at each angle that gives the torque, its loss, and whether it fits.

        Newton steps from ``seed``, kept within 0 and the current limit. The
        loss is infinite where the torque is not met within the limit or the
        voltage is over its limit.
        """
        size = np.clip(seed, 0.0, self.current)
        step = 1e-6 * self.current
        for _ in range(NEWTON):
            shaft, _ = self._shaft(size, angle)
            other = np.where(size + step > self.current, size - step, size + step)
            slope = (self._shaft(other, angle)[0] - shaft) / (other - size)
            with np.errstate(invalid="ignore", divide="ignore"):
                change = np.where(slope > 0.0, (shaft - self.torque) / slope, 0.0)
            size = np.clip(size - change, 0.0, self.current)
        shaft, state = self._shaft(size, angle)
        w = self.fmap.pole_pairs * self.speed
        volts = np.hypot(
            self.resistance * state["i_d"] - w * state["psi_q"],
            self.resistance * state["i_q"] + w * state["psi_d"],
        )
        fits = (np.abs(shaft - self.torque) <= self.tolerance) & (volts <= self.voltage)
        loss = 1.5 * self.resistance * size**2 + self.speed * _drag(state, self.speed)
        return size, np.where(fits, loss, np.inf), fits

    def refine(
        self, size: NDArray[np.float64], angle: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The least-loss size and angle near a grid choice, to full precision.

        Within one grid step either side of ``angle``: where the voltage
        limit cuts the interval, bisection finds where; then golden-section
        search finds the least loss on what is left. The grid choice is kept
        where nothing better is found.
        """
        grid_step = 0.5 * math.pi / (ANGLES - 1)
        low = np.maximum(angle - grid_step, 0.0)
        high = np.minimum(angle + grid_step, 0.5 * math.pi)
        size, loss, _ = self.solve(angle, size)
        # Move each end inwards onto the edge of what fits, if it is outside it.
        for end in (low, high):
            outside, edge = ~self.solve(end, size)[2], end.copy()
            inside = angle.copy()
            for _ in range(BISECTIONS):
                middle = 0.5 * (edge + inside)
                fits = self.solve(middle, size)[2]
                inside = np.where(outside & fits, middle, inside)
                edge = np.where(outside & ~fits, middle, edge)
            end[:] = np.where(outside, inside, end)
        # Golden-section search on the loss, which is flat at its least.
        ratio = 0.5 * (math.sqrt(5.0) - 1.0)
        a, b = low, high
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        loss_c, loss_d = self.solve(c, size)[1], self.solve(d, size)[1]
        for _ in range(GOLDEN):
            # Keep the side of the lower inner point; that point carries over
            # as the other inner point of the shorter interval.
            left = loss_c <= loss_d
            a, b = np.where(left, a, c), np.where(left, d, b)
            kept, kept_loss = np.where(left, c, d), np.where(left, loss_c, loss_d)
            fresh = np.where(left, b - ratio * (b - a), a + ratio * (b - a))
            fresh_loss = self.solve(fresh, size)[1]
            c, loss_c = np.where(left, fresh, kept), np.where(left, fresh_loss, kept_loss)
            d, loss_d = np.where(left, kept, fresh), np.where(left, kept_loss, fresh_loss)
        best = 0.5 * (a + b)
        best_size, best_loss, _ = self.solve(best, size)
        better = best_loss < loss
        return (
            np.where(better, best_size, size),
            np.where(better, best, angle),
            np.isfinite(np.minimum(best_loss, loss)),
        )


# Steps of the refinement: bisection onto the voltage limit, and golden-section
# search for the least loss; each shrinks its interval geometrically.
BISECTIONS = 40
GOLDEN = 50


def _state(
    fmap: FluxMap, size: ArrayLike, advance: ArrayLike, end_inductance: float
) -> dict[str, NDArray[np.float64]]:
    """Currents, linkages with the end turns, torque and core-loss parts at 1 rad/s."""
    i_d = -np.asarray(size) * np.sin(advance)
    i_q = np.asarray(size) * np.cos(advance)
    psi_d, psi_q, _ = fmap.at(i_d, i_q)
    psi_d = psi_d + end_inductance * i_d
    psi_q = psi_q + end_inductance * i_q
    if fmap.has_core_loss:
        h, e, x = fmap.core_loss(i_d, i_q, 1.0)
    else:
        h = e = x = np.zeros(np.shape(i_d))
    return {
        "i_d": i_d,
        "i_q": i_q,
        "psi_d": psi_d,
        "psi_q": psi_q,
        # The end turns add L i_d i_q to both products, which cancels.
        "t": 1.5 * fmap.pole_pairs * (psi_d * i_q - psi_q * i_d),
        "h": h,
        "e": e,
        "x": x,
    }


def _drag(state: dict[str, NDArray[np.float64]], speed: ArrayLike) -> NDArray[np.float64]:
    """Core loss over speed, as a torque in N m: h + e w + x w^0.5."""
    w = np.asarray(speed)
    return state["h"] + state["e"] * w + state["x"] * np.sqrt(w)


def _warn_frequencies(fmap: FluxMap, speeds: NDArray[np.float64]) -> None:
    if not fmap.has_core_loss or fmap.fitted_frequencies is None:
        return
    low, high = fmap.fitted_frequencies
    frequency = fmap.pole_pairs * speeds[speeds > 0.0] / (2.0 * math.pi)
    margin = 1e-9
    outside = frequency[(frequency < low * (1 - margin)) | (frequency > high * (1 + margin))]
    if outside.size:
        warnings.warn(
            f"core loss at {outside.min():g} to {outside.max():g} Hz, outside the "
            f"{low:g} to {high:g} Hz the loss coefficients were fitted over; those values "
            "are an extrapolation",
            LossRangeWarning,
            stacklevel=3,
        )
