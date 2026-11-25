"""Torque-speed curve: the most torque at each speed within a drive's limits.

The motor is taken in steady state in d and q terms (amplitude-invariant,
as in ``polecraft.studies.flux_map``), with w the electrical speed:

    v_d = R i_d - w psi_q,   v_q = R i_q + w psi_d
    |i| = sqrt(i_d^2 + i_q^2) <= current limit
    |v| = sqrt(v_d^2 + v_q^2) <= voltage limit

The flux linkages and torque come from a ``FluxMap``, so saturation and
cross-saturation are included; an end-turn inductance can be added to the
linkages. At each speed the study searches the current that gives the
most torque:

- Up to the base speed the voltage is enough for the torque per ampere at
  its best (MTPA) at the current limit, and that torque holds: the curve is
  flat.
- Above it the current is turned towards negative d (field weakening) until
  the voltage fits. With a fixed size of current, torque falls as the angle
  moves past the MTPA angle, so the best angle is the first one at which
  the voltage fits. Less than the full current can then give more torque
  (beyond the characteristic current), so all sizes of current are tried.

The limits are peak phase values. ``phase_limits`` turns a drive's DC bus
voltage and current limit into them, assuming space-vector modulation in
its linear range (peak line voltage up to the bus voltage).

What the curve is, and is not:

- Motoring only (positive torque and speed). For a rotor that looks the
  same along d and q, braking mirrors it.
- The torque is the electromagnetic torque of the map, the 2D model times
  the stack length. Core loss, friction and windage, which take some of it
  before the shaft, and the end effects of a short machine, are not taken
  off. Copper loss is in the voltage through R and in ``input_power``.
- Steady state with mean linkages: the drive's own voltage drops, dead
  time and the ripple of the linkages over the cycle are left out, so the
  voltage near the limit is a little optimistic.
- Speeds the limits cannot reach (with a strong magnet, above the speed at
  which even all the current along -d cannot cancel its flux) have no
  torque: NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, minimize_scalar

if TYPE_CHECKING:
    from polecraft.core.winding import PhaseWinding
    from polecraft.studies.flux_map import FluxMap

# Steps the search takes in the size of the current and in its angle; the
# best of them is then refined to full precision.
LEVELS = 48
ANGLES = 256
# A voltage this far over the limit, as a part of it, still counts as within.
SLACK = 1e-9


def phase_limits(
    winding: PhaseWinding, bus_voltage: float, line_current: float
) -> tuple[float, float]:
    """Peak phase voltage and current a drive allows, in V and A.

    ``bus_voltage`` is the drive's DC bus voltage and ``line_current`` the
    peak current it allows in each terminal. With space-vector modulation
    in its linear range the peak voltage between two terminals reaches the
    bus voltage; a star phase gets 1 / sqrt 3 of that, a delta phase all of
    it, and a delta phase carries 1 / sqrt 3 of the terminal current.
    """
    for name, value in (("bus voltage", bus_voltage), ("line current", line_current)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be a positive number, not {value}")
    return bus_voltage / winding.line_voltage(1.0), line_current / winding.line_current(1.0)


@dataclass(frozen=True, eq=False)
class TorqueSpeed:
    """The most torque at each speed, and the currents and voltages that give it.

    Arrays have one entry per speed; NaN where the speed cannot be reached.
    Currents and voltages are peak phase values in d and q terms.
    """

    speeds: NDArray[np.float64]  # mechanical rad/s
    torque: NDArray[np.float64]  # N m
    d_current: NDArray[np.float64]  # A
    q_current: NDArray[np.float64]  # A
    d_voltage: NDArray[np.float64]  # V
    q_voltage: NDArray[np.float64]  # V
    d_linkage: NDArray[np.float64]  # Wb, with the end turns
    q_linkage: NDArray[np.float64]  # Wb, with the end turns
    pole_pairs: int
    voltage_limit: float  # V, peak phase
    current_limit: float  # A, peak phase
    resistance: float  # ohm, phase
    end_inductance: float  # H, added to both axes
    base_speed: float  # mechanical rad/s; the curve is flat up to here
    base_torque: float  # N m, the torque on the flat part

    @property
    def voltage(self) -> NDArray[np.float64]:
        """Peak phase voltage in V."""
        return np.hypot(self.d_voltage, self.q_voltage)

    @property
    def current(self) -> NDArray[np.float64]:
        """Peak phase current in A."""
        return np.hypot(self.d_current, self.q_current)

    @property
    def power(self) -> NDArray[np.float64]:
        """Mechanical power, torque times speed, in W."""
        return self.torque * self.speeds

    @property
    def copper_loss(self) -> NDArray[np.float64]:
        """Resistive loss of the three phases, 1.5 R |i|^2, in W."""
        return 1.5 * self.resistance * (self.d_current**2 + self.q_current**2)

    @property
    def input_power(self) -> NDArray[np.float64]:
        """Electrical power into the three phases, 1.5 (v_d i_d + v_q i_q), in W.

        It is the copper loss plus 1.5 p w_m (psi_d i_q - psi_q i_d), the
        torque from the linkages times the speed; that torque agrees with
        the map's gap torque to the map's accuracy.
        """
        return 1.5 * (self.d_voltage * self.d_current + self.q_voltage * self.q_current)

    @property
    def max_power(self) -> float:
        """Largest mechanical power on the curve, in W."""
        return float(np.nanmax(self.power))


class _Machine:
    """Linkages, torque and voltage of a map at currents given by size and angle."""

    def __init__(self, fmap: FluxMap, resistance: float, end_inductance: float) -> None:
        self.fmap = fmap
        self.resistance = resistance
        self.end_inductance = end_inductance

    def state(
        self, size: ArrayLike, angle: ArrayLike, omega: float
    ) -> tuple[NDArray[np.float64], ...]:
        """i_d, i_q, psi_d, psi_q, torque, v_d, v_q for current ``size`` at advance ``angle``."""
        i_d = -np.asarray(size) * np.sin(angle)
        i_q = np.asarray(size) * np.cos(angle)
        psi_d, psi_q, torque = self.fmap.at(i_d, i_q)
        psi_d = psi_d + self.end_inductance * i_d
        psi_q = psi_q + self.end_inductance * i_q
        v_d = self.resistance * i_d - omega * psi_q
        v_q = self.resistance * i_q + omega * psi_d
        return i_d, i_q, psi_d, psi_q, torque, v_d, v_q

    def voltage(self, size: float, angle: float, omega: float) -> float:
        *_, v_d, v_q = self.state(size, angle, omega)
        return float(np.hypot(v_d, v_q))

    def torque(self, size: float, angle: float) -> float:
        return float(self.state(size, angle, 0.0)[4])

    def best_angle(self, size: float) -> float:
        """Advance between 0 and pi/2 that gives the most torque for a current (MTPA)."""
        grid = np.linspace(0.0, 0.5 * math.pi, ANGLES)
        torques = self.state(np.full(ANGLES, size), grid, 0.0)[4]
        k = int(np.argmax(torques))
        low, high = grid[max(k - 1, 0)], grid[min(k + 1, ANGLES - 1)]
        found = minimize_scalar(
            lambda angle: -self.torque(size, angle),
            bounds=(low, high),
            method="bounded",
            options={"xatol": 1e-12},
        )
        return float(found.x) if -found.fun >= torques[k] else float(grid[k])

    def weakened(self, size: float, omega: float, limit: float) -> tuple[float, float] | None:
        """Torque and advance at the first angle past MTPA where the voltage fits, or None."""
        start = self.best_angle(size)
        grid = np.linspace(start, 0.5 * math.pi, ANGLES)
        *_, torques, v_d, v_q = self.state(np.full(ANGLES, size), grid, omega)
        fits = np.hypot(v_d, v_q) <= limit * (1.0 + SLACK)
        if not fits.any():
            return None
        k = int(np.argmax(fits))
        if k == 0:
            return float(torques[0]), start
        if self.voltage(size, grid[k], omega) >= limit:
            # Over the limit by no more than the slack: as good as on it.
            return float(torques[k]), float(grid[k])
        angle = brentq(
            lambda a: self.voltage(size, a, omega) - limit, grid[k - 1], grid[k], xtol=1e-14
        )
        return self.torque(size, angle), float(angle)


def torque_speed(
    fmap: FluxMap,
    speeds: ArrayLike,
    voltage: float,
    current: float,
    resistance: float,
    end_inductance: float = 0.0,
) -> TorqueSpeed:
    """The most torque at each speed within a peak phase voltage and current.

    ``fmap`` is the machine's flux map; it must cover d currents from
    -``current`` to 0 and q currents from 0 to ``current`` (amperes).
    ``speeds`` are mechanical rad/s (rpm times pi / 30), 0 or more.
    ``voltage`` and ``current`` are the peak phase limits (see
    ``phase_limits``), ``resistance`` the phase resistance in ohms at the
    winding's temperature (``PhaseWinding.resistance``) and
    ``end_inductance`` the end-turn inductance in H that the 2D map leaves
    out (``polecraft.core.winding.end_winding_inductance``).

    Raises ValueError for limits or a resistance that are not positive
    numbers, a negative or non-finite speed, a map that does not cover the
    currents, or a voltage that cannot drive the current limit even at
    standstill.
    """
    for name, value in (("voltage", voltage), ("current", current)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} limit must be a positive number, not {value}")
    for name, value in (("resistance", resistance), ("end inductance", end_inductance)):
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be a number, 0 or more, not {value}")
    speeds = np.asarray(speeds, dtype=float)
    if speeds.ndim != 1 or speeds.size == 0:
        raise ValueError("speeds must be a list of at least one speed")
    if not (np.all(np.isfinite(speeds)) and np.all(speeds >= 0.0)):
        raise ValueError("speeds must be finite numbers of rad/s, 0 or more")
    corners_d = np.array([-current, -current, 0.0, 0.0])
    corners_q = np.array([0.0, current, 0.0, current])
    if not fmap.covers(corners_d, corners_q).all():
        raise ValueError(
            f"the flux map must cover d currents from {-current:g} to 0 A and q currents "
            f"from 0 to {current:g} A for a current limit of {current:g} A"
        )
    if resistance * current >= voltage:
        raise ValueError(
            f"a voltage of {voltage:g} V cannot drive {current:g} A through {resistance:g} ohm"
        )

    motor = _Machine(fmap, resistance, end_inductance)
    pole_pairs = fmap.pole_pairs
    rated_angle = motor.best_angle(current)
    i_d, i_q, psi_d, psi_q, base_torque, _, _ = motor.state(current, rated_angle, 0.0)
    # Base speed: |v| reaches the limit with the MTPA current at the limit.
    # |v|^2 is a quadratic in the electrical speed w.
    a = float(psi_d**2 + psi_q**2)
    b = float(2.0 * resistance * (i_q * psi_d - i_d * psi_q))
    c = resistance**2 * current**2 - voltage**2
    base_speed = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a) / pole_pairs

    levels = current * np.arange(1, LEVELS + 1) / LEVELS
    columns = {name: np.full(speeds.size, np.nan) for name in ("t", "size", "angle")}
    for n, speed in enumerate(speeds):
        omega = pole_pairs * speed
        if speed <= base_speed:
            best = (float(base_torque), current, rated_angle)
        else:
            found = []
            for size in levels:
                result = motor.weakened(float(size), omega, voltage)
                if result is not None:
                    found.append((result[0], float(size), result[1]))
            if not found:
                continue
            best = max(found)
            j = int(np.flatnonzero(levels == best[1])[0])
            low, high = levels[max(j - 1, 0)], levels[min(j + 1, LEVELS - 1)]
            if high > low:

                def lost(size: float, omega: float = omega) -> float:
                    # A current the voltage cannot fit counts as no torque,
                    # worse than any motoring point.
                    result = motor.weakened(size, omega, voltage)
                    return -result[0] if result else 0.0

                refined = minimize_scalar(
                    lost, bounds=(low, high), method="bounded", options={"xatol": 1e-9 * current}
                )
                if -refined.fun > best[0]:
                    result = motor.weakened(float(refined.x), omega, voltage)
                    best = (result[0], float(refined.x), result[1])
        columns["t"][n], columns["size"][n], columns["angle"][n] = best

    reached = np.isfinite(columns["t"])
    size = np.where(reached, columns["size"], 0.0)
    angle = np.where(reached, columns["angle"], 0.0)
    i_d, i_q, psi_d, psi_q, _, v_d, v_q = motor.state(size, angle, 0.0)
    omega = pole_pairs * speeds
    v_d = resistance * i_d - omega * psi_q
    v_q = resistance * i_q + omega * psi_d

    def masked(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(reached, values, np.nan)

    return TorqueSpeed(
        speeds=speeds,
        torque=columns["t"],
        d_current=masked(i_d),
        q_current=masked(i_q),
        d_voltage=masked(v_d),
        q_voltage=masked(v_q),
        d_linkage=masked(psi_d),
        q_linkage=masked(psi_q),
        pole_pairs=pole_pairs,
        voltage_limit=float(voltage),
        current_limit=float(current),
        resistance=float(resistance),
        end_inductance=float(end_inductance),
        base_speed=float(base_speed),
        base_torque=float(base_torque),
    )
