"""Stall heating: how long the winding can hold a current with the rotor still.

A QDD actuator often holds torque without turning: a leg standing, an arm
holding a load. The current then does not alternate. Whatever angle the
rotor stops at, the three phase currents of peak I add up to the same
copper loss, 1.5 I^2 R, but they are not shared evenly: at the worst angle
one phase carries the whole peak I all the time and makes I^2 R, twice its
share. That phase sets how long the hold can last, and the estimate is for
it.

The estimate takes the copper as heating alone (adiabatic): no heat reaches
the stator, the air or the housing. With the resistance rising as the
copper warms, R = R20 (1 + a (T - 20)), the temperature of the phase obeys

    m c dT/dt = I^2 R20 (1 + a (T - 20))

which has the exact solution

    1 + a (T - 20) = (1 + a (T0 - 20)) exp(a I^2 R20 t / (m c))

with m the copper mass of the phase and c its specific heat.

What the estimate is, and is not:

- An approximation that errs on the safe side for short holds: real
  windings shed heat into the stator and the time to the limit is longer.
  For holds of minutes the heat path, not the copper's capacity, decides,
  and this estimate says nothing about it (thermal simulation is out of
  scope).
- The temperature is the mean of the phase's copper; the hottest spot, in
  the middle of a coil, runs warmer.
- The specific heat is taken as constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polecraft.core.materials import REFERENCE_TEMPERATURE
from polecraft.studies.mechanical import copper_mass

if TYPE_CHECKING:
    from polecraft.core.winding import PhaseWinding


@dataclass(frozen=True)
class StallHeating:
    """Heating of the worst phase of a winding holding a current at standstill."""

    current: float  # A, peak phase current
    start_temperature: float  # degrees Celsius
    limit_temperature: float  # degrees Celsius, the insulation's limit
    resistance_20: float  # ohm, phase resistance at 20 C
    temperature_coefficient: float  # per kelvin, of the resistance, about 20 C
    heat_capacity: float  # J/K, of the copper of one phase
    torque: float | None  # N m, held, if a torque constant was given

    @property
    def _rate(self) -> float:
        # I^2 R20 / (m c), in K/s: the heating rate at 20 C.
        return self.current**2 * self.resistance_20 / self.heat_capacity

    def _scale(self, temperature: float) -> float:
        return 1.0 + self.temperature_coefficient * (temperature - REFERENCE_TEMPERATURE)

    @property
    def loss(self) -> float:
        """Copper loss of the worst phase at the start, I^2 R, in W."""
        return self.current**2 * self.resistance_20 * self._scale(self.start_temperature)

    @property
    def total_loss(self) -> float:
        """Copper loss of all three phases at the start, 1.5 I^2 R, in W."""
        return 1.5 * self.loss

    @property
    def heating_rate(self) -> float:
        """How fast the worst phase warms at the start, in K/s."""
        return self._rate * self._scale(self.start_temperature)

    @property
    def time_to_limit(self) -> float:
        """Seconds until the worst phase reaches the limit temperature; inf with no current."""
        if self.current == 0.0:
            return math.inf
        ratio = self._scale(self.limit_temperature) / self._scale(self.start_temperature)
        return math.log(ratio) / (self.temperature_coefficient * self._rate)

    def temperature_at(self, seconds: float) -> float:
        """Temperature of the worst phase after ``seconds`` of holding, in degrees Celsius."""
        if not (math.isfinite(seconds) and seconds >= 0.0):
            raise ValueError(f"seconds must be a number, 0 or more, not {seconds}")
        a = self.temperature_coefficient
        grown = self._scale(self.start_temperature) * math.exp(a * self._rate * seconds)
        return REFERENCE_TEMPERATURE + (grown - 1.0) / a


def _heat_capacity(winding: PhaseWinding, specific_heat: float | None) -> float:
    heat = winding.conductor.specific_heat if specific_heat is None else specific_heat
    if heat is None:
        raise ValueError(
            "the conductor has no specific heat: give specific_heat in J/(kg K)"
        )
    if not (math.isfinite(heat) and heat > 0.0):
        raise ValueError(f"specific heat must be a positive number, not {heat}")
    return copper_mass(winding) / 3.0 * heat


def _check_temperatures(start: float, limit: float) -> None:
    for name, value in (("start", start), ("limit", limit)):
        if not math.isfinite(value):
            raise ValueError(f"{name} temperature must be a finite number, not {value}")
    if limit <= start:
        raise ValueError(
            f"the limit temperature ({limit:g} C) must be above the start ({start:g} C)"
        )


def stall_heating(
    winding: PhaseWinding,
    current: float,
    start_temperature: float,
    limit_temperature: float,
    torque_constant: float | None = None,
    specific_heat: float | None = None,
) -> StallHeating:
    """Heating of a winding holding a peak phase current with the rotor still.

    ``current`` is the peak phase current in amperes, as in the studies.
    Temperatures are in degrees Celsius: ``start_temperature`` that of the
    winding when the hold begins, ``limit_temperature`` the most its
    insulation allows. ``torque_constant`` (N m per ampere of peak current)
    gives the torque held; ``specific_heat`` in J/(kg K) replaces the
    conductor's own.

    Raises ValueError for a negative or non-finite current, a limit not
    above the start, or a conductor with no specific heat when none is
    given.
    """
    if not (math.isfinite(current) and current >= 0.0):
        raise ValueError(f"current must be a number of amperes, 0 or more, not {current}")
    _check_temperatures(start_temperature, limit_temperature)
    torque = None
    if torque_constant is not None:
        if not math.isfinite(torque_constant):
            raise ValueError(f"torque constant must be a finite number, not {torque_constant}")
        torque = torque_constant * current
    return StallHeating(
        current=float(current),
        start_temperature=float(start_temperature),
        limit_temperature=float(limit_temperature),
        resistance_20=winding.resistance(REFERENCE_TEMPERATURE),
        temperature_coefficient=winding.conductor.temperature_coefficient,
        heat_capacity=_heat_capacity(winding, specific_heat),
        torque=torque,
    )


def stall_current(
    winding: PhaseWinding,
    seconds: float,
    start_temperature: float,
    limit_temperature: float,
    specific_heat: float | None = None,
) -> float:
    """Peak phase current that takes the worst phase to its limit in ``seconds``, in A.

    The inverse of ``stall_heating(...).time_to_limit``: the most a hold of
    that length can carry. Raises ValueError for a time that is not
    positive, or as ``stall_heating`` does.
    """
    if not (math.isfinite(seconds) and seconds > 0.0):
        raise ValueError(f"seconds must be a positive number, not {seconds}")
    _check_temperatures(start_temperature, limit_temperature)
    capacity = _heat_capacity(winding, specific_heat)
    a = winding.conductor.temperature_coefficient
    resistance = winding.resistance(REFERENCE_TEMPERATURE)
    start = 1.0 + a * (start_temperature - REFERENCE_TEMPERATURE)
    limit = 1.0 + a * (limit_temperature - REFERENCE_TEMPERATURE)
    return math.sqrt(math.log(limit / start) * capacity / (a * resistance * seconds))
