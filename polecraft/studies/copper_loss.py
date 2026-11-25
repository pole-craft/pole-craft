"""Copper loss of a machine at a given peak phase current and winding temperature.

For balanced sinusoidal currents of peak I in three phases the loss is

    P = 3 (I / sqrt 2)^2 R(T) = 1.5 I^2 R(T)

where R(T) is the phase resistance at the winding temperature. The loss
depends on the size of the current only, not on its angle (advance), so a
d-axis share of the current costs the same as a q-axis one.

The resistance is the DC value. Skin and proximity effects, which add loss
at high electrical frequency in thick wire, are not included, so the result
is a lower bound there. No solver is needed: the loss follows from the
winding alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polecraft.core.winding import phase_winding

if TYPE_CHECKING:
    from polecraft.core.materials import Conductor
    from polecraft.machines.base import Machine


@dataclass(frozen=True)
class CopperLoss:
    """Copper loss at one operating point."""

    peak_current: float  # A, peak of each phase current
    temperature: float  # degrees Celsius the resistance was taken at
    phase_resistance: float  # ohm, one phase at that temperature
    power: float  # W, all three phases

    @property
    def rms_current(self) -> float:
        """RMS phase current in amperes."""
        return self.peak_current / math.sqrt(2.0)


def copper_loss(
    machine: Machine,
    current: float,
    temperature: float | None = None,
    conductor: Conductor | None = None,
) -> CopperLoss:
    """Copper loss in watts for a peak phase current in amperes.

    ``temperature`` is in degrees Celsius and defaults to the winding
    temperature of the design. ``conductor`` is as for ``phase_winding``.

    Raises ValueError if ``current`` is negative or not finite.
    """
    if not math.isfinite(current) or current < 0.0:
        raise ValueError(f"current must be a finite number of amperes, 0 or more, not {current}")
    winding = phase_winding(machine, conductor)
    t = winding.temperature if temperature is None else temperature
    return CopperLoss(
        peak_current=float(current),
        temperature=float(t),
        phase_resistance=winding.resistance(t),
        power=winding.copper_loss(current / math.sqrt(2.0), t),
    )
