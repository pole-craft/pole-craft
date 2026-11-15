"""Slot and pole layout: the symmetry slice and which coil belongs to which phase.

Covers three-phase concentrated (tooth-wound) windings with a coil on every
tooth, the usual choice for QDD motors. Only slot and pole counts are needed,
so nothing here depends on the machine type or its dimensions.

Teeth are numbered from 0, counter-clockwise, with tooth 0 on the reference
axis. A tooth at mechanical angle ``a`` sits at electrical angle ``p * a``,
where ``p`` is the number of pole pairs.
"""

from __future__ import annotations

from cmath import exp
from dataclasses import dataclass
from math import gcd, pi, sin
from operator import index

PHASES = "ABC"

# The electrical circle cut into six 60 degree belts, centred on 0, 60, 120...
# A coil in the belt at 0 goes to phase A as it is. A coil in the belt at 180
# also goes to phase A, wound the other way, which turns its voltage by 180
# degrees so that it adds to the first. B is centred on 120 and C on 240.
_BELTS = (("A", 1), ("C", -1), ("B", 1), ("A", -1), ("C", 1), ("B", -1))


@dataclass(frozen=True)
class Symmetry:
    """The smallest slice of the machine that repeats around the air gap."""

    slots: int  # slots in the slice
    poles: int  # poles in the slice
    periods: int  # number of slices in the full machine
    antiperiodic: bool  # True if field and currents change sign in the next slice

    @property
    def angle_deg(self) -> float:
        """Mechanical angle covered by the slice."""
        return 360.0 / self.periods


@dataclass(frozen=True)
class Coil:
    """The coil wound around one tooth.

    Positive current in a coil with direction +1 drives flux along its tooth
    away from the machine axis. By the right-hand rule that current comes out
    of the plane in the slot on the counter-clockwise side of the tooth and
    goes into the plane in the slot on its clockwise side.
    """

    tooth: int  # 0-based, counter-clockwise
    phase: str  # "A", "B" or "C"
    direction: int  # +1, or -1 for a coil wound the opposite way


def _checked_counts(slots: int, poles: int) -> tuple[int, int]:
    """Return the counts as plain integers, or raise ValueError."""
    # NumPy integers of a small type would overflow in the arithmetic below.
    slots, poles = index(slots), index(poles)
    if slots < 3:
        raise ValueError(f"slots must be at least 3, got {slots}")
    if poles < 2 or poles % 2:
        raise ValueError(f"poles must be an even number of at least 2, got {poles}")
    return slots, poles


def symmetry(slots: int, poles: int) -> Symmetry:
    """Find the smallest repeating slice for the given slot and pole counts.

    Turning the machine by one slice must put slots on slots and poles on
    poles. If the slice holds an odd number of poles, north lands on south:
    the next slice is the same with every field and current reversed
    (antiperiodic), which a solver handles as easily as a plain repeat.
    """
    slots, poles = _checked_counts(slots, poles)
    periods = gcd(slots, poles)
    slice_poles = poles // periods
    return Symmetry(
        slots=slots // periods,
        poles=slice_poles,
        periods=periods,
        antiperiodic=slice_poles % 2 == 1,
    )


def combination_problem(slots: int, poles: int) -> str | None:
    """Say why a slot and pole combination cannot be wound, or None if it can.

    Coils sit at ``slots / gcd(slots, pole pairs)`` different electrical
    angles, evenly spread. Three equal phases 120 degrees apart need that
    number to be a multiple of 3.
    """
    slots, poles = _checked_counts(slots, poles)
    if slots % 3:
        return (
            f"{slots} slots with {poles} poles cannot carry a three-phase winding: "
            f"the slot count must be a multiple of 3"
        )
    angles = slots // gcd(slots, poles // 2)
    if angles % 3:
        positions = "position" if angles == 1 else "positions"
        return (
            f"{slots} slots with {poles} poles cannot carry a balanced three-phase "
            f"winding: the coils fall on only {angles} electrical {positions}, "
            f"which three phases cannot share equally"
        )
    return None


def concentrated_winding(slots: int, poles: int) -> tuple[Coil, ...]:
    """Assign every tooth coil to a phase and a winding direction.

    Each coil goes to the phase axis its electrical angle is closest to,
    reversed if it is closer to the opposite end of that axis. This gives the
    highest fundamental winding factor possible with a coil on every tooth.

    Phase B is centred 120 electrical degrees counter-clockwise of phase A,
    so with the rotor turning counter-clockwise the back-EMF of B follows
    that of A, and C follows B. With more poles than slots the phase groups
    appear around the stator in the order A, C, B; that is expected.

    The axis of a phase is the direction of its summed coil phasors. It need
    not pass through the centre of a tooth: with 12 slots and 10 poles phase
    A is two neighbouring teeth, and its axis lies on the slot between them.

    Raises ValueError if the combination cannot carry a balanced winding.
    """
    slots, poles = _checked_counts(slots, poles)
    problem = combination_problem(slots, poles)
    if problem:
        raise ValueError(problem)

    pole_pairs = poles // 2
    coils = []
    for tooth in range(slots):
        # Electrical angle as a whole number of 360/slots degree steps, so the
        # belt edges are decided exactly, not by rounding.
        steps = (tooth * pole_pairs) % slots
        # Belt number = floor((angle + 30 deg) / 60 deg).
        belt = ((12 * steps + slots) // (2 * slots)) % 6
        phase, direction = _BELTS[belt]
        coils.append(Coil(tooth=tooth, phase=phase, direction=direction))
    return tuple(coils)


def layout_string(coils: tuple[Coil, ...]) -> str:
    """The layout as one letter per tooth, lower case for a reversed coil."""
    return "".join(
        coil.phase if coil.direction > 0 else coil.phase.lower() for coil in coils
    )


def winding_factor(slots: int, poles: int) -> float:
    """Fundamental winding factor of the tooth-wound layout (0 to 1).

    The fraction of the ideal flux linkage the winding collects: the pitch
    factor of one coil times the distribution factor of a phase.
    """
    slots, poles = _checked_counts(slots, poles)
    coils = concentrated_winding(slots, poles)
    pole_pairs = poles // 2

    # A coil spans one slot pitch, 2 pi p / slots electrical radians, and
    # links sin(span / 2) of the flux a full-pitch coil would.
    pitch_factor = abs(sin(pi * (pole_pairs % slots) / slots))

    # Length of the sum of phase A's coil phasors over the number of coils.
    total = 0j
    count = 0
    for coil in coils:
        if coil.phase != "A":
            continue
        angle = 2 * pi * ((coil.tooth * pole_pairs) % slots) / slots
        total += coil.direction * exp(1j * angle)
        count += 1
    return pitch_factor * abs(total) / count
