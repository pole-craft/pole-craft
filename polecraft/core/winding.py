"""Slot and pole layout, and the winding as an electrical circuit.

Covers three-phase concentrated (tooth-wound) windings with a coil on every
tooth, the usual choice for QDD motors. The layout needs only slot and pole
counts. The electrical part (wire, slot fill, turn length, resistance) needs
the size of a coil, which a machine gives as a ``CoilShape``; nothing here
depends on the machine type.

Teeth are numbered from 0, counter-clockwise, with tooth 0 on the reference
axis. A tooth at mechanical angle ``a`` sits at electrical angle ``p * a``,
where ``p`` is the number of pole pairs.
"""

from __future__ import annotations

import math
from cmath import exp
from cmath import phase as angle_of
from dataclasses import dataclass
from math import gcd, pi, sin
from operator import index
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polecraft.core.materials import Conductor
    from polecraft.machines.base import Machine

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


def phase_axis_angle(slots: int, poles: int) -> float:
    """Electrical angle of the phase A axis, in radians.

    The direction of the summed phasors of phase A's coils, measured from
    tooth 0. A magnet pointing away from the machine axis links the most flux
    with phase A when its centre is at this electrical angle.

    A tooth links the field cos(p * angle) over one slot pitch, which gives
    sin(pi p / slots) times the field at its centre. With more pole pairs
    than slots that factor can be negative: the tooth then links flux of the
    opposite sign to the magnet facing it, and the axis is turned by half a
    cycle to account for it.
    """
    slots, poles = _checked_counts(slots, poles)
    pole_pairs = poles // 2
    total = 0j
    for coil in concentrated_winding(slots, poles):
        if coil.phase == "A":
            angle = 2 * pi * ((coil.tooth * pole_pairs) % slots) / slots
            total += coil.direction * exp(1j * angle)
    if sin(pi * (pole_pairs % (2 * slots)) / slots) < 0:
        total = -total
    return angle_of(total)


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


# Electrical model of the winding. Lengths in metres, areas in square metres.

CONNECTIONS = ("star", "delta")

# Densest packing of equal circles in a plane, pi / (2 sqrt 3): no winding
# method puts more round wire into an area than this.
MAX_PACKING = math.pi / (2.0 * math.sqrt(3.0))


@dataclass(frozen=True)
class CoilShape:
    """Size of the coil wound round one tooth, as seen in the cross-section.

    The coil wraps a tooth body of width ``tooth_width``. Each of its two
    sides fills ``side_area`` and is ``thickness`` thick across the slot, so
    along the tooth it is side_area / thickness deep.
    """

    tooth_width: float  # m
    thickness: float  # m, mean width of one coil side across the slot
    side_area: float  # m^2, cross-section of one coil side


def wire_area(diameter: float) -> float:
    """Cross-section of a round wire of ``diameter`` in metres, in square metres."""
    return math.pi * diameter * diameter / 4.0


@dataclass(frozen=True)
class SlotFill:
    """How full one coil side is.

    ``copper_fill`` is bare copper area over the coil side's area, the usual
    fill factor. ``packing`` is the same with the wire's overall (insulated)
    diameter where it is known, else the bare one; above ``MAX_PACKING`` the
    wires cannot fit at all.
    """

    conductors: int  # wires in one coil side: turns x strands
    side_area: float  # m^2
    copper_fill: float
    packing: float
    insulated: bool  # True if ``packing`` used the insulated diameter

    @property
    def fits(self) -> bool:
        """False if the wires could not fit even packed as tightly as circles can be."""
        return self.packing <= MAX_PACKING


def slot_fill(
    shape: CoilShape,
    turns: int,
    strands: int,
    diameter: float,
    insulated_diameter: float | None = None,
) -> SlotFill:
    """Fill of one coil side: ``turns`` turns of ``strands`` wires in hand.

    ``diameter`` is the bare copper diameter in metres, ``insulated_diameter``
    the overall diameter over the enamel, if known.
    """
    conductors = turns * strands
    copper = conductors * wire_area(diameter) / shape.side_area
    overall = insulated_diameter if insulated_diameter is not None else diameter
    return SlotFill(
        conductors=conductors,
        side_area=shape.side_area,
        copper_fill=copper,
        packing=conductors * wire_area(overall) / shape.side_area,
        insulated=insulated_diameter is not None,
    )


def mean_turn_length(
    shape: CoilShape, stack_length: float, end_overhang: float | None = None
) -> float:
    """Mean length of one turn of a tooth coil, in metres.

    The coil's centre line runs round the tooth body (stack_length by
    tooth_width) at half the coil thickness out from it, so the turn is a
    rectangle with rounded corners:

        2 stack_length + 2 tooth_width + pi thickness.

    At each end the coil stands out from the stack by its thickness. If
    ``end_overhang`` (how far it stands out, per end) is given instead, the
    corners take the mean of the two: radius (thickness + end_overhang) / 4,
    giving 2 L + 2 w + pi (thickness + end_overhang) / 2. Either way it is
    an estimate: wire bending, insulation and skew are left out.
    """
    if end_overhang is None:
        end_overhang = shape.thickness
    corners = math.pi * (shape.thickness + end_overhang) / 2.0
    return 2.0 * stack_length + 2.0 * shape.tooth_width + corners


@dataclass(frozen=True)
class PhaseWinding:
    """One phase of the winding as a circuit, with all its coils in series.

    Resistances are in ohms, currents in amperes and voltages in volts.
    Phase quantities are those of one phase winding; line quantities those
    at the motor's three terminals.
    """

    series_turns: int  # turns of one phase, all coils in series
    strands: int  # wires in hand
    wire_diameter: float  # m, bare
    turn_length: float  # m, mean length of one turn
    conductor: Conductor
    temperature: float  # degrees Celsius the winding runs at
    connection: str  # "star" or "delta"
    fill: SlotFill

    @property
    def wire_length(self) -> float:
        """Length of one strand through the whole phase, in metres."""
        return self.series_turns * self.turn_length

    def resistance(self, temperature: float | None = None) -> float:
        """Phase resistance in ohms, at the winding temperature unless one is given (C)."""
        t = self.temperature if temperature is None else temperature
        area = self.strands * wire_area(self.wire_diameter)
        return self.conductor.resistivity_at(t) * self.wire_length / area

    def line_resistance(self, temperature: float | None = None) -> float:
        """Resistance measured between two terminals, in ohms.

        Star: two phases in series, 2 R. Delta: one phase in parallel with
        the other two in series, R (2 R) / (3 R) = 2 R / 3.
        """
        phase = self.resistance(temperature)
        return 2.0 * phase if self.connection == "star" else 2.0 * phase / 3.0

    def line_current(self, phase_current: float) -> float:
        """Terminal current for a phase current; sqrt 3 times larger in delta."""
        return phase_current * (math.sqrt(3.0) if self.connection == "delta" else 1.0)

    def line_voltage(self, phase_voltage: float) -> float:
        """Voltage between terminals for a phase voltage; sqrt 3 times larger in star."""
        return phase_voltage * (math.sqrt(3.0) if self.connection == "star" else 1.0)

    def copper_loss(self, phase_current: float, temperature: float | None = None) -> float:
        """Resistive loss of all three phases in watts, for an RMS phase current."""
        return 3.0 * phase_current**2 * self.resistance(temperature)


def phase_winding(machine: Machine, conductor: Conductor | None = None) -> PhaseWinding:
    """The winding of a machine's design as a circuit.

    Every coil of a phase is taken to be in series, one path. The conductor
    is looked up in the material library by the name the design gives,
    unless one is passed. Raises ValueError if that name is not a conductor
    in the library.
    """
    # Imported here: the material library pulls in SciPy, which loading a
    # design file does not need.
    from polecraft.core.materials import Conductor, load_material

    design = machine.design
    winding = design.winding
    if conductor is None:
        found = load_material(winding.conductor)
        if not isinstance(found, Conductor):
            raise ValueError(f"'{winding.conductor}' is not a conductor in the material library")
        conductor = found
    if winding.connection not in CONNECTIONS:
        raise ValueError(
            f"connection must be one of {', '.join(CONNECTIONS)}, not {winding.connection!r}"
        )
    shape = machine.coil_shape()
    coils = concentrated_winding(design.slots, design.poles)
    per_phase = sum(1 for coil in coils if coil.phase == "A")
    mm = 1e-3
    overhang = design.axial.end_winding_overhang_mm
    insulated = winding.insulated_diameter_mm
    return PhaseWinding(
        series_turns=winding.turns_per_coil * per_phase,
        strands=winding.parallel_strands,
        wire_diameter=winding.wire_diameter_mm * mm,
        turn_length=mean_turn_length(
            shape,
            design.axial.stack_length_mm * mm,
            None if overhang is None else overhang * mm,
        ),
        conductor=conductor,
        temperature=winding.temperature_C,
        connection=winding.connection,
        fill=slot_fill(
            shape,
            winding.turns_per_coil,
            winding.parallel_strands,
            winding.wire_diameter_mm * mm,
            None if insulated is None else insulated * mm,
        ),
    )
