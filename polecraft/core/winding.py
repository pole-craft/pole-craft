"""Slot and pole layout, and the winding as an electrical circuit.

Covers three-phase concentrated (tooth-wound) windings with a coil on every
tooth, the usual choice for QDD motors. The layout needs only slot and pole
counts. The electrical part (wire, slot fill, turn length, resistance, the
inductance of the end turns) needs the size of a coil, which a machine gives
as a ``CoilShape``; nothing here depends on the machine type.

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
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

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


# Inductance of the end turns. The 2D field model has no ends: it gives the
# inductance of the coil sides per metre of stack, as if they were infinitely
# long. The end turns add flux that it cannot see. That part is estimated in
# air, as the inductance of the real 3D coil less the 2D value of the same
# coil sides over the stack length ("3D less 2D"), plus part of what the iron
# end face of the stator adds by acting as a magnetic mirror.

# The classical value, as in materials.MU0; not imported from there because
# that module loads SciPy, which loading a design file does not need.
MU0 = 4e-7 * math.pi

# Geometric mean distance of a square cross-section from itself, as a
# fraction of its side: the distance at which a thin filament has the same
# self-inductance as the square conductor it stands for.
SQUARE_GMD = 0.44705


def _neumann(
    middles: NDArray[np.float64],
    steps: NDArray[np.float64],
    other_middles: NDArray[np.float64],
    other_steps: NDArray[np.float64],
    nearest: float = 0.0,
) -> float:
    """Sum over all pairs of segments of (step . other step) / distance, by the midpoint rule.

    Distances under ``nearest`` are raised to it. Times mu0 / (4 pi) this is
    Neumann's mutual inductance of the two sets of segments, in H.
    """
    distance = np.linalg.norm(middles[:, None, :] - other_middles[None, :, :], axis=2)
    return float(np.sum((steps @ other_steps.T) / np.maximum(distance, nearest)))


def loop_inductance(
    loops: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]], cell: float
) -> float:
    """Inductance of a bundle of closed filaments that share one current equally, in H.

    Each loop is a pair of arrays (start points, end points) of its straight
    segments, shape (n, 3), in metres. The result is the mean of the mutual
    inductances of every pair of loops (Neumann's formula, by the midpoint
    rule): the inductance of one turn of a conductor whose cross-section the
    filaments share out, so a coil of N such turns has N^2 times it. Within a
    loop, distances are not allowed below the geometric mean distance of a
    square of side ``cell`` (the cross-section each filament stands for),
    which stands in for the conductor's own size. Segments must be short
    against the distance between neighbouring filaments.
    """
    if not loops:
        raise ValueError("at least one loop is needed")
    if not (math.isfinite(cell) and cell > 0.0):
        raise ValueError(f"cell must be a positive length in metres, not {cell}")
    middles = [0.5 * (a + b) for a, b in loops]
    steps = [b - a for a, b in loops]
    all_middles, all_steps = np.concatenate(middles), np.concatenate(steps)
    owner = np.concatenate([np.full(len(m), k) for k, m in enumerate(middles)])
    nearest = SQUARE_GMD * cell
    total = 0.0
    for k, (middle, step) in enumerate(zip(middles, steps)):
        total += _neumann(middle, step, middle, step, nearest)
        rest = owner != k
        total += _neumann(middle, step, all_middles[rest], all_steps[rest])
    return MU0 / (4.0 * math.pi) * total / len(loops) ** 2


def _parallel_mutual(length: float, distance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mutual inductance of two parallel straight filaments side by side, both ``length`` long, in H.

    The exact result of Neumann's formula: mu0 / (2 pi) (l asinh(l / d) -
    sqrt(l^2 + d^2) + d) for filaments a distance d apart.
    """
    return (
        MU0
        / (2.0 * math.pi)
        * (length * np.arcsinh(length / distance) - np.hypot(length, distance) + distance)
    )


def _graded(length: float, first: float, growth: float = 1.15) -> NDArray[np.float64]:
    """Points from 0 to ``length``, ``first`` apart at both ends and further apart in between."""
    half = [0.0]
    step = first
    while half[-1] + step < 0.5 * length:
        half.append(half[-1] + step)
        step *= growth
    half_array = np.array(half)
    return np.concatenate([half_array, [0.5 * length], length - half_array[::-1]])


def _end_path(
    half_width: float, half_length: float, offset: float, reach: float, height: float, step: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Segments (starts, ends) of a filament's two end turns: corners and the run across the tooth.

    Corners are quarter ellipses, ``offset`` sideways and ``reach`` along the
    stack; the run across the tooth is ``reach`` beyond the stack's end.
    """
    def count(length: float) -> int:
        return max(4, math.ceil(length / step))

    arcs = count(0.25 * math.pi * (offset + reach))
    across = count(2.0 * half_width)
    starts, ends = [], []
    for sign in (1.0, -1.0):
        # Corner from the side at x = +(w/2 + s), across, corner down to the other side.
        angle = (np.arange(arcs + 1) / arcs) * 0.5 * math.pi
        first = np.column_stack(
            [half_width + offset * np.cos(angle), np.full(arcs + 1, height),
             sign * (half_length + reach * np.sin(angle))]
        )
        run = np.linspace(half_width, -half_width, across + 1)
        middle = np.column_stack(
            [run, np.full(across + 1, height), np.full(across + 1, sign * (half_length + reach))]
        )
        second = np.column_stack(
            [-half_width - offset * np.sin(angle), np.full(arcs + 1, height),
             sign * (half_length + reach * np.cos(angle))]
        )
        path = np.concatenate([first, middle[1:], second[1:]])
        if sign < 0.0:
            path = path[::-1]  # the far end is run through the other way round
        starts.append(path[:-1])
        ends.append(path[1:])
    return np.concatenate(starts), np.concatenate(ends)


@dataclass(frozen=True)
class EndTurns:
    """Inductance the end turns of one tooth coil add, per turn squared, in H.

    ``air`` is the 3D coil in air less the 2D value of its sides over the
    stack length. ``face`` is what an ideal iron end face (a magnetic mirror)
    would add to that. The estimate takes half the face term, since the end
    turns lie partly over the tooth's end face and partly over the slots, and
    the spread covers both extremes.
    """

    air: float
    face: float

    @property
    def estimate(self) -> float:
        """Air value plus half the end-face term, H per turn squared."""
        return self.air + 0.5 * self.face

    @property
    def spread(self) -> float:
        """Half the end-face term: the estimate is this much from either extreme, H."""
        return 0.5 * self.face


def end_turn_inductance(
    shape: CoilShape,
    stack_length: float,
    end_overhang: float | None = None,
    grid: tuple[int, int] = (4, 12),
) -> EndTurns:
    """Inductance the end turns of one tooth coil add to its 2D value, per turn squared.

    The coil is straightened (the tooth's curvature is left out) and its
    cross-section, ``shape.thickness`` across the slot by side_area /
    thickness along the tooth, is shared out among grid[0] x grid[1]
    filaments; grids coarser than the default give values too high (by about
    10 % at (2, 3)), and finer ones change it by under 1 %. The corners are
    quarter ellipses, and the ends stand out from the stack by
    ``end_overhang`` per end (the coil thickness if not given), as in
    ``mean_turn_length``. The straight sides are coupled by the exact
    formula for parallel filaments, everything else by Neumann's formula on
    short segments.

    Left out: coupling between the end turns of neighbouring coils, the
    slots in the end face, saturation, and the fringing of the field at the
    ends of the stack and the magnets. The result is an estimate, good to
    perhaps 40 %; no 3D field check of it is possible here. For a short coil
    with tall, narrow sides it is a few percent of the 2D inductance. The
    simpler formula of a semicircular end turn in air (Hanselman, Brushless
    Permanent Magnet Motor Design, 2nd ed., eq. 4.29) does not apply to such
    coils: its logarithm's argument comes out near 1.
    """
    if end_overhang is None:
        end_overhang = shape.thickness
    for name, value in (
        ("stack length", stack_length),
        ("end overhang", end_overhang),
        ("coil thickness", shape.thickness),
        ("coil side area", shape.side_area),
        ("tooth width", shape.tooth_width),
    ):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be a positive length, not {value}")
    across, along = grid
    if across < 1 or along < 1:
        raise ValueError(f"grid must have at least one filament each way, not {grid}")
    depth = shape.side_area / shape.thickness
    offsets = (np.arange(across) + 0.5) / across * shape.thickness
    heights = (np.arange(along) + 0.5) / along * depth
    cell = math.sqrt(shape.thickness / across * depth / along)
    nearest = SQUARE_GMD * cell
    # Segments short against the distance between neighbouring filaments,
    # which the midpoint rule needs to judge their coupling.
    step = 0.5 * min(shape.thickness / across, depth / along)
    half_width, half_length = shape.tooth_width / 2.0, stack_length / 2.0

    # Filament k is at x = +-(w/2 + offset) on its sides, at height y.
    x = np.repeat(half_width + offsets, along)
    y = np.tile(heights, across)
    count = x.size
    reach = np.repeat(offsets * end_overhang / shape.thickness, along)

    # Sides against sides, exactly. The side at +x runs towards +z and the
    # one at -x back towards -z, so the two sides of different filaments
    # oppose each other and the two like sides add.
    same = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(same, nearest)
    cross = np.hypot(x[:, None] + x[None, :], y[:, None] - y[None, :])
    sides = 2.0 * float(np.sum(_parallel_mutual(stack_length, same)
                               - _parallel_mutual(stack_length, cross)))

    # Sides as segments, finer near the ends of the stack where the end
    # turns are, for their coupling with the end turns.
    z = _graded(stack_length, step) - half_length
    side_middles, side_steps, end_middles, end_steps, owner_side, owner_end = [], [], [], [], [], []
    for k in range(count):
        middle_z = 0.5 * (z[:-1] + z[1:])
        length = np.diff(z)
        for sign in (1.0, -1.0):
            side_middles.append(np.column_stack(
                [np.full(middle_z.size, sign * x[k]), np.full(middle_z.size, y[k]), sign * middle_z]))
            side_steps.append(np.column_stack(
                [np.zeros(length.size), np.zeros(length.size), sign * length]))
            owner_side.append(np.full(length.size, k))
        a, b = _end_path(half_width, half_length, float(offsets[k // along]), float(reach[k]),
                         float(y[k]), step)
        end_middles.append(0.5 * (a + b))
        end_steps.append(b - a)
        owner_end.append(np.full(len(a), k))
    all_ends_middle = np.concatenate(end_middles)
    all_ends_step = np.concatenate(end_steps)
    end_owner = np.concatenate(owner_end)
    side_middle = np.concatenate(side_middles)
    side_step = np.concatenate(side_steps)
    side_owner = np.concatenate(owner_side)

    others = 0.0
    for k in range(count):
        mine = side_owner == k
        # Each side segment with every end segment, counted both ways round;
        # only a filament's own segments meet closer than its cell.
        for theirs, closest in ((end_owner == k, nearest), (end_owner != k, 0.0)):
            others += 2.0 * _neumann(
                side_middle[mine], side_step[mine],
                all_ends_middle[theirs], all_ends_step[theirs], closest,
            )
        others += _neumann(end_middles[k], end_steps[k], end_middles[k], end_steps[k], nearest)
        rest = end_owner != k
        others += _neumann(end_middles[k], end_steps[k], all_ends_middle[rest], all_ends_step[rest])
    total = (sides + MU0 / (4.0 * math.pi) * others) / count**2

    # The 2D value of the same coil sides: a two-wire line with its outward
    # side at x = w/2 + s and its return side at -(w/2 + s).
    per_metre = MU0 / math.pi * float(np.mean(np.log(cross / same)))

    # The end face at z = +-L/2 as an ideal magnetic mirror: the image of a
    # current keeps its components along the face and reverses the one
    # across it, which is what reflecting the points does.
    face = 0.0
    for sign in (1.0, -1.0):
        plane = sign * half_length
        at_this_end = sign * all_ends_middle[:, 2] > half_length
        image_middle = all_ends_middle[at_this_end].copy()
        image_middle[:, 2] = 2.0 * plane - image_middle[:, 2]
        image_step = all_ends_step[at_this_end] * np.array([1.0, 1.0, -1.0])
        face += _neumann(all_ends_middle[at_this_end], all_ends_step[at_this_end],
                         image_middle, image_step, nearest)
    face *= MU0 / (4.0 * math.pi) / count**2
    return EndTurns(air=total - per_metre * stack_length, face=face)


@dataclass(frozen=True)
class EndWinding:
    """End-turn inductance of one phase, all its coils in series, in H."""

    estimate: float
    spread: float  # the estimate is uncertain by at least this much either way
    air: float  # in air alone, the low end
    face: float  # what an ideal iron end face adds to ``air``, the high end


def end_winding_inductance(machine: Machine, grid: tuple[int, int] = (4, 12)) -> EndWinding:
    """End-turn inductance of one phase of a machine's winding, in H.

    Each coil adds turns^2 times ``end_turn_inductance``; the coils of a
    phase are in series and the coupling between their end turns is left
    out. The end overhang is the design's ``end_winding_overhang_mm``, or the
    coil thickness if it gives none. Raises NotImplementedError if the
    machine does not describe its coils.
    """
    design = machine.design
    shape = machine.coil_shape()
    overhang = design.axial.end_winding_overhang_mm
    turns = design.winding.turns_per_coil
    coils = sum(1 for coil in concentrated_winding(design.slots, design.poles) if coil.phase == "A")
    one = end_turn_inductance(
        shape,
        design.axial.stack_length_mm * 1e-3,
        None if overhang is None else overhang * 1e-3,
        grid,
    )
    scale = coils * turns**2
    return EndWinding(
        estimate=scale * one.estimate,
        spread=scale * one.spread,
        air=scale * one.air,
        face=scale * one.face,
    )
