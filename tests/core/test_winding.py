"""Symmetry slice and tooth-wound winding layout, checked against known answers."""

import cmath
import math
from typing import Callable

import numpy as np
import pytest

from polecraft.core.materials import Conductor, load_material
from polecraft.core.winding import (
    MAX_PACKING,
    Coil,
    CoilShape,
    EndTurns,
    PhaseWinding,
    Symmetry,
    SQUARE_GMD,
    end_turn_inductance,
    end_winding_inductance,
    loop_inductance,
    mean_turn_length,
    phase_winding,
    slot_fill,
    wire_area,
    combination_problem,
    concentrated_winding,
    layout_string,
    phase_axis_angle,
    symmetry,
    winding_factor,
)

GRID = [(slots, poles) for slots in range(3, 49) for poles in range(2, 57, 2)]


def can_be_balanced(slots: int, poles: int) -> bool:
    """Count the distinct electrical angles of the teeth; three phases must share them."""
    angles = {(tooth * (poles // 2)) % slots for tooth in range(slots)}
    return slots % 3 == 0 and len(angles) % 3 == 0


VALID = [pair for pair in GRID if can_be_balanced(*pair)]


def failing(check: Callable[[int, int], bool]) -> list[tuple[int, int]]:
    """The valid combinations for which a property does not hold."""
    return [pair for pair in VALID if not check(*pair)]


def electrical_angle(coil: Coil, slots: int, poles: int) -> float:
    """Electrical position of a tooth in radians: pole pairs times its angle."""
    return (poles // 2) * 2 * math.pi * coil.tooth / slots


def phase_phasors(slots: int, poles: int) -> dict[str, complex]:
    """Sum of the coil phasors of each phase."""
    totals = {"A": 0j, "B": 0j, "C": 0j}
    for coil in concentrated_winding(slots, poles):
        angle = electrical_angle(coil, slots, poles)
        totals[coil.phase] += coil.direction * cmath.exp(1j * angle)
    return totals


def test_the_grid_holds_valid_and_invalid_combinations() -> None:
    assert len(VALID) == 336
    assert len(GRID) - len(VALID) == 952


# Symmetry


@pytest.mark.parametrize(
    "slots, poles, expected",
    [
        # 36 slots, 42 poles: one sixth of the machine, 6 slots and 7 poles.
        (36, 42, Symmetry(slots=6, poles=7, periods=6, antiperiodic=True)),
        # 12 slots, 10 poles: half the machine.
        (12, 10, Symmetry(slots=6, poles=5, periods=2, antiperiodic=True)),
        # 24 slots, 28 poles: a quarter of the machine.
        (24, 28, Symmetry(slots=6, poles=7, periods=4, antiperiodic=True)),
        # 12 slots, 8 poles: 3 slots and one pole pair, repeated 4 times.
        (12, 8, Symmetry(slots=3, poles=2, periods=4, antiperiodic=False)),
        # 9 slots, 8 poles: nothing repeats.
        (9, 8, Symmetry(slots=9, poles=8, periods=1, antiperiodic=False)),
        (9, 6, Symmetry(slots=3, poles=2, periods=3, antiperiodic=False)),
        (12, 14, Symmetry(slots=6, poles=7, periods=2, antiperiodic=True)),
        (3, 2, Symmetry(slots=3, poles=2, periods=1, antiperiodic=False)),
    ],
)
def test_symmetry_known_answers(slots: int, poles: int, expected: Symmetry) -> None:
    assert symmetry(slots, poles) == expected


@pytest.mark.parametrize(
    "slots, poles, degrees", [(36, 42, 60), (12, 10, 180), (24, 28, 90), (9, 8, 360)]
)
def test_slice_angle(slots: int, poles: int, degrees: float) -> None:
    assert symmetry(slots, poles).angle_deg == degrees


def test_slices_tile_the_machine() -> None:
    def tiles(slots: int, poles: int) -> bool:
        part = symmetry(slots, poles)
        return part.slots * part.periods == slots and part.poles * part.periods == poles

    assert failing(tiles) == []


def test_no_smaller_slice_exists() -> None:
    # Brute force: the magnet pattern cos(p * angle) must map onto plus or
    # minus itself when turned by a whole number of slots.
    def is_smallest(slots: int, poles: int) -> bool:
        pole_pairs = poles // 2
        smallest = next(
            n
            for n in range(1, slots + 1)
            if math.isclose(abs(math.cos(pole_pairs * 2 * math.pi * n / slots)), 1.0)
        )
        return symmetry(slots, poles).slots == smallest

    assert failing(is_smallest) == []


@pytest.mark.parametrize("slots, poles", [(2, 2), (0, 2), (-3, 2)])
def test_symmetry_rejects_too_few_slots(slots: int, poles: int) -> None:
    with pytest.raises(ValueError, match="slots must be at least 3"):
        symmetry(slots, poles)


@pytest.mark.parametrize("slots, poles", [(12, 0), (12, 7), (12, -2)])
def test_symmetry_rejects_bad_pole_counts(slots: int, poles: int) -> None:
    with pytest.raises(ValueError, match="poles must be an even"):
        symmetry(slots, poles)


def test_symmetry_returns_plain_python_values() -> None:
    part = symmetry(np.int64(36), np.int64(42))
    assert part == Symmetry(slots=6, poles=7, periods=6, antiperiodic=True)
    assert type(part.slots) is int and type(part.poles) is int
    assert type(part.periods) is int and part.antiperiodic is True


# Winding layout


def test_12_slot_10_pole_layout_matches_the_published_one() -> None:
    # Upper case is a coil wound in the positive direction, lower case negative.
    assert layout_string(concentrated_winding(12, 10)) == "AabBCcaABbcC"


def test_other_published_layouts() -> None:
    assert layout_string(concentrated_winding(3, 2)) == "ABC"
    assert layout_string(concentrated_winding(12, 8)) == "ABCABCABCABC"


def test_12_slot_14_pole_layout_is_the_10_pole_one_with_b_and_c_swapped() -> None:
    # Published as AacCBbaACcbB. The same winding may be listed starting from
    # another tooth, so look for the pattern anywhere around the stator.
    layout = layout_string(concentrated_winding(12, 14))
    assert "AacCBbaACcbB" in layout + layout


def test_9_slot_8_pole_layout_groups_three_coils_per_phase() -> None:
    # Published as AaABbBCcC. Besides starting on another tooth, the same
    # winding may be listed with every coil reversed.
    layout = layout_string(concentrated_winding(9, 8))
    doubled = layout + layout
    assert "AaABbBCcC" in doubled or "aAabBbcCc" in doubled


def test_24_slot_22_pole_layout_matches_the_published_one() -> None:
    layout = layout_string(concentrated_winding(24, 22))
    assert "AaAabBbBCcCcaAaABbBbcCcC" in layout + layout


def test_reference_motor_layout_is_three_copies_of_12_slot_14_pole() -> None:
    twelve = layout_string(concentrated_winding(12, 14))
    assert layout_string(concentrated_winding(36, 42)) == twelve * 3


def test_coils_are_listed_tooth_by_tooth() -> None:
    coils = concentrated_winding(12, 10)
    assert [coil.tooth for coil in coils] == list(range(12))
    assert coils[0] == Coil(tooth=0, phase="A", direction=1)
    assert coils[1] == Coil(tooth=1, phase="A", direction=-1)


def test_every_phase_gets_the_same_number_of_coils() -> None:
    def equal_shares(slots: int, poles: int) -> bool:
        coils = concentrated_winding(slots, poles)
        counts = [sum(coil.phase == phase for coil in coils) for phase in "ABC"]
        return counts == [slots // 3] * 3 and {c.direction for c in coils} <= {1, -1}

    assert failing(equal_shares) == []


def test_phases_are_equal_and_120_degrees_apart() -> None:
    # B sits 120 electrical degrees after A, and C 120 after B.
    def balanced(slots: int, poles: int) -> bool:
        totals = phase_phasors(slots, poles)
        third = 2 * math.pi / 3
        return (
            math.isclose(abs(totals["B"]), abs(totals["A"]))
            and math.isclose(abs(totals["C"]), abs(totals["A"]))
            and math.isclose(cmath.phase(totals["B"] / totals["A"]), third)
            and math.isclose(cmath.phase(totals["C"] / totals["B"]), third)
        )

    assert failing(balanced) == []


def test_layout_repeats_with_the_symmetry_slice() -> None:
    def repeats(slots: int, poles: int) -> bool:
        coils = concentrated_winding(slots, poles)
        part = symmetry(slots, poles)
        sign = -1 if part.antiperiodic else 1
        return all(
            coils[(coil.tooth + part.slots) % slots].phase == coil.phase
            and coils[(coil.tooth + part.slots) % slots].direction == sign * coil.direction
            for coil in coils
        )

    assert failing(repeats) == []


def test_every_coil_is_within_30_degrees_of_its_phase_axis() -> None:
    # A coil further than 30 degrees from its axis would link more flux in
    # another phase, so this is what makes the layout the best one.
    axes = {"A": 0.0, "B": 2 * math.pi / 3, "C": 4 * math.pi / 3}

    def aligned(slots: int, poles: int) -> bool:
        for coil in concentrated_winding(slots, poles):
            angle = electrical_angle(coil, slots, poles)
            if coil.direction < 0:
                angle += math.pi
            off_axis = cmath.phase(cmath.exp(1j * (angle - axes[coil.phase])))
            if abs(off_axis) > math.radians(30) + 1e-9:
                return False
        return True

    assert failing(aligned) == []


def test_small_numpy_integers_do_not_overflow() -> None:
    # 36 * 21 does not fit in an 8-bit integer.
    assert concentrated_winding(np.int8(36), np.int8(42)) == concentrated_winding(36, 42)
    assert winding_factor(np.int8(36), np.int8(42)) == winding_factor(36, 42)


# Winding factor


@pytest.mark.parametrize(
    "slots, poles, expected",
    [
        (3, 2, 0.866),
        (6, 4, 0.866),
        (9, 6, 0.866),
        (12, 8, 0.866),
        (9, 8, 0.945),
        (9, 10, 0.945),
        (12, 10, 0.933),
        (12, 14, 0.933),
        (18, 14, 0.902),
        (18, 16, 0.945),
        (24, 20, 0.933),
        (24, 22, 0.9495),
        (24, 28, 0.933),
        (27, 24, 0.945),
        (36, 42, 0.933),
    ],
)
def test_winding_factor_matches_published_tables(
    slots: int, poles: int, expected: float
) -> None:
    assert winding_factor(slots, poles) == pytest.approx(expected, abs=6e-4)


def test_winding_factor_of_12_slot_10_pole_by_hand() -> None:
    # Coil pitch factor sin(75 deg) times distribution factor cos(15 deg).
    by_hand = math.sin(math.radians(75)) * math.cos(math.radians(15))
    assert winding_factor(12, 10) == pytest.approx(by_hand)


def test_winding_factor_is_a_fraction_and_unchanged_by_repeating_the_machine() -> None:
    def sensible(slots: int, poles: int) -> bool:
        factor = winding_factor(slots, poles)
        part = symmetry(slots, poles)
        # The same slice repeated a different number of times is the same winding.
        smallest = part.periods // 2 if part.antiperiodic else part.periods
        base = winding_factor(slots // smallest, poles // smallest)
        return 0 < factor <= 1 and math.isclose(factor, base)

    assert failing(sensible) == []


# Combinations that cannot work


def test_exactly_the_balanced_combinations_are_accepted() -> None:
    wrong = [
        pair
        for pair in GRID
        if (combination_problem(*pair) is None) != can_be_balanced(*pair)
    ]
    assert wrong == []


@pytest.mark.parametrize(
    "slots, poles",
    [
        (10, 8),  # slots not a multiple of 3
        (12, 12),  # coils sit at only two electrical angles, 0 and 180 deg
        (6, 6),
        (12, 6),  # four electrical angles: cannot be shared by three phases
        (9, 18),  # every coil at the same electrical angle
        (24, 24),
        (36, 72),
    ],
)
def test_unbalanced_combinations_are_rejected(slots: int, poles: int) -> None:
    message = combination_problem(slots, poles)
    assert message is not None
    assert str(slots) in message and str(poles) in message
    with pytest.raises(ValueError, match="three-phase"):
        concentrated_winding(slots, poles)
    with pytest.raises(ValueError, match="three-phase"):
        winding_factor(slots, poles)


def test_rejection_message_gives_the_reason() -> None:
    assert "must be a multiple of 3" in combination_problem(10, 8)
    assert "only 2 electrical positions" in combination_problem(12, 12)
    assert "only 4 electrical positions" in combination_problem(12, 6)
    assert "only 1 electrical position," in combination_problem(9, 18)


@pytest.mark.parametrize("slots, poles", [(9, 7), (12, 7), (12, 0)])
def test_winding_rejects_bad_pole_counts(slots: int, poles: int) -> None:
    for function in (combination_problem, concentrated_winding, winding_factor):
        with pytest.raises(ValueError, match="poles must be an even"):
            function(slots, poles)


def test_winding_rejects_too_few_slots() -> None:
    for function in (combination_problem, concentrated_winding, winding_factor):
        with pytest.raises(ValueError, match="slots must be at least 3"):
            function(2, 2)


# Phase axis


def test_phase_axis_lies_between_the_two_teeth_of_a_phase_group() -> None:
    # 12 slots, 10 poles: phase A is tooth 0 at 0 degrees and tooth 1,
    # reversed, at 150 - 180 = -30 electrical degrees.
    assert phase_axis_angle(12, 10) == pytest.approx(math.radians(-15))
    assert phase_axis_angle(36, 42) == pytest.approx(math.radians(-15))
    assert phase_axis_angle(24, 22) == pytest.approx(math.radians(-7.5))


def test_phase_axis_is_on_tooth_0_when_a_phase_group_is_symmetric_about_it() -> None:
    assert phase_axis_angle(12, 8) == pytest.approx(0.0)
    assert phase_axis_angle(9, 8) == pytest.approx(0.0)


def test_phase_axis_rejects_unbalanced_combinations() -> None:
    with pytest.raises(ValueError, match="three-phase"):
        phase_axis_angle(12, 12)


# Electrical model

MM = 1e-3
COPPER = Conductor(resistivity=1.7241e-8, temperature_coefficient=0.00393, density=8890.0)
SHAPE = CoilShape(tooth_width=3.0 * MM, thickness=2.0 * MM, side_area=12.0 * MM**2)


def a_winding(connection: str = "star", strands: int = 1) -> PhaseWinding:
    return PhaseWinding(
        series_turns=240,
        strands=strands,
        wire_diameter=0.5 * MM,
        turn_length=mean_turn_length(SHAPE, 12.0 * MM),
        conductor=COPPER,
        temperature=100.0,
        connection=connection,
        fill=slot_fill(SHAPE, 20, strands, 0.5 * MM),
    )


def test_wire_area_of_half_a_millimetre() -> None:
    assert wire_area(0.5 * MM) == pytest.approx(0.19635 * MM**2, rel=1e-5)


def test_slot_fill_by_hand() -> None:
    # 20 turns x 2 strands of 0.5 mm: 40 x 0.19635 = 7.854 mm2 in 12 mm2.
    fill = slot_fill(SHAPE, 20, 2, 0.5 * MM)
    assert fill.conductors == 40
    assert fill.copper_fill == pytest.approx(7.854 / 12.0, rel=1e-4)
    assert fill.packing == fill.copper_fill and not fill.insulated
    # Over the enamel, 0.55 mm: 40 x 0.23758 = 9.503 mm2.
    insulated = slot_fill(SHAPE, 20, 2, 0.5 * MM, 0.55 * MM)
    assert insulated.copper_fill == fill.copper_fill
    assert insulated.packing == pytest.approx(9.503 / 12.0, rel=1e-4) and insulated.insulated


def test_packing_limit_is_that_of_circles_in_a_plane() -> None:
    assert MAX_PACKING == pytest.approx(0.9069, abs=1e-4)
    wires = MAX_PACKING * SHAPE.side_area / wire_area(0.5 * MM)  # just over 55 wires
    assert slot_fill(SHAPE, math.floor(wires), 1, 0.5 * MM).fits
    assert not slot_fill(SHAPE, math.ceil(wires), 1, 0.5 * MM).fits


def test_mean_turn_length_by_hand() -> None:
    # 2 x 12 + 2 x 3 + pi x 2 = 36.283 mm.
    assert mean_turn_length(SHAPE, 12.0 * MM) == pytest.approx(36.283 * MM, rel=1e-4)
    # The default end overhang is the coil's own thickness.
    assert mean_turn_length(SHAPE, 12.0 * MM, 2.0 * MM) == mean_turn_length(SHAPE, 12.0 * MM)
    # A 4 mm overhang: 2 x 12 + 2 x 3 + pi (2 + 4) / 2 = 39.425 mm.
    assert mean_turn_length(SHAPE, 12.0 * MM, 4.0 * MM) == pytest.approx(39.425 * MM, rel=1e-4)


def test_phase_resistance_matches_a_hand_calculation() -> None:
    # Copper at 100 C: 1.7241e-8 x (1 + 0.00393 x 80) = 2.26616e-8 ohm m.
    # Wire: 240 turns x 36.2832 mm = 8.70796 m; area 0.196350 mm2.
    # R = 2.26616e-8 x 8.70796 / 0.196350e-6 = 1.00502 ohm.
    winding = a_winding()
    assert winding.wire_length == pytest.approx(8.70796, rel=1e-5)
    assert winding.resistance() == pytest.approx(1.00502, rel=1e-5)
    # At 20 C the resistivity is the tabulated one: 1.00502 / 1.3144.
    assert winding.resistance(20.0) == pytest.approx(1.00502 / 1.3144, rel=1e-5)


def test_strands_in_hand_share_the_current() -> None:
    assert a_winding(strands=2).resistance() == pytest.approx(a_winding().resistance() / 2)


def test_star_line_quantities() -> None:
    star = a_winding("star")
    r = star.resistance()
    assert star.line_resistance() == pytest.approx(2.0 * r)
    assert star.line_current(10.0) == 10.0
    assert star.line_voltage(10.0) == pytest.approx(10.0 * math.sqrt(3.0))


def test_delta_line_quantities() -> None:
    delta = a_winding("delta")
    r = delta.resistance()
    # One phase in parallel with the other two: R x 2R / 3R.
    assert delta.line_resistance() == pytest.approx(r * 2.0 * r / (3.0 * r))
    assert delta.line_current(10.0) == pytest.approx(10.0 * math.sqrt(3.0))
    assert delta.line_voltage(10.0) == 10.0


def test_star_and_delta_deliver_the_same_power_for_the_same_phase_quantities() -> None:
    for winding in (a_winding("star"), a_winding("delta")):
        # Three phases: sqrt 3 x line voltage x line current = 3 x phase power.
        power = math.sqrt(3.0) * winding.line_voltage(48.0) * winding.line_current(5.0)
        assert power == pytest.approx(3.0 * 48.0 * 5.0)
        assert winding.copper_loss(5.0) == pytest.approx(3.0 * 25.0 * winding.resistance())


def test_copper_loss_takes_the_rms_current_not_the_peak(repo_root) -> None:
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    machine = machine_for(load_design(repo_root / "designs" / "qdd_outer_36s42p.yaml"))
    winding = phase_winding(machine)
    peak = 10.0
    # The loss at each rotor angle of a cycle, from the phase currents the
    # studies apply for a peak current: sum of i^2 R over the phases.
    angles = 2.0 * math.pi / (machine.design.poles // 2) * np.arange(24) / 24
    sampled = [
        sum(i * i for i in machine.excitation(float(angle), peak, 0.3).values())
        * winding.resistance()
        for angle in angles
    ]
    assert winding.copper_loss(rms_phase_current=peak / math.sqrt(2.0)) == pytest.approx(
        float(np.mean(sampled)), rel=1e-12
    )
    # Passing the peak current would double it.
    assert winding.copper_loss(peak) == pytest.approx(2.0 * float(np.mean(sampled)), rel=1e-12)
    assert winding.copper_loss(peak / math.sqrt(2.0), 20.0) == pytest.approx(
        1.5 * peak**2 * winding.resistance(20.0)
    )


def test_example_design_winding(repo_root) -> None:
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    machine = machine_for(load_design(repo_root / "designs" / "qdd_outer_36s42p.yaml"))
    winding = phase_winding(machine)
    shape = machine.coil_shape()
    # 36 slots: 12 coils of 20 turns in each phase.
    assert winding.series_turns == 240
    assert winding.conductor == load_material("copper")
    assert winding.temperature == 100.0 and winding.connection == "star"
    turn = 2 * 12.0 * MM + 2 * 3.0 * MM + math.pi * shape.thickness
    assert winding.turn_length == pytest.approx(turn, rel=1e-12)
    expected = COPPER.resistivity_at(100.0) * 240 * turn / wire_area(0.5 * MM)
    assert winding.resistance() == pytest.approx(expected, rel=1e-12)
    assert winding.fill.copper_fill == pytest.approx(20 * wire_area(0.5 * MM) / shape.side_area)


def test_winding_needs_a_conductor_from_the_library(repo_root) -> None:
    import dataclasses

    from polecraft.design import load_design
    from polecraft.machines import machine_for

    design = load_design(repo_root / "designs" / "qdd_outer_36s42p.yaml")
    for name, message in (("silver", "unknown material"), ("M270-35A", "not a conductor")):
        changed = dataclasses.replace(
            design, winding=dataclasses.replace(design.winding, conductor=name)
        )
        with pytest.raises(ValueError, match=message):
            phase_winding(machine_for(changed))


# Inductance of the end turns

MU0 = 4e-7 * math.pi
TOOTH_COIL = CoilShape(tooth_width=3.0 * MM, thickness=1.44 * MM, side_area=12.96 * MM**2)


def ring_of_round_wire(radius: float, wire: float, cells: int) -> tuple[list, float]:
    """Filaments filling a round wire bent into a ring, and the side of their cells."""
    cell = 2.0 * wire / cells
    angle = np.linspace(0.0, 2.0 * math.pi, 241)[:-1]
    loops = []
    for i in range(cells):
        for j in range(cells):
            u, v = -wire + (i + 0.5) * cell, -wire + (j + 0.5) * cell
            if u * u + v * v > wire * wire:
                continue
            points = np.column_stack(
                [(radius + u) * np.cos(angle), (radius + u) * np.sin(angle), np.full(angle.size, v)]
            )
            loops.append((points, np.roll(points, -1, axis=0)))
    return loops, cell


def test_loop_inductance_of_a_ring_matches_the_textbook_formula() -> None:
    # A ring of radius R in round wire of radius a, current spread evenly:
    # L = mu0 R (ln(8 R / a) - 7/4).
    radius, wire = 10.0 * MM, 1.0 * MM
    loops, cell = ring_of_round_wire(radius, wire, 8)
    exact = MU0 * radius * (math.log(8.0 * radius / wire) - 1.75)
    # Within 0.5 % with 52 filaments; what is left is the stepped edge of the
    # round section.
    assert loop_inductance(loops, cell) == pytest.approx(exact, rel=0.01)


def test_one_filament_has_the_inductance_of_a_thin_ring_at_its_mean_distance() -> None:
    # A single filament has no neighbours, so its own term is the whole
    # answer: a thin ring of radius R whose section has geometric mean
    # distance g from itself has L = mu0 R (ln(8 R / g) - 2), which is the
    # mutual inductance of two such rings a distance g apart. Terms of order
    # (g / R)^2 are left out of the formula; they are 0.06 % here.
    radius, cell = 10.0 * MM, 1.0 * MM
    angle = np.linspace(0.0, 2.0 * math.pi, 481)[:-1]
    points = np.column_stack(
        [radius * np.cos(angle), radius * np.sin(angle), np.zeros(angle.size)]
    )
    ring = (points, np.roll(points, -1, axis=0))
    assert SQUARE_GMD == pytest.approx(0.44705, abs=1e-5)
    exact = MU0 * radius * (math.log(8.0 * radius / (SQUARE_GMD * cell)) - 2.0)
    assert loop_inductance([ring], cell) == pytest.approx(exact, rel=2e-3)
    # A thinner conductor has more: ln(1 / 0.3) more for each mu0 R.
    thin = loop_inductance([ring], 0.3 * cell)
    assert thin - loop_inductance([ring], cell) == pytest.approx(
        MU0 * radius * math.log(1.0 / 0.3), rel=5e-3
    )


def test_loop_inductance_needs_loops_and_a_cell() -> None:
    loops, cell = ring_of_round_wire(10.0 * MM, 1.0 * MM, 2)
    with pytest.raises(ValueError, match="at least one loop"):
        loop_inductance([], cell)
    with pytest.raises(ValueError, match="cell"):
        loop_inductance(loops, 0.0)


def coil_as_closed_loops(
    shape: CoilShape, stack: float, overhang: float, grid: tuple[int, int]
) -> tuple[list, float, float]:
    """A straightened tooth coil as closed filaments, the side of their cells, and its 2D value.

    Each filament runs up one side of the tooth, round a quarter ellipse,
    across the end of the tooth, round a second quarter ellipse and back the
    same way. The 2D value, in H/m per turn squared, is that of the sides
    as a two-wire line: mu0 / pi times the mean of ln(distance to the return
    side / distance to the same side) over all pairs of filaments.
    """
    across, along = grid
    depth = shape.side_area / shape.thickness
    offsets = (np.arange(across) + 0.5) / across * shape.thickness
    heights = (np.arange(along) + 0.5) / along * depth
    cell = math.sqrt(shape.thickness / across * depth / along)
    step = 0.5 * min(shape.thickness / across, depth / along)
    half_width, half_length = shape.tooth_width / 2.0, stack / 2.0
    loops = []
    for offset in offsets:
        reach = offset * overhang / shape.thickness
        turn = np.linspace(0.0, 0.5 * math.pi, max(4, math.ceil((offset + reach) / step)) + 1)
        z = np.linspace(-half_length, half_length, math.ceil(stack / step) + 1)
        x = np.linspace(half_width, -half_width, math.ceil(shape.tooth_width / step) + 1)
        flat = np.concatenate(
            [
                np.column_stack([np.full(z.size, half_width + offset), z]),
                np.column_stack(
                    [half_width + offset * np.cos(turn), half_length + reach * np.sin(turn)]
                )[1:],
                np.column_stack([x, np.full(x.size, half_length + reach)])[1:],
                np.column_stack(
                    [-half_width - offset * np.sin(turn), half_length + reach * np.cos(turn)]
                )[1:],
            ]
        )
        # The second half is the first turned half a turn about the y axis.
        path = np.concatenate([flat, -flat[1:-1]])
        for height in heights:
            points = np.column_stack([path[:, 0], np.full(len(path), height), path[:, 1]])
            loops.append((points, np.roll(points, -1, axis=0)))
    x = np.repeat(half_width + offsets, along)
    y = np.tile(heights, across)
    same = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(same, SQUARE_GMD * cell)
    opposite = np.hypot(x[:, None] + x[None, :], y[:, None] - y[None, :])
    return loops, cell, MU0 / math.pi * float(np.mean(np.log(opposite / same)))


@pytest.mark.parametrize("overhang", [1.44 * MM, 4.0 * MM], ids=["1.44 mm", "4 mm"])
def test_end_turns_and_2d_value_add_up_to_the_whole_coil_in_air(overhang: float) -> None:
    # The same filaments as closed loops, every segment against every other
    # by Neumann's formula, give the inductance of the whole coil in air.
    # The end-turn part is defined as that less the 2D value of the sides.
    grid, stack = (2, 3), 12.0 * MM
    loops, cell, per_metre = coil_as_closed_loops(TOOTH_COIL, stack, overhang, grid)
    whole = loop_inductance(loops, cell)
    ends = end_turn_inductance(TOOTH_COIL, stack, overhang, grid)
    # Measured 0.2 % apart: the sides are summed exactly in one and by the
    # midpoint rule in the other.
    assert ends.air + per_metre * stack == pytest.approx(whole, rel=0.01)
    # The end turns are under a third of the whole, so they are held less
    # tightly by it; measured 1 % and 0.7 % apart.
    assert ends.air == pytest.approx(whole - per_metre * stack, rel=0.03)
    assert 0.1 < ends.air / whole < 0.35


def test_end_face_term_of_a_wide_flat_coil_is_that_of_two_parallel_wires() -> None:
    # One filament on a tooth far wider than the coil is thick: nearly all of
    # each end turn is the straight run across the tooth, 1 mm beyond the end
    # face. Its mirror image is a parallel run of the same length carrying
    # the same current 2 mm away, and two parallel filaments of length l a
    # distance d apart have M = mu0 / (2 pi) (l asinh(l / d) - sqrt(l^2 + d^2) + d).
    width, thickness, overhang = 100.0 * MM, 0.2 * MM, 2.0 * MM
    shape = CoilShape(tooth_width=width, thickness=thickness, side_area=thickness**2)
    # The filament is at mid-thickness, so its run is half the overhang out.
    apart = overhang
    mutual = MU0 / (2.0 * math.pi) * (
        width * math.asinh(width / apart) - math.hypot(width, apart) + apart
    )
    face = end_turn_inductance(shape, 20.0 * MM, overhang, grid=(1, 1)).face
    # Two ends. Measured 0.1 % under: the short corners add a little of
    # either sign.
    assert face == pytest.approx(2.0 * mutual, rel=0.01)


def test_end_face_term_of_a_narrow_coil_standing_far_out_is_negative() -> None:
    # One filament on a narrow tooth with a long overhang: each end turn is
    # a hairpin, two legs of length l = 8 mm standing on the end face
    # w = 4 mm apart, joined by the run across the tooth. The current in the
    # legs is across the face, so their images carry it the other way:
    #   each leg and its own image, in line, opposed:        -M_end(l, g)
    #   each leg and the image of the other leg, alike:      +M_end(l, w)
    #   the run and its image, side by side 2 l apart:       +M_side(w, 2 l)
    # and the run is at right angles to the legs. M_side(l, d) is the mutual
    # inductance of two parallel filaments of length l side by side, as in
    # the test above, and M_end(l, d) that of two such filaments one after
    # the other, on lines d apart: M_side(2 l, d) / 2 - M_side(l, d), since
    # two filaments of length 2 l side by side are two such pairs and two
    # side-by-side pairs of length l. A leg and its own image stand for one
    # conductor of side t, so g is its geometric mean distance from itself.
    width, thickness, overhang = 4.0 * MM, 0.1 * MM, 16.0 * MM
    shape = CoilShape(tooth_width=width, thickness=thickness, side_area=thickness**2)
    leg = overhang / 2.0  # the filament is at mid-thickness

    def side(length: float, apart: float) -> float:
        return MU0 / (2.0 * math.pi) * (
            length * math.asinh(length / apart) - math.hypot(length, apart) + apart
        )

    def end_to_end(length: float, apart: float) -> float:
        return 0.5 * side(2.0 * length, apart) - side(length, apart)

    one_end = (
        -2.0 * end_to_end(leg, SQUARE_GMD * thickness)
        + 2.0 * end_to_end(leg, width)
        + side(width, 2.0 * leg)
    )
    assert one_end < 0.0
    face = end_turn_inductance(shape, 20.0 * MM, overhang, grid=(1, 1)).face
    # Measured 0.4 % apart: the legs are quarter ellipses, 0.05 mm out of
    # true at the face. With the images' current along the legs left as it
    # is, not reversed, the sum would be +1.5 nH in place of -1.1 nH.
    assert face == pytest.approx(2.0 * one_end, rel=0.02)


def test_end_turn_part_settles_as_the_stack_grows() -> None:
    # The 2D part grows with the stack; what is left is the ends. The two
    # ends of a coil carry opposite currents across the tooth and take a
    # little from each other, less the further apart they are (as 1 / L),
    # so the change halves with each doubling of the stack.
    air = [end_turn_inductance(TOOTH_COIL, length * MM).air for length in (24.0, 48.0, 96.0, 192.0)]
    changes = np.diff(air)
    assert np.all(changes > 0.0)
    assert changes[1:] / changes[:-1] == pytest.approx([0.5, 0.5], abs=0.1)
    assert air[3] == pytest.approx(air[2], rel=0.02)
    # The end-face term does not see the stack length at all.
    faces = [end_turn_inductance(TOOTH_COIL, length * MM).face for length in (12.0, 48.0)]
    assert faces[1] == pytest.approx(faces[0], rel=1e-9)


def test_end_turn_inductance_settles_with_more_filaments() -> None:
    coarse = end_turn_inductance(TOOTH_COIL, 12.0 * MM, grid=(4, 12))
    fine = end_turn_inductance(TOOTH_COIL, 12.0 * MM, grid=(6, 18))
    assert fine.air == pytest.approx(coarse.air, rel=0.02)
    assert fine.face == pytest.approx(coarse.face, rel=0.02)


def test_end_turns_of_the_reference_coil() -> None:
    ends = end_turn_inductance(TOOTH_COIL, 12.0 * MM)
    assert ends.air > 0.0 and ends.face > 0.0
    assert ends.estimate == ends.air + 0.5 * ends.face and ends.spread == 0.5 * ends.face
    # About a nanohenry and a half per turn squared: 20 turns on each of 12
    # coils give 7 microhenries a phase, against 0.6 mH from the 2D model.
    # The air part is tied to the whole coil's inductance, and the face part
    # to parallel wires, by the tests above; these hold the values here.
    assert ends.air == pytest.approx(0.864e-9, rel=0.1)
    # Tight: softening the images of other filaments too would move it 0.4 %.
    assert ends.face == pytest.approx(1.1611e-9, rel=2e-3)
    assert ends.estimate == pytest.approx(1.445e-9, rel=0.1)


def test_default_overhang_is_the_coil_thickness() -> None:
    given = end_turn_inductance(TOOTH_COIL, 12.0 * MM, TOOTH_COIL.thickness)
    assert end_turn_inductance(TOOTH_COIL, 12.0 * MM) == given


def test_end_turns_that_stand_out_further_add_more() -> None:
    near = end_turn_inductance(TOOTH_COIL, 12.0 * MM, 1.0 * MM)
    far = end_turn_inductance(TOOTH_COIL, 12.0 * MM, 4.0 * MM)
    assert far.air > near.air


@pytest.mark.parametrize(
    ("stack", "overhang", "grid", "message"),
    [
        (0.0, None, (4, 12), "stack length"),
        (12.0 * MM, -1.0 * MM, (4, 12), "end overhang"),
        (math.nan, None, (4, 12), "stack length"),
        (12.0 * MM, None, (0, 12), "grid"),
    ],
)
def test_end_turn_inputs_are_checked(stack, overhang, grid, message) -> None:
    with pytest.raises(ValueError, match=message):
        end_turn_inductance(TOOTH_COIL, stack, overhang, grid)


def test_phase_end_winding_scales_with_coils_and_turns_squared(repo_root) -> None:
    from polecraft.design import design_from_dict, design_to_dict, load_design
    from polecraft.machines import machine_for

    design = load_design(repo_root / "designs" / "qdd_outer_36s42p.yaml")
    machine = machine_for(design)
    shape = machine.coil_shape()
    one = end_turn_inductance(shape, 12.0 * MM)
    phase = end_winding_inductance(machine)
    # 12 coils of 20 turns in series.
    assert phase.estimate == pytest.approx(12 * 20**2 * one.estimate, rel=1e-12)
    assert phase.spread == pytest.approx(12 * 20**2 * one.spread, rel=1e-12)
    data = design_to_dict(design)
    data["winding"]["turns_per_coil"] = 10
    data["axial"]["end_winding_overhang_mm"] = 3.0
    other = end_winding_inductance(machine_for(design_from_dict(data)))
    expected = end_turn_inductance(shape, 12.0 * MM, 3.0 * MM)
    assert other.air == pytest.approx(12 * 10**2 * expected.air, rel=1e-12)
    assert isinstance(one, EndTurns)
