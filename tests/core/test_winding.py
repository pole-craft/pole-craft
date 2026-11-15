"""Symmetry slice and tooth-wound winding layout, checked against known answers."""

import cmath
import math
from typing import Callable

import numpy as np
import pytest

from polecraft.core.winding import (
    Coil,
    Symmetry,
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
