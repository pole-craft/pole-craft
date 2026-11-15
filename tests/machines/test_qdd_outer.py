"""Outer-rotor parameters: derived radii and whether the parts fit."""

import math
from pathlib import Path

import pytest

from polecraft.design import design_from_dict, load_design
from polecraft.machines import machine_types, parameters_class
from polecraft.machines.qdd_outer import QddOuterParameters


def test_qdd_outer_is_registered() -> None:
    assert "qdd_outer" in machine_types()
    assert parameters_class("qdd_outer") is QddOuterParameters


def test_unregistered_type_raises_key_error() -> None:
    with pytest.raises(KeyError):
        parameters_class("no_such_machine")


def test_radii_follow_the_stack_from_the_outside_in(valid_data: dict) -> None:
    # 60 mm outer diameter, then 2 yoke, 2 magnet, 0.5 gap, 0.75 tip, 0.25 taper,
    # 8 slot, 3 stator yoke.
    geometry = design_from_dict(valid_data).geometry
    assert geometry.rotor_outer_radius_mm == pytest.approx(30.0)
    assert geometry.magnet_outer_radius_mm == pytest.approx(28.0)
    assert geometry.magnet_inner_radius_mm == pytest.approx(26.0)
    assert geometry.stator_outer_radius_mm == pytest.approx(25.5)
    assert geometry.slot_top_radius_mm == pytest.approx(24.5)
    assert geometry.slot_bottom_radius_mm == pytest.approx(16.5)
    assert geometry.bore_radius_mm == pytest.approx(13.5)


def test_reference_design_radii(repo_root: Path) -> None:
    geometry = load_design(repo_root / "designs" / "qdd_outer_36s42p.yaml").geometry
    assert geometry.rotor_outer_radius_mm == pytest.approx(45.0)
    assert geometry.magnet_outer_radius_mm == pytest.approx(42.5)
    assert geometry.magnet_inner_radius_mm == pytest.approx(40.0)
    assert geometry.stator_outer_radius_mm == pytest.approx(39.4)
    assert geometry.slot_top_radius_mm == pytest.approx(38.2)
    assert geometry.slot_bottom_radius_mm == pytest.approx(29.2)
    assert geometry.bore_radius_mm == pytest.approx(26.0)


# Parts that do not fit


def test_magnet_that_just_fits_inside_the_rotor_passes_that_check(
    changed, problems_of
) -> None:
    # 27.9 mm is under the 28 mm the rotor allows, so the magnet itself is not
    # blamed; the stator no longer fits, which is reported for the whole stack.
    problems = problems_of(changed("geometry.magnet.thickness_mm", 27.9))
    assert list(problems) == ["geometry"]


def test_rotor_yoke_thicker_than_the_outer_radius_is_rejected(
    changed, problems_of
) -> None:
    problems = problems_of(changed("geometry.rotor.yoke_thickness_mm", 30))
    assert list(problems) == ["geometry.rotor.yoke_thickness_mm"]


def test_layers_thicker_than_the_outer_radius_are_rejected(changed, problems_of) -> None:
    # 2 + 2 + 0.5 + 0.75 + 0.25 + 8 + 16.5 = 30 mm: the bore closes exactly.
    problems = problems_of(changed("geometry.stator.yoke_thickness_mm", 16.5))
    assert list(problems) == ["geometry"]
    message = problems["geometry"]
    assert "add up to 30 mm" in message
    assert "air gap 0.5 + tooth tip 1 + slot 8 + stator yoke 16.5" in message


def test_layers_just_inside_the_outer_radius_are_accepted(changed) -> None:
    geometry = design_from_dict(changed("geometry.stator.yoke_thickness_mm", 16.4)).geometry
    assert geometry.bore_radius_mm == pytest.approx(0.1)


def test_tooth_as_wide_as_the_slot_pitch_is_rejected(changed, problems_of) -> None:
    # Slot pitch chord at the slot bottom: 2 * 16.5 * sin(pi/12) = 8.541 mm.
    limit = 2 * 16.5 * math.sin(math.pi / 12)
    assert limit == pytest.approx(8.541, abs=1e-3)

    # Narrow the opening so it is not the first thing to fail.
    design_from_dict(
        changed("geometry.stator.tooth_width_mm", 8.5, "geometry.stator.slot_opening_mm", 0.01)
    )
    problems = problems_of(changed("geometry.stator.tooth_width_mm", 8.6))
    assert list(problems) == ["geometry.stator.tooth_width_mm"]
    assert "8.541" in problems["geometry.stator.tooth_width_mm"]


def test_slot_opening_wider_than_the_slot_is_rejected(changed, problems_of) -> None:
    # Tooth flank at the slot top sits asin(5 / (2 * 24.5)) from the tooth axis,
    # 5.857 deg, leaving 15 - 5.857 = 9.143 deg of half opening at the 25.5 mm
    # stator surface: 2 * 25.5 * sin(9.143 deg) = 8.104 mm.
    limit = 2 * 25.5 * math.sin(math.pi / 12 - math.asin(5.0 / (2 * 24.5)))
    assert limit == pytest.approx(8.104, abs=1e-3)

    design_from_dict(changed("geometry.stator.slot_opening_mm", 8.1))
    problems = problems_of(changed("geometry.stator.slot_opening_mm", 8.2))
    assert list(problems) == ["geometry.stator.slot_opening_mm"]
    assert "8.104" in problems["geometry.stator.slot_opening_mm"]


def test_fit_depends_on_the_slot_count(changed, problems_of) -> None:
    # The same 5 mm tooth fits 12 slots but not 24: 2 * 16.5 * sin(pi/24) = 4.31 mm.
    problems = problems_of(changed("slots", 24, "poles", 28))
    assert list(problems) == ["geometry.stator.tooth_width_mm"]
