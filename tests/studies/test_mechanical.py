"""Mass, rotor inertia, torque density and motor constant, from formulas worked by hand."""

import dataclasses
import math
from pathlib import Path

import pytest

from polecraft.core.geometry import Edge, Region, polar
from polecraft.core.winding import phase_winding
from polecraft.design import load_design
from polecraft.machines import machine_for
from polecraft.studies.mechanical import (
    Mechanical,
    mechanical,
    motor_constant,
    polar_moment_of_area,
)

DESIGNS = Path(__file__).resolve().parents[2] / "designs"
# Made-up densities for the tests, in kg/m3; the library's placeholders have none.
STEEL, MAGNET = 7650.0, 7500.0
DENSITIES = {"M270-35A": STEEL, "N42SH": MAGNET}


def machine(name: str = "qdd_outer_36s42p"):
    return machine_for(load_design(DESIGNS / f"{name}.yaml"))


def sector(inner: float, outer: float, start: float, sweep: float) -> Region:
    """A ring sector, counter-clockwise: out along the start side, round, and back."""
    end = start + sweep
    boundary = (
        Edge(polar(inner, start), polar(outer, start)),
        Edge(polar(outer, start), polar(outer, end), centre=(0.0, 0.0), ccw=True),
        Edge(polar(outer, end), polar(inner, end)),
        Edge(polar(inner, end), polar(inner, start), centre=(0.0, 0.0), ccw=False),
    )
    return Region("sector", "steel", "yoke", "M270-35A", boundary)


# The polar moment of an area


@pytest.mark.parametrize("sweep", [0.3, 1.0, 1.9])
def test_polar_moment_of_a_ring_sector(sweep: float) -> None:
    inner, outer = 0.03, 0.045
    region = sector(inner, outer, 0.4, sweep)
    assert region.area() == pytest.approx(0.5 * sweep * (outer**2 - inner**2), rel=1e-12)
    expected = sweep * (outer**4 - inner**4) / 4.0
    assert polar_moment_of_area(region) == pytest.approx(expected, rel=1e-9)


def test_polar_moment_of_a_whole_ring_from_its_pieces() -> None:
    inner, outer, pieces = 0.03, 0.045, 7
    sweep = 2.0 * math.pi / pieces
    total = sum(polar_moment_of_area(sector(inner, outer, k * sweep, sweep)) for k in range(pieces))
    assert total == pytest.approx(0.5 * math.pi * (outer**4 - inner**4), rel=1e-9)


def test_polar_moment_of_a_rectangle() -> None:
    # A w by h rectangle centred at (d, 0): I = w h (w^2 + h^2) / 12 + w h d^2.
    w, h, d = 0.004, 0.01, 0.03
    corners = [(d - w / 2, -h / 2), (d + w / 2, -h / 2), (d + w / 2, h / 2), (d - w / 2, h / 2)]
    edges = tuple(Edge(corners[k], corners[(k + 1) % 4]) for k in range(4))
    region = Region("bar", "steel", "tooth", "M270-35A", edges)
    assert polar_moment_of_area(region) == pytest.approx(
        w * h * (w**2 + h**2) / 12 + w * h * d**2, rel=1e-12
    )


# Mass and inertia of the example motor


@pytest.mark.parametrize("name", ["qdd_outer_36s42p", "qdd_outer_12s10p"])
def test_rotor_mass_and_inertia_match_the_ring_formulas(name: str) -> None:
    m = machine(name)
    design = m.design
    result = mechanical(m, DENSITIES)
    geometry = design.geometry
    stack = design.axial.stack_length_mm * 1e-3
    factor = design.steel.stacking_factor
    outer = geometry.rotor.outer_diameter_mm / 2e3
    yoke_inner = outer - geometry.rotor.yoke_thickness_mm * 1e-3
    magnet_inner = yoke_inner - geometry.magnet.thickness_mm * 1e-3
    # Whole turns of yoke; magnets cover the arc fraction of each pole.
    magnet_angle = geometry.magnet.arc_fraction * 2.0 * math.pi
    magnet_length = stack + 2.0 * design.axial.magnet_overhang_mm * 1e-3
    yoke = STEEL * stack * factor
    ring = yoke * math.pi * (outer**2 - yoke_inner**2)
    assert result.rotor_steel == pytest.approx(ring, rel=1e-9)
    assert result.magnets == pytest.approx(
        MAGNET * magnet_length * 0.5 * magnet_angle * (yoke_inner**2 - magnet_inner**2), rel=1e-9
    )
    expected = (
        yoke * 0.5 * math.pi * (outer**4 - yoke_inner**4)
        + MAGNET * magnet_length * magnet_angle * (yoke_inner**4 - magnet_inner**4) / 4.0
    )
    # The outlines of arcs are chords within 1e-9 m, so about 1e-7 of the inertia.
    assert result.rotor_inertia == pytest.approx(expected, rel=1e-6)
    assert result.rotor_mass == pytest.approx(result.rotor_steel + result.magnets)


def test_plain_steel_ring_rotor_has_the_inertia_of_a_ring() -> None:
    # With the magnets' density taken to nothing, the rotor is the yoke ring:
    # J = M (R^2 + r^2) / 2.
    m = machine()
    result = mechanical(m, {**DENSITIES, "N42SH": 1e-9})
    outer = m.design.geometry.rotor.outer_diameter_mm / 2e3
    inner = outer - m.design.geometry.rotor.yoke_thickness_mm * 1e-3
    assert result.rotor_inertia == pytest.approx(
        0.5 * result.rotor_steel * (outer**2 + inner**2), rel=1e-6
    )


def test_stator_steel_is_the_geometry_area_times_stack_and_density() -> None:
    m = machine()
    result = mechanical(m, DENSITIES)
    geometry = m.build_geometry(full=True)
    area = sum(r.area() for r in geometry.stator if r.kind == "steel")
    design = m.design
    expected = area * design.axial.stack_length_mm * 1e-3 * design.steel.stacking_factor * STEEL
    assert result.stator_steel == pytest.approx(expected, rel=1e-12)
    assert result.outer_radius == pytest.approx(design.geometry.rotor.outer_diameter_mm / 2e3)


def test_copper_mass_is_wire_length_times_area_and_density() -> None:
    m = machine()
    winding = phase_winding(m)
    result = mechanical(m, DENSITIES)
    area = math.pi * (m.design.winding.wire_diameter_mm * 1e-3) ** 2 / 4
    # One coil on every tooth: all the turns of all the phases.
    length = m.design.winding.turns_per_coil * m.design.slots * winding.turn_length
    assert result.copper == pytest.approx(length * area * winding.conductor.density, rel=1e-12)


def test_steel_density_comes_from_the_library_when_it_has_one() -> None:
    m = machine()
    from_library = mechanical(m, {"N42SH": MAGNET})
    assert from_library.stator_steel == pytest.approx(
        mechanical(m, DENSITIES).stator_steel * 7650.0 / STEEL, rel=1e-12
    )
    # A density given overrides the library's.
    assert mechanical(m, {**DENSITIES, "M270-35A": 7800.0}).rotor_steel == pytest.approx(
        from_library.rotor_steel * 7800.0 / 7650.0, rel=1e-12
    )


def test_masses_add_up_and_the_stack_cylinder_is_its_volume() -> None:
    result = mechanical(machine(), DENSITIES)
    assert result.mass == pytest.approx(
        result.stator_steel + result.rotor_steel + result.magnets + result.copper
    )
    assert result.volume == pytest.approx(math.pi * 0.045**2 * 0.012)
    assert 0.3 < result.mass < 3.0  # a 90 mm, 12 mm motor: a few hundred grams to a kilo
    assert result.rotor_inertia > 0.0


def test_torque_density() -> None:
    result = mechanical(machine(), DENSITIES)
    assert result.torque_density(2.0) == pytest.approx(2.0 / result.mass)
    assert result.torque_per_volume(2.0) == pytest.approx(2.0 / result.volume)
    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError, match="torque must be a positive number"):
            result.torque_density(bad)


@pytest.mark.parametrize(
    "densities, message",
    [
        ({}, "no density for the steel 'M270-35A'|no density for the magnet 'N42SH'"),
        ({"M270-35A": STEEL}, "no density for the magnet 'N42SH'"),
        ({"N42SH": MAGNET, "M270-35A": 0.0}, "density of 'M270-35A' must be a positive"),
        ({"N42SH": -1.0}, "density of 'N42SH' must be a positive"),
    ],
)
def test_missing_or_bad_densities_are_refused(densities, message) -> None:
    with pytest.raises(ValueError, match=message):
        mechanical(machine(), densities)


# Motor constant


def test_motor_constant_is_torque_over_the_root_of_the_copper_loss() -> None:
    winding = phase_winding(machine())
    kt = 0.05  # N m per ampere of peak phase current
    current = 7.0
    copper = winding.copper_loss(current / math.sqrt(2.0))
    assert motor_constant(kt, winding) == pytest.approx(kt * current / math.sqrt(copper), rel=1e-12)


@pytest.mark.parametrize("scale", [0.5, 2.0, 3.0])
def test_motor_constant_is_unchanged_when_turns_change_at_fixed_copper_fill(scale: float) -> None:
    base = machine()
    design = base.design
    winding = dataclasses.replace(
        design.winding,
        turns_per_coil=round(design.winding.turns_per_coil * scale),
        wire_diameter_mm=design.winding.wire_diameter_mm / math.sqrt(scale),
    )
    other = machine_for(dataclasses.replace(design, winding=winding))
    a, b = phase_winding(base), phase_winding(other)
    assert b.fill.copper_fill == pytest.approx(a.fill.copper_fill, rel=1e-9)
    # With the same field per ampere-turn, the torque constant follows the turns.
    kt_a = 0.05
    kt_b = kt_a * b.series_turns / a.series_turns
    assert b.resistance() == pytest.approx(a.resistance() * scale**2, rel=1e-9)
    assert motor_constant(kt_b, b) == pytest.approx(motor_constant(kt_a, a), rel=1e-9)
    # More turns of the same wire change it: the check is not empty.
    thick = machine_for(
        dataclasses.replace(
            design,
            winding=dataclasses.replace(
                design.winding, turns_per_coil=design.winding.turns_per_coil * 2
            ),
        )
    )
    kt_thick = kt_a * 2
    assert motor_constant(kt_thick, phase_winding(thick)) != pytest.approx(
        motor_constant(kt_a, a), rel=1e-3
    )


def test_motor_constant_refuses_a_bad_torque_constant() -> None:
    winding = phase_winding(machine())
    for bad in (0.0, -0.1, math.inf):
        with pytest.raises(ValueError, match="torque constant must be a positive number"):
            motor_constant(bad, winding)


def test_mechanical_is_a_plain_record() -> None:
    result = mechanical(machine("qdd_outer_12s10p"), DENSITIES)
    assert isinstance(result, Mechanical)
    assert result.stack_length == pytest.approx(0.010)
