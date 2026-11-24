"""Design changes move the results as the physics says they should."""

import dataclasses
from pathlib import Path

import pytest

from polecraft.core.materials import Magnet, load_material
from polecraft.core.winding import phase_winding, wire_area
from polecraft.design import load_design
from polecraft.machines import machine_for
from polecraft.studies.back_emf import back_emf
from polecraft.studies.loaded_torque import loaded_torque

DESIGN = Path(__file__).resolve().parents[2] / "designs" / "qdd_outer_36s42p.yaml"
STEEL = load_material("generic_35A_placeholder")  # placeholder, not the design's real steel


def materials(remanence: float) -> dict:
    return {
        "M270-35A": STEEL,
        "N42SH": Magnet(remanence=remanence, relative_permeability=1.05),
    }


@pytest.fixture(scope="module")
def machine():
    return machine_for(load_design(DESIGN))


@pytest.fixture(scope="module")
def mesh(machine):
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh

    return build_mesh(machine.build_geometry(), size=1.0)


def test_stronger_magnet_raises_torque_by_the_remanence_ratio(machine, mesh) -> None:
    # At 1 A the teeth carry little load flux, so torque follows the magnet
    # flux. The steel still saturates a little more under a stronger magnet:
    # the ratio falls about 0.5 % short at 0.9 T to 1.2 T, hence 1 %.
    weak = loaded_torque(machine, mesh, materials(0.9), 1.0)
    strong = loaded_torque(machine, mesh, materials(1.2), 1.0)
    assert strong.mean / weak.mean == pytest.approx(1.2 / 0.9, rel=0.01)
    assert strong.mean / weak.mean < 1.2 / 0.9  # saturation only ever takes away


def test_stronger_magnet_raises_back_emf_by_the_remanence_ratio(machine, mesh) -> None:
    weak = back_emf(machine, mesh, materials(0.9), 100.0)
    strong = back_emf(machine, mesh, materials(1.2), 100.0)
    assert strong.constant / weak.constant == pytest.approx(1.2 / 0.9, rel=0.01)


def with_winding(machine, **changes):
    design = machine.design
    return machine_for(
        dataclasses.replace(design, winding=dataclasses.replace(design.winding, **changes))
    )


def test_thicker_wire_lowers_resistance_by_the_area_ratio(machine) -> None:
    base = phase_winding(machine)
    thick = phase_winding(with_winding(machine, wire_diameter_mm=0.7))
    ratio = wire_area(0.5e-3) / wire_area(0.7e-3)
    assert thick.resistance() == pytest.approx(base.resistance() * ratio, rel=1e-12)
    assert thick.resistance(20.0) == pytest.approx(base.resistance(20.0) * ratio, rel=1e-12)
    # Same turns and path, so only the wire's cross-section changed.
    assert thick.wire_length == base.wire_length
    # The same copper that lowers resistance fills more of the slot.
    assert thick.fill.copper_fill == pytest.approx(base.fill.copper_fill / ratio, rel=1e-12)


def test_wire_size_does_not_change_the_torque_at_the_same_current(machine, mesh) -> None:
    # Torque depends on ampere-turns; wire size changes resistance, not the field.
    other = with_winding(machine, wire_diameter_mm=0.7)
    a = loaded_torque(machine, mesh, materials(1.2), 5.0).mean
    b = loaded_torque(other, mesh, materials(1.2), 5.0).mean
    assert b == pytest.approx(a, rel=1e-9)


def test_parallel_strands_cut_resistance_by_their_number(machine) -> None:
    base = phase_winding(machine)
    doubled = phase_winding(with_winding(machine, parallel_strands=2))
    assert doubled.resistance() == pytest.approx(base.resistance() / 2.0, rel=1e-12)
