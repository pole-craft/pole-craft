"""The U8-class motor of the Mini Cheetah actuator against its measured figures.

The design is calibrated on one number, the low-current torque constant,
through the turns per coil. Everything else is a check. Where the model and
the measurements part, the test holds the gap that docs/u8_comparison.md
explains, so that the document stays true.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from polecraft.core.materials import Magnet, load_material
from polecraft.core.winding import phase_winding
from polecraft.design import load_design
from polecraft.machines import machine_for
from polecraft.studies.cogging import cogging_period
from polecraft.studies.stall_heating import stall_heating

DATA = Path(__file__).resolve().parents[1] / "data" / "u8"
DESIGNS = Path(__file__).resolve().parents[2] / "designs"
PLACEHOLDERS = ("generic_35A_placeholder", "generic_NdFeB_placeholder")
MATERIALS = {name: load_material(name) for name in PLACEHOLDERS}


@pytest.fixture(scope="module")
def measured() -> dict:
    with open(DATA / "measured.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


@pytest.fixture(scope="module")
def machine():
    return machine_for(load_design(DATA / "u8_class_36s42p.yaml"))


@pytest.fixture(scope="module")
def mesh(machine):
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh

    return build_mesh(machine.build_geometry(), size=1.0)


@pytest.fixture(scope="module")
def saturation(machine, mesh):
    from polecraft.studies.saturation import torque_against_current

    return torque_against_current(machine, mesh, MATERIALS, [5.0, 40.0])


def low_current_constant(measured: dict) -> float:
    """Slope of the measured torque from 0 to 10 A, in N m/A.

    The drawn line does not start at zero (0.02 N m at 0 A), so the slope of
    a straight-line fit is used, not torque over current.
    """
    curve = measured["torque_against_current"]
    current, torque = np.array(curve["current_A"]), np.array(curve["torque_Nm"])
    low = current <= 10.0
    return float(np.polyfit(current[low], torque[low], 1)[0])


def test_digitised_figures_agree_with_the_thesis_text(measured: dict) -> None:
    constant = low_current_constant(measured)
    curve = measured["torque_against_current"]
    fall = 1.0 - curve["torque_Nm"][-1] / curve["current_A"][-1] / constant
    assert fall == pytest.approx(curve["stated_constant_drop"], abs=0.015)
    cogging = measured["cogging"]
    assert cogging["digitised_rms_Nm"] == pytest.approx(cogging["rms_Nm"], rel=0.02)
    thermal = measured["thermal"]
    rise = thermal["resistance_K_per_W"] * thermal["continuous_loss_W"]
    assert thermal["ambient_C"] + rise == pytest.approx(thermal["limit_C"], abs=1.5)


def test_design_has_the_stated_figures(machine) -> None:
    design, geometry = machine.design, machine.build_geometry()
    gap_diameter = geometry.band_inner_radius + geometry.band_outer_radius
    assert gap_diameter == pytest.approx(81e-3, abs=0.3e-3)
    assert design.axial.stack_length_mm == 8.2
    assert (design.slots, design.poles) == (36, 42)
    assert (design.steel.stator, design.magnet.material) == PLACEHOLDERS
    # 36 slots cog 12 times an electrical cycle, as the thesis measured.
    assert cogging_period(36, 42) * 21 == pytest.approx(2.0 * math.pi / 12)


def test_the_two_torque_constants_differ_by_root_3(measured: dict) -> None:
    # The thesis divides by line current, the U8 figure by the current in
    # one delta phase, which is line current / sqrt(3).
    ratio = measured["u8"]["torque_constant_Nm_per_A"] / low_current_constant(measured)
    assert ratio == pytest.approx(math.sqrt(3.0), rel=0.03)


def test_turns_are_calibrated_on_the_low_current_constant(machine, measured, saturation) -> None:
    target, model = low_current_constant(measured), saturation.constant[0]
    turns = machine.design.winding.turns_per_coil
    assert model == pytest.approx(target, rel=0.04)
    # At low current the constant goes as the turns: no other whole number is closer.
    for other in (turns - 1, turns + 1):
        assert abs(model * other / turns - target) > abs(model - target)
    # As a delta, these turns would be fewer than the U8's counted 16.
    assert turns * math.sqrt(3.0) / measured["u8"]["turns_per_coil"] == pytest.approx(0.76, abs=0.02)


def test_model_saturates_far_less_than_measured(measured, saturation) -> None:
    curve = measured["torque_against_current"]
    measured_fall = 1.0 - curve["torque_Nm"][-1] / curve["current_A"][-1] / low_current_constant(measured)
    assert 0.01 < saturation.fall[-1] < 0.04
    assert saturation.fall[-1] < measured_fall / 3.0


def test_inductance_is_above_both_measurements(machine, mesh, measured) -> None:
    from polecraft.studies.inductance import inductance

    result = inductance(machine, mesh, MATERIALS, 10.0)
    assert result.d / measured["inductance_H"] == pytest.approx(1.7, abs=0.1)
    star = measured["u8"]["effective_inductance_H"] / 3.0  # a delta phase as a star phase
    assert result.d / star == pytest.approx(2.2, abs=0.15)
    assert result.q == pytest.approx(result.d, rel=0.03)


def test_cogging_of_the_estimated_design_is_far_below_the_measured(machine, measured) -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.studies.cogging import cogging_torque

    cogging = cogging_torque(machine, build_mesh(machine.build_geometry(), size=0.5), MATERIALS)
    peak = float(np.max(np.abs(cogging.torque - cogging.mean)))
    assert peak < measured["cogging"]["harmonic_12_Nm"] / 5.0


def test_resistance_lies_between_the_two_sources(machine, measured) -> None:
    thermal = measured["thermal"]
    # 60 W at 15.4 A peak with the resistance 30 % up, so at 25 C:
    thesis = thermal["continuous_loss_W"] / (1.5 * thermal["continuous_current_A"] ** 2) / 1.3
    u8 = measured["u8"]["phase_resistance_ohm"] / 3.0  # delta phase as a star phase
    assert u8 < phase_winding(machine).resistance(25.0) < thesis


def test_stall_estimate_is_on_the_safe_side_of_the_thesis_thermal_mass(machine, measured) -> None:
    thermal = measured["thermal"]
    winding = phase_winding(machine)
    current = thermal["continuous_current_A"]
    hold = stall_heating(winding, current, thermal["ambient_C"], thermal["limit_C"])
    # Copper alone holds well under the thesis's lumped 32 J/K ...
    assert 3.0 * hold.heat_capacity / thermal["capacity_J_per_K"] == pytest.approx(0.29, abs=0.03)
    # ... so the worst phase reaches the limit sooner than the lumped
    # winding would, even with no heat leaving it.
    lumped = thermal["capacity_J_per_K"] * (thermal["limit_C"] - thermal["ambient_C"])
    assert hold.time_to_limit < lumped / hold.total_loss


def test_reference_design_is_of_the_same_class(machine, saturation) -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.studies.loaded_torque import loaded_torque

    reference = machine_for(load_design(DESIGNS / "qdd_outer_36s42p.yaml"))
    materials = {"M270-35A": MATERIALS[PLACEHOLDERS[0]], "N42SH": Magnet(remanence=1.2, relative_permeability=1.05)}
    mesh = build_mesh(reference.build_geometry(), size=1.0)
    constant = loaded_torque(reference, mesh, materials, 5.0, points=24).constant

    def per_turn_and_metre(m, kt: float) -> float:
        return kt / m.design.winding.turns_per_coil / (m.design.axial.stack_length_mm * 1e-3)

    assert per_turn_and_metre(reference, constant) == pytest.approx(
        per_turn_and_metre(machine, saturation.constant[0]), rel=0.1
    )
