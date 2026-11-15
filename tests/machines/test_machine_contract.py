"""The machine contract: creating a machine from a design, symmetry, excitation."""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from polecraft.core.winding import phase_axis_angle, symmetry
from polecraft.design import load_design
from polecraft.machines import machine_for, machine_types, parameters_class
from polecraft.machines.base import Machine
from polecraft.machines.qdd_outer import QddOuter, QddOuterParameters


@pytest.fixture
def machine(repo_root: Path) -> Machine:
    return machine_for(load_design(repo_root / "designs" / "qdd_outer_36s42p.yaml"))


def test_machine_is_created_from_a_design(machine: Machine, repo_root: Path) -> None:
    assert isinstance(machine, QddOuter)
    assert isinstance(machine, Machine)
    assert machine.design == load_design(repo_root / "designs" / "qdd_outer_36s42p.yaml")


def test_parameters_are_the_design_geometry(machine: Machine) -> None:
    assert machine.parameters() is machine.design.geometry
    assert isinstance(machine.parameters(), QddOuterParameters)


def test_registry_still_serves_the_design_loader() -> None:
    assert machine_types() == ["qdd_outer"]
    assert parameters_class("qdd_outer") is QddOuterParameters
    assert QddOuter.Parameters is QddOuterParameters


def test_symmetry_comes_from_the_slot_and_pole_counts(machine: Machine) -> None:
    assert machine.symmetry() == symmetry(36, 42)


def test_a_machine_without_geometry_cannot_be_created(machine: Machine) -> None:
    class Unfinished(Machine):
        pass

    with pytest.raises(TypeError):
        Unfinished(machine.design)


@pytest.mark.parametrize(
    "first", ["polecraft.machines", "polecraft.design", "polecraft.core.geometry",
              "polecraft.machines.base", "polecraft.machines.qdd_outer"]
)
def test_modules_import_in_any_order(first: str) -> None:
    code = f"import {first}; import polecraft; import polecraft.machines"
    subprocess.run([sys.executable, "-c", code], check=True)


# Excitation


def test_phase_currents_sum_to_zero(machine: Machine) -> None:
    for angle in (0.0, 0.01, 0.1, 1.0):
        currents = machine.excitation(angle, 10.0)
        assert set(currents) == {"A", "B", "C"}
        assert sum(currents.values()) == pytest.approx(0.0, abs=1e-12)
        peak = math.sqrt(2 / 3 * sum(i * i for i in currents.values()))
        assert peak == pytest.approx(10.0)


def test_phase_a_axis_of_the_reference_winding() -> None:
    # Phase A of a 12 slot, 14 pole unit is two neighbouring teeth, one at
    # electrical angle 0 and one, reversed, at -30 degrees: axis at -15 degrees.
    assert phase_axis_angle(36, 42) == pytest.approx(math.radians(-15))
    assert phase_axis_angle(12, 10) == pytest.approx(math.radians(-15))
    assert phase_axis_angle(12, 8) == pytest.approx(0.0)


def test_current_is_a_quarter_cycle_ahead_of_the_flux(machine: Machine) -> None:
    # Rotor placed so that phase A links the most flux: its current is zero
    # there and the other two are equal and opposite.
    aligned = phase_axis_angle(36, 42) / 21
    currents = machine.excitation(aligned, 10.0)
    assert currents["A"] == pytest.approx(0.0, abs=1e-12)
    assert currents["B"] == pytest.approx(10.0 * math.cos(math.pi / 2 - 2 * math.pi / 3))
    assert currents["C"] == pytest.approx(-currents["B"])


def test_advance_shifts_the_current(machine: Machine) -> None:
    aligned = phase_axis_angle(36, 42) / 21
    currents = machine.excitation(aligned, 10.0, advance=-math.pi / 2)
    assert currents == pytest.approx({"A": 10.0, "B": -5.0, "C": -5.0})


def test_currents_repeat_every_pole_pair(machine: Machine) -> None:
    one_pole_pair = 2 * math.pi / 21
    assert machine.excitation(0.02 + one_pole_pair, 5.0) == pytest.approx(
        machine.excitation(0.02, 5.0)
    )


def test_b_follows_a_by_a_third_of_a_cycle(machine: Machine) -> None:
    third = 2 * math.pi / 3 / 21  # rotor angle for 120 electrical degrees
    assert machine.excitation(0.03 + third, 5.0)["B"] == pytest.approx(
        machine.excitation(0.03, 5.0)["A"]
    )
    assert machine.excitation(0.03 + 2 * third, 5.0)["C"] == pytest.approx(
        machine.excitation(0.03, 5.0)["A"]
    )
