"""Torque against current up to peak: saturation lowers the torque constant."""

import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import Magnet, load_material
from polecraft.studies.saturation import TorqueAgainstCurrent, torque_against_current

MATERIALS = {
    # Placeholders: not the design's real steel or magnet grade.
    "M270-35A": load_material("generic_35A_placeholder"),
    "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
}
CURRENTS = [1.0, 10.0, 40.0, 80.0]  # A peak; 10 A is what the examples run at


@pytest.fixture(scope="module")
def small() -> dict:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / "qdd_outer_12s10p.yaml"))
    return {"machine": machine, "mesh": build_mesh(machine.build_geometry(), size=1.0)}


@pytest.fixture(scope="module")
def curve(small: dict) -> TorqueAgainstCurrent:
    return torque_against_current(small["machine"], small["mesh"], MATERIALS, CURRENTS)


def test_constant_at_low_current_is_that_of_the_back_emf(small: dict, curve) -> None:
    from polecraft.studies.back_emf import back_emf

    constant = back_emf(small["machine"], small["mesh"], MATERIALS, 100.0, points=24).constant
    # At advance 0, with the steel far from saturation: T = 1.5 k_e I.
    assert curve.constant[0] == pytest.approx(1.5 * constant, rel=3e-3)


def test_torque_constant_falls_at_high_current(curve: TorqueAgainstCurrent) -> None:
    assert np.all(np.diff(curve.torque) > 0.0)  # more current still gives more torque
    assert np.all(np.diff(curve.constant) < 0.0)  # but less of it per ampere
    assert curve.fall[0] == 0.0
    assert curve.fall[1] < 0.01  # at the examples' current, under 1 %
    assert curve.fall[-1] > 0.15  # at eight times that, saturation shows
    # Each extra ampere gives less than the mean ampere before it.
    assert np.all(curve.incremental_constant < curve.constant[:-1])
    assert curve.incremental_constant[-1] < 0.8 * curve.constant[0]


def test_each_point_is_the_loaded_torque_study(curve: TorqueAgainstCurrent) -> None:
    for current, torque, run in zip(curve.currents, curve.torque, curve.runs):
        assert run.current == current and run.mean == torque
        assert len(run.angles) == 24
    assert np.all(curve.peak_to_peak > 0.0)
    assert curve.timing_report().startswith("Timing (4 currents, 96 rotor angles)")


@pytest.mark.parametrize(
    "currents, message",
    [
        ([], "at least one"),
        ([0.0, 1.0], "above 0"),
        ([1.0, math.inf], "above 0"),
        ([2.0, 1.0], "rise"),
    ],
)
def test_bad_currents_are_refused(currents, message) -> None:
    # Checked before anything is solved, so no machine is needed.
    with pytest.raises(ValueError, match=message):
        torque_against_current(None, None, MATERIALS, currents)
