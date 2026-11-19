"""Loaded torque: phase currents locked to the rotor, over an electrical cycle."""

import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import Magnet, load_material
from polecraft.core.motion import rotate
from polecraft.core.post import torque
from polecraft.core.problem import build_problem
from polecraft.core.solver import solve
from polecraft.studies.back_emf import back_emf
from polecraft.studies.loaded_torque import LoadedTorque, loaded_torque

MATERIALS = {
    # Placeholders: not the designs' real steel or magnet grade.
    "M270-35A": load_material("generic_35A_placeholder"),
    "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
}


def _motor(name: str, size: float = 1.0) -> dict:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / f"{name}.yaml"))
    return {"machine": machine, "mesh": build_mesh(machine.build_geometry(), size=size)}


def run(motor: dict, current: float, **options: object) -> LoadedTorque:
    return loaded_torque(motor["machine"], motor["mesh"], MATERIALS, current, **options)


@pytest.fixture(scope="module")
def reference() -> dict:
    return _motor("qdd_outer_36s42p")


@pytest.fixture(scope="module")
def low(reference: dict) -> LoadedTorque:
    """The reference motor at 1 A, well below saturation."""
    return run(reference, 1.0)


@pytest.fixture(scope="module")
def high(reference: dict) -> LoadedTorque:
    """The reference motor at 40 A, four times its 10 A, with the steel saturated."""
    return run(reference, 40.0)


@pytest.fixture(scope="module")
def small() -> dict:
    return _motor("qdd_outer_12s10p")


# What the study solves


def test_angles_cover_one_electrical_cycle(low: LoadedTorque) -> None:
    cycle = 2 * math.pi / 21
    assert low.pole_pairs == 21
    assert np.allclose(low.angles, cycle * np.arange(48) / 48, rtol=0, atol=1e-15)
    assert len(low.torque) == 48


def test_phase_currents_follow_the_rotor_as_the_machine_says(reference: dict) -> None:
    advance = math.radians(25.0)
    study = run(reference, 3.0, advance=advance, points=6)
    for k, angle in enumerate(study.angles):
        expected = reference["machine"].excitation(angle, 3.0, advance)
        for phase in "ABC":
            assert study.currents[phase][k] == pytest.approx(expected[phase], rel=1e-12, abs=1e-12)
    # Balanced: the three phase currents add up to nothing at every angle.
    total = sum(study.currents[phase] for phase in "ABC")
    assert np.abs(total).max() < 1e-12 * 3.0


def test_torque_at_an_angle_is_that_of_a_plain_solve(small: dict) -> None:
    study = run(small, 5.0, advance=0.3, points=6)
    k = 4
    problem = build_problem(
        small["machine"], rotate(small["mesh"], study.angles[k]), MATERIALS, current=5.0, advance=0.3
    )
    assert study.torque[k] == pytest.approx(torque(problem, solve(problem)), rel=1e-6)


def test_summary_values_follow_from_the_torque(high: LoadedTorque) -> None:
    assert high.mean == pytest.approx(float(np.mean(high.torque)), rel=1e-12)
    assert high.peak_to_peak == pytest.approx(float(np.ptp(high.torque)), rel=1e-12)
    assert high.ripple == pytest.approx(high.peak_to_peak / high.mean, rel=1e-12)
    assert high.mean > 0.0
    assert high.constant == pytest.approx(high.mean / 40.0, rel=1e-12)


# The plan's checks


def test_torque_is_linear_in_current_at_low_current(reference: dict, low: LoadedTorque) -> None:
    assert low.mean > 0.3  # N m: positive current at advance 0 drives the rotor forwards
    for current in (0.5, 2.0, -1.0):
        other = run(reference, current)
        assert other.mean == pytest.approx(low.mean * current, rel=1e-3)
        assert other.constant == pytest.approx(low.constant, rel=1e-3)
        # Ripple is a size: the same whichever way the torque acts.
        assert other.ripple > 0.0


def test_saturation_lowers_the_torque_constant_at_high_current(
    low: LoadedTorque, high: LoadedTorque
) -> None:
    # The steel saturates: 40 A gives about three quarters of the torque per
    # ampere of 1 A (0.74 measured). The low-current check above means something.
    assert high.constant < 0.85 * low.constant
    assert high.constant > 0.6 * low.constant


def test_torque_constant_matches_the_back_emf_constant(reference: dict, low: LoadedTorque) -> None:
    # Power balance with current in phase with the back-EMF: three phases of
    # peak EMF E and peak current I give 3/2 E I = T omega. With E = ke omega,
    # T / I = 3/2 ke, in N m/A and V s/rad, the same SI unit.
    ke = back_emf(reference["machine"], reference["mesh"], MATERIALS, speed=100.0).constant
    assert low.constant == pytest.approx(1.5 * ke, rel=3e-3)  # 0.16 % measured


@pytest.mark.parametrize("fixture", ["low", "high"])
def test_stress_and_flux_linkage_give_the_same_mean_torque(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    # Over a whole cycle the stored energy comes back to where it started, so
    # the mean of sum(i dpsi/dangle) is the mean torque, saturated or not.
    study = request.getfixturevalue(fixture)
    assert study.mean_from_linkage == pytest.approx(study.mean, rel=6e-3)  # 0.15 and 0.31 % measured


def test_the_two_methods_and_the_mean_settle_with_a_finer_mesh(reference: dict) -> None:
    coarse = run(reference, 10.0, points=24)
    fine = run(_motor("qdd_outer_36s42p", size=0.5), 10.0, points=24)
    assert fine.mean == pytest.approx(coarse.mean, rel=2e-3)  # 0.02 % measured
    gap = abs(fine.mean_from_linkage / fine.mean - 1)
    assert gap < 1e-3  # 0.05 % measured, from 0.18 % at size 1
    assert gap < abs(coarse.mean_from_linkage / coarse.mean - 1)


# Advance and ripple


@pytest.fixture(scope="module")
def straight(reference: dict) -> LoadedTorque:
    """2 A in phase with the back-EMF, to compare other advances with."""
    return run(reference, 2.0, points=24)


@pytest.mark.parametrize("degrees", [30.0, 60.0, -60.0])
def test_torque_falls_as_the_cosine_of_the_advance(
    reference: dict, straight: LoadedTorque, degrees: float
) -> None:
    # Surface magnets: no reluctance torque, so only the part of the current
    # in phase with the back-EMF makes torque.
    turned = run(reference, 2.0, advance=math.radians(degrees), points=24)
    assert turned.mean / straight.mean == pytest.approx(math.cos(math.radians(degrees)), abs=3e-3)
    # Away from advance 0 a phase error between currents and linkage would
    # show at first order; the two methods still agree (0.18 % measured).
    assert turned.mean_from_linkage == pytest.approx(turned.mean, rel=6e-3)


def test_current_at_right_angles_to_the_back_emf_makes_no_torque(
    reference: dict, straight: LoadedTorque
) -> None:
    across = run(reference, 2.0, advance=math.pi / 2, points=24)
    assert abs(across.mean) < 2e-3 * straight.mean
    assert abs(across.mean_from_linkage) < 2e-3 * straight.mean


def test_ripple_comes_at_multiples_of_six_times_the_electrical_frequency(
    high: LoadedTorque,
) -> None:
    # Three balanced phases with half-wave symmetric back-EMF give torque
    # harmonics at 6, 12, 18... per electrical cycle; cogging here (252 per
    # turn, 12 per electrical cycle) falls on the same ones.
    waves = np.abs(np.fft.rfft(high.torque)) / (len(high.torque) / 2)
    order = np.arange(len(waves))
    sixes = (order % 6 == 0) & (order > 0)
    assert waves[6] > 5e-3 * high.mean
    assert waves[(order > 0) & ~sixes].max() < 5e-4 * high.mean
    assert 0.01 < high.ripple < 0.06  # 0.032 measured


# Inputs and output


@pytest.mark.parametrize(
    ("current", "options", "message"),
    [
        (0.0, {}, "current"),
        (math.nan, {}, "current"),
        (1.0, {"points": 3}, "points"),
        (1.0, {"advance": math.inf}, "advance"),
    ],
)
def test_loaded_torque_needs_a_current_enough_points_and_a_finite_advance(
    small: dict, current: float, options: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run(small, current, **options)


def test_loaded_torque_reports_its_timing(small: dict) -> None:
    report = run(small, 1.0, points=4).timing_report()
    for text in ("4 rotor angles", "turn rotor", "linear solve", "total"):
        assert text in report
