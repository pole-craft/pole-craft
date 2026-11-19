"""Cogging torque: the torque of the magnets alone over one cogging period."""

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import Magnet, load_material
from polecraft.core.motion import rotate, step_angle
from polecraft.core.post import torque
from polecraft.core.problem import build_problem
from polecraft.core.solver import solve
from polecraft.studies.cogging import CoarseMeshWarning, Cogging, cogging_period, cogging_torque

MATERIALS = {
    # Placeholders: not the designs' real steel or magnet grade.
    "M270-35A": load_material("generic_35A_placeholder"),
    "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
}


def _motor(name: str, size: float) -> dict:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / f"{name}.yaml"))
    return {"machine": machine, "mesh": build_mesh(machine.build_geometry(), size=size)}


def run(motor: dict, **options: object) -> Cogging:
    return cogging_torque(motor["machine"], motor["mesh"], MATERIALS, **options)


def harmonics(values: np.ndarray) -> np.ndarray:
    """Amplitude of each harmonic of samples taken evenly over the window."""
    return np.abs(np.fft.rfft(values)) / (len(values) / 2)


# The period: 360 / LCM(slots, poles) mechanical degrees


@pytest.mark.parametrize(
    ("slots", "poles", "degrees"),
    [(36, 42, 360 / 252), (12, 10, 6.0), (24, 28, 360 / 168), (9, 8, 5.0), (12, 8, 15.0), (3, 2, 60.0)],
)
def test_period_is_a_full_turn_over_the_lcm_of_slots_and_poles(
    slots: int, poles: int, degrees: float
) -> None:
    assert math.degrees(cogging_period(slots, poles)) == pytest.approx(degrees, rel=1e-12)


@pytest.mark.parametrize(("slots", "poles"), [(2, 2), (12, 7), (12, 0), (0, 10)])
def test_period_needs_a_real_slot_and_pole_count(slots: int, poles: int) -> None:
    with pytest.raises(ValueError):
        cogging_period(slots, poles)


@pytest.fixture(scope="module")
def reference() -> dict:
    """The reference motor at the coarsest mesh that is not warned about.

    Its cogging period and mean come out right there, but not its size: that
    still moves by up to a third on finer meshes (see the study's docstring).
    """
    return _motor("qdd_outer_36s42p", size=0.5)


@pytest.fixture(scope="module")
def reference_cogging(reference: dict) -> Cogging:
    return run(reference, points=12, periods=2)


@pytest.fixture(scope="module")
def small() -> dict:
    return _motor("qdd_outer_12s10p", size=1.0)


def test_cogging_is_solved_at_evenly_spaced_angles_with_no_current(
    reference: dict, reference_cogging: Cogging
) -> None:
    period = cogging_period(36, 42)
    assert reference_cogging.period == pytest.approx(period, rel=1e-12)
    assert np.allclose(reference_cogging.angles, period * np.arange(24) / 12, rtol=0, atol=1e-15)
    assert len(reference_cogging.torque) == 24
    for step in reference_cogging.run.steps:
        assert not np.any(step.problem.current_density)


def test_reference_motor_cogging_repeats_with_the_lcm_period(reference_cogging: Cogging) -> None:
    torque = reference_cogging.torque
    first, second = torque[:12], torque[12:]
    # The same torque one period on, to the mesh noise.
    assert np.abs(first - second).max() < 0.05 * np.ptp(torque)
    # Over two periods, the second harmonic (the period itself) is the
    # strongest; the first (twice the period) is not there at all.
    waves = harmonics(torque)
    assert 1e-3 < waves[2] < 4e-3  # N m: 1.8 to 2.5 mN m on meshes of size 0.5 to 0.2
    assert waves[1] < 0.1 * waves[2]  # about 0.02 to 0.03 measured
    assert np.delete(waves[1:], 1).max() < 0.1 * waves[2]


def test_small_motor_cogging_has_the_lcm_period_over_a_slot_pitch(small: dict) -> None:
    # A slot pitch of 12s10p (30 degrees) holds five cogging periods. Sampled
    # over the slot pitch, the fifth harmonic is the strongest, and only its
    # own multiples (the waveform's harmonics) may be there besides.
    cogging = run(small, points=8, periods=5)
    assert cogging.angles[-1] < math.radians(30.0)
    waves = harmonics(cogging.torque)
    assert waves[5] > 2e-3  # N m
    assert np.argmax(waves[1:]) + 1 == 5
    order = np.arange(len(waves))
    assert waves[(order > 0) & (order % 5 != 0)].max() < 0.01 * waves[5]


def test_mean_cogging_torque_is_close_to_zero(reference_cogging: Cogging, small: dict) -> None:
    # Torque without current is minus the rate of change of stored energy with
    # angle, and the energy comes back to where it was after a period.
    for cogging in (reference_cogging, run(small, points=12)):
        assert abs(cogging.mean) < 0.01 * cogging.peak_to_peak
        assert cogging.mean == pytest.approx(float(np.mean(cogging.torque)), rel=1e-12, abs=1e-15)
        assert cogging.peak_to_peak == pytest.approx(float(np.ptp(cogging.torque)), rel=1e-12)


def test_cogging_barely_changes_with_a_finer_mesh() -> None:
    # 12s10p has its cogging resolved at size 0.5 (24 node spacings a period).
    # The reference motor's is not: see the module docstring of the study.
    coarse = run(_motor("qdd_outer_12s10p", size=0.5), points=12)
    fine = run(_motor("qdd_outer_12s10p", size=0.35), points=12)
    a, b = harmonics(coarse.torque)[1], harmonics(fine.torque)[1]
    assert 4e-3 < a < 6e-3  # N m
    assert a == pytest.approx(b, rel=0.1)  # about 3 % measured


def test_a_mesh_too_coarse_for_the_period_is_warned_about(reference: dict) -> None:
    coarse = _motor("qdd_outer_36s42p", size=1.0)
    spacings = cogging_period(36, 42) / step_angle(coarse["mesh"])
    assert spacings == pytest.approx(4.0)
    with pytest.warns(CoarseMeshWarning, match="4 node spacings"):
        cogging = run(coarse, points=4)
    assert cogging.spacings_per_period == pytest.approx(4.0)


def test_a_fine_enough_mesh_is_not_warned_about(reference: dict, small: dict) -> None:
    for motor, spacings in ((reference, 10.0), (small, 12.0)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cogging = run(motor, points=4)
        assert cogging.spacings_per_period == pytest.approx(spacings)


@pytest.mark.parametrize(
    ("options", "message"),
    [({"points": 3}, "points"), ({"points": 0}, "points"), ({"periods": 0}, "periods")],
)
def test_cogging_needs_enough_points_and_at_least_one_period(
    small: dict, options: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run(small, **options)


def test_cogging_reports_its_timing(small: dict) -> None:
    report = run(small, points=4).timing_report()
    for text in ("4 rotor angles", "turn rotor", "linear solve", "total"):
        assert text in report


def test_cogging_torque_is_the_torque_of_a_plain_solve_at_each_angle(small: dict) -> None:
    # The study's numbers are those of torque() on a field solved on its own:
    # whole machine, on the rotor, counter-clockwise positive.
    cogging = run(small, points=4)
    for k in (1, 3):
        problem = build_problem(small["machine"], rotate(small["mesh"], cogging.angles[k]), MATERIALS)
        assert cogging.torque[k] == pytest.approx(torque(problem, solve(problem)), rel=1e-6)
    assert abs(cogging.torque).max() > 1e-3  # N m: not comparing zeros


def test_the_warning_starts_just_below_its_limit() -> None:
    eight = _motor("qdd_outer_36s42p", size=0.7)
    with pytest.warns(CoarseMeshWarning, match="8 node spacings"):
        cogging = run(eight, points=4)
    assert cogging.spacings_per_period == pytest.approx(8.0)
