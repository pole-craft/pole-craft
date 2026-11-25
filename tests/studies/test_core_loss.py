"""Core loss of a machine from the field waveform in each steel triangle."""

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import (
    CoreLossCoefficients,
    LossRangeWarning,
    Magnet,
    load_core_loss,
    load_material,
)
from polecraft.core.motion import solve_angles
from polecraft.core.post import core_loss_density, material_flux_density
from polecraft.core.problem import build_problem
from polecraft.studies.core_loss import CoreLoss, core_loss

STEEL = load_material("M270-35A")
MATERIALS = {
    "M270-35A": STEEL,
    # A placeholder: not the designs' real magnet grade.
    "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
}
DATA = load_core_loss("M270-35A")
LOSSES = {"M270-35A": DATA}
FIT = DATA.fit()
FREQUENCY = 200.0  # Hz, electrical


def _motor(name: str, size: float = 1.0) -> dict:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / f"{name}.yaml"))
    return {"machine": machine, "mesh": build_mesh(machine.build_geometry(), size=size)}


def run(motor: dict, current: float = 0.0, **options: object) -> CoreLoss:
    speed = 2 * math.pi * FREQUENCY / (motor["machine"].design.poles // 2)
    options.setdefault("speed", speed)
    return core_loss(
        motor["machine"], motor["mesh"], MATERIALS, LOSSES, current=current, **options
    )


@pytest.fixture(scope="module")
def small() -> dict:
    return _motor("qdd_outer_12s10p")


@pytest.fixture(scope="module")
def no_load(small: dict) -> CoreLoss:
    return run(small)


@pytest.fixture(scope="module")
def loaded(small: dict) -> CoreLoss:
    return run(small, 10.0)


# What the study solves


def test_solves_one_electrical_cycle_and_its_end(no_load: CoreLoss) -> None:
    cycle = 2 * math.pi / 5
    assert no_load.pole_pairs == 5
    assert no_load.frequency == pytest.approx(FREQUENCY)
    assert np.allclose(no_load.run.angles, cycle * np.arange(49) / 48, rtol=0, atol=1e-15)


def test_stator_field_closes_after_the_cycle(no_load: CoreLoss) -> None:
    still = no_load.triangles[~no_load.turning]
    first = no_load.run.steps[0].solution.flux_density[still]
    last = no_load.run.steps[-1].solution.flux_density[still]
    assert np.abs(last - first).max() < 2e-3  # tesla, against fields of 1.5 T


def test_only_steel_is_counted_and_all_of_it(small: dict, no_load: CoreLoss) -> None:
    problem = no_load.run.steps[0].problem
    steel = np.flatnonzero(
        np.array([problem.materials[n] is STEEL for n in problem.triangle_material])
    )
    assert np.array_equal(no_load.triangles, steel)
    design = small["machine"].design
    geometry = small["machine"].build_geometry(full=True)
    area = sum(region.area() for region in geometry.regions if region.material == "M270-35A")
    expected = area * design.axial.stack_length_mm * 1e-3 * design.steel.stacking_factor * 7650.0
    # Meshed arcs are chords, so the meshed steel is a little short of the exact area.
    assert no_load.mass.sum() == pytest.approx(expected, rel=5e-3)


def test_stator_and_rotor_steel_are_told_apart(small: dict, no_load: CoreLoss) -> None:
    problem = no_load.run.steps[0].problem
    centre = problem.nodes[problem.triangles[no_load.triangles]].mean(axis=1)
    radius = np.hypot(centre[:, 0], centre[:, 1])
    gap = problem.gap
    # An outer rotor: the turning steel is outside the gap.
    assert np.all(radius[no_load.turning] > gap.outer_radius)
    assert np.all(radius[~no_load.turning] < gap.inner_radius)
    assert no_load.turning.any() and not no_load.turning.all()


# The losses


def test_parts_add_up(loaded: CoreLoss) -> None:
    for term in (loaded.hysteresis_density, loaded.eddy_density, loaded.excess_density):
        assert np.all(term >= 0.0)
    assert loaded.hysteresis > 0 and loaded.eddy > 0 and loaded.excess > 0
    assert loaded.total == pytest.approx(loaded.hysteresis + loaded.eddy + loaded.excess)
    assert loaded.total == pytest.approx(loaded.stator + loaded.rotor)
    assert loaded.total == pytest.approx(float(loaded.mass @ loaded.density))


def test_stator_loss_is_near_a_sine_wave_at_each_triangles_peak(no_load: CoreLoss) -> None:
    # A sine wave of the same peak is the least a stator triangle can lose;
    # a field that also turns, as behind the teeth, loses up to twice that
    # before harmonics.
    still = ~no_load.turning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LossRangeWarning)
        sine = float(no_load.mass[still] @ FIT.loss(FREQUENCY, no_load.peak[still]))
    assert 1.0 * sine < no_load.stator < 1.5 * sine


def test_rotor_loss_is_small_without_current_and_grows_with_it(
    no_load: CoreLoss, loaded: CoreLoss
) -> None:
    # The rotor's steel sees a nearly steady field: only the slots passing,
    # and with current the stator's field harmonics.
    assert 0.0 < no_load.rotor < 0.02 * no_load.stator
    assert loaded.rotor > 1.5 * no_load.rotor
    assert loaded.total > no_load.total


def test_rotor_steel_sees_a_steady_field_with_a_small_ripple(no_load: CoreLoss) -> None:
    steps = [(step.problem, step.solution) for step in no_load.run.steps]
    turning = no_load.triangles[no_load.turning]
    field, _ = material_flux_density(steps, turning)
    size = np.hypot(field[..., 0], field[..., 1])
    # In the fixed frame the same triangle's field turns with it, by the 72
    # degrees the rotor moves in one electrical cycle.
    fixed = np.stack([step.solution.flux_density[turning] for step in no_load.run.steps])
    busiest = np.argmax(size.mean(axis=0))
    assert np.ptp(fixed[:, busiest, 0]) > 0.8
    assert np.ptp(field[:, busiest, 0]) < 0.1


def test_rotor_loss_from_one_cycle_matches_its_whole_period(small: dict) -> None:
    # The rotor of a 12-slot, 10-pole machine sees a field that repeats
    # every 5/6 of an electrical cycle, so one cycle is not a whole number
    # of its periods; five cycles are six of them.
    machine, mesh = small["machine"], small["mesh"]
    one = run(small, 10.0, points=24)
    cycle, cycles = 2 * math.pi / 5, 5
    long = solve_angles(
        mesh,
        cycle * np.arange(24 * cycles + 1) / 24,
        lambda turned: build_problem(machine, turned, MATERIALS, current=10.0),
    )
    turning = one.triangles[one.turning]
    field, _ = material_flux_density([(s.problem, s.solution) for s in long.steps], turning)
    _, eddy, excess = core_loss_density(field, cycles / FREQUENCY, FIT, periodic=False)
    mass = one.mass[one.turning]
    for whole, cycle_only in ((eddy, one.eddy_density), (excess, one.excess_density)):
        assert float(mass @ cycle_only[one.turning]) == pytest.approx(float(mass @ whole), rel=0.01)


def test_more_points_change_little(small: dict, no_load: CoreLoss) -> None:
    fine = run(small, points=96)
    assert fine.stator == pytest.approx(no_load.stator, rel=2e-3)
    assert fine.rotor == pytest.approx(no_load.rotor, rel=0.05)


def test_finer_mesh_changes_little(no_load: CoreLoss) -> None:
    fine = run(_motor("qdd_outer_12s10p", size=0.6))
    assert fine.total == pytest.approx(no_load.total, rel=0.015)
    assert fine.hysteresis == pytest.approx(no_load.hysteresis, rel=0.015)
    assert fine.eddy == pytest.approx(no_load.eddy, rel=0.02)


def test_reference_motor(no_load: CoreLoss) -> None:
    reference = run(_motor("qdd_outer_36s42p"), 10.0)
    assert reference.pole_pairs == 21
    assert reference.total == pytest.approx(reference.stator + reference.rotor)
    assert 0.0 < reference.rotor < 0.05 * reference.stator
    still = ~reference.turning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LossRangeWarning)
        sine = float(reference.mass[still] @ FIT.loss(FREQUENCY, reference.peak[still]))
    assert 1.0 * sine < reference.stator < 1.5 * sine


# Speed


def test_loss_at_another_speed_follows_from_the_same_fields(small: dict, no_load: CoreLoss) -> None:
    double = no_load.at_speed(2 * no_load.speed)
    assert double.frequency == pytest.approx(2 * FREQUENCY)
    assert double.hysteresis == pytest.approx(2 * no_load.hysteresis)
    assert double.eddy == pytest.approx(4 * no_load.eddy)
    assert double.excess == pytest.approx(2**1.5 * no_load.excess)
    assert double.run is no_load.run
    solved = run(small, speed=2 * no_load.speed)
    assert solved.total == pytest.approx(double.total, rel=1e-9)
    assert solved.rotor == pytest.approx(double.rotor, rel=1e-9)


def test_direction_of_turning_does_not_matter_to_the_scaling(no_load: CoreLoss) -> None:
    assert no_load.at_speed(-no_load.speed).total == pytest.approx(no_load.total)


def test_warns_outside_the_fitted_frequencies(no_load: CoreLoss) -> None:
    assert no_load.fitted_frequencies == (50.0, 1000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        no_load.at_speed(no_load.speed * 1000.0 / FREQUENCY)
    with pytest.warns(LossRangeWarning, match="1500 Hz"):
        no_load.at_speed(no_load.speed * 1500.0 / FREQUENCY)
    with pytest.warns(LossRangeWarning, match="20 Hz"):
        no_load.at_speed(no_load.speed * 20.0 / FREQUENCY)


def test_study_warns_outside_the_fitted_frequencies(small: dict) -> None:
    with pytest.warns(LossRangeWarning, match="2000 Hz"):
        run(small, speed=2 * math.pi * 2000.0 / 5, points=8)


def test_share_of_loss_beyond_the_fitted_flux_density(no_load: CoreLoss, loaded: CoreLoss) -> None:
    assert np.array_equal(no_load.beyond_fit, no_load.peak > 1.8)
    assert 0.0 <= no_load.share_beyond_fit < loaded.share_beyond_fit < 0.3


# Options


def test_coefficients_given_by_hand_are_used(small: dict) -> None:
    eddy_only = CoreLossCoefficients(hysteresis=0.0, eddy=5e-5, excess=0.0)
    result = run(small, points=8, coefficients={"M270-35A": eddy_only})
    assert result.hysteresis == 0.0 and result.excess == 0.0 and result.eddy > 0.0
    assert result.fitted_frequencies is None
    assert not result.beyond_fit.any()


def test_stacking_factor_scales_the_mass(small: dict) -> None:
    design = run(small, points=8)
    solid = run(small, points=8, stacking_factor=1.0)
    assert design.total == pytest.approx(0.95 * solid.total)


def test_timing_report_includes_the_loss_step(no_load: CoreLoss) -> None:
    report = no_load.timing_report()
    assert "49 rotor angles" in report and "core loss" in report


@pytest.mark.parametrize(
    "options, message",
    [
        ({"speed": 0.0}, "speed"),
        ({"speed": math.inf}, "speed"),
        ({"points": 7}, "points"),
        ({"stacking_factor": 0.0}, "stacking factor"),
        ({"stacking_factor": 1.2}, "stacking factor"),
        ({"current": math.nan}, "finite"),
    ],
)
def test_refuses_bad_input(small: dict, options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run(small, **options)


def test_refuses_loss_data_that_fits_no_steel(small: dict) -> None:
    machine, mesh = small["machine"], small["mesh"]
    with pytest.raises(ValueError, match="no material model for 'M400-50A'"):
        core_loss(machine, mesh, MATERIALS, {"M400-50A": DATA}, 100.0, points=8)
    extra = {**MATERIALS, "M400-50A": load_material("generic_35A_placeholder")}
    with pytest.raises(ValueError, match="not a material of this machine"):
        core_loss(machine, mesh, extra, {"M400-50A": DATA}, 100.0, points=8)
    with pytest.raises(ValueError, match="at least one steel"):
        core_loss(machine, mesh, MATERIALS, {}, 100.0, points=8)
    same = {**MATERIALS, "other": STEEL}
    with pytest.raises(ValueError, match="shares its material model"):
        core_loss(
            machine,
            mesh,
            same,
            {"M270-35A": DATA, "other": load_core_loss("M270-35A")},
            100.0,
            points=8,
        )
