"""Back-EMF: phase flux linkage over an electrical cycle and its rate of change."""

import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import Magnet, load_material
from polecraft.core.motion import rotate, step_angle
from polecraft.core.post import flux_linkage, gap_flux_density
from polecraft.core.problem import build_problem
from polecraft.core.solver import solve
from polecraft.core.winding import concentrated_winding, winding_factor
from polecraft.studies.back_emf import BackEmf, _derivative, back_emf

MATERIALS = {
    # Placeholders: not the designs' real steel or magnet grade.
    "M270-35A": load_material("generic_35A_placeholder"),
    "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
}
SPEED = 100.0  # rad/s, mechanical


def _motor(name: str, size: float = 1.0) -> dict:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / f"{name}.yaml"))
    return {"machine": machine, "mesh": build_mesh(machine.build_geometry(), size=size)}


def run(motor: dict, speed: float = SPEED, **options: object) -> BackEmf:
    return back_emf(motor["machine"], motor["mesh"], MATERIALS, speed, **options)


def fundamental(values: np.ndarray, cycles: int = 1) -> complex:
    """Fundamental of samples over whole cycles, as a complex peak amplitude."""
    return np.fft.rfft(values)[cycles] / (len(values) / 2)


@pytest.fixture(scope="module")
def reference() -> dict:
    return _motor("qdd_outer_36s42p")


@pytest.fixture(scope="module")
def reference_emf(reference: dict) -> BackEmf:
    return run(reference)


@pytest.fixture(scope="module")
def small() -> dict:
    return _motor("qdd_outer_12s10p")


# The rate of change over a cycle, from the samples


def test_derivative_of_a_sum_of_harmonics_is_exact() -> None:
    period = 0.3
    x = period * np.arange(16) / 16
    w = 2 * math.pi / period
    values = 0.5 + 2.0 * np.cos(w * x - 0.4) + 0.3 * np.sin(3 * w * x) - 0.1 * np.cos(7 * w * x)
    slope = -2.0 * w * np.sin(w * x - 0.4) + 0.9 * w * np.cos(3 * w * x) + 0.7 * w * np.sin(7 * w * x)
    assert np.allclose(_derivative(values, period), slope, rtol=0, atol=1e-12 * w)


def test_derivative_leaves_out_the_harmonic_that_cannot_be_told_apart() -> None:
    # At exactly half the sampling rate, cos samples as +-1 and sin as 0: the
    # slope there is unknown, and is taken as zero.
    values = np.cos(math.pi * np.arange(8))
    assert np.allclose(_derivative(values, 1.0), 0.0, atol=1e-12)


# What the study solves


def test_angles_cover_whole_electrical_cycles_with_no_current(
    reference: dict, reference_emf: BackEmf
) -> None:
    cycle = 2 * math.pi / 21
    assert reference_emf.pole_pairs == 21
    assert np.allclose(reference_emf.angles, cycle * np.arange(48) / 48, rtol=0, atol=1e-15)
    assert np.allclose(reference_emf.times, reference_emf.angles / SPEED, rtol=0, atol=1e-15)
    assert sorted(reference_emf.linkage) == sorted(reference_emf.emf) == ["A", "B", "C"]
    for step in reference_emf.run.steps:
        assert not np.any(step.problem.current_density)


def test_linkage_is_that_of_a_plain_solve(small: dict) -> None:
    emf = run(small, points=8)
    k = 3
    problem = build_problem(small["machine"], rotate(small["mesh"], emf.angles[k]), MATERIALS)
    linked = flux_linkage(problem, solve(problem))
    for phase in "ABC":
        assert emf.linkage[phase][k] == pytest.approx(linked[phase], rel=1e-6)


def test_emf_is_the_rate_of_change_of_linkage(reference: dict, reference_emf: BackEmf) -> None:
    # Against a central difference from solves one node spacing either side,
    # which is good to about 0.3 % at 48 steps a cycle.
    k = 7
    h = step_angle(reference["mesh"])
    linked = []
    for angle in (reference_emf.angles[k] - h, reference_emf.angles[k] + h):
        problem = build_problem(reference["machine"], rotate(reference["mesh"], angle), MATERIALS)
        linked.append(flux_linkage(problem, solve(problem)))
    size = np.abs(reference_emf.emf["A"]).max()
    for phase in "ABC":
        difference = SPEED * (linked[1][phase] - linked[0][phase]) / (2 * h)
        assert reference_emf.emf[phase][k] == pytest.approx(difference, abs=0.01 * size)


# The plan's checks


def test_the_three_phases_are_120_electrical_degrees_apart(reference_emf: BackEmf) -> None:
    waves = {phase: fundamental(reference_emf.emf[phase]) for phase in "ABC"}
    size = abs(waves["A"])
    for k, phase in enumerate("ABC"):
        assert abs(waves[phase]) == pytest.approx(size, rel=1e-3)
        lag = math.remainder(np.angle(waves["A"]) - np.angle(waves[phase]), 2 * math.pi)
        expected = math.remainder(k * 2 * math.pi / 3, 2 * math.pi)
        assert math.degrees(lag) == pytest.approx(math.degrees(expected), abs=0.1)


def test_emf_is_in_phase_with_the_current_the_excitation_gives_at_no_advance(
    reference: dict, reference_emf: BackEmf
) -> None:
    # Machine.excitation puts current in phase with the back-EMF at advance 0,
    # which is what gives most torque per ampere.
    machine = reference["machine"]
    for phase in "ABC":
        current = np.array([machine.excitation(a, 1.0)[phase] for a in reference_emf.angles])
        shift = np.angle(fundamental(reference_emf.emf[phase])) - np.angle(fundamental(current))
        assert math.degrees(math.remainder(shift, 2 * math.pi)) == pytest.approx(0.0, abs=0.5)


def test_fundamental_frequency_is_pole_pairs_times_the_mechanical_frequency(small: dict) -> None:
    # Over one whole turn of the rotor, time series of the EMF: its strongest
    # frequency is p times the turns per second.
    pole_pairs = 5
    emf = run(small, points=8, cycles=pole_pairs)
    turn_time = 2 * math.pi / SPEED
    step = emf.times[1] - emf.times[0]
    assert len(emf.times) * step == pytest.approx(turn_time, rel=1e-12)
    for phase in "ABC":
        spectrum = np.abs(np.fft.rfft(emf.emf[phase]))
        frequencies = np.fft.rfftfreq(len(emf.times), d=step)
        strongest = frequencies[np.argmax(spectrum)]
        assert strongest == pytest.approx(pole_pairs / turn_time, rel=1e-9)
    assert emf.frequency == pytest.approx(pole_pairs * SPEED / (2 * math.pi), rel=1e-12)


def test_amplitude_scales_linearly_with_speed(reference: dict, reference_emf: BackEmf) -> None:
    for speed in (250.0, -100.0):
        other = run(reference, speed=speed, points=48)
        for phase in "ABC":
            assert np.array_equal(other.linkage[phase], reference_emf.linkage[phase])
            assert np.allclose(
                other.emf[phase], reference_emf.emf[phase] * speed / SPEED, rtol=1e-12, atol=0
            )
        assert other.amplitude["A"] == pytest.approx(
            reference_emf.amplitude["A"] * abs(speed) / SPEED, rel=1e-12
        )
        assert other.frequency == pytest.approx(21 * abs(speed) / (2 * math.pi), rel=1e-12)
        assert other.constant == pytest.approx(reference_emf.constant, rel=1e-12)
    assert reference_emf.constant == pytest.approx(reference_emf.amplitude["A"] / SPEED, rel=1e-3)


def test_the_three_phases_sum_to_their_triplen_harmonics_only(reference_emf: BackEmf) -> None:
    # Phases 120 degrees apart cancel in the sum at every harmonic except the
    # third and its multiples, which are in step in all three. The sum is
    # three times those; a star connection keeps them out of the line voltages.
    emf = reference_emf.emf
    total = emf["A"] + emf["B"] + emf["C"]
    size = reference_emf.amplitude["A"]
    order = np.arange(len(total) // 2 + 1)
    triplen = (order % 3 == 0) & (order > 0)
    total_waves = np.fft.rfft(total) / (len(total) / 2)
    a_waves = np.fft.rfft(emf["A"]) / (len(total) / 2)
    assert np.abs(total_waves[~triplen]).max() < 1e-3 * size
    assert np.allclose(total_waves[triplen], 3 * a_waves[triplen], rtol=0, atol=1e-3 * size)
    assert abs(a_waves[3]) > 0.03 * size  # this motor has a third harmonic to cancel
    lines = (emf["A"] - emf["B"], emf["B"] - emf["C"], emf["C"] - emf["A"])
    assert np.abs(sum(lines)).max() < 1e-12 * size
    # Line to line, the fundamental is root 3 times the phase one.
    assert abs(fundamental(lines[0])) == pytest.approx(math.sqrt(3) * size, rel=1e-3)


# A known answer, and the mesh


@pytest.mark.parametrize("name", ["qdd_outer_36s42p", "qdd_outer_12s10p", "qdd_outer_24s28p"])
def test_emf_agrees_with_the_formula_from_the_gap_field(name: str, reference: dict) -> None:
    # E1 = 2 omega Ns kw B1 r L: the fundamental B1 of the radial gap field at
    # radius r, linked by Ns series turns with winding factor kw. The formula
    # counts all of that flux as linking whole coils, while some of it enters
    # the slot openings and links only part of a coil, so it comes out a
    # little high (0.5 to 2.4 % measured).
    motor = reference if name == "qdd_outer_36s42p" else _motor(name)
    emf = run(motor, points=24)
    design = motor["machine"].design
    first = emf.run.steps[0]
    around = np.linspace(0.0, 2 * math.pi, 3600, endpoint=False)
    radial, _ = gap_flux_density(first.problem, first.solution, around)
    b1 = abs(np.fft.rfft(radial)[emf.pole_pairs]) / 1800
    gap = first.problem.gap
    radius = 0.5 * (gap.inner_radius + gap.outer_radius)
    series = design.winding.turns_per_coil * sum(
        coil.phase == "A" for coil in concentrated_winding(design.slots, design.poles)
    )
    kw = winding_factor(design.slots, design.poles)
    formula = 2 * SPEED * series * kw * b1 * radius * first.problem.depth
    assert emf.amplitude["A"] < formula
    assert emf.amplitude["A"] == pytest.approx(formula, rel=0.04)


def test_emf_barely_changes_with_a_finer_mesh(reference: dict) -> None:
    coarse = run(reference, points=24)
    fine = run(_motor("qdd_outer_36s42p", size=0.5), points=24)
    assert fine.amplitude["A"] == pytest.approx(coarse.amplitude["A"], rel=0.005)  # 0.09 % measured


# Inputs and output


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"points": 3}, "points"),
        ({"cycles": 0}, "cycles"),
        ({"speed": math.nan}, "speed"),
        ({"speed": math.inf}, "speed"),
        ({"speed": 0.0}, "speed"),
    ],
)
def test_back_emf_needs_enough_points_whole_cycles_and_a_finite_speed(
    small: dict, options: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run(small, **options)


def test_back_emf_reports_its_timing(small: dict) -> None:
    report = run(small, points=4).timing_report()
    for text in ("4 rotor angles", "turn rotor", "linear solve", "total"):
        assert text in report


def test_several_cycles_give_the_emf_of_one_cycle_again_and_again(small: dict) -> None:
    one = run(small, points=8)
    five = run(small, points=8, cycles=5)
    size = one.amplitude["A"]
    assert size > 1.0  # V
    # The mesh repeats every slice, not every electrical cycle, so the cycles
    # differ by mesh noise (a few parts in 100,000 measured).
    for phase in "ABC":
        for cycle in range(5):
            part = five.emf[phase][8 * cycle : 8 * (cycle + 1)]
            assert np.allclose(part, one.emf[phase], rtol=0, atol=1e-3 * size)
        assert five.amplitude[phase] == pytest.approx(one.amplitude[phase], rel=1e-3)
    assert five.constant == pytest.approx(one.constant, rel=1e-3)
