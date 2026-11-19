"""Cogging and back-EMF waveforms against FEMM, from stored results.

FEMM solved the whole machine with no current at a series of rotor angles,
turning the rotor and meshing again at each (tests/data/femm/make_waveforms.py
made the file). The studies here solve the repeating slice at the same angles.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import Magnet, load_material
from polecraft.design import design_to_dict
from polecraft.studies._cycle import derivative
from polecraft.studies.back_emf import back_emf
from polecraft.studies.cogging import cogging_torque

DATA = Path(__file__).resolve().parents[1] / "data" / "femm"
REMANENCE, PERMEABILITY = 1.2, 1.05
MATERIALS = {
    # Placeholders: not the designs' real steel or magnet grade.
    "M270-35A": load_material("generic_35A_placeholder"),
    "N42SH": Magnet(remanence=REMANENCE, relative_permeability=PERMEABILITY),
}
SPEED = 100.0  # rad/s


@pytest.fixture(scope="module")
def femm() -> dict:
    return json.loads((DATA / "waveforms.json").read_text(encoding="utf-8"))


def _motor(name: str, size: float) -> dict:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / f"{name}.yaml"))
    return {"machine": machine, "mesh": build_mesh(machine.build_geometry(), size=size)}


def harmonic(values, k: int = 1) -> complex:
    """Harmonic k of samples over one whole period, as a complex peak amplitude."""
    values = np.asarray(values, dtype=float)
    return np.fft.rfft(values)[k] / (len(values) / 2)


def phase_gap(a: complex, b: complex) -> float:
    """Angle between two harmonics in degrees, -180 to 180."""
    return math.degrees(math.remainder(np.angle(a) - np.angle(b), 2 * math.pi))


# The stored results are for the model as it is now


def test_femm_was_given_the_steel_and_magnet_the_tests_use(femm: dict) -> None:
    steel = MATERIALS["M270-35A"]
    assert femm["steel"]["bh_curve"] == [[float(b), float(h)] for b, h in zip(steel.b, steel.h)]
    assert femm["magnet"] == {"remanence": REMANENCE, "relative_permeability": PERMEABILITY}


@pytest.mark.parametrize("name", ["qdd_outer_36s42p", "qdd_outer_12s10p"])
def test_femm_was_run_on_the_designs_as_they_are_now(femm: dict, name: str) -> None:
    from polecraft.design import load_design

    root = Path(__file__).resolve().parents[2]
    # If this fails, run tests/data/femm/make_waveforms.py again.
    assert femm["designs"][name] == design_to_dict(load_design(root / "designs" / f"{name}.yaml"))


def test_femm_runs_are_at_the_angles_the_studies_use(
    femm: dict, reference_emf, small_cogging, reference_cogging
) -> None:
    studies = {
        "reference_back_emf": reference_emf,
        "reference_cogging": reference_cogging,
        "reference_cogging_fine": reference_cogging,
        "small_cogging": small_cogging,
        "small_cogging_fine": small_cogging,
    }
    assert sorted(femm["runs"]) == sorted(studies)
    for name, run in femm["runs"].items():
        angles = studies[name].angles
        assert run["angles"] == pytest.approx(list(angles), rel=1e-12, abs=1e-15)
        assert len(run["torque"]) == len(angles)
        assert all(len(values) == len(angles) for values in run["flux_linkage"].values())


@pytest.mark.parametrize(
    ("run", "static_run"),
    [
        ("reference_back_emf", "normal"),
        ("reference_cogging", "normal"),
        ("reference_cogging_fine", "fine"),
    ],
)
def test_turning_the_rotor_in_femm_starts_from_the_static_model(
    femm: dict, run: str, static_run: str
) -> None:
    # At angle 0 the turned model is the one checked in B8: the same mesh,
    # torque and linkage. The torque would show anything that the grouping
    # for the turn put on the wrong side.
    static = json.loads((DATA / "qdd_outer_36s42p.json").read_text(encoding="utf-8"))
    idle = static["runs"][static_run]["cases"][0]
    turned = femm["runs"][run]
    assert turned["nodes"] == static["runs"][static_run]["nodes"]
    assert turned["torque"][0] == pytest.approx(idle["torque"], rel=1e-6, abs=1e-9)
    for phase, value in idle["flux_linkage"].items():
        assert turned["flux_linkage"][phase][0] == pytest.approx(value, rel=1e-6, abs=1e-9)


# Back-EMF of the reference motor


@pytest.fixture(scope="module")
def reference_emf():
    motor = _motor("qdd_outer_36s42p", size=1.0)
    return back_emf(motor["machine"], motor["mesh"], MATERIALS, SPEED, points=24)


def femm_emf(femm: dict) -> dict:
    run = femm["runs"]["reference_back_emf"]
    cycle = 2 * math.pi / (femm["designs"]["qdd_outer_36s42p"]["poles"] // 2)
    return {p: SPEED * derivative(values, cycle) for p, values in run["flux_linkage"].items()}


def test_back_emf_fundamental_matches_femm(femm: dict, reference_emf) -> None:
    theirs = femm_emf(femm)
    for phase in "ABC":
        ours, other = harmonic(reference_emf.emf[phase]), harmonic(theirs[phase])
        assert abs(ours) == pytest.approx(abs(other), rel=5e-3)  # 0.14 % measured
        assert phase_gap(ours, other) == pytest.approx(0.0, abs=0.1)  # 0.01 deg measured


def test_back_emf_third_harmonic_matches_femm(femm: dict, reference_emf) -> None:
    # Both programs find the same third harmonic, so it is the model's, not a solver's.
    theirs = femm_emf(femm)
    for phase in "ABC":
        ours, other = harmonic(reference_emf.emf[phase], 3), harmonic(theirs[phase], 3)
        assert abs(other) > 0.05 * abs(harmonic(theirs[phase]))
        assert abs(ours) == pytest.approx(abs(other), rel=0.03)  # 0.2 % measured
        assert phase_gap(ours, other) == pytest.approx(0.0, abs=1.0)


def test_back_emf_waveform_matches_femm_point_by_point(femm: dict, reference_emf) -> None:
    theirs = femm_emf(femm)
    peak = np.abs(theirs["A"]).max()
    for phase in "ABC":
        assert np.abs(reference_emf.emf[phase] - theirs[phase]).max() < 0.01 * peak  # 0.19 %
        linked = np.array(femm["runs"]["reference_back_emf"]["flux_linkage"][phase])
        assert np.abs(reference_emf.linkage[phase] - linked).max() < 5e-3 * np.abs(linked).max()


# Cogging torque


def cogging_of(femm: dict, run: str) -> np.ndarray:
    return np.array(femm["runs"][run]["torque"])


def test_femm_settles_on_the_12s10p_cogging(femm: dict) -> None:
    normal, fine = cogging_of(femm, "small_cogging"), cogging_of(femm, "small_cogging_fine")
    assert abs(harmonic(normal)) == pytest.approx(abs(harmonic(fine)), rel=0.03)  # 0.6 % measured
    for run in (normal, fine):
        assert abs(run.mean()) < 0.03 * np.ptp(run)  # cogging averages zero over a period


@pytest.fixture(scope="module")
def small_cogging():
    motor = _motor("qdd_outer_12s10p", size=0.35)
    return cogging_torque(motor["machine"], motor["mesh"], MATERIALS, points=12)


def test_12s10p_cogging_matches_femm(femm: dict, small_cogging) -> None:
    fine = cogging_of(femm, "small_cogging_fine")
    ours, other = harmonic(small_cogging.torque), harmonic(fine)
    assert abs(other) > 4e-3  # N m
    assert abs(ours) == pytest.approx(abs(other), rel=0.06)  # 3.0 % measured
    assert phase_gap(ours, other) == pytest.approx(0.0, abs=2.0)  # 0.2 deg measured
    assert np.abs(small_cogging.torque - fine).max() < 0.1 * np.ptp(fine)


@pytest.fixture(scope="module")
def reference_cogging():
    motor = _motor("qdd_outer_36s42p", size=0.35)
    return cogging_torque(motor["machine"], motor["mesh"], MATERIALS, points=12)


def test_reference_cogging_is_hard_for_femm_too(femm: dict) -> None:
    # About 2 mN m against 3.19 N m at 10 A. FEMM's normal mesh does not
    # settle it (its mean is off zero and its size 41 % above the fine run's);
    # its fine mesh does. So the comparison below is with the fine run.
    normal = cogging_of(femm, "reference_cogging")
    fine = cogging_of(femm, "reference_cogging_fine")
    assert abs(fine.mean()) < 0.03 * np.ptp(fine)
    assert abs(normal.mean()) > 0.1 * np.ptp(normal)
    assert abs(harmonic(normal)) > 1.2 * abs(harmonic(fine))


def test_reference_cogging_matches_femm(femm: dict, reference_cogging) -> None:
    # Ours moves with the mesh (1.94 mN m at size 0.35, 2.14 at 0.25, 2.36
    # at 0.5: 6, 16 and 28 % above FEMM's fine run), so the size is held to
    # 15 % at size 0.35 only; that is no sign of a settled answer.
    fine = cogging_of(femm, "reference_cogging_fine")
    ours, other = harmonic(reference_cogging.torque), harmonic(fine)
    assert abs(other) > 1.5e-3  # N m
    assert abs(ours) == pytest.approx(abs(other), rel=0.15)  # 5.6 % measured
    assert phase_gap(ours, other) == pytest.approx(0.0, abs=2.0)  # 0.4 deg measured
