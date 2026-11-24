"""d- and q-axis inductance against FEMM, from stored results.

FEMM solved the whole reference machine at rotor angle 0. The q-axis cases
(0, 10 and 40 A at advance 0) are those of tests/data/femm/make_reference.py;
the d-axis cases are from tests/data/femm/make_inductance.py. The study
solves the repeating slice at the same angle on a mesh of size 0.5, where
its inductance is within a few tenths of a percent of a much finer mesh.
Both programs model the same 2D machine, so the end turns are left out.
"""

import json
import math
from pathlib import Path

import pytest

from polecraft.core.materials import Magnet, load_material
from polecraft.core.post import flux_linkage
from polecraft.core.problem import build_problem
from polecraft.core.solver import solve
from polecraft.design import design_to_dict, load_design
from polecraft.machines import machine_for
from polecraft.studies.inductance import Inductance, dq_components, inductance

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data" / "femm"
REMANENCE, PERMEABILITY = 1.2, 1.05
MATERIALS = {
    # Placeholders: not the design's real steel or magnet grade.
    "M270-35A": load_material("generic_35A_placeholder"),
    "N42SH": Magnet(remanence=REMANENCE, relative_permeability=PERMEABILITY),
}
DESIGN = ROOT / "designs" / "qdd_outer_36s42p.yaml"


@pytest.fixture(scope="module")
def machine():
    return machine_for(load_design(DESIGN))


@pytest.fixture(scope="module")
def static() -> dict:
    return json.loads((DATA / "qdd_outer_36s42p.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def femm() -> dict:
    return json.loads((DATA / "inductance.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mesh(machine):
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh

    return build_mesh(machine.build_geometry(), size=0.5)


@pytest.fixture(scope="module")
def ours_10(machine, mesh) -> Inductance:
    return inductance(machine, mesh, MATERIALS, 10.0, points=1)


@pytest.fixture(scope="module")
def ours_40(machine, mesh) -> Inductance:
    return inductance(machine, mesh, MATERIALS, 40.0, points=1)


def dq_of(machine, case: dict) -> tuple[float, float]:
    return dq_components(machine, 0.0, case["flux_linkage"])


# The stored results are for the model as it is now


def test_femm_was_given_the_design_steel_and_magnet_the_tests_use(femm: dict) -> None:
    assert femm["design_values"] == design_to_dict(load_design(DESIGN))
    steel = MATERIALS["M270-35A"]
    assert femm["steel"]["bh_curve"] == [[float(b), float(h)] for b, h in zip(steel.b, steel.h)]
    assert femm["magnet"] == {"remanence": REMANENCE, "relative_permeability": PERMEABILITY}
    assert femm["rotor_angle"] == 0.0


def test_femm_was_given_the_currents_the_study_uses(machine, femm: dict) -> None:
    for run in femm["runs"].values():
        for case in run["cases"].values():
            expected = machine.excitation(0.0, case["current"], case["advance"])
            assert case["phase_currents"] == pytest.approx(expected, abs=1e-12)


def test_new_q_case_repeats_the_earlier_reference(machine, femm: dict, static: dict) -> None:
    earlier = dq_of(machine, static["runs"]["fine"]["cases"][1])
    again = dq_of(machine, femm["runs"]["fine"]["cases"]["q_10"])
    assert again[1] == pytest.approx(earlier[1], rel=5e-4)


# The comparison


def test_magnet_linkage_matches_femm(machine, femm: dict, ours_10: Inductance) -> None:
    magnet = dq_of(machine, femm["runs"]["fine"]["cases"]["none"])[0]
    assert ours_10.magnet_flux == pytest.approx(magnet, rel=5e-3)


@pytest.mark.parametrize(("index", "current"), [(1, 10.0), (2, 40.0)])
def test_q_axis_inductance_matches_femm(
    machine, static: dict, ours_10: Inductance, ours_40: Inductance, index: int, current: float
) -> None:
    cases = static["runs"]["fine"]["cases"]
    assert cases[index]["current"] == current and static["advance"] == 0.0
    femm_lq = (dq_of(machine, cases[index])[1] - dq_of(machine, cases[0])[1]) / current
    ours = ours_10 if current == 10.0 else ours_40
    assert ours.q_2d == pytest.approx(femm_lq, rel=0.01)


@pytest.mark.parametrize(("case", "current"), [("d_10", 10.0), ("d_40", 40.0)])
def test_d_axis_inductance_matches_femm(
    machine, femm: dict, ours_10: Inductance, ours_40: Inductance, case: str, current: float
) -> None:
    cases = femm["runs"]["fine"]["cases"]
    femm_ld = (dq_of(machine, cases[case])[0] - dq_of(machine, cases["none"])[0]) / -current
    ours = ours_10 if current == 10.0 else ours_40
    assert ours.d_2d == pytest.approx(femm_ld, rel=0.01)


def test_d_current_that_aids_the_magnet_gives_less_inductance(machine, mesh, femm: dict) -> None:
    # Aiding the magnet pushes the teeth further into saturation.
    cases = femm["runs"]["fine"]["cases"]
    none = dq_of(machine, cases["none"])[0]
    femm_aiding = (dq_of(machine, cases["d_aiding_10"])[0] - none) / 10.0
    femm_opposing = (dq_of(machine, cases["d_10"])[0] - none) / -10.0
    assert femm_aiding < femm_opposing

    def linked_d(current: float, advance: float) -> float:
        problem = build_problem(machine, mesh, MATERIALS, current=current, advance=advance)
        return dq_components(machine, 0.0, flux_linkage(problem, solve(problem)))[0]

    ours_aiding = (linked_d(10.0, -0.5 * math.pi) - linked_d(0.0, 0.0)) / 10.0
    assert ours_aiding == pytest.approx(femm_aiding, rel=0.01)
