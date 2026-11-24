"""d- and q-axis inductance from flux-linkage changes."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import LinearMaterial, Magnet, load_material
from polecraft.core.post import flux_linkage
from polecraft.core.problem import build_problem
from polecraft.core.solver import solve
from polecraft.core.winding import end_winding_inductance
from polecraft.design import design_from_dict, design_to_dict, load_design
from polecraft.machines import machine_for
from polecraft.studies.back_emf import back_emf
from polecraft.studies.inductance import Inductance, dq_components, inductance
from polecraft.studies.loaded_torque import loaded_torque

ROOT = Path(__file__).resolve().parents[2]
MATERIALS = {
    # Placeholders: not the designs' real steel or magnet grade.
    "M270-35A": load_material("generic_35A_placeholder"),
    "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
}
# Steel that does not saturate and a magnet with no remanence that is as
# permeable as air: a rotor that looks the same along d and q.
ISOTROPIC = {
    "M270-35A": LinearMaterial(1000.0),
    "N42SH": Magnet(remanence=0.0, relative_permeability=1.0),
}


def machine_of(name: str, turns: int | None = None):
    design = load_design(ROOT / "designs" / f"{name}.yaml")
    if turns is not None:
        data = design_to_dict(design)
        data["winding"]["turns_per_coil"] = turns
        design = design_from_dict(data)
    return machine_for(design)


def _motor(name: str, size: float = 1.0) -> dict:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh

    machine = machine_of(name)
    return {"machine": machine, "mesh": build_mesh(machine.build_geometry(), size=size)}


def run(motor: dict, current: float, materials=MATERIALS, **options) -> Inductance:
    return inductance(motor["machine"], motor["mesh"], materials, current, **options)


@pytest.fixture(scope="module")
def reference() -> dict:
    return _motor("qdd_outer_36s42p")


@pytest.fixture(scope="module")
def small() -> dict:
    return _motor("qdd_outer_12s10p")


@pytest.fixture(scope="module")
def at_10(reference: dict) -> Inductance:
    return run(reference, 10.0)


@pytest.fixture(scope="module")
def at_40(reference: dict) -> Inductance:
    return run(reference, 40.0)


# The d and q axes, from the machine's excitation


@pytest.mark.parametrize("name", ["qdd_outer_36s42p", "qdd_outer_12s10p"])
def test_phase_currents_turn_into_the_expected_d_and_q_currents(name: str) -> None:
    machine = machine_of(name)
    current = 7.0
    for k in range(20):
        angle = 0.37 * k - 2.0
        advance = (-0.5 * math.pi, 0.0, 0.5 * math.pi, 0.3 * k - 3.0)[k % 4]
        d, q = dq_components(machine, angle, machine.excitation(angle, current, advance))
        assert d == pytest.approx(-current * math.sin(advance), abs=1e-12 * current)
        assert q == pytest.approx(current * math.cos(advance), abs=1e-12 * current)


def test_values_must_name_the_phases_of_the_excitation() -> None:
    machine = machine_of("qdd_outer_12s10p")
    with pytest.raises(ValueError, match="phases A, B, C"):
        dq_components(machine, 0.0, {"A": 1.0, "B": 2.0})


def test_an_unbalanced_excitation_is_refused() -> None:
    machine = machine_of("qdd_outer_12s10p")

    class Lopsided(type(machine)):
        def excitation(self, angle, current, advance=0.0):
            values = super().excitation(angle, current, advance)
            return {**values, "A": 1.5 * values["A"]}

    with pytest.raises(ValueError, match="not a balanced set"):
        dq_components(Lopsided(machine.design), 0.3, {"A": 1.0, "B": 0.0, "C": -1.0})


@pytest.mark.parametrize("current", [0.0, -1.0, math.nan, math.inf])
def test_current_must_be_a_positive_number(current: float) -> None:
    machine = machine_of("qdd_outer_12s10p")
    with pytest.raises(ValueError, match="current"):
        inductance(machine, None, MATERIALS, current)


def test_at_least_one_point_is_needed() -> None:
    machine = machine_of("qdd_outer_12s10p")
    with pytest.raises(ValueError, match="points"):
        inductance(machine, None, MATERIALS, 1.0, points=0)


# Known answers


def test_frame_puts_the_magnet_on_the_d_axis(reference: dict, at_10: Inductance) -> None:
    assert at_10.magnet_flux > 0.0
    machine = reference["machine"]
    for angle, step in zip(at_10.angles, at_10.runs["none"].steps):
        d, q = dq_components(machine, float(angle), flux_linkage(step.problem, step.solution))
        assert abs(q) < 1e-3 * d


def test_magnet_flux_matches_the_back_emf_constant(reference: dict, at_10: Inductance) -> None:
    emf = back_emf(reference["machine"], reference["mesh"], MATERIALS, speed=100.0, points=24)
    assert at_10.pole_pairs * at_10.magnet_flux == pytest.approx(emf.constant, rel=2e-3)


@pytest.mark.parametrize(("current", "advance"), [(10.0, 0.25 * math.pi), (40.0, 0.0)])
def test_dq_torque_equals_the_torque_from_the_phase_linkages(
    reference: dict, current: float, advance: float
) -> None:
    machine = reference["machine"]
    loaded = loaded_torque(machine, reference["mesh"], MATERIALS, current, advance, points=12)
    power = []
    for k, angle in enumerate(loaded.angles):
        i_d, i_q = dq_components(machine, float(angle), {p: v[k] for p, v in loaded.currents.items()})
        psi_d, psi_q = dq_components(machine, float(angle), {p: v[k] for p, v in loaded.linkage.items()})
        power.append(1.5 * loaded.pole_pairs * (psi_d * i_q - psi_q * i_d))
    # Exact over a whole cycle: the rest of sum(i dpsi/dangle) averages to zero.
    assert np.mean(power) == pytest.approx(loaded.mean_from_linkage, rel=1e-9)
    assert np.mean(power) == pytest.approx(loaded.mean, rel=5e-3)


def test_inductance_scales_with_turns_squared(reference: dict) -> None:
    # 10 turns at 80 A make the same field as 20 turns at 40 A, saturation
    # and all, so the flux linkage halves and the inductance falls by four.
    mesh = reference["mesh"]
    twenty = inductance(machine_of("qdd_outer_36s42p", 20), mesh, MATERIALS, 40.0, points=3)
    ten = inductance(machine_of("qdd_outer_36s42p", 10), mesh, MATERIALS, 80.0, points=3)
    assert twenty.d_2d / ten.d_2d == pytest.approx(4.0, rel=1e-6)
    assert twenty.q_2d / ten.q_2d == pytest.approx(4.0, rel=1e-6)
    assert twenty.magnet_flux / ten.magnet_flux == pytest.approx(2.0, rel=1e-9)
    assert twenty.end_winding.estimate / ten.end_winding.estimate == pytest.approx(4.0, rel=1e-12)


@pytest.mark.parametrize("motor", ["reference", "small"])
def test_rotor_that_looks_the_same_both_ways_has_equal_d_and_q(request, motor: str) -> None:
    found = run(request.getfixturevalue(motor), 10.0, ISOTROPIC, points=4)
    assert found.d_2d / found.q_2d == pytest.approx(1.0, abs=1e-3)
    # Linear materials: the same at any current.
    again = run(request.getfixturevalue(motor), 40.0, ISOTROPIC, points=4)
    assert again.d_2d == pytest.approx(found.d_2d, rel=1e-9)
    assert again.q_2d == pytest.approx(found.q_2d, rel=1e-9)


def test_isotropic_inductance_is_self_less_mutual(reference: dict) -> None:
    # With three phases at 120 degrees, d or q current I in a winding whose
    # phases each have self-inductance L and mutual M links (L - M) I.
    machine, mesh = reference["machine"], reference["mesh"]
    blank = build_problem(machine, mesh, ISOTROPIC, current=0.0)
    matrix = np.zeros((3, 3))
    for column, phase in enumerate(sorted(blank.windings)):
        driven = dataclasses.replace(blank, current_density=blank.windings[phase] * 1.0)
        linked = flux_linkage(driven, solve(driven))
        matrix[:, column] = [linked[p] for p in sorted(linked)]
    assert matrix == pytest.approx(matrix.T, rel=1e-9)  # reciprocity
    self_less_mutual = np.mean(np.diag(matrix)) - np.mean(matrix[~np.eye(3, dtype=bool)])
    found = run(reference, 10.0, ISOTROPIC, points=1)
    assert found.d_2d == pytest.approx(self_less_mutual, rel=1e-3)
    assert found.q_2d == pytest.approx(self_less_mutual, rel=1e-3)


# Surface magnets with saturating steel


def test_d_and_q_are_close_below_saturation(at_10: Inductance) -> None:
    assert at_10.d_2d / at_10.q_2d == pytest.approx(1.0, abs=0.05)


def test_q_current_saturates_the_teeth(at_10: Inductance, at_40: Inductance) -> None:
    # At 40 A the q current adds to the magnet's flux in the teeth.
    assert at_40.q_2d < 0.8 * at_40.d_2d
    assert at_40.q_2d < 0.8 * at_10.q_2d


def test_q_current_lowers_the_magnets_linkage(at_40: Inductance) -> None:
    assert np.mean(at_40.d_under_q) < 0.9 * at_40.magnet_flux
    assert np.all(np.abs(at_40.q_under_d) < 0.05 * at_40.magnet_flux)


def test_inductance_barely_changes_with_rotor_angle(at_10: Inductance) -> None:
    assert np.ptp(at_10.q_axis_2d) < 0.01 * at_10.q_2d
    assert np.ptp(at_10.d_axis_2d) < 0.01 * at_10.d_2d


def test_end_turns_add_a_few_percent(reference: dict, small: dict, at_10: Inductance) -> None:
    ends = at_10.end_winding
    assert ends.estimate == end_winding_inductance(reference["machine"]).estimate
    assert at_10.d == at_10.d_2d + ends.estimate and at_10.q == at_10.q_2d + ends.estimate
    assert 0.005 < ends.estimate / at_10.q_2d < 0.06
    other = run(small, 10.0, points=1)
    assert 0.005 < other.end_winding.estimate / other.q_2d < 0.06


def test_machine_without_coil_shape_has_no_end_turns(small: dict) -> None:
    machine = small["machine"]

    class Plain(type(machine)):
        def coil_shape(self):
            raise NotImplementedError

    found = inductance(Plain(machine.design), small["mesh"], MATERIALS, 10.0, points=1)
    assert found.end_winding is None and found.d == found.d_2d and found.q == found.q_2d


def test_one_point_gives_the_value_at_the_mesh_angle(at_10: Inductance, reference: dict) -> None:
    single = run(reference, 10.0, points=1)
    assert list(single.angles) == [0.0]
    assert single.d_2d == pytest.approx(at_10.d_axis_2d[0], rel=1e-9)
    assert single.q_2d == pytest.approx(at_10.q_axis_2d[0], rel=1e-9)


def test_finer_mesh_changes_the_inductance_little(reference: dict) -> None:
    fine = _motor("qdd_outer_36s42p", size=0.5)
    coarse_value = run(reference, 10.0, points=1)
    fine_value = run(fine, 10.0, points=1)
    assert fine_value.d_2d == pytest.approx(coarse_value.d_2d, rel=0.02)
    assert fine_value.q_2d == pytest.approx(coarse_value.q_2d, rel=0.02)


def test_timing_report_names_the_three_runs(at_10: Inductance) -> None:
    report = at_10.timing_report()
    assert all(name in report for name in ("none:", "d:", "q:"))
