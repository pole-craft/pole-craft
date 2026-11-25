"""Flux and torque maps over d and q current, interpolated between solved points."""

import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import Magnet, load_material
from polecraft.studies.flux_map import (
    FluxMap,
    current_and_advance,
    flux_map,
    operating_point,
)
from polecraft.studies.loaded_torque import loaded_torque
from polecraft.timing import Timings

MATERIALS = {
    # Placeholders: not the designs' real steel or magnet grade.
    "M270-35A": load_material("generic_35A_placeholder"),
    "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
}
D_AXIS = [-24.0, -16.0, -8.0, 0.0]
Q_AXIS = [-16.0, -8.0, 0.0, 8.0, 16.0, 24.0]
# Rotor angles at each grid point; the check solves use the same. With 6 the
# torque's 6th harmonic would show in the means and spoil the mirror check.
POINTS = 12


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
def solved(small: dict) -> FluxMap:
    return flux_map(small["machine"], small["mesh"], MATERIALS, D_AXIS, Q_AXIS, points=POINTS)


def direct(small: dict, i_d: float, i_q: float, points: int = POINTS):
    return operating_point(small["machine"], small["mesh"], MATERIALS, i_d, i_q, points)


def linear_map(d_axis, q_axis) -> FluxMap:
    """A map made by hand for a machine with no saturation, where the answer is known."""
    magnet, l_d, l_q, pole_pairs = 0.006, 2.0e-5, 2.4e-5, 5
    i_d, i_q = np.meshgrid(d_axis, q_axis, indexing="ij")
    psi_d, psi_q = magnet + l_d * i_d, l_q * i_q
    return FluxMap(
        d_currents=np.array(d_axis, dtype=float),
        q_currents=np.array(q_axis, dtype=float),
        pole_pairs=pole_pairs,
        points=1,
        d_linkage=psi_d,
        q_linkage=psi_q,
        torque=1.5 * pole_pairs * (psi_d * i_q - psi_q * i_d),
        timings=Timings(),
    )


# d and q currents


@pytest.mark.parametrize(
    "i_d, i_q, size, advance",
    [
        (0.0, 10.0, 10.0, 0.0),
        (-10.0, 0.0, 10.0, 0.5 * math.pi),
        (10.0, 0.0, 10.0, -0.5 * math.pi),
        (0.0, -10.0, 10.0, math.pi),
        (-3.0, 4.0, 5.0, math.atan2(3.0, 4.0)),
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_current_and_advance(i_d, i_q, size, advance) -> None:
    found = current_and_advance(i_d, i_q)
    assert found == pytest.approx((size, advance))
    # The inverse, as in the inductance study: i_d = -I sin a, i_q = I cos a.
    assert -found[0] * math.sin(found[1]) == pytest.approx(i_d, abs=1e-12)
    assert found[0] * math.cos(found[1]) == pytest.approx(i_q, abs=1e-12)


# Interpolation, on a map made by hand


def test_linear_machine_is_reproduced_everywhere() -> None:
    known = linear_map([-30.0, -20.0, -10.0, 0.0], [-20.0, -10.0, 0.0, 10.0, 20.0])
    i_d = np.array([-27.3, -14.1, -0.2, -30.0, 0.0])
    i_q = np.array([-19.9, 3.7, 17.5, 20.0, -20.0])
    psi_d, psi_q, torque = known.at(i_d, i_q)
    assert psi_d == pytest.approx(0.006 + 2.0e-5 * i_d, rel=1e-12)
    assert psi_q == pytest.approx(2.4e-5 * i_q, rel=1e-12, abs=1e-15)
    # Torque is a product of two linear parts, which a bicubic spline holds exactly.
    assert torque == pytest.approx(7.5 * (psi_d * i_q - psi_q * i_d), rel=1e-10, abs=1e-12)
    assert np.allclose(known.torque_from_linkage, known.torque, rtol=1e-12, atol=1e-15)


def test_two_points_an_axis_give_straight_lines() -> None:
    known = linear_map([-10.0, 0.0], [0.0, 10.0])
    psi_d, psi_q, _ = known.at(-4.0, 6.0)
    assert float(psi_d) == pytest.approx(0.006 - 8.0e-5)
    assert float(psi_q) == pytest.approx(1.44e-4)


def test_shape_of_the_answer_follows_the_question() -> None:
    known = linear_map([-10.0, -5.0, 0.0], [0.0, 5.0, 10.0])
    psi_d, psi_q, torque = known.at(np.full((2, 3), -2.0), 4.0)
    assert psi_d.shape == psi_q.shape == torque.shape == (2, 3)
    assert np.ndim(known.at(-2.0, 4.0)[0]) == 0


def test_outside_the_map_is_refused() -> None:
    known = linear_map([-10.0, -5.0, 0.0], [0.0, 5.0, 10.0])
    assert known.covers([-10.0, 0.0, 0.1, -5.0], [5.0, 10.0, 5.0, -0.1]).tolist() == [
        True,
        True,
        False,
        False,
    ]
    with pytest.raises(ValueError, match="i_d = 0.1 A, i_q = 5 A is outside the map"):
        known.at([-5.0, 0.1], 5.0)
    # The edges themselves are in, also after rounding.
    known.at(-10.0 - 1e-12, 10.0 + 1e-12)


@pytest.mark.parametrize(
    "d_axis, q_axis, message",
    [
        ([0.0], [0.0, 1.0], "at least two"),
        ([[0.0, 1.0]], [0.0, 1.0], "at least two"),
        ([0.0, math.nan], [0.0, 1.0], "finite"),
        ([0.0, 1.0], [1.0, 0.0], "rise"),
        ([0.0, 1.0], [1.0, 1.0], "rise"),
    ],
)
def test_bad_axes_are_refused(d_axis, q_axis, message) -> None:
    # Checked before anything is solved, so no machine is needed.
    with pytest.raises(ValueError, match=message):
        flux_map(None, None, MATERIALS, d_axis, q_axis)


# Solved maps


def test_map_holds_the_solved_values_at_its_grid_points(small: dict, solved: FluxMap) -> None:
    assert solved.d_linkage.shape == solved.torque.shape == (4, 6)
    point = direct(small, -16.0, 8.0)
    row, column = D_AXIS.index(-16.0), Q_AXIS.index(8.0)
    assert solved.torque[row, column] == pytest.approx(point.torque, rel=1e-9)
    assert solved.d_linkage[row, column] == pytest.approx(point.d_linkage, rel=1e-9)
    i_d, i_q = np.meshgrid(D_AXIS, Q_AXIS, indexing="ij")
    grids = (solved.d_linkage, solved.q_linkage, solved.torque)
    for found, grid in zip(solved.at(i_d, i_q), grids):
        assert np.allclose(found, grid, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "i_d, i_q",
    [(-4.0, 4.0), (-12.0, 20.0), (-21.0, -11.0), (-5.5, 13.3), (-19.0, 2.5), (-2.0, -14.0)],
)
def test_interpolated_values_match_direct_solves(small: dict, solved: FluxMap, i_d, i_q) -> None:
    point = direct(small, i_d, i_q)
    psi_d, psi_q, torque = (float(value) for value in solved.at(i_d, i_q))
    # Against the largest value on the map: q linkage and torque pass through
    # zero. On this 8 A grid the worst seen is 0.3 %, q linkage near an edge.
    assert psi_d == pytest.approx(point.d_linkage, abs=5e-3 * np.abs(solved.d_linkage).max())
    assert psi_q == pytest.approx(point.q_linkage, abs=5e-3 * np.abs(solved.q_linkage).max())
    assert torque == pytest.approx(point.torque, abs=5e-3 * np.abs(solved.torque).max())


def test_map_shows_the_physics_of_a_surface_magnet_machine(solved: FluxMap) -> None:
    zero_d, zero_q = D_AXIS.index(0.0), Q_AXIS.index(0.0)
    magnet = solved.d_linkage[zero_d, zero_q]
    scale = np.abs(solved.torque).max()
    assert magnet > 0.0
    assert abs(solved.q_linkage[zero_d, zero_q]) < 0.01 * np.abs(solved.q_linkage).max()
    assert abs(solved.torque[zero_d, zero_q]) < 0.01 * scale
    # Negative d current weakens the magnet's flux; q current makes torque.
    assert np.all(np.diff(solved.d_linkage, axis=0) > 0.0)
    assert np.all(np.diff(solved.torque[zero_d]) > 0.0)
    # A rotor that looks the same either way: braking mirrors motoring.
    up, down = Q_AXIS.index(16.0), Q_AXIS.index(-16.0)
    assert solved.torque[:, down] == pytest.approx(-solved.torque[:, up], abs=0.01 * scale)
    assert solved.d_linkage[:, down] == pytest.approx(solved.d_linkage[:, up], rel=0.01)


def test_mean_torque_is_that_of_the_loaded_torque_study(small: dict) -> None:
    point = direct(small, -12.0, 9.0, points=12)
    size, advance = current_and_advance(-12.0, 9.0)
    study = loaded_torque(small["machine"], small["mesh"], MATERIALS, size, advance, points=12)
    assert point.torque == pytest.approx(study.mean, rel=1e-9)


def test_gap_torque_agrees_with_the_linkage_formula(small: dict) -> None:
    # At the default 24 angles; with fewer the torque's 12th harmonic spoils the mean.
    point = direct(small, -20.0, 20.0, points=24)
    formula = 1.5 * 5 * (point.d_linkage * 20.0 + point.q_linkage * 20.0)
    assert point.torque == pytest.approx(formula, rel=3e-3)


def test_timing_report(solved: FluxMap) -> None:
    report = solved.timing_report()
    assert report.startswith("Timing (24 operating points, 288 rotor angles)")
    assert "linear solve" in report and "total" in report


def test_operating_point_refuses_bad_input(small: dict) -> None:
    with pytest.raises(ValueError, match="d current"):
        direct(small, math.inf, 1.0)
    with pytest.raises(ValueError, match="q current"):
        direct(small, 0.0, math.nan)
    with pytest.raises(ValueError, match="points"):
        direct(small, 0.0, 1.0, points=0)
