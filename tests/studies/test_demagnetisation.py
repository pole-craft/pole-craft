"""Magnet working point against the knee of its demagnetisation curve, at temperature."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import MU0, Magnet, load_material
from polecraft.studies.demagnetisation import Demagnetisation, demagnetisation

# A magnet made up for the tests, of the size of a sintered NdFeB grade's
# numbers; not taken from any datasheet.
MAGNET = Magnet(
    remanence=1.2,
    relative_permeability=1.05,
    remanence_coefficient=-0.0012,
    intrinsic_coercivity=1.6e6,
    intrinsic_coercivity_coefficient=-0.005,
    knee_ratio=0.9,
)
MATERIALS = {
    # Placeholder steel: not the design's real steel.
    "M270-35A": load_material("generic_35A_placeholder"),
    "N42SH": MAGNET,
}
POINTS = 12
D_AXIS = 0.5 * math.pi  # advance that puts the current along -d


def _motor(size: float) -> dict:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / "qdd_outer_12s10p.yaml"))
    return {"machine": machine, "mesh": build_mesh(machine.build_geometry(), size=size)}


@pytest.fixture(scope="module")
def small() -> dict:
    return _motor(1.0)


def check(motor: dict, current: float, advance: float = 0.0, temperature=None, **options):
    return demagnetisation(
        motor["machine"], motor["mesh"], MATERIALS, current, advance, temperature,
        points=POINTS, **options,
    )


@pytest.fixture(scope="module")
def rated(small: dict) -> Demagnetisation:
    """10 A at advance 0, the current the examples run at, at the design's 80 C."""
    return check(small, 10.0)


@pytest.fixture(scope="module")
def abused(small: dict) -> Demagnetisation:
    """Six times that current, all along -d, with the magnets at 150 C."""
    return check(small, 60.0, D_AXIS, 150.0)


def test_rated_case_is_not_flagged(rated: Demagnetisation) -> None:
    assert rated.temperature == 80.0  # the design's magnet temperature
    assert not rated.flagged
    assert rated.share_beyond == 0.0
    assert rated.margin > 0.5


def test_over_current_high_temperature_case_is_flagged(abused: Demagnetisation) -> None:
    assert abused.flagged
    assert abused.margin < 0.0
    assert abused.share_beyond > 0.5  # most of the magnet, not a corner
    assert abused.worst > abused.knee


def test_the_same_current_cold_is_not_flagged(small: dict, abused: Demagnetisation) -> None:
    # Hcj falls far faster with temperature than Br: the current that
    # passes the knee at 150 C does not at 80 C.
    cold = check(small, 60.0, D_AXIS)
    assert not cold.flagged
    assert cold.knee == pytest.approx(2 * abused.knee)


def test_knee_and_remanence_at_temperature(abused: Demagnetisation) -> None:
    assert abused.knee == pytest.approx(0.9 * 1.6e6 * (1 - 0.005 * 130))
    assert abused.remanence == pytest.approx(1.2 * (1 - 0.0012 * 130))
    expected = abused.remanence - 1.05 * MU0 * abused.knee
    assert abused.knee_flux_density == pytest.approx(expected)
    # The knee is passed where the flux density along the magnet falls below it.
    assert np.array_equal(abused.beyond, abused.flux_density < abused.knee_flux_density - 1e-12)


def test_d_current_demagnetises_more_than_q_current(small: dict, rated: Demagnetisation) -> None:
    along_d = check(small, 10.0, D_AXIS)
    none = check(small, 0.0)
    assert none.worst < rated.worst < along_d.worst
    # Opposing the magnet, the d current lowers the flux density in it.
    assert along_d.flux_density.mean() < none.flux_density.mean()


def test_no_load_working_point_is_near_the_simple_magnetic_circuit(small: dict) -> None:
    # Magnet of length lm in series with a gap g: B = Br lm / (lm + mu_r g).
    # Slot openings and leakage between poles take some flux away, so the
    # mean over the magnet lies a little below that.
    none = check(small, 0.0)
    design = small["machine"].design
    lm, g = design.geometry.magnet.thickness_mm, design.geometry.air_gap_mm
    circuit = none.remanence * lm / (lm + 1.05 * g)
    mean = float(np.average(none.flux_density, weights=none.area))
    assert 0.85 * circuit < mean < circuit


def test_finer_mesh_changes_little(abused: Demagnetisation) -> None:
    fine = check(_motor(0.6), 60.0, D_AXIS, 150.0)
    assert fine.worst == pytest.approx(abused.worst, rel=0.01)
    assert fine.share_beyond == pytest.approx(abused.share_beyond, abs=0.05)


def test_knee_ratio_can_be_given(small: dict) -> None:
    magnet = dataclasses.replace(MAGNET, knee_ratio=None)
    materials = {**MATERIALS, "N42SH": magnet}
    machine, mesh = small["machine"], small["mesh"]
    with pytest.raises(ValueError, match="knee of the demagnetisation curve is not known"):
        demagnetisation(machine, mesh, materials, 10.0, points=1)
    given = demagnetisation(machine, mesh, materials, 10.0, knee_ratio=0.8, points=1)
    assert given.knee == pytest.approx(0.8 * MAGNET.intrinsic_coercivity_at(80.0))


@pytest.mark.parametrize(
    "magnet, message",
    [
        (load_material("generic_NdFeB_placeholder"), "no intrinsic coercivity"),
        (MAGNET.at_temperature(80.0), "give the magnet at 20 C"),
        (load_material("copper"), "needs a Magnet model"),
    ],
)
def test_magnet_without_the_data_is_refused(small: dict, magnet, message) -> None:
    materials = {**MATERIALS, "N42SH": magnet}
    with pytest.raises(ValueError, match=message):
        demagnetisation(small["machine"], small["mesh"], materials, 10.0, points=1)


def test_bad_input_is_refused(small: dict) -> None:
    with pytest.raises(ValueError, match="finite"):
        check(small, math.nan)
    with pytest.raises(ValueError, match="points"):
        demagnetisation(small["machine"], small["mesh"], MATERIALS, 10.0, points=0)
