"""Timing a solve section by section."""

import math
import os
import time
from pathlib import Path

import pytest

from polecraft.benchmark import SECTIONS, machine_info, time_machine
from polecraft.core.materials import Magnet, load_material
from polecraft.design import load_design
from polecraft.machines import machine_for

pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")

ROOT = Path(__file__).resolve().parents[1]
SIZE = 2.0  # a coarse mesh: the tests are about the timing, not the field


@pytest.fixture(scope="module")
def machine():
    return machine_for(load_design(ROOT / "designs" / "qdd_outer_12s10p.yaml"))


@pytest.fixture(scope="module")
def materials(machine) -> dict:
    # Placeholders: not the design's real steel or magnet grade.
    design = machine.design
    steel = load_material("generic_35A_placeholder")
    return {
        design.steel.stator: steel,
        design.steel.rotor: steel,
        design.magnet.material: Magnet(remanence=1.2, relative_permeability=1.05),
    }


@pytest.fixture(scope="module")
def timed(machine, materials):
    began = time.perf_counter()
    run = time_machine(machine, materials, size=SIZE, current=5.0)
    return run, time.perf_counter() - began


def test_sections_sum_to_the_total(timed) -> None:
    run, _ = timed
    assert [name for name, _ in run.sections.stages] == list(SECTIONS)
    assert run.total == pytest.approx(sum(run.seconds(name) for name in SECTIONS))
    assert run.total == pytest.approx(run.wall)


def test_the_run_is_timed_from_outside_and_little_is_left_untimed(timed) -> None:
    run, outside = timed
    assert 0.8 * outside <= run.wall <= outside
    assert all(run.seconds(name) >= 0.0 for name in SECTIONS)
    assert run.seconds("mesh") > 0.0 and run.seconds("solve") > 0.0
    assert run.seconds("other") <= 0.25 * run.wall


def test_a_section_that_did_not_occur_takes_no_time(timed) -> None:
    assert timed[0].seconds("no such section") == 0.0


def test_a_sweep_solves_every_angle_with_the_same_sections(machine, materials) -> None:
    angles = [0.0, 0.01, 0.02]
    sweep = time_machine(machine, materials, angles, current=5.0, size=SIZE)
    assert sweep.angles == 3 and len(sweep.torque) == 3
    assert [name for name, _ in sweep.sections.stages] == list(SECTIONS)
    assert sweep.total == pytest.approx(sweep.wall)
    static = time_machine(machine, materials, [0.0], current=5.0, size=SIZE)
    assert static.angles == 1
    assert sweep.torque[0] == pytest.approx(static.torque[0], rel=1e-9)
    assert sweep.nodes == static.nodes and sweep.elements == static.elements


def test_the_whole_machine_has_more_nodes_than_its_slice_and_the_same_torque(machine, materials) -> None:
    slice_run = time_machine(machine, materials, current=5.0, size=SIZE)
    whole = time_machine(machine, materials, current=5.0, size=SIZE, full=True)
    assert whole.nodes > slice_run.nodes
    assert whole.torque[0] == pytest.approx(slice_run.torque[0], rel=5e-3)
    assert math.isfinite(whole.torque[0])


def test_the_report_lists_the_sections_and_the_size(timed) -> None:
    text = timed[0].report()
    assert text.splitlines()[0].startswith(f"Benchmark ({timed[0].nodes} nodes")
    for name in SECTIONS:
        assert name in text
    assert text.splitlines()[-1].lstrip().startswith("total")


def test_machine_info_describes_the_machine_and_the_software() -> None:
    info = machine_info()
    assert info["logical_cores"] == os.cpu_count()
    assert info["python"] and info["numpy"] and info["scipy"] and info["polecraft"]
    assert info["gmsh"]  # this module needs it
    assert info["ram_gb"] is None or info["ram_gb"] > 0.0
    assert info["cpu"] is None or isinstance(info["cpu"], str)
