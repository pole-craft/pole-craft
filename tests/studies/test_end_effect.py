"""End factor of a finite stack: tends to 1 as the stack grows."""

import math
from pathlib import Path

import numpy as np
import pytest

import polecraft.studies.end_effect as end_module
from polecraft.design import load_design
from polecraft.machines import machine_for
from polecraft.studies.end_effect import end_effect, end_factor

GAP, MAGNET = 0.6e-3, 2.5e-3  # m, as in the 36-slot, 42-pole example


@pytest.fixture(scope="module")
def lengths() -> tuple:
    stacks = np.array([2e-3, 5e-3, 12e-3, 50e-3, 200e-3, 1.0])
    return stacks, [end_factor(stack, GAP, MAGNET) for stack in stacks]


def test_factor_tends_to_1_as_the_stack_grows(lengths) -> None:
    stacks, results = lengths
    misses = np.abs(np.array([r.factor for r in results]) - 1.0)
    assert np.all(np.diff(misses) < 0.0)
    assert misses[-1] < 1e-4
    assert misses[2] < 0.05  # a 12 mm stack is still near the 2D answer here


def test_ends_add_a_fixed_length_each(lengths) -> None:
    # Far from the end the cut is the 1D circuit, so the ends give the same
    # length gained (or lost) whatever the stack: factor = 1 + 2 d / L.
    stacks, results = lengths
    extensions = np.array([r.extension for r in results])
    assert extensions[2:] == pytest.approx(np.full(len(stacks) - 2, extensions[-1]), rel=1e-3)
    for stack, result in zip(stacks, results):
        assert result.factor == pytest.approx(1.0 + 2.0 * result.extension / stack, rel=1e-12)
    # Less than a gap either way: fringing into the stator's end face and
    # leakage round the magnet's end nearly cancel when they line up.
    assert abs(extensions[-1]) < GAP


def test_long_stack_carries_the_magnetic_circuit_flux() -> None:
    result = end_factor(1.0, GAP, MAGNET, remanence=1.2, relative_permeability=1.05)
    middle = 1.2 * MAGNET / (MAGNET + 1.05 * GAP)
    assert result.flux_2d == pytest.approx(middle * 0.5)
    assert result.flux == pytest.approx(result.flux_2d, rel=1e-4)


def test_factor_does_not_depend_on_the_remanence() -> None:
    weak = end_factor(12e-3, GAP, MAGNET, remanence=0.4)
    strong = end_factor(12e-3, GAP, MAGNET, remanence=1.3)
    assert strong.factor == pytest.approx(weak.factor, rel=1e-10)
    assert strong.flux == pytest.approx(weak.flux * 1.3 / 0.4, rel=1e-10)


def test_magnet_overhang_raises_the_factor() -> None:
    factors = [end_factor(12e-3, GAP, MAGNET, magnet_overhang=o).factor for o in (0, 5e-4, 1e-3)]
    assert factors[0] < factors[1] < factors[2]
    assert factors[1] > 1.0


def test_finer_grid_changes_little(monkeypatch) -> None:
    standard = end_factor(12e-3, GAP, MAGNET).factor
    monkeypatch.setattr(end_module, "FINE", 0.025)
    monkeypatch.setattr(end_module, "GROWTH", 1.05)
    monkeypatch.setattr(end_module, "REACH", 60.0)
    assert end_factor(12e-3, GAP, MAGNET).factor == pytest.approx(standard, abs=1e-4)


def test_end_effect_of_a_design() -> None:
    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / "qdd_outer_36s42p.yaml"))
    result = end_effect(machine)
    design = machine.design
    assert result.stack_length == pytest.approx(design.axial.stack_length_mm * 1e-3)
    assert result.gap == pytest.approx(design.geometry.air_gap_mm * 1e-3)
    assert result.magnet_thickness == pytest.approx(design.geometry.magnet.thickness_mm * 1e-3)
    assert result.corrected(2.0) == pytest.approx(2.0 * result.factor)
    assert 0.9 < result.factor < 1.1


@pytest.mark.parametrize(
    "arguments, message",
    [
        ((0.0, GAP, MAGNET), "stack length"),
        ((0.01, -GAP, MAGNET), "gap"),
        ((0.01, GAP, math.nan), "magnet thickness"),
    ],
)
def test_bad_lengths_are_refused(arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        end_factor(*arguments)
    with pytest.raises(ValueError, match="overhang"):
        end_factor(0.01, GAP, MAGNET, magnet_overhang=-1e-3)
