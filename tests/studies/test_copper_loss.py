"""Copper loss matches I squared R worked by hand, at the design temperature and others."""

import math
from pathlib import Path

import pytest

from polecraft.core.materials import Conductor, load_material
from polecraft.core.winding import phase_winding
from polecraft.design import load_design
from polecraft.machines import machine_for
from polecraft.studies.copper_loss import copper_loss

DESIGN = Path(__file__).resolve().parents[2] / "designs" / "qdd_outer_36s42p.yaml"


@pytest.fixture(scope="module")
def machine():
    return machine_for(load_design(DESIGN))


def by_hand(machine, peak: float, temperature: float) -> tuple[float, float]:
    """Phase resistance and total loss from the design's numbers, step by step."""
    design = machine.design
    copper = load_material(design.winding.conductor)
    assert isinstance(copper, Conductor)
    turns = design.winding.turns_per_coil * design.slots // 3  # coils of a phase in series
    length = turns * phase_winding(machine).turn_length
    diameter = design.winding.wire_diameter_mm * 1e-3
    area = design.winding.parallel_strands * math.pi * diameter**2 / 4
    rho = copper.resistivity * (1.0 + copper.temperature_coefficient * (temperature - 20.0))
    resistance = rho * length / area
    return resistance, 3.0 * (peak / math.sqrt(2.0)) ** 2 * resistance


@pytest.mark.parametrize("temperature", [20.0, 60.0, 100.0, 150.0])
def test_matches_hand_calculation(machine, temperature) -> None:
    resistance, loss = by_hand(machine, 25.0, temperature)
    result = copper_loss(machine, 25.0, temperature)
    assert result.temperature == temperature
    assert result.phase_resistance == pytest.approx(resistance, rel=1e-12)
    assert result.power == pytest.approx(loss, rel=1e-12)
    assert result.power == pytest.approx(1.5 * 25.0**2 * resistance, rel=1e-12)


def test_defaults_to_the_design_temperature(machine) -> None:
    result = copper_loss(machine, 10.0)
    assert result.temperature == machine.design.winding.temperature_C
    assert result.power == pytest.approx(by_hand(machine, 10.0, result.temperature)[1], rel=1e-12)


def test_scales_with_current_squared(machine) -> None:
    assert copper_loss(machine, 30.0).power == pytest.approx(
        9.0 * copper_loss(machine, 10.0).power, rel=1e-12
    )
    assert copper_loss(machine, 0.0).power == 0.0


def test_rises_with_temperature_by_the_resistivity_ratio(machine) -> None:
    copper = load_material(machine.design.winding.conductor)
    alpha = copper.temperature_coefficient
    ratio = (1.0 + alpha * 80.0) / (1.0 + alpha * 20.0)
    hot = copper_loss(machine, 10.0, 100.0).power
    warm = copper_loss(machine, 10.0, 40.0).power
    assert hot / warm == pytest.approx(ratio, rel=1e-12)


def test_rms_current_is_the_peak_over_root_two(machine) -> None:
    assert copper_loss(machine, 10.0).rms_current == pytest.approx(10.0 / math.sqrt(2.0))


@pytest.mark.parametrize("current", [-1.0, math.nan, math.inf])
def test_rejects_a_bad_current(machine, current) -> None:
    with pytest.raises(ValueError, match="current"):
        copper_loss(machine, current)
