"""Stall heating of the worst phase, adiabatic, with the resistance rising as it warms."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.winding import phase_winding
from polecraft.design import load_design
from polecraft.machines import machine_for
from polecraft.studies.mechanical import copper_mass
from polecraft.studies.stall_heating import stall_current, stall_heating


@pytest.fixture(scope="module")
def winding():
    root = Path(__file__).resolve().parents[2]
    return phase_winding(machine_for(load_design(root / "designs" / "qdd_outer_36s42p.yaml")))


def test_start_matches_i_squared_r_over_heat_capacity(winding) -> None:
    hold = stall_heating(winding, 20.0, 40.0, 155.0)
    capacity = copper_mass(winding) / 3.0 * 384.5
    assert hold.heat_capacity == pytest.approx(capacity)
    assert hold.loss == pytest.approx(20.0**2 * winding.resistance(40.0))
    assert hold.total_loss == pytest.approx(winding.copper_loss(20.0 / math.sqrt(2.0), 40.0))
    assert hold.heating_rate == pytest.approx(hold.loss / capacity)


def test_time_to_limit_matches_a_step_by_step_integration(winding) -> None:
    hold = stall_heating(winding, 20.0, 40.0, 155.0)
    # Integrate m c dT/dt = I^2 R(T) in small steps (classic Runge-Kutta).
    capacity, current = hold.heat_capacity, hold.current

    def rate(t: float) -> float:
        return current**2 * winding.resistance(t) / capacity

    temperature, time, step = 40.0, 0.0, hold.time_to_limit / 20000
    while time < hold.time_to_limit - 0.5 * step:
        k1 = rate(temperature)
        k2 = rate(temperature + 0.5 * step * k1)
        k3 = rate(temperature + 0.5 * step * k2)
        k4 = rate(temperature + step * k3)
        temperature += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        time += step
    assert temperature == pytest.approx(155.0, rel=1e-8)
    assert hold.temperature_at(hold.time_to_limit) == pytest.approx(155.0, rel=1e-12)
    assert hold.temperature_at(0.0) == pytest.approx(40.0, rel=1e-12)


def test_heating_speeds_up_as_the_resistance_rises(winding) -> None:
    hold = stall_heating(winding, 20.0, 40.0, 155.0)
    # With the resistance held at its starting value it would take longer.
    linear = (155.0 - 40.0) / hold.heating_rate
    assert hold.time_to_limit < linear
    temperatures = [hold.temperature_at(t) for t in np.linspace(0.0, hold.time_to_limit, 5)]
    assert np.all(np.diff(np.diff(temperatures)) > 0.0)


def test_time_goes_as_one_over_current_squared(winding) -> None:
    one = stall_heating(winding, 10.0, 40.0, 155.0).time_to_limit
    two = stall_heating(winding, 20.0, 40.0, 155.0).time_to_limit
    assert one == pytest.approx(4.0 * two)
    assert stall_heating(winding, 0.0, 40.0, 155.0).time_to_limit == math.inf


def test_more_copper_holds_longer(winding) -> None:
    thick = dataclasses.replace(winding, wire_diameter=winding.wire_diameter * math.sqrt(2.0))
    # Twice the copper area: half the resistance and twice the heat capacity.
    assert stall_heating(thick, 20.0, 40.0, 155.0).time_to_limit == pytest.approx(
        4.0 * stall_heating(winding, 20.0, 40.0, 155.0).time_to_limit
    )


def test_stall_current_is_the_inverse(winding) -> None:
    current = stall_current(winding, 5.0, 40.0, 155.0)
    assert stall_heating(winding, current, 40.0, 155.0).time_to_limit == pytest.approx(5.0)


def test_torque_held(winding) -> None:
    held = stall_heating(winding, 20.0, 40.0, 155.0, torque_constant=0.1)
    assert held.torque == pytest.approx(2.0)
    assert stall_heating(winding, 20.0, 40.0, 155.0).torque is None


def test_specific_heat_can_be_given(winding) -> None:
    given = stall_heating(winding, 20.0, 40.0, 155.0, specific_heat=769.0)
    assert given.time_to_limit == pytest.approx(
        2.0 * stall_heating(winding, 20.0, 40.0, 155.0).time_to_limit
    )
    bare = dataclasses.replace(
        winding, conductor=dataclasses.replace(winding.conductor, specific_heat=None)
    )
    with pytest.raises(ValueError, match="no specific heat"):
        stall_heating(bare, 20.0, 40.0, 155.0)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ((-1.0, 40.0, 155.0), "current"),
        ((20.0, 155.0, 40.0), "must be above the start"),
        ((20.0, math.nan, 155.0), "start temperature"),
    ],
)
def test_bad_input_is_refused(winding, arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        stall_heating(winding, *arguments)
    with pytest.raises(ValueError, match="seconds"):
        stall_current(winding, 0.0, 40.0, 155.0)
