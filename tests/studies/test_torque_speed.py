"""Torque-speed curve under a drive's voltage and current limits."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import Magnet, load_material
from polecraft.studies.flux_map import FluxMap, flux_map, operating_point
from polecraft.studies.torque_speed import TorqueSpeed, phase_limits, torque_speed
from polecraft.timing import Timings

POLE_PAIRS = 5


def linear_map(magnet: float, inductance: float, top: float = 20.0) -> FluxMap:
    """A map made by hand: no saturation, Ld = Lq, so the answer is known."""
    d_axis, q_axis = np.linspace(-top, 0.0, 5), np.linspace(0.0, top, 5)
    i_d, i_q = np.meshgrid(d_axis, q_axis, indexing="ij")
    psi_d, psi_q = magnet + inductance * i_d, inductance * i_q
    return FluxMap(
        d_currents=d_axis,
        q_currents=q_axis,
        pole_pairs=POLE_PAIRS,
        points=1,
        d_linkage=psi_d,
        q_linkage=psi_q,
        torque=1.5 * POLE_PAIRS * (psi_d * i_q - psi_q * i_d),
        timings=Timings(),
    )


def linkage_torque(curve: TorqueSpeed) -> np.ndarray:
    return (
        1.5
        * curve.pole_pairs
        * (curve.d_linkage * curve.q_current - curve.q_linkage * curve.d_current)
    )


# A machine with no saturation and no resistance: the textbook answer


MAGNET, INDUCTANCE, VOLTAGE, CURRENT = 0.006, 2e-4, 6.0, 20.0


@pytest.fixture(scope="module")
def textbook() -> TorqueSpeed:
    speeds = np.array([0.0, 50.0, 120.0, 166.0, 170.0, 200.0, 300.0, 450.0, 590.0, 610.0, 800.0])
    return torque_speed(linear_map(MAGNET, INDUCTANCE), speeds, VOLTAGE, CURRENT, 0.0)


def test_flat_torque_up_to_the_base_speed(textbook: TorqueSpeed) -> None:
    base = VOLTAGE / (POLE_PAIRS * math.hypot(MAGNET, INDUCTANCE * CURRENT))
    assert textbook.base_speed == pytest.approx(base, rel=1e-7)
    below = textbook.speeds <= base
    assert below.sum() == 4
    rated = 1.5 * POLE_PAIRS * MAGNET * CURRENT
    assert textbook.base_torque == pytest.approx(rated, rel=1e-9)
    assert np.all(textbook.torque[below] == textbook.base_torque)
    assert np.all(textbook.voltage[below] < VOLTAGE)
    assert textbook.voltage[1] == pytest.approx(
        POLE_PAIRS * 50.0 * math.hypot(MAGNET, INDUCTANCE * CURRENT), rel=1e-7
    )


def test_field_weakening_follows_the_circle_and_ellipse(textbook: TorqueSpeed) -> None:
    # At the current limit, |v| = w |psi| with psi_d = magnet + L i_d, psi_q = L i_q:
    # i_d = (V^2 / w^2 - magnet^2 - L^2 I^2) / (2 L magnet).
    for speed, torque, i_d in zip(textbook.speeds, textbook.torque, textbook.d_current):
        if not textbook.base_speed < speed < 600.0:
            continue
        omega = POLE_PAIRS * speed
        expected_d = (
            VOLTAGE**2 / omega**2 - MAGNET**2 - (INDUCTANCE * CURRENT) ** 2
        ) / (2 * INDUCTANCE * MAGNET)
        expected_q = math.sqrt(CURRENT**2 - expected_d**2)
        assert i_d == pytest.approx(expected_d, rel=1e-6)
        assert torque == pytest.approx(1.5 * POLE_PAIRS * MAGNET * expected_q, rel=1e-7)
    above = textbook.speeds > textbook.base_speed
    reached = above & np.isfinite(textbook.torque)
    assert textbook.voltage[reached] == pytest.approx(np.full(reached.sum(), VOLTAGE), rel=1e-9)
    assert textbook.current[reached] == pytest.approx(np.full(reached.sum(), CURRENT), rel=1e-6)
    assert np.all(np.diff(textbook.torque[reached]) < 0.0)


def test_speeds_out_of_reach_have_no_torque(textbook: TorqueSpeed) -> None:
    # Even all the current along -d leaves magnet - L I of flux to turn.
    top = VOLTAGE / (POLE_PAIRS * (MAGNET - INDUCTANCE * CURRENT))
    assert top == pytest.approx(600.0)
    assert np.isfinite(textbook.torque[textbook.speeds < top]).all()
    out = textbook.speeds > top
    for values in (textbook.torque, textbook.d_current, textbook.voltage, textbook.power):
        assert np.isnan(values[out]).all()
    assert textbook.max_power == pytest.approx(np.nanmax(textbook.torque * textbook.speeds))


def test_beyond_the_characteristic_current_less_current_gives_more_torque() -> None:
    # magnet / L = 10 A, under the 20 A limit: at high speed the best is
    # i_d = -10 A, where the flux along d is zero, and |v| = w L i_q.
    magnet = 0.002
    curve = torque_speed(
        linear_map(magnet, INDUCTANCE), [800.0, 1500.0, 6000.0], VOLTAGE, CURRENT, 0.0
    )
    omega = POLE_PAIRS * curve.speeds
    assert curve.d_current == pytest.approx(np.full(3, -magnet / INDUCTANCE), rel=1e-6)
    assert curve.q_current == pytest.approx(VOLTAGE / (omega * INDUCTANCE), rel=1e-6)
    assert curve.torque == pytest.approx(
        1.5 * POLE_PAIRS * magnet * VOLTAGE / (omega * INDUCTANCE), rel=1e-6
    )
    assert np.all(curve.current < CURRENT)


def test_resistance_lowers_the_base_speed_as_the_voltage_drop_says() -> None:
    resistance = 0.1
    curve = torque_speed(
        linear_map(MAGNET, INDUCTANCE), [0.0, 100.0], VOLTAGE, CURRENT, resistance
    )
    # |v|^2 = (R I + w magnet)^2 + (w L I)^2 = V^2 at the base speed, i_d = 0.
    a = MAGNET**2 + (INDUCTANCE * CURRENT) ** 2
    b = 2 * resistance * CURRENT * MAGNET
    c = (resistance * CURRENT) ** 2 - VOLTAGE**2
    base = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a) / POLE_PAIRS
    assert curve.base_speed == pytest.approx(base, rel=1e-7)
    assert curve.voltage[0] == pytest.approx(resistance * CURRENT, rel=1e-7)
    assert curve.copper_loss == pytest.approx(np.full(2, 1.5 * resistance * CURRENT**2), rel=1e-7)


def test_power_in_is_copper_loss_plus_torque_times_speed() -> None:
    curve = torque_speed(
        linear_map(MAGNET, INDUCTANCE), np.linspace(0.0, 550.0, 12), VOLTAGE, CURRENT, 0.1
    )
    assert curve.power == pytest.approx(curve.torque * curve.speeds)
    assert curve.input_power - curve.copper_loss == pytest.approx(curve.power, rel=1e-9, abs=1e-9)
    assert curve.input_power - curve.copper_loss == pytest.approx(
        linkage_torque(curve) * curve.speeds, rel=1e-9, abs=1e-9
    )


def test_end_inductance_adds_to_the_maps_inductance() -> None:
    speeds = np.array([100.0, 200.0, 400.0])
    with_ends = torque_speed(linear_map(MAGNET, 1.5e-4), speeds, VOLTAGE, CURRENT, 0.05, 0.5e-4)
    whole = torque_speed(linear_map(MAGNET, 2e-4), speeds, VOLTAGE, CURRENT, 0.05)
    assert with_ends.torque == pytest.approx(whole.torque, rel=1e-6)
    assert with_ends.d_linkage == pytest.approx(whole.d_linkage, rel=1e-6)
    assert with_ends.base_speed == pytest.approx(whole.base_speed, rel=1e-9)


# Limits of a drive


@dataclasses.dataclass
class Connection:
    connection: str

    def line_voltage(self, phase: float) -> float:
        return phase * (math.sqrt(3.0) if self.connection == "star" else 1.0)

    def line_current(self, phase: float) -> float:
        return phase * (math.sqrt(3.0) if self.connection == "delta" else 1.0)


def test_phase_limits_of_star_and_delta() -> None:
    assert phase_limits(Connection("star"), 48.0, 30.0) == pytest.approx((48 / math.sqrt(3), 30.0))
    assert phase_limits(Connection("delta"), 48.0, 30.0) == pytest.approx((48.0, 30 / math.sqrt(3)))
    with pytest.raises(ValueError, match="bus voltage"):
        phase_limits(Connection("star"), 0.0, 30.0)
    with pytest.raises(ValueError, match="line current"):
        phase_limits(Connection("star"), 48.0, math.nan)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"voltage": 0.0}, "voltage limit"),
        ({"current": -1.0}, "current limit"),
        ({"resistance": -0.1}, "resistance"),
        ({"end_inductance": math.inf}, "end inductance"),
        ({"speeds": [-1.0]}, "speeds"),
        ({"speeds": []}, "at least one"),
        ({"current": 25.0}, "must cover"),
        ({"resistance": 0.5}, "cannot drive 20 A"),
    ],
)
def test_bad_input_is_refused(options: dict, message: str) -> None:
    arguments = {
        "speeds": [0.0, 100.0],
        "voltage": VOLTAGE,
        "current": CURRENT,
        "resistance": 0.0,
        **options,
    }
    with pytest.raises(ValueError, match=message):
        torque_speed(linear_map(MAGNET, INDUCTANCE), **arguments)


# The 12-slot, 10-pole example motor


MATERIALS = {
    # Placeholders: not the design's real steel or magnet grade.
    "M270-35A": load_material("generic_35A_placeholder"),
    "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
}


@pytest.fixture(scope="module")
def motor() -> dict:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.core.winding import end_winding_inductance, phase_winding
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / "qdd_outer_12s10p.yaml"))
    mesh = build_mesh(machine.build_geometry(), size=1.0)
    winding = phase_winding(machine)
    voltage, current = phase_limits(winding, 24.0, 20.0)
    return {
        "machine": machine,
        "mesh": mesh,
        "map": flux_map(
            machine, mesh, MATERIALS, np.linspace(-20, 0, 5), np.linspace(0, 20, 5), points=12
        ),
        "voltage": voltage,
        "current": current,
        "resistance": winding.resistance(),
        "ends": end_winding_inductance(machine).estimate,
    }


def curve_of(motor: dict, speeds, current: float | None = None) -> TorqueSpeed:
    return torque_speed(
        motor["map"],
        speeds,
        motor["voltage"],
        motor["current"] if current is None else current,
        motor["resistance"],
        motor["ends"],
    )


@pytest.fixture(scope="module")
def example(motor: dict) -> TorqueSpeed:
    return curve_of(motor, np.linspace(0.0, 600.0, 13))


def test_example_is_flat_then_falls(example: TorqueSpeed) -> None:
    below = example.speeds <= example.base_speed
    assert 2 <= below.sum() < len(example.speeds) - 3
    assert np.all(example.torque[below] == example.base_torque)
    assert example.base_torque == pytest.approx(float(np.nanmax(example.torque)))
    reached = ~below & np.isfinite(example.torque)
    assert np.all(np.diff(example.torque[reached]) < 0.0)
    assert example.voltage[reached] == pytest.approx(
        np.full(reached.sum(), example.voltage_limit), rel=1e-9
    )
    assert np.all(example.voltage[below] < example.voltage_limit)
    drawn = example.current[np.isfinite(example.current)]
    assert np.all(drawn <= example.current_limit * (1 + 1e-9))


def test_example_base_speed_matches_the_back_emf_constant(motor: dict) -> None:
    from polecraft.studies.back_emf import back_emf

    constant = back_emf(motor["machine"], motor["mesh"], MATERIALS, 100.0, points=12).constant
    # A small current: the voltage is the back-EMF plus the resistance's drop,
    # the inductance's part of it being under 0.01 %.
    current = 0.2
    curve = curve_of(motor, [0.0], current)
    expected = (curve.voltage_limit - motor["resistance"] * current) / constant
    assert curve.base_speed == pytest.approx(expected, rel=3e-3)


def test_example_power_balance(example: TorqueSpeed) -> None:
    reached = np.isfinite(example.torque)
    gained = (example.input_power - example.copper_loss)[reached]
    assert gained == pytest.approx((linkage_torque(example) * example.speeds)[reached], abs=1e-9)
    # The gap torque and the linkage torque of the map agree to its accuracy.
    assert gained == pytest.approx(example.power[reached], rel=1e-2, abs=1e-9)


def test_example_working_point_holds_when_solved_directly(
    motor: dict, example: TorqueSpeed
) -> None:
    n = int(np.flatnonzero(example.speeds > 1.5 * example.base_speed)[0])
    i_d, i_q = example.d_current[n], example.q_current[n]
    point = operating_point(motor["machine"], motor["mesh"], MATERIALS, i_d, i_q, points=12)
    omega = example.pole_pairs * example.speeds[n]
    v_d = motor["resistance"] * i_d - omega * (point.q_linkage + motor["ends"] * i_q)
    v_q = motor["resistance"] * i_q + omega * (point.d_linkage + motor["ends"] * i_d)
    assert math.hypot(v_d, v_q) == pytest.approx(example.voltage_limit, rel=5e-3)
    assert point.torque == pytest.approx(example.torque[n], rel=5e-3)
