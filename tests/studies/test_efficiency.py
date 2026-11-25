"""Efficiency map: least-loss current at each speed and shaft torque."""

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import LossRangeWarning, Magnet, load_core_loss, load_material
from polecraft.studies.efficiency import EfficiencyMap, efficiency_map
from polecraft.studies.flux_map import FluxMap, current_and_advance, flux_map
from polecraft.studies.torque_speed import phase_limits, torque_speed
from polecraft.timing import Timings

POLE_PAIRS = 5
MAGNET, INDUCTANCE, RESISTANCE = 0.006, 2e-4, 0.1
VOLTAGE, CURRENT = 6.0, 20.0
# Core loss at 1 mechanical rad/s, W: hysteresis, eddy current, excess.
H, E, X = 0.002, 1e-5, 2e-4


def linear_map(core: str | None = "constant") -> FluxMap:
    """A map made by hand: no saturation, Ld = Lq, so the answer is known.

    ``core`` is None for no core loss, "constant" for the same core loss at
    every current, or "d" for a core loss that rises with d current, which
    makes field weakening cost more than copper.
    """
    d_axis, q_axis = np.linspace(-CURRENT, 0.0, 9), np.linspace(0.0, CURRENT, 9)
    i_d, i_q = np.meshgrid(d_axis, q_axis, indexing="ij")
    psi_d, psi_q = MAGNET + INDUCTANCE * i_d, INDUCTANCE * i_q
    parts = {}
    if core is not None:
        rise = 1.0 + 2.0 * (i_d / CURRENT) ** 2 if core == "d" else np.ones_like(i_d)
        parts = {"hysteresis": H * rise, "eddy": E * rise, "excess": X * rise}
    return FluxMap(
        d_currents=d_axis,
        q_currents=q_axis,
        pole_pairs=POLE_PAIRS,
        points=1,
        d_linkage=psi_d,
        q_linkage=psi_q,
        torque=1.5 * POLE_PAIRS * (psi_d * i_q - psi_q * i_d),
        timings=Timings(),
        **parts,
    )


SPEEDS = np.linspace(0.0, 500.0, 26)
TORQUES = np.linspace(0.0, 0.9, 19)


def run(fmap: FluxMap, speeds=SPEEDS, torques=TORQUES, **options) -> EfficiencyMap:
    arguments = {"voltage": VOLTAGE, "current": CURRENT, "resistance": RESISTANCE, **options}
    return efficiency_map(fmap, speeds, torques, **arguments)


@pytest.fixture(scope="module")
def constant() -> EfficiencyMap:
    return run(linear_map("constant"))


def check_balance_and_bounds(result: EfficiencyMap) -> None:
    reached = result.reached
    assert reached.any()
    # Power in = power out + losses, at every point.
    balance = result.input_power - result.output_power - result.losses
    assert np.all(np.abs(balance[reached]) <= 1e-9 * np.maximum(result.input_power[reached], 1.0))
    efficiency = result.efficiency[reached]
    assert np.all((efficiency >= 0.0) & (efficiency <= 1.0))
    assert np.all(result.losses[reached] >= 0.0)
    # The torque asked for, within the limits.
    asked = np.broadcast_to(result.torques, result.torque.shape)
    assert result.torque[reached] == pytest.approx(asked[reached], abs=1e-9)
    assert np.all(result.current[reached] <= result.current_limit * (1 + 1e-9))
    assert np.all(result.voltage[reached] <= result.voltage_limit * (1 + 1e-6))
    for values in (result.efficiency, result.input_power, result.d_current, result.core_loss):
        assert np.isnan(values[~reached]).all()


# A machine made by hand


@pytest.mark.parametrize("core", [None, "constant", "d"])
def test_energy_balance_and_efficiency_between_0_and_1(core) -> None:
    check_balance_and_bounds(run(linear_map(core), end_inductance=3e-5))


def test_low_speed_takes_q_current_alone(constant: EfficiencyMap) -> None:
    # Torque does not depend on d current here, so the least current is all
    # along q, enough for the torque asked for and the drag of the core loss.
    for n in (1, 4, 6):
        speed = SPEEDS[n]
        drag = H + E * speed + X * math.sqrt(speed)
        current = (TORQUES + drag) / (1.5 * POLE_PAIRS * MAGNET)
        within = current <= CURRENT
        assert not within.all() and not constant.reached[n][~within].any()
        # The loss is flat about i_d = 0, so the search ends within about
        # 1e-8 rad of it, which costs about 1e-16 of the loss.
        assert constant.d_current[n][within] == pytest.approx(0.0, abs=1e-6)
        assert constant.q_current[n][within] == pytest.approx(current[within], rel=1e-9)
        copper = 1.5 * RESISTANCE * current**2
        core = H * speed + E * speed**2 + X * speed**1.5
        assert constant.core_loss[n][within] == pytest.approx(core, rel=1e-12)
        output = TORQUES * speed
        expected = output / (output + copper + core)
        assert constant.efficiency[n][within] == pytest.approx(expected[within], rel=1e-9)


def test_without_core_loss_the_efficiency_is_the_copper_alone() -> None:
    result = run(linear_map(None), speeds=[100.0], torques=[0.3, 0.6])
    current = np.array([0.3, 0.6]) / (1.5 * POLE_PAIRS * MAGNET)
    assert np.all(result.core_loss == 0.0)
    output = np.array([30.0, 60.0])
    expected = output / (output + 1.5 * RESISTANCE * current**2)
    assert result.efficiency[0] == pytest.approx(expected, rel=1e-9)


def test_standstill_and_zero_torque_give_zero_efficiency(constant: EfficiencyMap) -> None:
    assert np.all(constant.efficiency[0][constant.reached[0]] == 0.0)
    assert np.all(constant.efficiency[:, 0][constant.reached[:, 0]] == 0.0)
    # At standstill the copper still carries the current for the torque.
    assert np.all(constant.copper_loss[0, 1:][constant.reached[0, 1:]] > 0.0)
    assert np.all(constant.core_loss[0][constant.reached[0]] == 0.0)


def test_reach_matches_the_torque_speed_curve() -> None:
    fmap = linear_map(None)
    result = run(fmap)
    curve = torque_speed(fmap, SPEEDS, VOLTAGE, CURRENT, RESISTANCE)
    for n in range(len(SPEEDS)):
        top = curve.torque[n]
        if not np.isfinite(top):
            assert not result.reached[n].any()
            continue
        # Zero torque past the no-load speed is out of reach without core loss.
        assert result.reached[n][(TORQUES > 0.0) & (TORQUES <= 0.995 * top)].all()
        assert not result.reached[n][TORQUES > 1.001 * top].any()


def least_loss(fmap: FluxMap, speed: float, torque: float, end_inductance: float = 0.0) -> float:
    """The least loss for a torque, by brute force.

    4001 angles over the quarter turn, then 4001 within a step either side
    of the best, each solved for the current by bisection. Fine steps
    matter: on the voltage limit the loss changes fast with the angle.
    """
    angles = np.linspace(0.0, 0.5 * math.pi, 4001)
    loss, best = _least_loss_at(fmap, speed, torque, end_inductance, angles)
    if not math.isfinite(loss):
        return loss
    step = angles[1] - angles[0]
    fine = np.linspace(max(best - step, 0.0), min(best + step, 0.5 * math.pi), 4001)
    return min(loss, _least_loss_at(fmap, speed, torque, end_inductance, fine)[0])


def _least_loss_at(fmap, speed, torque, end_inductance, angle) -> tuple[float, float]:
    def shaft(size: np.ndarray) -> tuple:
        i_d, i_q = -size * np.sin(angle), size * np.cos(angle)
        psi_d, psi_q, _ = fmap.at(i_d, i_q)
        h, e, x = fmap.core_loss(i_d, i_q, speed) if fmap.has_core_loss else (0.0, 0.0, 0.0)
        core = h + e + x
        drag = core / speed
        torque_em = 1.5 * fmap.pole_pairs * (psi_d * i_q - psi_q * i_d)
        psi_d, psi_q = psi_d + end_inductance * i_d, psi_q + end_inductance * i_q
        w = fmap.pole_pairs * speed
        volts = np.hypot(RESISTANCE * i_d - w * psi_q, RESISTANCE * i_q + w * psi_d)
        return torque_em - drag, 1.5 * RESISTANCE * size**2 + core, volts

    low, high = np.zeros_like(angle), np.full_like(angle, CURRENT)
    reachable = shaft(high)[0] >= torque
    for _ in range(60):
        middle = 0.5 * (low + high)
        above = shaft(middle)[0] >= torque
        high = np.where(above, middle, high)
        low = np.where(above, low, middle)
    _, loss, volts = shaft(high)
    fits = reachable & (volts <= VOLTAGE)
    if not fits.any():
        return math.inf, math.nan
    k = int(np.argmin(np.where(fits, loss, np.inf)))
    return float(loss[k]), float(angle[k])


@pytest.mark.parametrize(
    "speed, torque",
    [(60.0, 0.5), (180.0, 0.6), (240.0, 0.3), (300.0, 0.25), (400.0, 0.1), (450.0, 0.05)],
)
def test_chosen_current_loses_least(speed, torque) -> None:
    # Core loss that rises with d current: at speed a trade between copper,
    # core loss and the voltage limit.
    fmap = linear_map("d")
    result = run(fmap, speeds=[speed], torques=[torque], end_inductance=3e-5)
    best = least_loss(fmap, speed, torque, 3e-5)
    assert math.isfinite(best) == bool(result.reached[0, 0])
    if math.isfinite(best):
        # The brute force is itself a fine grid, so a little above the least.
        assert float(result.losses[0, 0]) <= best * (1 + 1e-7)
        assert float(result.losses[0, 0]) == pytest.approx(best, rel=2e-5)


def test_field_weakening_is_used_where_the_voltage_needs_it(constant: EfficiencyMap) -> None:
    reached = constant.reached
    weakened = reached & (constant.d_current < -1e-6)
    assert weakened.any()
    # Negative d current costs copper and gains nothing but voltage, so it
    # is only drawn with the voltage at its limit.
    assert constant.voltage[weakened] == pytest.approx(
        np.full(weakened.sum(), VOLTAGE), rel=2e-3
    )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"voltage": 0.0}, "voltage limit"),
        ({"current": math.inf}, "current limit"),
        ({"resistance": -1.0}, "resistance"),
        ({"end_inductance": -1e-6}, "end inductance"),
        ({"speeds": [-1.0]}, "speeds"),
        ({"torques": [math.nan]}, "torques"),
        ({"torques": []}, "at least one"),
        ({"current": 25.0}, "must cover"),
    ],
)
def test_bad_input_is_refused(options, message) -> None:
    with pytest.raises(ValueError, match=message):
        run(linear_map(), **options)


def test_warns_outside_the_fitted_frequencies() -> None:
    fmap = linear_map()
    object.__setattr__(fmap, "fitted_frequencies", (50.0, 1000.0))
    # 5 pole pairs: 100 rad/s is 79.6 Hz, 20 rad/s 15.9 Hz, 2000 rad/s 1592 Hz.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(fmap, speeds=[0.0, 100.0, 1000.0], torques=[0.1])
    with pytest.warns(LossRangeWarning, match="15.9155 to 1591.55 Hz"):
        run(fmap, speeds=[20.0, 100.0, 2000.0], torques=[0.1])


# The 12-slot, 10-pole example motor


MATERIALS = {
    # Placeholders: not the design's real steel or magnet grade.
    "M270-35A": load_material("generic_35A_placeholder"),
    "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
}
LOSSES = {"M270-35A": load_core_loss("M270-35A")}


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
    fmap = flux_map(
        machine,
        mesh,
        MATERIALS,
        np.linspace(-20, 0, 5),
        np.linspace(0, 20, 5),
        points=12,
        losses=LOSSES,
    )
    speeds, torques = np.linspace(70.0, 600.0, 12), np.linspace(0.0, 1.0, 11)
    return {
        "machine": machine,
        "mesh": mesh,
        "map": fmap,
        "result": efficiency_map(
            fmap,
            speeds,
            torques,
            voltage,
            current,
            winding.resistance(),
            end_winding_inductance(machine).estimate,
        ),
    }


def test_example_balance_and_bounds(motor: dict) -> None:
    result = motor["result"]
    check_balance_and_bounds(result)
    assert 0.8 < np.nanmax(result.efficiency) < 0.95
    # A 24 V bus reaches less torque at the top speed than at the lowest.
    assert result.reached[-1].sum() < result.reached[0].sum()


def test_example_core_loss_matches_the_core_loss_study(motor: dict) -> None:
    from polecraft.studies.core_loss import core_loss

    result = motor["result"]
    n, k = 3, 6  # a point between the grid's currents
    i_d, i_q = float(result.d_current[n, k]), float(result.q_current[n, k])
    size, advance = current_and_advance(i_d, i_q)
    speed = float(result.speeds[n])
    study = core_loss(
        motor["machine"], motor["mesh"], MATERIALS, LOSSES, speed, size, advance, points=12
    )
    assert float(result.core_loss[n, k]) == pytest.approx(study.total, rel=0.01)
    assert float(result.hysteresis_loss[n, k]) == pytest.approx(study.hysteresis, rel=0.01)
    assert float(result.eddy_loss[n, k]) == pytest.approx(study.eddy, rel=0.01)


def test_example_loss_rises_with_torque_and_core_loss_with_speed(motor: dict) -> None:
    result = motor["result"]
    for row in range(len(result.speeds)):
        reached = result.reached[row]
        assert np.all(np.diff(result.losses[row][reached]) > 0.0)
    # Below the voltage limit; with field weakening the flux in the steel,
    # and with it the core loss, can fall as the speed rises.
    column = result.reached[:, 2] & (result.voltage[:, 2] < 0.999 * result.voltage_limit)
    assert column.sum() >= 4
    assert np.all(np.diff(result.core_loss[column, 2]) > 0.0)
