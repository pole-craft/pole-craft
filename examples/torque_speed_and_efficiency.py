"""Torque-speed curve and efficiency map of a motor on a 24 V drive.

Run from any folder: python examples/torque_speed_and_efficiency.py
Needs the optional Gmsh package (pip install "polecraft[mesh]").

The 12-slot, 10-pole example motor is solved once over a grid of d and q
current (a flux map) with the library's M270-35A steel and its core-loss
data. The magnet is a placeholder, not a datasheet grade, so the numbers
show how the studies work, not the performance of this design. The curve
and the map are then read from the flux map without solving the field
again: the most torque at each speed within the drive's voltage and
current limits, and the copper and core loss and the efficiency at the
current that loses least.

The drive's limits are set at the top of the script: a 24 V DC bus and
20 A in each terminal. The end turns add inductance that the 2D field does
not hold; it is estimated and added.
"""

import math
import sys
from pathlib import Path

import numpy as np

from polecraft import load_design, machine_for
from polecraft.core.materials import load_core_loss, load_material
from polecraft.core.mesh import build_mesh
from polecraft.core.winding import end_winding_inductance, phase_winding
from polecraft.studies.efficiency import efficiency_map
from polecraft.studies.flux_map import flux_map
from polecraft.studies.torque_speed import phase_limits, torque_speed

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "qdd_outer_12s10p.yaml"
MAGNET = "generic_NdFeB_placeholder"
BUS_VOLTAGE = 24.0  # V, DC
LINE_CURRENT = 20.0  # A, peak in each terminal
CURVE_RPM = (0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0)
MAP_RPM = (600.0, 1200.0, 1800.0, 2400.0, 3000.0)  # 50 Hz or more: inside the loss data
TORQUES = (0.1, 0.3, 0.5, 0.7)  # N m at the shaft


def main() -> None:
    design = load_design(DESIGN)
    machine = machine_for(design)
    try:
        mesh = build_mesh(machine.build_geometry())
    except ImportError as error:
        sys.exit(str(error))  # Gmsh is missing: the message says how to install it
    steel = design.steel.stator
    materials = {steel: load_material("M270-35A"), design.magnet.material: load_material(MAGNET)}
    winding = phase_winding(machine)
    voltage, current = phase_limits(winding, BUS_VOLTAGE, LINE_CURRENT)
    resistance = winding.resistance()
    end_inductance = end_winding_inductance(machine).estimate

    print(f"{design.slots} slots, {design.poles} poles, M270-35A steel, placeholder magnet {MAGNET}")
    print(f"Drive: {BUS_VOLTAGE:.0f} V bus, {LINE_CURRENT:.0f} A per terminal; "
          f"phase limits {voltage:.2f} V and {current:.1f} A peak")
    print(f"Phase resistance {1e3 * resistance:.1f} mOhm at {design.winding.temperature_C:.0f} C, "
          f"end-turn inductance {1e6 * end_inductance:.1f} uH")

    fmap = flux_map(
        machine, mesh, materials,
        np.linspace(-current, 0.0, 5), np.linspace(0.0, current, 5),
        points=12, losses={steel: load_core_loss("M270-35A")},
    )
    to_rad = math.pi / 30.0  # rpm to rad/s
    curve = torque_speed(fmap, np.array(CURVE_RPM) * to_rad, voltage, current, resistance, end_inductance)
    print(f"\nMost torque at each speed; flat up to {curve.base_speed / to_rad:.0f} rpm "
          f"at {curve.base_torque:.3f} N m:")
    print("   rpm   torque   current   power")
    print("            N m         A       W")
    for rpm, torque, size, power in zip(CURVE_RPM, curve.torque, curve.current, curve.power):
        print(f"{rpm:6.0f} {torque:8.3f} {size:9.2f} {power:7.1f}")

    result = efficiency_map(fmap, np.array(MAP_RPM) * to_rad, TORQUES, voltage, current, resistance, end_inductance)
    print("\nEfficiency in percent (- where the drive cannot reach the point):")
    print("   rpm  " + "".join(f"{torque:7.1f} N m" for torque in TORQUES))
    for row, rpm in enumerate(MAP_RPM):
        cells = "".join(
            f"{100.0 * result.efficiency[row, col]:10.1f}" if result.reached[row, col]
            else f"{'-':>10}"
            for col in range(len(TORQUES))
        )
        print(f"{rpm:6.0f}  {cells}")

    row, col = np.unravel_index(np.nanargmax(result.efficiency), result.efficiency.shape)
    print(f"\nBest point: {100.0 * result.efficiency[row, col]:.1f} % at {MAP_RPM[row]:.0f} rpm "
          f"and {TORQUES[col]:.1f} N m, with {result.copper_loss[row, col]:.1f} W of copper loss "
          f"and {result.core_loss[row, col]:.1f} W of core loss.")
    print("Above the base speed the drive spends current on field weakening, so the torque falls")
    print("and the copper loss no longer follows the torque alone. Not counted: friction, windage,")
    print("magnet and winding eddy currents, and the drive's own loss.")


if __name__ == "__main__":
    main()
