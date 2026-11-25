"""Torque and back-EMF of a motor as its magnets warm up.

Run from any folder: python examples/magnet_temperature.py
Needs the optional Gmsh package (pip install "polecraft[mesh]").

The 12-slot, 10-pole example motor is solved with the library's M270-35A
steel and its placeholder NdFeB magnet at three magnet temperatures. The
magnet is a placeholder, not a datasheet grade, so the numbers show how the
remanence temperature coefficient carries through to the motor, not the
performance of this design. The same loop works for any other magnet entry
in the library.
"""

import math
import sys
from pathlib import Path

from polecraft import load_design, machine_for
from polecraft.core.materials import load_material
from polecraft.core.mesh import build_mesh
from polecraft.studies.back_emf import back_emf
from polecraft.studies.loaded_torque import loaded_torque

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "qdd_outer_12s10p.yaml"
MAGNET = "generic_NdFeB_placeholder"
TEMPERATURES = (20.0, 60.0, 100.0)  # degrees Celsius
CURRENT = 5.0  # peak phase current in amperes
SPEED_RPM = 1000.0


def main() -> None:
    design = load_design(DESIGN)
    machine = machine_for(design)
    try:
        mesh = build_mesh(machine.build_geometry())
    except ImportError as error:
        sys.exit(str(error))  # Gmsh is missing: the message says how to install it
    steel = load_material("M270-35A")
    magnet = load_material(MAGNET)
    speed = SPEED_RPM * math.pi / 30.0  # rad/s
    print(f"{design.slots} slots, {design.poles} poles, M270-35A steel, placeholder magnet {MAGNET}")
    print(f"remanence coefficient {100.0 * magnet.remanence_coefficient:+.3f} % per kelvin, "
          "recoil permeability constant")
    print(f"\nTorque at {CURRENT:.0f} A peak and back-EMF at {SPEED_RPM:.0f} rpm:")
    print("  temp  remanence   torque  back-EMF")
    print("     C          T      N m         V")

    results = []
    for temperature in TEMPERATURES:
        warm = magnet.at_temperature(temperature)
        materials = {design.steel.stator: steel, design.magnet.material: warm}
        emf = back_emf(machine, mesh, materials, speed, points=12)
        loaded = loaded_torque(machine, mesh, materials, CURRENT, points=12)
        results.append((temperature, warm.remanence, loaded.mean, emf.amplitude["A"]))
        print(f"{temperature:6.0f} {warm.remanence:10.4f} {loaded.mean:8.4f} {emf.amplitude['A']:9.3f}")

    cold, hot = results[0], results[-1]
    print(f"\nFrom {cold[0]:.0f} C to {hot[0]:.0f} C the remanence changes by "
          f"{100.0 * (hot[1] / cold[1] - 1.0):+.2f} %, the torque by "
          f"{100.0 * (hot[2] / cold[2] - 1.0):+.2f} % and the back-EMF by "
          f"{100.0 * (hot[3] / cold[3] - 1.0):+.2f} %.")
    print("Torque and back-EMF change by somewhat less than the remanence does; how much")
    print("less depends on the design and on how far its steel saturates. The magnet's straight")
    print("line holds only up to its grade's working temperature, and nothing here checks for")
    print("demagnetisation.")


if __name__ == "__main__":
    main()
