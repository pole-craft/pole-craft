"""Cogging torque, back-EMF and loaded torque of a motor as its rotor turns.

Run from any folder: python examples/motor_over_a_cycle.py
Needs the optional Gmsh package (pip install "polecraft[mesh]").

Three studies on the 12-slot, 10-pole example motor, all from one mesh. Each
turns the rotor without meshing again and solves the field at a series of
angles, every solve starting from the field at the angle before. The steel
curve and the magnet are placeholders, so the numbers show the behaviour,
not the performance of this design.
"""

import math
import sys
from pathlib import Path

import numpy as np

from polecraft import load_design, machine_for
from polecraft.core.materials import Magnet, load_material
from polecraft.core.mesh import build_mesh
from polecraft.studies.back_emf import back_emf
from polecraft.studies.cogging import cogging_torque
from polecraft.studies.loaded_torque import loaded_torque

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "qdd_outer_12s10p.yaml"
SPEED_RPM = 1000.0
CURRENT = 5.0  # peak phase current in amperes


def main() -> None:
    design = load_design(DESIGN)
    machine = machine_for(design)
    try:
        mesh = build_mesh(machine.build_geometry())
    except ImportError as error:
        sys.exit(str(error))  # Gmsh is missing: the message says how to install it
    materials = {
        design.steel.stator: load_material("generic_35A_placeholder"),
        design.magnet.material: Magnet(remanence=1.2, relative_permeability=1.05),
    }
    print(f"{design.slots} slots, {design.poles} poles, placeholder steel and magnet")

    cogging = cogging_torque(machine, mesh, materials, points=12)
    print(f"\nCogging torque with no current. One period is {math.degrees(cogging.period):.1f} degrees "
          f"of rotor angle, 360 / LCM({design.slots}, {design.poles}):")
    print("  angle   torque")
    print("   deg     mN m")
    for angle, value in zip(cogging.angles, cogging.torque):
        print(f"{math.degrees(angle):7.2f}  {value * 1e3:7.3f}")
    print(f"peak to peak {cogging.peak_to_peak * 1e3:.2f} mN m, mean {cogging.mean * 1e3:.3f} mN m")

    speed = SPEED_RPM * math.pi / 30.0  # rad/s
    emf = back_emf(machine, mesh, materials, speed, points=12)
    print(f"\nBack-EMF at {SPEED_RPM:.0f} rpm, over one electrical cycle "
          f"({360.0 / emf.pole_pairs:.0f} degrees of rotor angle):")
    print("  angle      A       B       C")
    print("   deg       V       V       V")
    for k, angle in enumerate(emf.angles):
        a, b, c = (emf.emf[phase][k] for phase in "ABC")
        print(f"{math.degrees(angle):7.1f}  {a:7.3f} {b:7.3f} {c:7.3f}")
    fundamental = emf.amplitude["A"]
    waves = np.abs(np.fft.rfft(emf.emf["A"]))
    print(f"phase voltage {fundamental:.3f} V peak at {emf.frequency:.1f} Hz; "
          f"back-EMF constant {emf.constant:.5f} V s/rad")
    print(f"third harmonic {100.0 * waves[3] / waves[1]:.1f} % of the fundamental, "
          "in step in all three phases")

    loaded = loaded_torque(machine, mesh, materials, CURRENT, points=24)
    print(f"\nTorque with {CURRENT:.0f} A peak in the phases, in step with the back-EMF, over one cycle:")
    print(f"mean {loaded.mean:.4f} N m, ripple {100.0 * loaded.ripple:.2f} % peak to peak")
    print(f"torque constant {loaded.constant:.5f} N m/A; "
          f"1.5 x back-EMF constant {1.5 * emf.constant:.5f} N m/A")
    print(f"mean torque from the flux linkages instead of the air-gap stress: "
          f"{loaded.mean_from_linkage:.4f} N m")

    print("\nCogging torque averages to zero over its period. The third harmonic is the same")
    print("in all three phases, so it is missing from the voltage between two phases. Torque")
    print("per ampere is 1.5 times the back-EMF constant, as power balance requires.")


if __name__ == "__main__":
    main()
