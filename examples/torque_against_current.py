"""Torque of a motor at several currents, to see the steel saturate.

Run from any folder: python examples/torque_against_current.py
Needs the optional Gmsh package (pip install "polecraft[mesh]").

The mesh is made once. Each solve starts from the field of the one before,
which takes fewer iterations than starting from nothing. The steel curve and
the magnet are placeholders, so the numbers show the behaviour, not the
performance of this design.
"""

import sys
from pathlib import Path

from polecraft import load_design, machine_for
from polecraft.core.materials import Magnet, load_material
from polecraft.core.mesh import build_mesh
from polecraft.core.post import torque
from polecraft.core.problem import build_problem
from polecraft.core.solver import solve

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "qdd_outer_36s42p.yaml"
CURRENTS = [0.0, 5.0, 10.0, 20.0, 40.0]  # peak phase current in amperes


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
    print("current   torque   torque per ampere   iterations")
    print("   A       N m          N m/A")
    start = None
    for current in CURRENTS:
        problem = build_problem(machine, mesh, materials, current=current)
        solution = solve(problem, start=start)
        start = solution.potential  # the next solve begins from this field
        newton_metres = torque(problem, solution)
        per_ampere = f"{newton_metres / current:8.4f}" if current else "       -"
        print(f"{current:6.1f}  {newton_metres:8.4f}       {per_ampere}        {solution.iterations:5d}")

    print("\nWith no current the torque is the cogging torque, which is zero at this")
    print("rotor position. Torque per ampere falls as the current rises: the steel")
    print("saturates.")


if __name__ == "__main__":
    main()
