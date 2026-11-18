"""Solve the magnetic field of a motor at one rotor position and read off the results.

Run from any folder: python examples/solve_a_motor.py
Needs the optional Gmsh package (pip install "polecraft[mesh]").

The steel curve and the magnet used here are placeholders, not the grades the
design names, so the numbers show how the tool behaves, not how this motor
performs.
"""

import sys
from pathlib import Path

import numpy as np

from polecraft import load_design, machine_for
from polecraft.core.materials import Magnet, load_material
from polecraft.core.mesh import build_mesh
from polecraft.core.post import flux_linkage, torque
from polecraft.core.problem import build_problem
from polecraft.core.solver import solve

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "qdd_outer_36s42p.yaml"
CURRENT = 10.0  # peak phase current in amperes


def main() -> None:
    design = load_design(DESIGN)
    machine = machine_for(design)
    print(f"{design.slots} slots, {design.poles} poles, placeholder steel and magnet")

    # Only the slice of the motor that repeats is meshed and solved.
    geometry = machine.build_geometry()
    try:
        mesh = build_mesh(geometry)
    except ImportError as error:
        sys.exit(str(error))  # Gmsh is missing: the message says how to install it
    print(f"mesh: {len(mesh.nodes)} nodes, {len(mesh.triangles)} triangles")

    # A model for every grade the design names. These two are stand-ins: a
    # steel curve worked out from a formula, and a magnet of 1.2 T.
    materials = {
        design.steel.stator: load_material("generic_35A_placeholder"),
        design.magnet.material: Magnet(remanence=1.2, relative_permeability=1.05),
    }

    # The problem at rotor angle 0, with current in phase with the back-EMF.
    problem = build_problem(machine, mesh, materials, current=CURRENT)
    solution = solve(problem)
    print(f"\ncurrent: {CURRENT:g} A peak")
    print(solution.report())

    # Results are for the whole machine, not the slice.
    print(f"\ntorque: {torque(problem, solution):.3f} N m")
    for phase, linked in flux_linkage(problem, solution).items():
        print(f"flux linkage of phase {phase}: {linked * 1e3:+.3f} mWb")

    # Flux density is one value per triangle; the steel is the saturating material.
    b = np.linalg.norm(solution.flux_density, axis=1)
    steel = np.array([not material.linear for material in problem.materials])
    in_steel = steel[problem.triangle_material]
    print(f"flux density in the steel: {np.percentile(b[in_steel], 99):.2f} T at the 99th percentile")

    print()
    print(solution.timing_report())


if __name__ == "__main__":
    main()
