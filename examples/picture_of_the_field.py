"""Solve a motor and save a picture of its field: flux density and field lines.

Run from any folder: python examples/picture_of_the_field.py
Needs the optional Gmsh package (pip install "polecraft[mesh]").

Writes qdd_outer_36s42p_field.png in the current folder. The steel curve and
the magnet are placeholders, not the grades the design names. The command
``polecraft solve`` does the same in one line.
"""

import sys
from pathlib import Path

import numpy as np

from polecraft import load_design, machine_for
from polecraft.core.materials import Magnet, load_material
from polecraft.core.mesh import build_mesh
from polecraft.core.problem import build_problem
from polecraft.core.solver import solve
from polecraft.plot import plot_field

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "qdd_outer_36s42p.yaml"
CURRENT = 10.0  # peak phase current in amperes


def main() -> None:
    design = load_design(DESIGN)
    machine = machine_for(design)
    geometry = machine.build_geometry()
    try:
        mesh = build_mesh(geometry)
    except ImportError as error:
        sys.exit(str(error))  # Gmsh is missing: the message says how to install it
    materials = {
        design.steel.stator: load_material("generic_35A_placeholder"),
        design.magnet.material: Magnet(remanence=1.2, relative_permeability=1.05),
    }
    problem = build_problem(machine, mesh, materials, current=CURRENT)
    solution = solve(problem)

    # Flux density as colour, 30 field lines over it, and the outline of
    # every region. The top of the colour scale is set, so that pictures at
    # different currents can be compared.
    picture = plot_field(
        problem,
        solution,
        f"{DESIGN.stem}_field.png",
        title=f"{DESIGN.name}: flux density at {CURRENT:g} A, placeholder steel and magnet",
        lines=30,
        largest=2.2,
        outlines=geometry,
    )
    flux_density = np.linalg.norm(solution.flux_density, axis=1)
    print(f"largest flux density: {flux_density.max():.2f} T, with placeholder steel and magnet")
    print(f"picture: {picture}")


if __name__ == "__main__":
    main()
