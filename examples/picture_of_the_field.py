"""Solve a motor and save a picture of its field: flux density and field lines.

Run from any folder: python examples/picture_of_the_field.py
Needs the optional Gmsh package (pip install "polecraft[mesh]").

Writes qdd_outer_36s42p_field.png in the current folder. The steel curve and
the magnet are placeholders, not the grades the design names.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file; no window is needed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

from polecraft import load_design, machine_for
from polecraft.core.materials import Magnet, load_material
from polecraft.core.mesh import build_mesh
from polecraft.core.problem import build_problem
from polecraft.core.solver import solve

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "qdd_outer_36s42p.yaml"
CURRENT = 10.0  # peak phase current in amperes


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
    problem = build_problem(machine, mesh, materials, current=CURRENT)
    solution = solve(problem)

    # Positions in millimetres for the picture; the solver works in metres.
    x, y = 1e3 * mesh.nodes[:, 0], 1e3 * mesh.nodes[:, 1]
    surface = Triangulation(x, y, mesh.triangles)
    flux_density = np.linalg.norm(solution.flux_density, axis=1)

    figure, axes = plt.subplots(figsize=(9, 6))
    # One colour per triangle: the flux density is constant over each.
    shading = axes.tripcolor(surface, facecolors=flux_density, cmap="viridis", vmin=0.0, vmax=2.2)
    # Lines of equal potential are the field lines.
    axes.tricontour(
        surface, solution.potential, levels=25, colors="white", linewidths=0.4, linestyles="solid"
    )
    axes.set_aspect("equal")
    axes.set_xlabel("x (mm)")
    axes.set_ylabel("y (mm)")
    axes.set_title(
        f"{DESIGN.name}: flux density at {CURRENT:g} A, placeholder steel and magnet"
    )
    figure.colorbar(shading, ax=axes, label="flux density (T)")

    picture = Path(f"{DESIGN.stem}_field.png")
    figure.savefig(picture, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"largest flux density: {flux_density.max():.2f} T, with placeholder steel and magnet")
    print(f"picture: {picture}")


if __name__ == "__main__":
    main()
