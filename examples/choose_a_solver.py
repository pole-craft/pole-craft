"""List the sparse solvers on this machine and time a field solve with each.

Run from any folder: python examples/choose_a_solver.py
Needs the optional Gmsh package (pip install "polecraft[mesh]").

Every solver gives the same field; they differ in speed. With no solver
named, the first one listed (SuperLU) is used. The steel curve and the magnet of the
model are placeholders, which does not matter for comparing solvers.
"""

import sys
import time
from pathlib import Path

import numpy as np

from polecraft import load_design, machine_for
from polecraft.core.compute import (
    available_solvers,
    install_hint,
    known_solvers,
    solver_descriptions,
)
from polecraft.core.materials import Magnet, load_material
from polecraft.core.mesh import build_mesh
from polecraft.core.problem import build_problem
from polecraft.core.solver import solve

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "qdd_outer_36s42p.yaml"
REPEATS = 5


def main() -> None:
    print("solvers on this machine (the first is used when none is named):")
    for name, description in solver_descriptions().items():
        print(f"  {name}: {description}")
    for name in known_solvers():
        if name not in available_solvers():
            # An optional solver that is not here. On a machine it does not
            # exist for, there is no command to give.
            how = install_hint(name)
            print(f"  {name}: not installed" + (f"; to install it: {how}" if how else ""))

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
    problem = build_problem(machine, mesh, materials, current=20.0)
    print(f"\n{DESIGN.name}: {len(mesh.nodes)} nodes, placeholder steel and magnet")

    reference = None
    for name in available_solvers():
        solve(problem, solver=name)  # the first call also loads the solver
        # One solve is too short to time well: take the best of several.
        best = None
        for _ in range(REPEATS):
            began = time.perf_counter()
            solution = solve(problem, solver=name)
            seconds = time.perf_counter() - began
            if best is None or seconds < best[0]:
                best = (seconds, dict(solution.timings.stages)["linear solve"])
        if reference is None:
            reference = solution.potential
        difference = np.abs(solution.potential - reference).max() / np.abs(reference).max()
        # The solution records the solver that ran.
        print(
            f"  {solution.solver:18s} {best[0] * 1e3:7.1f} ms, of which {best[1] * 1e3:6.1f} ms in "
            f"the solver; {solution.iterations} iterations; differs from the first by {difference:.0e}"
        )
    print(f"\nBest of {REPEATS} solves each. On a model this small the solvers are close;")
    print("the differences show on larger ones.")


if __name__ == "__main__":
    main()
