"""Solve the repeating slice of a motor and the whole motor, and compare.

Run from any folder: python examples/slice_against_whole_machine.py
Needs the optional Gmsh package (pip install "polecraft[mesh]").

A motor with 36 slots and 42 poles repeats six times round, with the field
reversed from one slice to the next. Solving one slice with its two cut
edges coupled gives the results of the whole machine from a model a sixth
of the size. The steel curve and the magnet are placeholders, not the grades
the design names.
"""

import sys
import time
from pathlib import Path

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
    repeat = machine.symmetry()
    kind = "reversed" if repeat.antiperiodic else "the same"
    print(f"{design.slots} slots, {design.poles} poles, placeholder steel and magnet")
    print(f"repeats {repeat.periods} times; the field in the next slice is {kind}")
    materials = {
        design.steel.stator: load_material("generic_35A_placeholder"),
        design.magnet.material: Magnet(remanence=1.2, relative_permeability=1.05),
    }

    print("\n                 nodes   mesh and solve   torque     flux linkage of phase A")
    results = []
    for name, full in (("slice", False), ("whole machine", True)):
        began = time.perf_counter()
        try:
            mesh = build_mesh(machine.build_geometry(full=full))
        except ImportError as error:
            sys.exit(str(error))  # Gmsh is missing: the message says how to install it
        # The problem knows how its cut edges are coupled, and how many
        # times it repeats, so the results come out for the whole machine.
        problem = build_problem(machine, mesh, materials, current=CURRENT)
        solution = solve(problem)
        seconds = time.perf_counter() - began
        newton_metres = torque(problem, solution)
        linked = flux_linkage(problem, solution)["A"]
        results.append((newton_metres, linked))
        print(
            f"{name:15s} {len(mesh.nodes):6d}   {seconds:10.2f} s    {newton_metres:7.4f} N m"
            f"   {linked * 1e3:+9.4f} mWb"
        )

    (slice_torque, slice_linked), (whole_torque, whole_linked) = results
    print(f"\ntorque differs by {100.0 * abs(slice_torque / whole_torque - 1.0):.3f} %")
    print(f"flux linkage differs by {100.0 * abs(slice_linked / whole_linked - 1.0):.3f} %")
    print("The two meshes are made separately, so they agree to the accuracy of the mesh.")


if __name__ == "__main__":
    main()
