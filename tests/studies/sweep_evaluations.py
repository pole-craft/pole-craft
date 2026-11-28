"""Evaluations for the parallel sweep tests.

A sweep with several workers imports its evaluation in each process, so
these are top-level functions of a module, not closures in a test.
"""

import math
import os

FIELD_MESH_SIZE = 2.0


def copper(design):
    """A quick evaluation with no field solve; fails for 0.35 mm wire."""
    winding = design.winding
    if winding.wire_diameter_mm == 0.35:
        raise RuntimeError("this case does not solve")
    area = winding.parallel_strands * math.pi * winding.wire_diameter_mm**2 / 4.0
    return {"copper_mm2": area, "turns": winding.turns_per_coil, "grade": design.magnet.material, "pid": os.getpid()}


def crash(design):
    """Stops its process without an error for 0.4 mm wire, as a crash in a compiled library would."""
    if design.winding.wire_diameter_mm == 0.4:
        os._exit(3)
    return {"copper_mm2": math.pi * design.winding.wire_diameter_mm**2 / 4.0}


def field(design):
    """Torque from one field solve and the phase resistance, with placeholder magnets."""
    from polecraft.core.materials import Magnet, load_material
    from polecraft.core.mesh import build_mesh
    from polecraft.core.post import torque
    from polecraft.core.problem import build_problem
    from polecraft.core.solver import solve
    from polecraft.core.winding import phase_winding
    from polecraft.machines import machine_for

    # Placeholder magnets, not datasheet grades: only their order matters here.
    grades = {"N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
              "N52": Magnet(remanence=1.4, relative_permeability=1.05)}
    machine = machine_for(design)
    mesh = build_mesh(machine.build_geometry(), size=FIELD_MESH_SIZE)
    materials = {design.steel.stator: load_material("generic_35A_placeholder"),
                 design.magnet.material: grades[design.magnet.material]}
    problem = build_problem(machine, mesh, materials, current=5.0)
    return {"torque": torque(problem, solve(problem)), "resistance": phase_winding(machine).resistance()}
