"""The ``polecraft`` command."""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Sequence

from polecraft import __version__
from polecraft.design import Design, load_design
from polecraft.machines import machine_for
from polecraft.timing import Timings
from polecraft.validation import DesignError, warning_text

# Added to the help of every command that meshes. Such a command also calls
# require_gmsh() before anything else.
GMSH_NOTE = 'Needs Gmsh: pip install "polecraft[mesh]"'

# What the solve command uses until it takes a design's materials from the
# material library: a steel curve made from a formula, and a magnet given by
# two numbers.
STAND_IN_STEEL = "generic_35A_placeholder"
STAND_IN_REMANENCE = 1.2  # tesla
STAND_IN_PERMEABILITY = 1.05


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polecraft", description="Design and simulation tool for QDD motors."
    )
    parser.add_argument("--version", action="version", version=f"polecraft {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    mesh = commands.add_parser(
        "mesh", help="mesh a design and save a picture of it (needs Gmsh)",
        description="Build the geometry of a design, mesh it and save a coloured picture. "
        + GMSH_NOTE,
    )
    mesh.add_argument("design", type=Path, help="design file (YAML)")
    mesh.add_argument(
        "-o", "--output", type=Path,
        help="picture to write; .png, .svg or .pdf (default: <design name>_mesh.png)",
    )
    mesh.add_argument(
        "--full", action="store_true",
        help="mesh the whole machine, not only the slice that repeats",
    )
    mesh.add_argument(
        "--size", type=float, default=1.0,
        help="scale of the triangles; 0.5 halves their edges (default: 1)",
    )

    solve = commands.add_parser(
        "solve", help="solve the field of a design and save a picture of it (needs Gmsh)",
        description="Mesh a design, solve its magnetic field at rotor angle 0, print torque and "
        "flux linkage and save a picture of the field. It does not use the material library "
        "yet: the steel and the magnet are stand-ins, so the numbers show how the tool "
        "behaves, not how the motor performs. " + GMSH_NOTE,
    )
    solve.add_argument("design", type=Path, help="design file (YAML)")
    solve.add_argument(
        "-o", "--output", type=Path,
        help="picture to write; .png, .svg or .pdf (default: <design name>_field.png)",
    )
    solve.add_argument(
        "--current", type=float, default=0.0,
        help="peak phase current in amperes (default: 0, magnets only)",
    )
    solve.add_argument(
        "--advance", type=float, default=0.0,
        help="angle of the current ahead of the back-EMF, in electrical degrees (default: 0)",
    )
    solve.add_argument(
        "--full", action="store_true",
        help="solve the whole machine, not only the slice that repeats",
    )
    solve.add_argument(
        "--size", type=float, default=1.0,
        help="scale of the triangles; 0.5 halves their edges (default: 1)",
    )
    solve.add_argument(
        "--solver", help="sparse solver to use (default: superlu); see polecraft.core.compute",
    )
    solve.add_argument(
        "--remanence", type=float, default=STAND_IN_REMANENCE,
        help=f"remanence of the stand-in magnet in tesla (default: {STAND_IN_REMANENCE})",
    )
    return parser


def _mesh(arguments: argparse.Namespace) -> int:
    timings = Timings()
    # Imported here so that other commands start without loading Matplotlib.
    # Gmsh comes first: without it nothing below is of use, so say so at once.
    with timings.stage("load libraries"):
        from polecraft.core.mesh import build_mesh, require_gmsh

        require_gmsh()
        from polecraft.plot import picture_path, plot_mesh

    # Checked first, so that a mistyped file name does not cost a mesh.
    output = picture_path(arguments.output or Path(f"{arguments.design.stem}_mesh.png"))

    with timings.stage("load design"):
        design = load_design(arguments.design)
    _print_warnings(design, arguments.design)
    with timings.stage("geometry"):
        geometry = machine_for(design).build_geometry(full=arguments.full)
    with timings.stage("mesh"):
        mesh = build_mesh(geometry, size=arguments.size)

    shown = "whole machine" if arguments.full else "repeating slice"
    title = f"{arguments.design.name}: {design.slots} slots, {design.poles} poles, {shown}"
    with timings.stage("picture"):
        written = plot_mesh(mesh, geometry, output, title=title)

    print(f"{arguments.design.name}: {design.slots} slots, {design.poles} poles")
    print(f"  {shown}: {len(geometry.regions)} regions")
    print(f"  mesh: {len(mesh.nodes)} nodes, {len(mesh.triangles)} triangles")
    print(f"  smallest triangle angle: {mesh.smallest_angles().min():.1f} degrees")
    print(f"  picture: {written}")
    print(timings.report())
    return 0


def _print_warnings(design: Design, path: Path) -> None:
    """Advice found in the design file; the command carries on."""
    for warning in design.warnings:
        print(warning_text(warning, str(path)), file=sys.stderr)


def _solve(arguments: argparse.Namespace) -> int:
    timings = Timings()
    with timings.stage("load libraries"):
        from polecraft.core.mesh import build_mesh, require_gmsh

        require_gmsh()
        import numpy as np

        from polecraft.core.compute import solver_name
        from polecraft.core.materials import Magnet, load_material
        from polecraft.core.post import flux_linkage, torque
        from polecraft.core.problem import MeshMismatch, build_problem
        from polecraft.core.solver import ConvergenceError, solve
        from polecraft.plot import picture_path, plot_field

    # Checked before any work, so that a mistyped option does not cost a solve.
    output = picture_path(arguments.output or Path(f"{arguments.design.stem}_field.png"))
    for name in ("current", "advance"):
        if not math.isfinite(getattr(arguments, name)):
            raise ValueError(f"{name} must be a finite number, not {getattr(arguments, name)}")
    magnet = Magnet(remanence=arguments.remanence, relative_permeability=STAND_IN_PERMEABILITY)
    # A solver that is not installed is replaced, and that is said in one line.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = solver_name(arguments.solver)
    for warning in caught:
        print(f"polecraft: {warning.message}", file=sys.stderr)

    with timings.stage("load design"):
        design = load_design(arguments.design)
    _print_warnings(design, arguments.design)
    steels = sorted({design.steel.stator, design.steel.rotor})
    if design.magnet.material in steels:
        raise ValueError(
            f"{design.magnet.material} is named as the magnet and as a steel; "
            "give them different names"
        )
    with timings.stage("geometry"):
        machine = machine_for(design)
        geometry = machine.build_geometry(full=arguments.full)
    with timings.stage("mesh"):
        mesh = build_mesh(geometry, size=arguments.size)
    steel = load_material(STAND_IN_STEEL)
    materials = {**{grade: steel for grade in steels}, design.magnet.material: magnet}
    with timings.stage("solve"):
        try:
            problem = build_problem(
                machine, mesh, materials,
                current=arguments.current, advance=math.radians(arguments.advance),
            )
        except MeshMismatch:
            # The mesh was made from this machine, so only its coarseness is left.
            raise ValueError(
                f"the mesh is too coarse to solve at --size {arguments.size:g}: its regions "
                "no longer have the areas of the design; use a smaller --size"
            ) from None
        try:
            solution = solve(problem, solver=solver)
        except ConvergenceError as error:
            print(f"polecraft: {error}", file=sys.stderr)
            return 1
    with timings.stage("results"):
        newton_metres = torque(problem, solution)
        linked = flux_linkage(problem, solution)
        b = np.hypot(solution.flux_density[:, 0], solution.flux_density[:, 1])
        saturating = np.array([not material.linear for material in problem.materials])
        in_steel = b[saturating[problem.triangle_material]]

    shown = "whole machine" if arguments.full else "repeating slice"
    title = (
        f"{arguments.design.name}: flux density at {arguments.current:g} A, "
        "stand-in steel and magnet"
    )
    with timings.stage("picture"):
        written = plot_field(problem, solution, output, title=title, outlines=geometry)

    print(f"{arguments.design.name}: {design.slots} slots, {design.poles} poles")
    print("  materials: stand-ins, not the grades the design names")
    for grade in steels:
        print(f"    steel {grade}: {STAND_IN_STEEL}, a curve from a formula")
    print(
        f"    magnet {design.magnet.material}: remanence {magnet.remanence:g} T, "
        f"relative permeability {magnet.relative_permeability:g}"
    )
    print(f"  {shown}: {len(geometry.regions)} regions")
    print(f"  mesh: {len(mesh.nodes)} nodes, {len(mesh.triangles)} triangles")
    if arguments.current == 0.0:
        print("  current: 0 A peak (give --current for torque), rotor angle 0")
    else:
        print(
            f"  current: {arguments.current:g} A peak, advance {arguments.advance:g} electrical degrees, "
            "rotor angle 0"
        )
    print(solution.report())
    print("Results (whole machine, 2D, no end effects)")
    # Adding zero turns a rounded -0.000 into 0.000.
    print(f"  torque: {round(newton_metres, 3) + 0.0:.3f} N m")
    for phase, value in linked.items():
        print(f"  flux linkage of phase {phase}: {value * 1e3:+.3f} mWb")
    print(f"  flux density in the steel: {np.percentile(in_steel, 99):.2f} T at the 99th percentile")
    print(f"  picture: {written}")
    print("  With stand-in materials these show how the tool behaves, not how this motor performs.")
    print(f"  sparse solver: {solution.solver}")
    print(timings.report())
    return 0


COMMANDS = {"mesh": _mesh, "solve": _solve}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the exit code: 0 for success, 1 for failure.

    Stopping it with Ctrl-C returns 130.
    """
    arguments = _parser().parse_args(argv)
    try:
        return COMMANDS[arguments.command](arguments)
    except DesignError as error:
        print(error, file=sys.stderr)
    except (ImportError, ValueError, OSError) as error:
        print(f"polecraft: {error}", file=sys.stderr)
    except KeyboardInterrupt:
        print("polecraft: interrupted", file=sys.stderr)
        return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
