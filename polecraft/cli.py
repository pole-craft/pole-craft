"""The ``polecraft`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from polecraft import __version__
from polecraft.design import load_design
from polecraft.machines import machine_for
from polecraft.timing import Timings
from polecraft.validation import DesignError

# Added to the help of every command that meshes. Such a command also calls
# require_gmsh() before anything else.
GMSH_NOTE = 'Needs Gmsh: pip install "polecraft[mesh]"'


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the exit code: 0 for success, 1 for failure.

    Stopping it with Ctrl-C returns 130.
    """
    arguments = _parser().parse_args(argv)
    try:
        return _mesh(arguments)
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
