"""Load a design, look at its winding, mesh it and save a picture.

Run from any folder: python examples/mesh_a_design.py
Needs the optional Gmsh package (pip install "polecraft[mesh]").
"""

import sys
from pathlib import Path

from polecraft import load_design, machine_for
from polecraft.core.mesh import build_mesh
from polecraft.core.winding import concentrated_winding, layout_string, winding_factor
from polecraft.plot import plot_mesh

DESIGN = Path(__file__).resolve().parents[1] / "designs" / "qdd_outer_36s42p.yaml"


def main() -> None:
    design = load_design(DESIGN)
    machine = machine_for(design)

    # The smallest slice of the motor that repeats; only this is meshed.
    slice_ = machine.symmetry()
    print(f"{design.slots} slots, {design.poles} poles")
    print(f"repeats {slice_.periods} times, every {slice_.angle_deg:g} degrees")

    # One letter per tooth round the stator: the phase of its coil; lower
    # case is a coil wound the opposite way.
    coils = concentrated_winding(design.slots, design.poles)
    print(f"winding layout: {layout_string(coils)}")
    print(f"winding factor: {winding_factor(design.slots, design.poles):.4f}")

    # Lengths are in metres here; the design file is in millimetres.
    geometry = machine.build_geometry()  # full=True for the whole machine
    magnet = geometry.region("magnet_0")
    print(f"{len(geometry.regions)} regions, magnet area {magnet.area() * 1e6:.2f} mm^2")

    try:
        mesh = build_mesh(geometry, size=1.0)  # smaller size, finer mesh
    except ImportError as error:
        sys.exit(str(error))  # Gmsh is missing: the message says how to install it
    print(f"{len(mesh.nodes)} nodes, {len(mesh.triangles)} triangles")

    picture = plot_mesh(mesh, geometry, "qdd_outer_36s42p_mesh.png", title=DESIGN.name)
    print(f"picture: {picture}")


if __name__ == "__main__":
    main()
