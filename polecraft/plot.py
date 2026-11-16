"""Pictures of a machine: the mesh, coloured by what each region is."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from polecraft.core.geometry import Geometry, Region
from polecraft.core.mesh import Mesh

STEEL = "#9aa1ab"
AIR = "#eef4fa"
BAND_COLOUR = "#7fd6e3"
PHASE = {"A": "#8e5bd0", "B": "#e0a030", "C": "#3f8f5f"}
MAGNET_OUT = "#c0392b"
MAGNET_IN = "#2a6fb0"
EDGE = "#30363d"

# Neutral colour used in the legend to explain the two shades of a coil.
COIL_OUT = "#5a6068"

FILE_TYPES = (".png", ".svg", ".pdf")


def lighter(colour: str, amount: float = 0.55) -> str:
    """The colour mixed with white."""
    channels = [int(colour[i : i + 2], 16) for i in (1, 3, 5)]
    mixed = [round(c + (255 - c) * amount) for c in channels]
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def region_colour(region: Region) -> str:
    """Colour for a region: steel grey, air pale, coils by phase, magnets by polarity.

    A coil side carrying positive phase current out of the page is drawn in
    the full phase colour, the returning side in a lighter shade.
    """
    if region.coil_side is not None:
        colour = PHASE[region.coil_side.phase]
        return colour if region.coil_side.sign > 0 else lighter(colour)
    if region.magnetisation is not None:
        return MAGNET_OUT if region.magnetisation.polarity > 0 else MAGNET_IN
    return STEEL if region.kind == "steel" else AIR


def draw_mesh(mesh: Mesh, geometry: Geometry, title: str = "") -> Figure:
    """Draw the mesh, coloured by region, and return the figure.

    The rotor is drawn at rotor angle 0. Axes are in millimetres.
    """
    colours = [region_colour(region) for region in geometry.regions] + [BAND_COLOUR]
    if len(colours) != len(mesh.region_names):
        raise ValueError("the mesh was not built from this geometry")

    millimetres = mesh.nodes * 1e3
    triangles = millimetres[mesh.triangles]
    faces = np.array(colours)[mesh.triangle_region]

    figure = Figure(figsize=(10, 10))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    axes.add_collection(
        PolyCollection(triangles, facecolors=faces, edgecolors=EDGE, linewidths=0.15)
    )
    axes.set_aspect("equal")
    axes.autoscale_view()
    axes.set_xlabel("x (mm)")
    axes.set_ylabel("y (mm)")
    if title:
        axes.set_title(title)

    legend = [
        Patch(facecolor=STEEL, label="steel"),
        Patch(facecolor=AIR, edgecolor=EDGE, linewidth=0.3, label="air"),
        Patch(facecolor=BAND_COLOUR, label="air-gap band"),
        Patch(facecolor=MAGNET_OUT, label="magnet, outwards"),
        Patch(facecolor=MAGNET_IN, label="magnet, inwards"),
    ]
    legend += [Patch(facecolor=colour, label=f"phase {name}") for name, colour in PHASE.items()]
    legend += [
        Patch(facecolor=COIL_OUT, label="full colour: current out of the page"),
        Patch(facecolor=lighter(COIL_OUT), label="light colour: current into the page"),
    ]
    axes.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    return figure


def picture_path(path: str | Path) -> Path:
    """Check where a picture is to be written. Returns the path as given.

    Raises ValueError if the file type is not one of ``.png``, ``.svg`` and
    ``.pdf``, if the path is a folder, or if its folder does not exist.
    """
    path = Path(path)
    if path.is_dir():
        raise ValueError(f"{path} is a folder; give the name of a picture file")
    if path.suffix.lower() not in FILE_TYPES:
        raise ValueError(
            f"{path}: the picture must end in {', '.join(FILE_TYPES[:-1])} or {FILE_TYPES[-1]}"
        )
    if not path.parent.is_dir():
        raise ValueError(f"{path}: the folder {path.parent} does not exist")
    return path


def plot_mesh(mesh: Mesh, geometry: Geometry, path: str | Path, title: str = "") -> Path:
    """Save a picture of the mesh, coloured by region. Returns the file written.

    The file type follows the extension: ``.png``, ``.svg`` or ``.pdf``.
    """
    path = picture_path(path)
    figure = draw_mesh(mesh, geometry, title=title)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    return path
