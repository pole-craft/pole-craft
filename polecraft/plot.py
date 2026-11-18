"""Pictures of a machine: its mesh, coloured by region, and its solved field."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from polecraft.core.geometry import Geometry, Region
from polecraft.core.mesh import Mesh

if TYPE_CHECKING:
    from polecraft.core.problem import Problem
    from polecraft.core.solver import Solution

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

# Flux density from pale yellow to mid blue. The darkest end of the colour
# map is left out, so that the dark field lines and region outlines drawn
# over it still show at the top of the scale.
FIELD_COLOURS = ListedColormap(colormaps["YlGnBu"](np.linspace(0.0, 0.7, 256)), name="field")
FIELD_LINE = "#101418"


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


def draw_field(
    problem: Problem,
    solution: Solution,
    title: str = "",
    lines: int = 25,
    largest: float | None = None,
    outlines: Geometry | None = None,
) -> Figure:
    """Draw a solved field and return the figure.

    Each triangle is coloured by its flux density, with a scale in tesla,
    and ``lines`` field lines are drawn over it. They are lines of equal
    potential, evenly spaced in potential, so where they crowd the flux
    density is high. ``largest`` is the top of the colour scale in tesla;
    left out, it is the largest flux density in the model. With
    ``outlines``, the outline of every region of that geometry is drawn
    too, at rotor angle 0. Axes are in millimetres.
    """
    if solution.potential.shape != (len(problem.nodes),) or solution.flux_density.shape != (
        len(problem.triangles),
        2,
    ):
        raise ValueError("the solution does not belong to this problem: the sizes differ")
    if isinstance(lines, bool) or not isinstance(lines, (int, np.integer)) or lines < 0:
        raise ValueError(f"lines must be a whole number, zero or more, not {lines!r}")
    flux_density = np.hypot(solution.flux_density[:, 0], solution.flux_density[:, 1])
    if largest is None:
        # With no field at all there is still a scale to draw.
        largest = float(flux_density.max()) if flux_density.size and flux_density.max() > 0 else 1.0
    elif not (math.isfinite(largest) and largest > 0.0):
        raise ValueError(f"largest must be a flux density above zero, in tesla, not {largest}")

    millimetres = problem.nodes * 1e3
    surface = Triangulation(millimetres[:, 0], millimetres[:, 1], problem.triangles)
    figure = Figure(figsize=(10, 8))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    # One colour per triangle: the flux density is constant over each.
    shading = axes.tripcolor(
        surface, facecolors=flux_density, cmap=FIELD_COLOURS, vmin=0.0, vmax=largest
    )
    if outlines is not None:
        loops = [region.polygon(1e-6) * 1e3 for region in outlines.regions]
        axes.add_collection(
            LineCollection(
                [np.vstack([loop, loop[:1]]) for loop in loops],
                colors=EDGE, linewidths=0.8, label="outlines",
            )
        )
    low, high = float(solution.potential.min()), float(solution.potential.max())
    if lines and high > low:
        levels = np.linspace(low, high, lines + 2)[1:-1]
        drawn = axes.tricontour(
            surface, solution.potential, levels=levels,
            colors=FIELD_LINE, linewidths=0.35, linestyles="solid",
        )
        drawn.set_label("field lines")
    axes.set_aspect("equal")
    axes.set_xlabel("x (mm)")
    axes.set_ylabel("y (mm)")
    if title:
        axes.set_title(title)
    # A scale as tall as the picture, in its own place beside it, so that a
    # long title passes above it and not behind it.
    scale = make_axes_locatable(axes).append_axes("right", size="4%", pad=0.15)
    figure.colorbar(
        shading, cax=scale, label="flux density (T)",
        extend="max" if flux_density.size and flux_density.max() > largest else "neither",
    )
    return figure


def plot_field(
    problem: Problem,
    solution: Solution,
    path: str | Path,
    title: str = "",
    lines: int = 25,
    largest: float | None = None,
    outlines: Geometry | None = None,
) -> Path:
    """Save a picture of a solved field. Returns the file written.

    Flux density as colour with field lines over it, as ``draw_field``
    draws them. The file type follows the extension: ``.png``, ``.svg`` or
    ``.pdf``.
    """
    path = picture_path(path)
    figure = draw_field(
        problem, solution, title=title, lines=lines, largest=largest, outlines=outlines
    )
    figure.savefig(path, dpi=200, bbox_inches="tight")
    return path
