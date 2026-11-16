"""Mesh pictures."""

import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.geometry import CoilSide, Geometry, Magnetisation, Region, ring_sector
from polecraft.core.mesh import BAND, Mesh
from polecraft.plot import (
    AIR,
    BAND_COLOUR,
    MAGNET_IN,
    MAGNET_OUT,
    PHASE,
    STEEL,
    draw_mesh,
    lighter,
    picture_path,
    plot_mesh,
    region_colour,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def make_region(name: str, kind: str, inner: float, outer: float, **extra) -> Region:
    return Region(
        name=name, kind=kind, role=name, material="any",
        boundary=ring_sector(inner, outer, 0.0, 1.0), **extra,
    )


@pytest.fixture
def geometry() -> Geometry:
    """Four ring sectors stacked outwards: steel, coil, then magnet beyond the band."""
    return Geometry(
        stator=(
            make_region("yoke", "steel", 1.0, 2.0),
            make_region("coil", "coil", 2.0, 3.0, coil_side=CoilSide("B", 0, 1, 10)),
        ),
        rotor=(
            make_region("magnet", "magnet", 4.0, 5.0,
                        magnetisation=Magnetisation("radial", 0.5, -1)),
        ),
        inner_radius=1.0, outer_radius=5.0,
        band_inner_radius=3.0, band_outer_radius=4.0,
        stator_start_angle=0.0, rotor_start_angle=0.0,
        span_angle=1.0, antiperiodic=False, stack_length=0.01,
    )


@pytest.fixture
def mesh() -> Mesh:
    """Two triangles for each of the four layers of the geometry fixture."""
    radii = [1.0, 2.0, 3.0, 4.0, 5.0]
    nodes = np.array([[r * math.cos(a), r * math.sin(a)] for r in radii for a in (0.0, 1.0)])
    triangles, labels = [], []
    for layer in range(4):
        a, b, c, d = 2 * layer, 2 * layer + 1, 2 * layer + 2, 2 * layer + 3
        triangles += [[a, c, d], [a, d, b]]
        labels += [layer, layer]
    order = [0, 1, 3, 2]  # layers outwards: yoke, coil, band, magnet
    return Mesh(
        nodes=nodes,
        triangles=np.array(triangles),
        triangle_region=np.array([order[label] for label in labels]),
        region_names=("yoke", "coil", "magnet", BAND),
        rotor_nodes=np.array([False] * 6 + [True] * 4),
        periodic=np.zeros((0, 2), dtype=int),
        antiperiodic=False,
        band_divisions=1,
    )


# Colours


def test_steel_and_air_colours() -> None:
    assert region_colour(make_region("yoke", "steel", 1.0, 2.0)) == STEEL
    assert region_colour(make_region("gap", "air", 1.0, 2.0)) == AIR


def test_magnets_are_coloured_by_polarity() -> None:
    outwards = make_region("m", "magnet", 1.0, 2.0, magnetisation=Magnetisation("parallel", 0.0, 1))
    inwards = make_region("m", "magnet", 1.0, 2.0, magnetisation=Magnetisation("parallel", 0.0, -1))
    assert region_colour(outwards) == MAGNET_OUT
    assert region_colour(inwards) == MAGNET_IN


@pytest.mark.parametrize("phase", ["A", "B", "C"])
def test_coil_sides_are_coloured_by_phase_and_direction(phase: str) -> None:
    out = make_region("c", "coil", 1.0, 2.0, coil_side=CoilSide(phase, 0, 1, 10))
    back = make_region("c", "coil", 1.0, 2.0, coil_side=CoilSide(phase, 0, -1, 10))
    assert region_colour(out) == PHASE[phase]
    assert region_colour(back) != PHASE[phase]
    # The returning side is the same hue, lighter.
    assert int(region_colour(back)[1:3], 16) > int(PHASE[phase][1:3], 16)


def channels(colour: str) -> tuple[int, int, int]:
    return tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))


def test_all_colours_are_clearly_different() -> None:
    colours = [STEEL, AIR, BAND_COLOUR, MAGNET_OUT, MAGNET_IN]
    colours += list(PHASE.values()) + [lighter(colour) for colour in PHASE.values()]
    for i, first in enumerate(colours):
        for second in colours[i + 1 :]:
            apart = max(abs(a - b) for a, b in zip(channels(first), channels(second)))
            assert apart > 40, (first, second)


def pixel_colour(figure, x_mm: float, y_mm: float) -> tuple[int, int, int]:
    """Colour drawn at a point given in the axes' millimetre coordinates."""
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())
    column, row = figure.axes[0].transData.transform((x_mm, y_mm))
    # Image rows run downwards, figure coordinates upwards.
    return tuple(int(v) for v in image[image.shape[0] - 1 - round(row), round(column), :3])


def test_each_triangle_is_drawn_in_the_colour_of_its_region(
    mesh: Mesh, geometry: Geometry
) -> None:
    figure = draw_mesh(mesh, geometry)
    expected = {"yoke": STEEL, "coil": PHASE["B"], "magnet": MAGNET_IN, BAND: BAND_COLOUR}
    centres = mesh.nodes[mesh.triangles].mean(axis=1) * 1e3
    for centre, label in zip(centres, mesh.triangle_region):
        name = mesh.region_names[label]
        assert pixel_colour(figure, *centre) == channels(expected[name]), name


def test_picture_is_in_millimetres_with_equal_scales(mesh: Mesh, geometry: Geometry) -> None:
    axes = draw_mesh(mesh, geometry, title="Test motor").axes[0]
    assert axes.get_aspect() == 1.0
    assert (axes.get_xlabel(), axes.get_ylabel()) == ("x (mm)", "y (mm)")
    assert axes.get_title() == "Test motor"
    # The fixture reaches a radius of 5 m, which is 5000 mm.
    assert axes.get_xlim()[1] > 5000


def test_legend_names_every_colour(mesh: Mesh, geometry: Geometry) -> None:
    from matplotlib.colors import to_hex

    legend = draw_mesh(mesh, geometry).axes[0].get_legend()
    shown = {
        text.get_text(): to_hex(patch.get_facecolor())
        for text, patch in zip(legend.get_texts(), legend.get_patches())
    }
    assert shown == {
        "steel": STEEL,
        "air": AIR,
        "air-gap band": BAND_COLOUR,
        "magnet, outwards": MAGNET_OUT,
        "magnet, inwards": MAGNET_IN,
        "phase A": PHASE["A"],
        "phase B": PHASE["B"],
        "phase C": PHASE["C"],
        "full colour: current out of the page": "#5a6068",
        "light colour: current into the page": lighter("#5a6068"),
    }


# Picture files


def test_picture_is_written_as_png(mesh: Mesh, geometry: Geometry, tmp_path: Path) -> None:
    written = plot_mesh(mesh, geometry, tmp_path / "motor.png", title="Test motor")
    assert written == tmp_path / "motor.png"
    data = written.read_bytes()
    assert data.startswith(PNG_SIGNATURE)
    assert len(data) > 5000


def test_path_may_be_a_string(mesh: Mesh, geometry: Geometry, tmp_path: Path) -> None:
    written = plot_mesh(mesh, geometry, str(tmp_path / "motor.png"))
    assert written.is_file()


def test_file_type_follows_the_extension(mesh: Mesh, geometry: Geometry, tmp_path: Path) -> None:
    svg = plot_mesh(mesh, geometry, tmp_path / "motor.svg", title="Test motor")
    text = svg.read_text(encoding="utf-8")
    assert "<svg" in text and "Test motor" in text
    assert "x (mm)" in text and "phase B" in text and "magnet, inwards" in text


@pytest.mark.parametrize("name", ["motor", "motor.xyz", "motor.jpg"])
def test_unknown_file_type_is_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="must end in .png, .svg or .pdf"):
        picture_path(tmp_path / name)


def test_folder_as_picture_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is a folder"):
        picture_path(tmp_path)


def test_picture_in_a_missing_folder_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        picture_path(tmp_path / "missing" / "motor.png")


def test_good_picture_path_is_returned(tmp_path: Path) -> None:
    assert picture_path(str(tmp_path / "Motor.SVG")) == tmp_path / "Motor.SVG"


def test_reference_slice_is_coloured_correctly(tmp_path: Path) -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[1]
    geometry = machine_for(load_design(root / "designs" / "qdd_outer_36s42p.yaml")).build_geometry()
    mesh = build_mesh(geometry)
    figure = draw_mesh(mesh, geometry)

    # Tooth 0 carries a phase A coil wound the positive way: current out of
    # the page on its counter-clockwise side, back in on the other.
    expected = {
        "tooth_0": STEEL,
        "rotor_yoke_3": STEEL,
        "magnet_0": MAGNET_OUT,
        "magnet_1": MAGNET_IN,
        "coil_0_ccw": PHASE["A"],
        "coil_0_cw": lighter(PHASE["A"]),
        "gap_stator_2": AIR,
        BAND: BAND_COLOUR,
    }
    areas = mesh.triangle_areas()
    for name, colour in expected.items():
        chosen = np.flatnonzero(mesh.triangle_region == mesh.region_names.index(name))
        # The largest triangle of the region: its centre is clear of the edge lines.
        largest = chosen[np.argmax(areas[chosen])]
        centre = mesh.nodes[mesh.triangles[largest]].mean(axis=0) * 1e3
        # Band and gap triangles are only a few pixels across, so the thin
        # edge lines tint them slightly.
        drawn = pixel_colour(figure, *centre)
        assert max(abs(a - b) for a, b in zip(drawn, channels(colour))) <= 20, name


def test_mesh_from_another_geometry_is_rejected(
    mesh: Mesh, geometry: Geometry, tmp_path: Path
) -> None:
    other = Geometry(**{**geometry.__dict__, "rotor": ()})
    with pytest.raises(ValueError, match="not built from this geometry"):
        plot_mesh(mesh, other, tmp_path / "motor.png")
    assert not (tmp_path / "motor.png").exists()
