"""Mesh pictures."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.geometry import CoilSide, Geometry, Magnetisation, Region, ring_sector
from polecraft.core.materials import LinearMaterial
from polecraft.core.mesh import BAND, Mesh
from polecraft.core.problem import Problem
from polecraft.core.solver import Solution, solve
from polecraft.plot import (
    AIR,
    BAND_COLOUR,
    FIELD_LINE,
    MAGNET_IN,
    MAGNET_OUT,
    PHASE,
    STEEL,
    draw_field,
    draw_mesh,
    lighter,
    picture_path,
    plot_field,
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


# Pictures of the field


def field_model(bx: float = 0.0, by: float = 0.8) -> tuple[Problem, Solution]:
    """A 6 by 4 mm rectangle of air in a uniform field, set by its edge potentials.

    The potential on the edges is A = bx y - by x, so the field inside is
    B = (dA/dy, -dA/dx) = (bx, by) everywhere.
    """
    columns, rows, width, height = 6, 4, 0.006, 0.004
    x, y = np.meshgrid(np.linspace(0.0, width, columns + 1), np.linspace(0.0, height, rows + 1))
    nodes = np.column_stack([x.ravel(), y.ravel()])
    triangles = []
    for j in range(rows):
        for i in range(columns):
            a = j * (columns + 1) + i
            b, c, d = a + 1, a + columns + 2, a + columns + 1
            triangles += [(a, b, c), (a, c, d)]
    edge = np.flatnonzero(
        (nodes[:, 0] == 0.0) | (nodes[:, 0] == width) | (nodes[:, 1] == 0.0) | (nodes[:, 1] == height)
    )
    problem = Problem(
        nodes=nodes,
        triangles=np.array(triangles),
        materials=(LinearMaterial(1.0),),
        fixed_nodes=edge,
        fixed_values=bx * nodes[edge, 1] - by * nodes[edge, 0],
    )
    return problem, solve(problem)


def test_field_model_has_the_field_it_was_given() -> None:
    problem, solution = field_model(bx=0.3, by=0.8)
    assert solution.flux_density == pytest.approx(np.tile([0.3, 0.8], (len(problem.triangles), 1)))


def shading_of(figure):
    (shading,) = [c for c in figure.axes[0].collections if c.get_array() is not None and c.colorbar]
    return shading


def test_field_picture_colours_each_triangle_by_its_flux_density() -> None:
    # Both parts of the field count: the size is 0.5 T, not 0.3 or 0.4.
    problem, solution = field_model(bx=0.3, by=0.4)
    # Three different values, to tell the triangles apart.
    b = solution.flux_density.copy()
    b[:16] *= 0.25
    b[16:32] *= 0.5
    uneven = dataclasses.replace(solution, flux_density=b)
    figure = draw_field(problem, uneven)
    shading = shading_of(figure)
    assert np.asarray(shading.get_array()) == pytest.approx(np.linalg.norm(b, axis=1))
    assert shading.get_clim() == pytest.approx((0.0, 0.5))
    centres = problem.nodes[problem.triangles].mean(axis=1) * 1e3
    for number in (0, 20, 40):
        expected = shading.cmap(shading.norm(np.linalg.norm(b[number])))
        seen = pixel_colour(figure, *centres[number])
        assert seen == pytest.approx([round(255 * v) for v in expected[:3]], abs=3)
    assert pixel_colour(figure, *centres[0]) != pixel_colour(figure, *centres[40])


def test_field_picture_has_a_colour_scale_in_tesla_and_axes_in_millimetres() -> None:
    problem, solution = field_model()
    figure = draw_field(problem, solution, title="A field")
    picture, scale = figure.axes
    assert scale.get_ylabel() == "flux density (T)"
    assert picture.get_xlabel() == "x (mm)" and picture.get_ylabel() == "y (mm)"
    assert picture.get_aspect() == 1.0
    assert picture.get_title() == "A field"
    (x0, x1), (y0, y1) = picture.get_xlim(), picture.get_ylim()
    assert x0 <= 0.0 and x1 >= 6.0 and y0 <= 0.0 and y1 >= 4.0
    assert x1 - x0 < 8.0  # millimetres, not metres or more


def field_lines(figure) -> list[np.ndarray]:
    """The points of every field line drawn, in millimetres."""
    found = []
    for collection in figure.axes[0].collections:
        if collection.get_label() == "field lines":
            found += [path.vertices for path in collection.get_paths() if len(path.vertices)]
    return found


def test_field_lines_run_along_the_field() -> None:
    # The field points along y, so every line is a vertical one.
    problem, solution = field_model(bx=0.0, by=0.8)
    lines = field_lines(draw_field(problem, solution, lines=5))
    assert len(lines) == 5
    for points in lines:
        assert np.ptp(points[:, 0]) < 1e-9
        assert np.ptp(points[:, 1]) == pytest.approx(4.0)
    # Evenly spaced, none on the edge of the model.
    across = sorted(points[0, 0] for points in lines)
    assert across == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_the_number_of_field_lines_can_be_chosen_and_none_is_allowed() -> None:
    problem, solution = field_model()
    assert len(field_lines(draw_field(problem, solution, lines=11))) == 11
    assert field_lines(draw_field(problem, solution, lines=0)) == []
    assert len(field_lines(draw_field(problem, solution))) == 25


def test_a_model_with_no_field_is_drawn_without_lines() -> None:
    problem, solution = field_model(bx=0.0, by=0.0)
    figure = draw_field(problem, solution)
    assert field_lines(figure) == []
    assert shading_of(figure).get_clim()[1] > 0.0  # still a scale to show


def test_the_top_of_the_colour_scale_can_be_set() -> None:
    problem, solution = field_model()
    figure = draw_field(problem, solution, largest=2.0)
    assert shading_of(figure).get_clim() == pytest.approx((0.0, 2.0))
    assert shading_of(figure).colorbar.extend == "neither"
    # A field above the top of the scale is marked on the scale.
    clipped = draw_field(problem, solution, largest=0.5)
    assert shading_of(clipped).colorbar.extend == "max"
    for wrong in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError, match="largest"):
            draw_field(problem, solution, largest=wrong)
    with pytest.raises(ValueError, match="lines"):
        draw_field(problem, solution, lines=-1)


def test_region_outlines_can_be_drawn_over_the_field(geometry: Geometry) -> None:
    problem, solution = field_model()
    plain = draw_field(problem, solution)
    outlined = draw_field(problem, solution, outlines=geometry)
    (drawn,) = [c for c in outlined.axes[0].collections if c.get_label() == "outlines"]
    assert len(drawn.get_paths()) == len(geometry.regions) == 3
    assert not [c for c in plain.axes[0].collections if c.get_label() == "outlines"]
    # In millimetres, like the rest of the picture.
    radius = np.hypot(*np.vstack([path.vertices for path in drawn.get_paths()]).T)
    assert radius.min() == pytest.approx(1000.0, rel=1e-6) and radius.max() == pytest.approx(5000.0, rel=1e-6)
    for path in drawn.get_paths():  # each outline is a closed loop
        assert path.vertices[0] == pytest.approx(path.vertices[-1])


def test_field_lines_show_even_at_the_top_of_the_colour_scale() -> None:
    problem, solution = field_model()
    figure = draw_field(problem, solution)
    top = shading_of(figure).cmap(1.0)[:3]
    line = [int(FIELD_LINE[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    assert sum(top) - sum(line) > 0.8  # clearly lighter than the lines


def test_a_long_title_does_not_run_into_the_colour_scale() -> None:
    problem, solution = field_model()
    figure = draw_field(problem, solution, title="a long name of a design file: " + "x" * 70)
    figure.canvas.draw()
    picture, scale = figure.axes
    title = picture.title.get_window_extent()
    assert not title.overlaps(scale.get_window_extent())


def test_field_picture_passes_its_options_to_the_drawing(
    geometry: Geometry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polecraft.plot as plot

    problem, solution = field_model()
    seen = {}
    real = plot.draw_field

    def recording(*arguments, **keywords):
        seen.update(keywords)
        return real(*arguments, **keywords)

    monkeypatch.setattr(plot, "draw_field", recording)
    plot_field(problem, solution, tmp_path / "f.png", title="t", lines=7, largest=2.0, outlines=geometry)
    assert seen == {"title": "t", "lines": 7, "largest": 2.0, "outlines": geometry}


def test_field_picture_is_written_in_each_file_type(tmp_path: Path) -> None:
    problem, solution = field_model()
    png = plot_field(problem, solution, tmp_path / "field.png", title="A field")
    assert png == tmp_path / "field.png"
    assert png.read_bytes().startswith(PNG_SIGNATURE) and png.stat().st_size > 5000
    svg = plot_field(problem, solution, str(tmp_path / "field.svg"), title="A field")
    text = svg.read_text(encoding="utf-8")
    assert "<svg" in text and "A field" in text and "flux density (T)" in text
    pdf = plot_field(problem, solution, tmp_path / "field.pdf")
    assert pdf.read_bytes().startswith(b"%PDF")


def test_field_picture_path_is_checked_before_drawing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, solution = field_model()
    monkeypatch.setattr(
        "polecraft.plot.draw_field",
        lambda *arguments, **keywords: pytest.fail("drawn before the path was checked"),
    )
    with pytest.raises(ValueError, match="must end in .png, .svg or .pdf"):
        plot_field(problem, solution, tmp_path / "field.jpg")
    with pytest.raises(ValueError, match="does not exist"):
        plot_field(problem, solution, tmp_path / "missing" / "field.png")
    assert list(tmp_path.iterdir()) == []


def test_a_solution_of_another_problem_is_refused() -> None:
    problem, solution = field_model()
    for wrong in (
        dataclasses.replace(solution, potential=solution.potential[:-1]),
        dataclasses.replace(solution, flux_density=solution.flux_density[:-1]),
    ):
        with pytest.raises(ValueError, match="solution does not belong to this problem"):
            draw_field(problem, wrong)


def test_field_picture_of_the_reference_motor(tmp_path: Path) -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.materials import Magnet, load_material
    from polecraft.core.mesh import build_mesh
    from polecraft.core.problem import build_problem
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[1]
    machine = machine_for(load_design(root / "designs" / "qdd_outer_36s42p.yaml"))
    geometry = machine.build_geometry()
    mesh = build_mesh(geometry)
    # Placeholders: not the design's real steel or magnet grade.
    materials = {
        "M270-35A": load_material("generic_35A_placeholder"),
        "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
    }
    problem = build_problem(machine, mesh, materials, current=10.0)
    solution = solve(problem)
    figure = draw_field(problem, solution, outlines=geometry)
    assert len(field_lines(figure)) >= 25  # a level may be drawn in several pieces
    top = shading_of(figure).get_clim()[1]
    assert top == pytest.approx(np.linalg.norm(solution.flux_density, axis=1).max())
    assert 1.5 < top < 3.0
    written = plot_field(problem, solution, tmp_path / "field.png", outlines=geometry)
    assert written.stat().st_size > 50_000
