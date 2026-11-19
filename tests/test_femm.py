"""The model written for FEMM is the model the solver is given."""

import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from polecraft.core.geometry import Edge, Geometry, Region, closed_loop, ring_sector
from polecraft.core.materials import MU0, FrohlichSteel, LinearMaterial, Magnet, load_material
from polecraft.design import design_to_dict, load_design
from polecraft.femm import _edges, femm_script, inside_point, read_results
from polecraft.machines import machine_for

BR, MU_R = 1.2, 1.05
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def machine():
    return machine_for(load_design(ROOT / "designs" / "qdd_outer_36s42p.yaml"))


@pytest.fixture(scope="module")
def geometry(machine):
    return machine.build_geometry(full=True)


@pytest.fixture(scope="module")
def materials() -> dict:
    # Placeholders: not the design's real steel or magnet grade.
    return {
        "M270-35A": load_material("generic_35A_placeholder"),
        "N42SH": Magnet(remanence=BR, relative_permeability=MU_R),
    }


@pytest.fixture(scope="module")
def cases(machine) -> list:
    return [machine.excitation(0.0, current) for current in (0.0, 10.0)]


@pytest.fixture(scope="module")
def script(geometry, materials, cases) -> str:
    return femm_script(geometry, materials, cases, "results.txt", gap_points=360)


def calls(script: str, name: str) -> list[list]:
    """The arguments of every call of one FEMM function, numbers as floats."""
    found = []
    for text in re.findall(rf"^\s*{name}\((.*)\)$", script, flags=re.MULTILINE):
        row = []
        for item in re.findall(r'"[^"]*"|[^,\s]+', text):
            row.append(item[1:-1] if item.startswith('"') else float(item))
        found.append(row)
    return found


def labels(script: str) -> list[dict]:
    """Every block label: where it is and what it was given."""
    pattern = (
        r"mi_addblocklabel\((.*), (.*)\)\nmi_selectlabel\(\1, \2\)\n"
        r'mi_setblockprop\("(.*)", 0, (.*), "(.*)", (.*), (.*), (.*)\)\nmi_clearselected\(\)'
    )
    return [
        {
            "point": (float(x) * 1e-3, float(y) * 1e-3),
            "material": material,
            "size": float(size) * 1e-3,
            "circuit": circuit,
            "direction": float(direction),
            "group": int(group),
            "turns": int(turns),
        }
        for x, y, material, size, circuit, direction, group, turns in re.findall(pattern, script)
    ]


def contains(region: Region, point: tuple) -> bool:
    outline = region.polygon(1e-7)
    x, y = point
    inside = False
    for (x0, y0), (x1, y1) in zip(outline, np.roll(outline, -1, axis=0)):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


# The drawing


def test_lengths_are_in_millimetres_and_the_depth_is_the_stack_length(script: str, geometry) -> None:
    (definition,) = calls(script, "mi_probdef")
    assert definition[:3] == [0.0, "millimeters", "planar"]
    assert definition[4] == pytest.approx(12.0)  # the stack length of the design
    assert definition[3] == 1e-8 and definition[5] == 30.0  # solver precision, smallest angle
    radius = [math.hypot(x, y) for x, y in calls(script, "mi_addnode")]
    assert min(radius) == pytest.approx(geometry.inner_radius * 1e3, rel=1e-9)
    assert max(radius) == pytest.approx(geometry.outer_radius * 1e3, rel=1e-9)
    assert max(radius) == pytest.approx(45.0)


def test_every_corner_and_edge_is_drawn_once(script: str) -> None:
    nodes = [tuple(row) for row in calls(script, "mi_addnode")]
    assert len(set(nodes)) == len(nodes)
    lines = [frozenset([tuple(row[:2]), tuple(row[2:])]) for row in calls(script, "mi_addsegment")]
    arcs = [(tuple(row[:2]), tuple(row[2:4])) for row in calls(script, "mi_addarc")]
    assert len(set(lines)) == len(lines)
    assert len(set(arcs)) == len(arcs) and not {(b, a) for a, b in arcs} & set(arcs)
    # Every edge ends on drawn corners.
    ends = {point for line in lines for point in line} | {point for arc in arcs for point in arc}
    assert ends == set(nodes)


def test_the_drawing_encloses_one_area_for_each_region_the_band_and_the_bore(
    script: str, geometry
) -> None:
    # Euler's rule for a flat drawing in two separate pieces (stator and
    # rotor, with the band between): closed areas = edges - corners + 2.
    corners = len(calls(script, "mi_addnode"))
    edges = len(calls(script, "mi_addsegment")) + len(calls(script, "mi_addarc"))
    assert edges - corners + 2 == len(geometry.regions) + 2
    assert len(labels(script)) == len(geometry.regions) + 2


def test_no_corner_lies_partway_along_an_edge(script: str) -> None:
    # FEMM would not join an arc to a corner in its middle.
    nodes = np.array(calls(script, "mi_addnode"))
    radius = np.hypot(nodes[:, 0], nodes[:, 1])
    angle = np.arctan2(nodes[:, 1], nodes[:, 0])
    for x0, y0, x1, y1, sweep, _ in calls(script, "mi_addarc"):
        arc_radius = math.hypot(x0, y0)
        assert math.hypot(x1, y1) == pytest.approx(arc_radius, rel=1e-9)  # about the origin
        turned = (angle - math.atan2(y0, x0)) % (2.0 * math.pi)
        within = (np.abs(radius - arc_radius) < 1e-6) & (turned > 1e-7)
        assert not np.any(within & (turned < math.radians(sweep) - 1e-7))


def test_arcs_turn_counter_clockwise_through_less_than_half_a_turn(script: str, geometry) -> None:
    band = geometry.band_outer_radius - geometry.band_inner_radius
    for x0, y0, x1, y1, sweep, chord in calls(script, "mi_addarc"):
        turned = math.degrees((math.atan2(y1, x1) - math.atan2(y0, x0)) % (2.0 * math.pi))
        assert sweep == pytest.approx(turned, abs=1e-6)
        assert 0.0 < sweep < 180.0
        # Chords as long as the triangles beside them: fine near the gap.
        radius = math.hypot(x0, y0) * 1e-3
        near = geometry.band_inner_radius - 2 * band <= radius <= geometry.band_outer_radius + 2 * band
        size = (0.5 if near else 4.0) * band
        assert chord == pytest.approx(min(10.0, max(0.02, math.degrees(size / radius))), abs=1e-4)


def test_the_mesh_sizes_are_left_to_decide_the_mesh(script: str) -> None:
    line = "if mi_smartmesh then mi_smartmesh(0) end"
    assert line in script
    assert script.index(line) < script.index("mi_addnode")


def test_the_inner_and_outer_surface_are_field_lines_and_nothing_else_is(
    script: str, geometry
) -> None:
    assert calls(script, "mi_addboundprop") == [["field line"] + [0.0] * 9]
    pattern = (
        r"mi_addarc\((.*), (.*), .*, .*, (.*), .*\)\nmi_selectarcsegment\(.*\)\n"
        r'mi_setarcsegmentprop\(.*, "(.*)", 0, 0\)'
    )
    swept = {geometry.inner_radius: 0.0, geometry.outer_radius: 0.0}
    for x, y, sweep, boundary in re.findall(pattern, script):
        radius = math.hypot(float(x), float(y)) * 1e-3
        on_surface = [r for r in swept if math.isclose(radius, r, rel_tol=1e-9)]
        assert (boundary == "field line") == bool(on_surface)
        assert boundary in ("field line", "")
        if on_surface:
            swept[on_surface[0]] += float(sweep)
    # Both surfaces all the way round.
    assert swept[geometry.inner_radius] == pytest.approx(360.0)
    assert swept[geometry.outer_radius] == pytest.approx(360.0)


def test_arcs_are_selected_at_their_own_middle(script: str) -> None:
    pattern = r"mi_addarc\((.*), (.*), (.*), (.*), (.*), .*\)\nmi_selectarcsegment\((.*), (.*)\)"
    found = re.findall(pattern, script)
    assert len(found) == len(calls(script, "mi_addarc"))
    for row in found:
        x0, y0, x1, y1, sweep, x, y = map(float, row)
        assert math.hypot(x, y) == pytest.approx(math.hypot(x0, y0), rel=1e-8)
        half = (math.atan2(y, x) - math.atan2(y0, x0)) % (2.0 * math.pi)
        assert math.degrees(half) == pytest.approx(0.5 * sweep, abs=1e-5)


# Labels: materials, coils and magnets


def test_every_region_has_one_label_inside_it(script: str, geometry) -> None:
    found = labels(script)[: len(geometry.regions)]
    for region, label in zip(geometry.regions, found):
        assert label["material"] == region.material, region.name
        assert contains(region, label["point"]), region.name
    # No label has strayed into a neighbour: spot-check each against a few others.
    regions = geometry.regions
    for number, label in enumerate(found):
        for other in (regions[number - 1], regions[(number + 1) % len(regions)]):
            assert not contains(other, label["point"])


def test_labels_keep_clear_of_the_outline(geometry) -> None:
    for region in geometry.regions:
        x, y = inside_point(region)
        outline = region.polygon(1e-7)
        nearest = np.hypot(outline[:, 0] - x, outline[:, 1] - y).min()
        assert nearest > 0.03 * math.sqrt(region.area()), region.name


def test_the_band_is_air_and_the_bore_is_left_out(script: str, geometry) -> None:
    band, bore = labels(script)[len(geometry.regions) :]
    radius = math.hypot(*band["point"])
    assert geometry.band_inner_radius < radius < geometry.band_outer_radius
    assert band["material"] == "air" and band["group"] == 0 and band["circuit"] == "<None>"
    width = geometry.band_outer_radius - geometry.band_inner_radius
    assert band["size"] == pytest.approx(0.5 * width, rel=1e-5)
    # Halfway between two of the 360 points the field is read at.
    assert math.atan2(band["point"][1], band["point"][0]) == pytest.approx(math.pi / 360, rel=1e-6)
    assert bore["point"] == (0.0, 0.0) and bore["material"] == "<No Mesh>"


def test_stator_and_rotor_are_two_groups_and_torque_is_taken_on_the_rotor(
    script: str, geometry
) -> None:
    found = labels(script)[: len(geometry.regions)]
    groups = [label["group"] for label in found]
    assert groups == [1] * len(geometry.stator) + [2] * len(geometry.rotor)
    assert calls(script, "mo_groupselectblock") == [[2.0]] * 2  # once for each case
    assert script.count("mo_blockintegral(22)") == 2


def test_coil_sides_carry_their_turns_in_their_phase(script: str, geometry) -> None:
    found = labels(script)[: len(geometry.regions)]
    total = {"A": 0, "B": 0, "C": 0}
    sides = 0
    for region, label in zip(geometry.regions, found):
        if region.coil_side is None:
            assert label["circuit"] == "<None>" and label["turns"] == 0
            continue
        sides += 1
        assert label["circuit"] == region.coil_side.phase
        assert label["turns"] == region.coil_side.sign * region.coil_side.turns
        assert abs(label["turns"]) == 20
        total[label["circuit"]] += label["turns"]
    assert sides == 72  # two sides for each of 36 teeth
    assert total == {"A": 0, "B": 0, "C": 0}  # every coil goes out and comes back
    assert calls(script, "mi_addcircprop") == [["A", 0.0, 1.0], ["B", 0.0, 1.0], ["C", 0.0, 1.0]]


def test_magnets_point_along_their_own_axis_out_and_in_by_turns(script: str, geometry) -> None:
    found = labels(script)[: len(geometry.regions)]
    magnets = [
        (region, label)
        for region, label in zip(geometry.regions, found)
        if region.magnetisation is not None
    ]
    assert len(magnets) == 42
    for number, (region, label) in enumerate(magnets):
        axis = math.degrees(region.magnetisation.axis_angle)
        expected = axis if number % 2 == 0 else axis + 180.0
        assert (label["direction"] - expected) % 360.0 == pytest.approx(0.0, abs=1e-6) or (
            label["direction"] - expected
        ) % 360.0 == pytest.approx(360.0, abs=1e-6)
        assert label["material"] == "N42SH"
    assert magnets[0][1]["direction"] == pytest.approx(0.0, abs=1e-9)  # magnet 0, outwards along x


def test_air_is_meshed_finer_than_the_rest_and_fineness_divides_both(
    geometry, materials, cases
) -> None:
    band = geometry.band_outer_radius - geometry.band_inner_radius
    for fineness in (1.0, 2.0):
        text = femm_script(geometry, materials, cases, "r.txt", fineness=fineness, gap_points=16)
        for region, label in zip(geometry.regions, labels(text)):
            factor = 0.5 if region.kind == "air" else 4.0
            assert label["size"] == pytest.approx(factor * band / fineness, rel=1e-5)


# Materials


def test_the_steel_curve_is_the_table_of_the_material_file_point_for_point(
    script: str, package_dir: Path
) -> None:
    path = package_dir / "data" / "materials" / "generic_35A_placeholder.yaml"
    table = yaml.safe_load(path.read_text(encoding="utf-8"))["bh_curve"]
    points = calls(script, "mi_addbhpoint")
    assert len(points) == len(table) == 25
    for (name, b, h), (b_file, h_file) in zip(points, table):
        assert name == "M270-35A"
        assert b == b_file and h == h_file  # the very numbers, not rounded ones
    # And they are what the solver's own curve holds.
    curve = load_material("generic_35A_placeholder")
    assert [row[1] for row in points] == list(curve.b)
    assert [row[2] for row in points] == list(curve.h)


def test_the_magnet_has_the_remanence_and_permeability_of_its_model(script: str) -> None:
    by_name = {row[0]: row for row in calls(script, "mi_addmaterial")}
    assert sorted(by_name) == ["M270-35A", "N42SH", "air", "copper"]
    magnet = by_name["N42SH"]
    assert magnet[1] == magnet[2] == MU_R
    # FEMM: B = mu_r mu0 (H + Hc), which is Br at H = 0.
    assert MU_R * MU0 * magnet[3] == pytest.approx(BR, rel=1e-14)
    for name in ("air", "copper"):
        assert by_name[name][1:4] == [1.0, 1.0, 0.0]
    assert by_name["M270-35A"][3] == 0.0


def test_no_material_carries_a_current_or_a_conductivity_of_its_own(script: str) -> None:
    for row in calls(script, "mi_addmaterial"):
        assert row[4:8] == [0.0, 0.0, 0.0, 0.0]  # J, conductivity, lamination, hysteresis
        assert row[8] == 1.0  # fill factor


# The cases and what is written


def test_each_case_sets_the_phase_currents_then_solves(script: str, cases: list) -> None:
    changed = calls(script, "mi_modifycircprop")
    assert len(changed) == 3 * len(cases)
    for number, case in enumerate(cases):
        for (phase, what, amps), name in zip(changed[3 * number : 3 * number + 3], "ABC"):
            assert phase == name and what == 1.0
            assert amps == case[name]
    assert script.count("mi_analyze(1)") == script.count("mi_loadsolution()") == len(cases)
    assert script.count("mo_close()") == len(cases)
    pieces = re.split(r'write\(out, "case \d+', script)[1:]
    assert len(pieces) == len(cases)
    for piece in pieces:
        order = [
            piece.rindex("mi_modifycircprop("),
            piece.index("mi_analyze(1)"),
            piece.index("mi_loadsolution()"),
            piece.index("mo_groupselectblock("),
            piece.index("mo_close()"),
        ]
        assert order == sorted(order)
    assert script.splitlines()[-4:] == ['write(out, "end\\n")', "closefile(out)", "mi_close()", "quit()"]
    opened = [line for line in script.splitlines() if line.startswith("out = openfile(")]
    assert len(opened) == 1 and opened[0].endswith('", "w")')


def test_the_model_is_saved_before_it_is_solved(script: str) -> None:
    assert script.index("mi_saveas(") < script.index("mi_analyze(1)")
    (saved,) = calls(script, "mi_saveas")
    assert saved[0].endswith("results.fem") and "\\" not in saved[0]


def test_gap_points_are_on_the_middle_circle_all_the_way_round(script: str, geometry) -> None:
    x = [float(v) for v in re.search(r"^gx = \{(.*)\}$", script, flags=re.MULTILINE)[1].split(",")]
    y = [float(v) for v in re.search(r"^gy = \{(.*)\}$", script, flags=re.MULTILINE)[1].split(",")]
    assert len(x) == len(y) == 360
    middle = 0.5 * (geometry.band_inner_radius + geometry.band_outer_radius) * 1e3
    assert np.hypot(x, y) == pytest.approx(middle, rel=1e-8)
    angles = np.unwrap(np.arctan2(y, x))
    assert angles == pytest.approx(np.radians(np.arange(360)), abs=1e-8)
    assert "for k = 1, 360 do" in script


# What is refused


def test_a_slice_is_refused(machine, materials, cases) -> None:
    with pytest.raises(ValueError, match="whole machine"):
        femm_script(machine.build_geometry(), materials, cases, "r.txt")


def test_bad_requests_are_refused(geometry, materials, cases) -> None:
    with pytest.raises(ValueError, match="at least one case"):
        femm_script(geometry, materials, [], "r.txt")
    with pytest.raises(ValueError, match="each phase"):
        femm_script(geometry, materials, [{"A": 1.0, "B": 0.0}], "r.txt")
    with pytest.raises(ValueError, match="no material model for 'N42SH'"):
        femm_script(geometry, {"M270-35A": materials["M270-35A"]}, cases, "r.txt")
    with pytest.raises(ValueError, match="fineness"):
        femm_script(geometry, materials, cases, "r.txt", fineness=0.0)
    with pytest.raises(ValueError, match="gap_points"):
        femm_script(geometry, materials, cases, "r.txt", gap_points=4)
    with pytest.raises(ValueError, match="surround"):
        femm_script(geometry, materials, cases, "r.txt", surround=1.0)


def test_radial_magnets_are_refused_for_now(geometry, materials, cases) -> None:
    radial = tuple(
        replace(region, magnetisation=replace(region.magnetisation, pattern="radial"))
        if region.magnetisation is not None
        else region
        for region in geometry.rotor
    )
    with pytest.raises(ValueError, match="only parallel magnetisation"):
        femm_script(replace(geometry, rotor=radial), materials, cases, "r.txt")


def test_a_name_that_would_break_the_script_is_refused(geometry, materials, cases) -> None:
    quoted = 'N42 "hot"'
    rotor = tuple(
        replace(region, material=quoted) if region.magnetisation is not None else region
        for region in geometry.rotor
    )
    with pytest.raises(ValueError, match="cannot hold a double quote"):
        femm_script(
            replace(geometry, rotor=rotor), {**materials, quoted: materials["N42SH"]}, cases, "r.txt"
        )


def solid_machine() -> Geometry:
    """A steel disc inside a steel ring, with a band of air between: no bore."""

    def half_disc(start: float) -> tuple:
        return closed_loop(
            [(0.0, 0.0, "line"), (0.010, start, "ccw"), (0.010, start + math.pi, "line")]
        )

    def steel(name: str, boundary: tuple) -> Region:
        return Region(name=name, kind="steel", role="test", material="iron", boundary=boundary)

    return Geometry(
        stator=(steel("disc_0", half_disc(0.0)), steel("disc_1", half_disc(math.pi))),
        rotor=(
            steel("ring_0", ring_sector(0.011, 0.015, 0.0, math.pi)),
            steel("ring_1", ring_sector(0.011, 0.015, math.pi, 2.0 * math.pi)),
        ),
        inner_radius=0.0,
        outer_radius=0.015,
        band_inner_radius=0.010,
        band_outer_radius=0.011,
        stator_start_angle=0.0,
        rotor_start_angle=0.0,
        span_angle=2.0 * math.pi,
        antiperiodic=False,
        stack_length=0.01,
    )


@pytest.mark.parametrize("surround", [None, 1.5])
def test_a_machine_with_no_bore_gets_no_label_at_its_centre(surround: float | None) -> None:
    text = femm_script(
        solid_machine(), {"iron": LinearMaterial(1000.0)}, [{}], "r.txt", gap_points=16, surround=surround
    )
    found = labels(text)
    assert all(math.hypot(*label["point"]) > 1e-4 for label in found)
    assert len(found) == (6 if surround else 5)  # four regions, the band, and the air outside
    assert "<No Mesh>" not in text


def test_an_edge_too_short_to_draw_is_refused() -> None:
    sliver = Region(
        name="sliver",
        kind="air",
        role="test",
        material="air",
        boundary=(
            Edge((0.0, 0.0), (0.001, 0.0)),
            Edge((0.001, 0.0), (0.001, 5e-10)),
            Edge((0.001, 5e-10), (0.0, 0.0)),
        ),
    )
    with pytest.raises(ValueError, match="sliver has an edge shorter than"):
        _edges([sliver])


def test_an_arc_shared_by_two_regions_is_listed_once_whatever_its_centre() -> None:
    # The same arc about a point off the origin, worked out twice: its
    # centre differs in the last digit, and it runs the other way.
    centre, other = (0.02, 0.01), (0.02 + 4e-18, 0.01 - 2e-18)
    start, end = (0.03, 0.01), (0.02, 0.02)

    def region(name: str, edges: tuple) -> Region:
        return Region(name=name, kind="air", role="test", material="air", boundary=edges)

    inside = region("inside", (Edge(centre, start), Edge(start, end, centre), Edge(end, centre)))
    outside = region(
        "outside",
        (Edge(start, (0.03, 0.02)), Edge((0.03, 0.02), end), Edge(end, start, other, ccw=False)),
    )
    arcs = [edge for edge in _edges([inside, outside]) if edge[2] is not None]
    assert len(arcs) == 1
    assert arcs[0][:2] == (start, end)


def test_a_steel_given_by_a_formula_is_refused(geometry, materials, cases) -> None:
    # FEMM takes a table; a formula would have to be turned into one first.
    formula = {**materials, "M270-35A": FrohlichSteel(saturation=2.0, half_field=300.0)}
    with pytest.raises(ValueError, match="linear, a magnet or a BH table, not FrohlichSteel"):
        femm_script(geometry, formula, cases, "r.txt")


def test_a_linear_steel_is_written_with_its_permeability(geometry, materials, cases) -> None:
    linear = {**materials, "M270-35A": LinearMaterial(2000.0)}
    text = femm_script(geometry, linear, cases, "r.txt", gap_points=16)
    by_name = {row[0]: row for row in calls(text, "mi_addmaterial")}
    assert by_name["M270-35A"][1:3] == [2000.0, 2000.0]
    assert calls(text, "mi_addbhpoint") == []


# Air round the machine


def test_with_air_round_it_the_machine_surfaces_are_free(geometry, materials, cases) -> None:
    text = femm_script(geometry, materials, cases, "r.txt", gap_points=16, surround=1.5)
    pattern = (
        r"mi_addarc\((.*), (.*), .*, .*, .*, .*\)\nmi_selectarcsegment\(.*\)\n"
        r'mi_setarcsegmentprop\(.*, "(.*)", 0, 0\)'
    )
    held = {round(math.hypot(float(x), float(y)), 6) for x, y, name in re.findall(pattern, text) if name}
    assert held == {round(1.5 * geometry.outer_radius * 1e3, 6)}
    extra = labels(text)[len(geometry.regions) + 1 :]
    assert [label["material"] for label in extra] == ["air", "air"]
    band = geometry.band_outer_radius - geometry.band_inner_radius
    assert [label["size"] for label in extra] == pytest.approx([8.0 * band] * 2, rel=1e-5)
    assert labels(text)[len(geometry.regions)]["size"] == pytest.approx(0.5 * band, rel=1e-5)
    assert geometry.outer_radius < math.hypot(*extra[0]["point"]) < 1.5 * geometry.outer_radius
    assert extra[1]["point"] == (0.0, 0.0)
    assert "<No Mesh>" not in text


# Reading results back


RESULTS = """radius_mm 42.300000000
case 0
nodes 1200
elements 2300
torque 0.25
circuit A 0 0.01
circuit B 0 -0.005
b 0.5 0
b 0 0.25
b -0.5 0
b 0.1 -0.25
case 1
nodes 1200
elements 2300
torque 3.5
circuit A 10 0.008
circuit B -5 0.004
b 0 0
b 0 0
b 0 0
b 0 0
end
"""


def test_results_are_read_back(tmp_path: Path) -> None:
    path = tmp_path / "results.txt"
    path.write_text(RESULTS, encoding="utf-8")
    first, second = read_results(path)
    assert first["torque"] == 0.25 and second["torque"] == 3.5
    assert first["nodes"] == 1200 and first["elements"] == 2300
    assert first["flux_linkage"] == {"A": 0.01, "B": -0.005}
    assert second["current"] == {"A": 10.0, "B": -5.0}
    assert first["radius"] == pytest.approx(0.0423)
    assert first["angles"] == pytest.approx([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi])
    # (Bx, By) turned into the part along the radius and the part round it.
    assert first["radial"] == pytest.approx([0.5, 0.25, 0.5, 0.25])
    assert first["tangential"] == pytest.approx([0.0, 0.0, 0.0, 0.1], abs=1e-15)


def test_results_that_stop_early_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "results.txt"
    path.write_text(RESULTS.replace("end\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        read_results(path)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        read_results(path)
    path.write_text(RESULTS.replace("torque 3.5\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="no torque"):
        read_results(path)
    path.write_text(RESULTS.replace("nodes 1200", "knots 1200", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown line 'knots 1200'"):
        read_results(path)


def test_the_script_writes_what_the_reader_reads(script: str) -> None:
    written = set(re.findall(r'write\(out, (?:format\()?"(\w+)', script))
    assert written == {"radius_mm", "case", "nodes", "elements", "torque", "circuit", "b", "end"}


# Turning the rotor

TURNS = [0.0, 0.05, 0.05, 0.3, -0.2]


@pytest.fixture(scope="module")
def turning(geometry, materials, machine) -> str:
    cases = [machine.excitation(angle, 0.0) for angle in TURNS]
    return femm_script(geometry, materials, cases, "r.txt", gap_points=8, rotor_angles=TURNS)


def per_case(script: str) -> list[str]:
    """The script split at each case, first piece (the drawing) dropped."""
    return re.split(r'write\(out, "case \d+\\n"\)\n', script)[1:]


def test_without_rotor_angles_nothing_is_turned(script: str) -> None:
    for name in ("mi_selectcircle", "mi_setgroup", "mi_selectgroup", "mi_moverotate"):
        assert name not in script
    assert "rotor_angle" not in script


def test_everything_past_the_middle_of_the_gap_becomes_the_rotor_group(turning: str, geometry) -> None:
    # Outer rotor: all of it first, then what lies inside the middle circle
    # is given back to the stator.
    middle = 0.5 * (geometry.band_inner_radius + geometry.band_outer_radius)
    circles = calls(turning, "mi_selectcircle")
    assert calls(turning, "mi_setgroup")[:2] == [[2.0], [1.0]]
    assert circles[0][:2] == [0.0, 0.0] and circles[0][2] > geometry.outer_radius * 1e3
    assert circles[1] == pytest.approx([0.0, 0.0, middle * 1e3, 4.0])
    assert all(circle[3] == 4.0 for circle in circles)  # nodes, edges, arcs and labels
    # Before the model is saved and solved.
    assert turning.index("mi_setgroup") < turning.index("mi_saveas")


@pytest.mark.parametrize("surround", [None, 1.5])
def test_labels_of_no_region_stay_in_no_group_when_the_rotor_turns(
    surround: float | None, geometry, materials, machine
) -> None:
    # The band's label lies on the dividing circle. In the rotor group it
    # would add the band to the torque integral (8 mN m at angle 0 when it did).
    cases = [machine.excitation(0.0, 0.0)] * 2
    text = femm_script(
        geometry, materials, cases, "r.txt", gap_points=8, surround=surround, rotor_angles=[0.0, 0.1]
    )
    regionless = [
        (label["point"][0] * 1e3, label["point"][1] * 1e3)
        for label in labels(text)[len(geometry.regions) :]
    ]
    assert len(regionless) == (3 if surround else 2)  # band, bore, and the air outside
    drawing = text[: text.index("mi_saveas")]
    resets = re.findall(r"mi_selectlabel\((.*), (.*)\)\nmi_setgroup\(0\)", drawing)
    assert [(float(x), float(y)) for x, y in resets] == regionless
    # After the circles that set the groups.
    assert drawing.rindex("mi_selectcircle") < drawing.index("mi_setgroup(0)")
    assert calls(drawing, "mi_setgroup") == [[2.0], [1.0]] + [[0.0]] * len(regionless)


def test_each_case_turns_the_rotor_by_what_it_still_needs(turning: str) -> None:
    turned, previous = 0.0, 0.0
    for angle, text in zip(TURNS, per_case(turning)):
        steps = calls(text, "mi_moverotate")
        if angle == previous:
            assert steps == []
        else:
            assert len(steps) == 1 and steps[0][:2] == [0.0, 0.0]
            turned += math.radians(steps[0][2])
            assert calls(text, "mi_selectgroup") == [[2.0]]
        assert turned == pytest.approx(angle, abs=1e-12)
        previous = angle
        assert f'"rotor_angle {angle!r}\\n"' in text
        # The rotor is turned before the case is solved.
        if steps:
            assert text.index("mi_moverotate") < text.index("mi_analyze")


def test_magnets_are_turned_with_the_rotor(turning: str, geometry) -> None:
    magnets = [r for r in geometry.rotor if r.magnetisation is not None]
    first = {
        label["point"]: label["direction"]
        for label in labels(turning)[: len(geometry.regions)]
        if label["material"] == "N42SH"
    }
    pieces = per_case(turning)
    assert len(pieces) == len(TURNS)
    for angle, text in zip(TURNS[1:], pieces[1:]):
        picked = calls(text, "mi_selectlabel")
        props = calls(text, "mi_setblockprop")
        if "mi_moverotate" not in text:
            assert picked == [] and props == []  # the same angle again: nothing to do
            continue
        assert len(picked) == len(props) == len(magnets) == 42
        # Picked where the labels are after the turn, and before the solve.
        assert text.index("mi_moverotate") < text.index("mi_selectlabel") < text.index("mi_analyze")
        c, s = math.cos(angle), math.sin(angle)
        for (x, y), row in zip(picked, props):
            # Back to where the label was drawn, and its direction then.
            px, py = (c * x + s * y) * 1e-3, (-s * x + c * y) * 1e-3
            match = min(first, key=lambda p: math.dist(p, (px, py)))
            assert math.dist(match, (px, py)) < 1e-9
            assert row[0] == "N42SH" and row[3] == "<None>" and row[5:] == [2.0, 0.0]
            assert row[4] == pytest.approx(first[match] + math.degrees(angle), abs=1e-6)


def test_rotor_angles_must_match_the_cases(geometry, materials, cases) -> None:
    for angles in ([0.0], [0.0, 0.1, 0.2], [0.0, math.nan]):
        with pytest.raises(ValueError, match="rotor_angles"):
            femm_script(geometry, materials, cases, "r.txt", gap_points=8, rotor_angles=angles)


def test_the_turned_script_writes_what_the_reader_reads(turning: str, tmp_path: Path) -> None:
    written = set(re.findall(r'write\(out, (?:format\()?"(\w+)', turning))
    assert written == {
        "radius_mm", "case", "rotor_angle", "nodes", "elements", "torque", "circuit", "b", "end"
    }
    path = tmp_path / "results.txt"
    path.write_text(RESULTS.replace("case 1\n", "case 1\nrotor_angle 0.25\n"), encoding="utf-8")
    first, second = read_results(path)
    assert first["rotor_angle"] == 0.0 and second["rotor_angle"] == 0.25


# A point inside a region


def test_inside_point_of_a_shape_whose_middle_is_outside_it() -> None:
    # A thin arc of ring: its centroid's neighbourhood is fine, but the
    # middle of its corners is not inside it.
    arc = Region(
        name="arc",
        kind="air",
        role="test",
        material="air",
        boundary=closed_loop(
            [(0.040, 0.0, "line"), (0.041, 0.0, "ccw"), (0.041, 1.9, "line"), (0.040, 1.9, "cw")]
        ),
    )
    corners = np.array([edge.start for edge in arc.boundary])
    assert not contains(arc, tuple(corners.mean(axis=0)))
    x, y = inside_point(arc)
    assert contains(arc, (x, y))
    assert 0.0402 < math.hypot(x, y) < 0.0408


# The solver against stored FEMM results
#
# FEMM solved the same model: the reference design at rotor angle 0, whole
# machine, with the placeholder steel table and magnet. Its results are in
# tests/data/femm/, made by make_reference.py there. The limits below are
# what the two programs were found to agree within, with room to spare; they
# were not adjusted to make anything pass.

TORQUE_LIMIT = 0.01  # of FEMM's torque
LINKAGE_LIMIT = 0.01  # of the largest phase linkage of the case
GAP_LIMIT = 0.01  # of the main wave of the gap field
REFINEMENT_LIMIT = 0.005  # change of our torque when the mesh is made twice as fine
GAP_SAMPLES = 14400


@pytest.fixture(scope="module")
def reference() -> dict:
    import json

    return json.loads((ROOT / "tests" / "data" / "femm" / "qdd_outer_36s42p.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def solved(machine, materials, reference) -> dict:
    """Our results for the stored cases: on the slice, at mesh sizes 1 and 0.5."""
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.core.post import flux_linkage, gap_flux_density, torque
    from polecraft.core.problem import build_problem
    from polecraft.core.solver import solve

    # Our gap field steps from triangle to triangle, so its main wave is
    # taken from many more points than there are triangles round the gap.
    angles = 2.0 * math.pi * np.arange(GAP_SAMPLES) / GAP_SAMPLES
    found: dict = {}
    for size in (1.0, 0.5):
        mesh = build_mesh(machine.build_geometry(), size=size)
        for case in reference["runs"]["fine"]["cases"]:
            problem = build_problem(machine, mesh, materials, current=case["current"])
            solution = solve(problem)
            found[size, case["current"]] = {
                "torque": torque(problem, solution),
                "flux_linkage": flux_linkage(problem, solution),
                "radial": gap_flux_density(problem, solution, angles)[0] if size == 1.0 else None,
                "middle_radius": 0.5 * (problem.gap.inner_radius + problem.gap.outer_radius),
            }
    return found


def main_wave(radial: np.ndarray) -> complex:
    """The part of the gap field with one wave for each of the 21 pole pairs."""
    return 2.0 * np.fft.fft(radial)[21] / len(radial)


def test_the_reference_was_made_with_the_materials_the_tests_use(reference: dict) -> None:
    # If the material file changes, FEMM has to be run again.
    curve = load_material(reference["steel"]["name"])
    assert reference["steel"]["name"] == "generic_35A_placeholder"
    assert [row[0] for row in reference["steel"]["bh_curve"]] == list(curve.b)
    assert [row[1] for row in reference["steel"]["bh_curve"]] == list(curve.h)
    assert reference["magnet"] == {"remanence": BR, "relative_permeability": MU_R}
    assert (ROOT / reference["made_by"]).is_file()


def test_the_reference_was_made_for_the_design_as_it_is_now(reference: dict) -> None:
    # If the design file changes, FEMM has to be run again.
    design = load_design(ROOT / reference["design"])
    assert reference["design_values"] == design_to_dict(design)


def test_the_reference_cases_carry_the_currents_the_machine_asks_for(
    reference: dict, machine
) -> None:
    for run in reference["runs"].values():
        assert [case["current"] for case in run["cases"]] == [0.0, 10.0, 40.0]
        for case in run["cases"]:
            wanted = machine.excitation(reference["rotor_angle"], case["current"], reference["advance"])
            assert case["phase_currents"] == pytest.approx(wanted, abs=1e-9)


def test_femm_itself_has_settled(reference: dict) -> None:
    # Its finer mesh has more than three times the nodes and gives the same answers.
    normal, fine = reference["runs"]["normal"], reference["runs"]["fine"]
    assert normal["nodes"] > 50_000 and fine["nodes"] > 3 * normal["nodes"]
    assert (normal["fineness"], fine["fineness"]) == (1.0, 2.0)
    for coarse, finer in zip(normal["cases"][1:], fine["cases"][1:]):
        assert coarse["torque"] == pytest.approx(finer["torque"], rel=0.005)
        assert coarse["gap_fundamental"] == pytest.approx(finer["gap_fundamental"], rel=0.005)


@pytest.mark.parametrize("current", [10.0, 40.0])
def test_torque_agrees_with_femm(reference: dict, solved: dict, current: float) -> None:
    (femm,) = [c for c in reference["runs"]["fine"]["cases"] if c["current"] == current]
    assert femm["torque"] > 3.0
    assert solved[1.0, current]["torque"] == pytest.approx(femm["torque"], rel=TORQUE_LIMIT)
    assert solved[0.5, current]["torque"] == pytest.approx(femm["torque"], rel=TORQUE_LIMIT)


def test_neither_program_finds_torque_without_current(reference: dict, solved: dict) -> None:
    # At this rotor position the motor is its own mirror image.
    at_ten = reference["runs"]["fine"]["cases"][1]["torque"]
    assert abs(reference["runs"]["fine"]["cases"][0]["torque"]) < 0.002 * at_ten
    assert abs(solved[1.0, 0.0]["torque"]) < 0.002 * at_ten


@pytest.mark.parametrize("current", [0.0, 10.0, 40.0])
def test_flux_linkage_agrees_with_femm(reference: dict, solved: dict, current: float) -> None:
    (femm,) = [c for c in reference["runs"]["fine"]["cases"] if c["current"] == current]
    ours = solved[1.0, current]["flux_linkage"]
    largest = max(abs(value) for value in femm["flux_linkage"].values())
    assert largest > 5e-3
    for phase in "ABC":
        assert abs(ours[phase] - femm["flux_linkage"][phase]) < LINKAGE_LIMIT * largest, phase


@pytest.mark.parametrize("current", [0.0, 10.0, 40.0])
def test_gap_flux_density_agrees_with_femm(reference: dict, solved: dict, current: float) -> None:
    (femm,) = [c for c in reference["runs"]["fine"]["cases"] if c["current"] == current]
    assert reference["runs"]["fine"]["gap_radius"] == pytest.approx(
        solved[1.0, current]["middle_radius"], rel=1e-9
    )
    theirs = main_wave(np.array(femm["gap_radial"]))
    ours = main_wave(solved[1.0, current]["radial"])
    assert abs(theirs) == pytest.approx(femm["gap_fundamental"], rel=1e-4)
    assert 0.9 < abs(theirs) < 1.1  # tesla
    # Size and position of the wave both.
    assert abs(ours - theirs) < GAP_LIMIT * abs(theirs)


@pytest.mark.parametrize("current", [10.0, 40.0])
def test_a_finer_mesh_barely_changes_the_torque(solved: dict, current: float) -> None:
    normal, fine = solved[1.0, current]["torque"], solved[0.5, current]["torque"]
    assert fine == pytest.approx(normal, rel=REFINEMENT_LIMIT)


def test_air_round_the_machine_makes_no_difference_that_matters(reference: dict) -> None:
    # The solver ends the model at the inner and outer surface of the iron.
    # FEMM, given air inside the bore and out to 1.5 times the outer radius,
    # finds the same torque and flux linkage: at these currents the yokes of
    # this design let almost no flux out.
    closed, open_ = reference["runs"]["normal"], reference["runs"]["air_outside"]
    assert open_["air_round_the_machine"] == 1.5 and closed["air_round_the_machine"] is None
    for inside, outside in zip(closed["cases"][1:], open_["cases"][1:]):
        assert outside["torque"] == pytest.approx(inside["torque"], rel=0.002)
        for phase in "ABC":
            assert outside["flux_linkage"][phase] == pytest.approx(
                inside["flux_linkage"][phase], abs=0.002 * 0.0166
            )
