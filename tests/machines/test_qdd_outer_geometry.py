"""Outer-rotor geometry: areas against hand formulas, tiling, labels."""

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from matplotlib.path import Path as Outline
from scipy.spatial import cKDTree

from polecraft.core.geometry import Geometry, Region
from polecraft.core.winding import concentrated_winding, symmetry
from polecraft.design import Design, design_from_dict, design_to_dict, load_design
from polecraft.machines import machine_for

MM2 = 1e-6  # square millimetres in square metres

# With fewer slots the tips overhang further, and the reference taper becomes
# too shallow; a square tip fits any slot count.
SQUARE_TIP = {"geometry__stator__tooth_tip_taper_mm": 0}


@pytest.fixture(scope="module")
def reference() -> Design:
    root = Path(__file__).resolve().parents[2]
    return load_design(root / "designs" / "qdd_outer_36s42p.yaml")


def variant(design: Design, **changes: object) -> Design:
    """The design with some values replaced.

    Sections are joined with a double underscore, for example
    ``geometry__magnet__arc_fraction=1``.
    """
    data = design_to_dict(design)
    for path, value in changes.items():
        *sections, key = path.split("__")
        block = data
        for name in sections:
            block = block[name]
        block[key] = value
    return design_from_dict(data)


def build(design: Design, full: bool = False) -> Geometry:
    return machine_for(design).build_geometry(full=full)


def by_role(regions: tuple[Region, ...], role: str) -> list[Region]:
    return [region for region in regions if region.role == role]


def frames_of(geometry: Geometry) -> list[tuple[tuple[Region, ...], float, float, float]]:
    """Each frame's regions with its start angle and radial extent."""
    return [
        (geometry.stator, geometry.stator_start_angle,
         geometry.inner_radius, geometry.band_inner_radius),
        (geometry.rotor, geometry.rotor_start_angle,
         geometry.band_outer_radius, geometry.outer_radius),
    ]


def unshared_edges(regions: tuple[Region, ...]) -> list[tuple[tuple, tuple]]:
    """Edges that are not matched by the same edge of a neighbour, reversed.

    Two regions that meet without a gap or overlap share each edge between
    them exactly: one runs it forwards, the other backwards.
    """
    edges = [edge for region in regions for edge in region.boundary]
    points = np.array([p for edge in edges for p in (edge.start, edge.end)])
    tree = cKDTree(points)
    ids = [min(tree.query_ball_point(point, 1e-10)) for point in points]
    directed = Counter(
        (ids[2 * i], ids[2 * i + 1], edge.centre is not None)
        for i, edge in enumerate(edges)
    )
    assert max(directed.values()) == 1, "two regions run the same edge the same way"
    return [
        (tuple(points[a]), tuple(points[b]))
        for (a, b, arc) in directed
        if (b, a, arc) not in directed
    ]


def assert_tiles(geometry: Geometry) -> None:
    """Stator and rotor each fill their ring sector, with no gap and no overlap."""
    for regions, start, inner, outer in frames_of(geometry):
        assert all(region.area() > 0 for region in regions)
        sector = 0.5 * geometry.span_angle * (outer**2 - inner**2)
        assert sum(region.area() for region in regions) == pytest.approx(sector, rel=1e-11)

        # The only edges without a partner are those on the outline of the part.
        end = start + geometry.span_angle
        for a, b in unshared_edges(regions):
            radii = [math.hypot(*a), math.hypot(*b)]
            angles = [math.atan2(a[1], a[0]), math.atan2(b[1], b[0])]
            on_circle = any(
                all(abs(r - circle) < 1e-10 for r in radii) for circle in (inner, outer)
            )
            on_ray = any(
                all(abs(math.remainder(angle - ray, 2 * math.pi)) < 1e-9 for angle in angles)
                for ray in (start, end)
            )
            assert on_circle or on_ray, f"edge {a} to {b} is not shared"
            if geometry.span_angle == 2 * math.pi:
                assert on_circle, "a full machine has no cut edges"

    total = sum(region.area() for region in geometry.regions) + geometry.band_area()
    assert total == pytest.approx(geometry.area(), rel=1e-11)


def assert_no_overlap(geometry: Geometry, samples: int = 3000) -> None:
    """Random points in the stator and in the rotor lie in exactly one region."""
    random = np.random.default_rng(1)
    # Arcs are tested as chords within 1e-8 m of them; keeping the points that
    # far from the two outline circles keeps them out of the slivers between.
    margin = 1e-7
    for regions, start, inner, outer in frames_of(geometry):
        radius = np.sqrt(random.uniform((inner + margin) ** 2, (outer - margin) ** 2, samples))
        angle = start + random.uniform(0, geometry.span_angle, samples)
        points = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
        hits = np.zeros(samples, dtype=int)
        for region in regions:
            hits += Outline(region.polygon(1e-8)).contains_points(points)
        assert np.all(hits == 1), f"{np.sum(hits != 1)} points are not in exactly one region"


# Counts


@pytest.mark.parametrize(
    "slots, poles, slice_slots, slice_poles",
    [(36, 42, 6, 7), (12, 10, 6, 5), (24, 28, 6, 7), (12, 8, 3, 2)],
)
def test_slice_holds_the_slots_and_magnets_of_the_symmetry(
    reference: Design, slots: int, poles: int, slice_slots: int, slice_poles: int
) -> None:
    design = variant(reference, slots=slots, poles=poles, **SQUARE_TIP)
    geometry = build(design)
    repeat = symmetry(slots, poles)
    assert (repeat.slots, repeat.poles) == (slice_slots, slice_poles)

    assert len(by_role(geometry.stator, "tooth")) == slice_slots
    assert len(by_role(geometry.stator, "coil")) == 2 * slice_slots
    assert len(by_role(geometry.stator, "stator_yoke")) == slice_slots
    assert len(by_role(geometry.rotor, "magnet")) == slice_poles
    assert len(by_role(geometry.rotor, "rotor_yoke")) == slice_poles
    assert geometry.span_angle == pytest.approx(2 * math.pi / repeat.periods)
    assert geometry.antiperiodic == repeat.antiperiodic


def test_region_kinds_and_parts(reference: Design) -> None:
    geometry = build(reference)
    assert Counter(region.role for region in geometry.stator) == {
        "stator_yoke": 6, "tooth": 6, "coil": 12, "slot_air": 12, "air_gap": 6,
    }
    assert Counter(region.role for region in geometry.rotor) == {
        "magnet": 7, "magnet_gap": 14, "rotor_yoke": 7, "air_gap": 7,
    }
    kinds = {region.role: region.kind for region in geometry.regions}
    assert kinds == {
        "stator_yoke": "steel", "tooth": "steel", "coil": "coil", "slot_air": "air",
        "air_gap": "air", "magnet": "magnet", "magnet_gap": "air", "rotor_yoke": "steel",
    }


def test_region_names_are_unique(reference: Design) -> None:
    for full in (False, True):
        names = [region.name for region in build(reference, full=full).regions]
        assert len(names) == len(set(names))


# Areas against hand formulas (reference design, lengths in mm)


def test_magnet_area(reference: Design) -> None:
    # 85 % of a pole pitch of the ring between 40 and 42.5 mm radius.
    expected = 0.85 * (math.pi / 42) * (42.5**2 - 40.0**2)
    assert expected == pytest.approx(13.1133, abs=1e-4)
    for magnet in by_role(build(reference).rotor, "magnet"):
        assert magnet.area() == pytest.approx(expected * MM2, rel=1e-10)


def test_yoke_areas(reference: Design) -> None:
    geometry = build(reference)
    slice_angle = math.radians(60)
    rotor_yoke = 0.5 * slice_angle * (45.0**2 - 42.5**2)
    stator_yoke = 0.5 * slice_angle * (29.2**2 - 26.0**2)
    assert sum(r.area() for r in by_role(geometry.rotor, "rotor_yoke")) == pytest.approx(
        rotor_yoke * MM2, rel=1e-10
    )
    assert sum(r.area() for r in by_role(geometry.stator, "stator_yoke")) == pytest.approx(
        stator_yoke * MM2, rel=1e-10
    )


def test_coil_side_area(reference: Design) -> None:
    # Half a slot is the wedge between the slot centre line and the tooth axis,
    # less the half tooth. The half tooth out to radius r, a strip of half
    # width h = 1.5 mm capped by the circle, has area
    #   F(r) = (h / 2) sqrt(r^2 - h^2) + (r^2 / 2) asin(h / r).
    h, top, bottom = 1.5, 38.2, 29.2

    def half_tooth(r: float) -> float:
        return 0.5 * h * math.sqrt(r * r - h * h) + 0.5 * r * r * math.asin(h / r)

    wedge = 0.5 * (math.pi / 36) * (top**2 - bottom**2)
    expected = wedge - (half_tooth(top) - half_tooth(bottom))
    assert expected == pytest.approx(12.9634, abs=1e-4)
    for coil in by_role(build(reference).stator, "coil"):
        assert coil.area() == pytest.approx(expected * MM2, rel=1e-10)


def test_tooth_area(reference: Design) -> None:
    # Body: the strip of half width 1.5 mm between the slot-bottom and
    # slot-top circles. Tip: everything above the slot-top circle.
    h, bottom, top, tip, surface = 1.5, 29.2, 38.2, 38.8, 39.4

    def half_tooth(r: float) -> float:
        return 0.5 * h * math.sqrt(r * r - h * h) + 0.5 * r * r * math.asin(h / r)

    flank = math.asin(h / top)
    edge = math.pi / 36 - math.asin(1.5 / (2 * surface))
    body = 2 * (half_tooth(top) - half_tooth(bottom))
    # Half the tip: the sector out to the air-gap surface over the edge
    # angle, less the triangle (origin, tip corner, tooth corner) that the
    # taper cuts off, less the sector inside the slot-top circle.
    half_tip = (
        0.5 * surface**2 * edge
        - 0.5 * tip * top * math.sin(edge - flank)
        - 0.5 * top**2 * flank
    )
    expected = body + 2 * half_tip
    assert expected == pytest.approx(32.7050, abs=1e-4)
    for tooth in by_role(build(reference).stator, "tooth"):
        assert tooth.area() == pytest.approx(expected * MM2, rel=1e-10)


def test_slot_air_area_by_what_is_left_of_the_slot_pitch(reference: Design) -> None:
    geometry = build(reference)
    pitch = (math.pi / 36) * (39.4**2 - 29.2**2) * MM2  # one slot pitch, bottom to surface
    tooth = geometry.region("tooth_0").area()
    coils = geometry.region("coil_0_cw").area() + geometry.region("coil_0_ccw").area()
    air = geometry.region("slot_air_0_cw").area() + geometry.region("slot_air_0_ccw").area()
    assert air == pytest.approx(pitch - tooth - coils, rel=1e-9)

    # Half the air: the ring between the slot-top circle and the air-gap
    # surface over half a slot pitch, less the half tooth tip (as in
    # test_tooth_area).
    flank = math.asin(1.5 / 38.2)
    edge = math.pi / 36 - math.asin(1.5 / (2 * 39.4))
    half_tip = (
        0.5 * 39.4**2 * edge
        - 0.5 * 38.8 * 38.2 * math.sin(edge - flank)
        - 0.5 * 38.2**2 * flank
    )
    expected = 0.5 * (math.pi / 36) * (39.4**2 - 38.2**2) - half_tip
    assert expected == pytest.approx(1.2151, abs=1e-4)
    assert geometry.region("slot_air_0_ccw").area() == pytest.approx(expected * MM2, rel=1e-9)


def test_air_gap_layers_are_a_third_of_the_gap_each(reference: Design) -> None:
    geometry = build(reference)
    assert geometry.band_inner_radius == pytest.approx(39.6e-3)
    assert geometry.band_outer_radius == pytest.approx(39.8e-3)
    slice_angle = math.radians(60)
    stator_layer = 0.5 * slice_angle * (39.6**2 - 39.4**2) * MM2
    rotor_layer = 0.5 * slice_angle * (40.0**2 - 39.8**2) * MM2
    gaps = by_role(geometry.stator, "air_gap"), by_role(geometry.rotor, "air_gap")
    assert sum(r.area() for r in gaps[0]) == pytest.approx(stator_layer, rel=1e-10)
    assert sum(r.area() for r in gaps[1]) == pytest.approx(rotor_layer, rel=1e-10)
    assert geometry.band_area() == pytest.approx(
        0.5 * slice_angle * (39.8**2 - 39.6**2) * MM2, rel=1e-10
    )


# Tiling: areas add up, nothing overlaps


def test_region_areas_sum_to_the_slice(reference: Design) -> None:
    geometry = build(reference)
    slice_area = 0.5 * math.radians(60) * (45.0**2 - 26.0**2) * MM2
    assert geometry.area() == pytest.approx(slice_area, rel=1e-12)
    assert_tiles(geometry)


def test_no_two_regions_overlap(reference: Design) -> None:
    assert_no_overlap(build(reference))


@pytest.mark.parametrize(
    "changes",
    [
        {"slots": 12, "poles": 10, "geometry__stator__tooth_tip_taper_mm": 1.2},
        {"slots": 24, "poles": 28},
        {"slots": 12, "poles": 8, **SQUARE_TIP},
        SQUARE_TIP,
        {"geometry__magnet__arc_fraction": 1},
        {"geometry__magnet__arc_fraction": 0.5, "geometry__magnet__magnetisation": "radial"},
        # Opening at its widest: the tip is no wider than the tooth.
        {"geometry__stator__slot_opening_mm": 3.78},
        {"geometry__stator__slot_opening_mm": 3.7, "geometry__stator__tooth_tip_taper_mm": 0},
        {"slots": 3, "poles": 2, "geometry__stator__tooth_width_mm": 20, **SQUARE_TIP},
        {"slots": 3, "poles": 2, "geometry__magnet__arc_fraction": 1, **SQUARE_TIP},
    ],
    ids=lambda changes: ",".join(f"{k.split('__')[-1]}={v}" for k, v in changes.items()),
)
def test_other_designs_tile_too(reference: Design, changes: dict) -> None:
    design = variant(reference, **changes)
    assert_tiles(build(design, full=True))
    piece = build(design)
    assert_tiles(piece)
    assert_no_overlap(piece, samples=1500)


def test_full_machine_is_the_slice_repeated(reference: Design) -> None:
    piece, whole = build(reference), build(reference, full=True)
    assert whole.span_angle == pytest.approx(2 * math.pi)
    assert whole.antiperiodic is False
    assert len(whole.stator) == 6 * len(piece.stator)
    assert len(whole.rotor) == 6 * len(piece.rotor)
    assert whole.area() == pytest.approx(math.pi * (45.0**2 - 26.0**2) * MM2)
    assert_tiles(whole)
    # The slice is the first part of the full machine, unchanged.
    assert whole.stator[: len(piece.stator)] == piece.stator
    assert whole.rotor[: len(piece.rotor)] == piece.rotor


def test_every_corner_lies_inside_its_part(reference: Design) -> None:
    geometry = build(reference)
    for regions, start, inner, outer in frames_of(geometry):
        for region in regions:
            for edge in region.boundary:
                radius = math.hypot(*edge.start)
                angle = math.atan2(edge.start[1], edge.start[0])
                assert inner - 1e-12 <= radius <= outer + 1e-12
                assert start - 1e-9 <= angle <= start + geometry.span_angle + 1e-9


def test_cut_edges_have_matching_corners(reference: Design) -> None:
    # The mesher needs the same corner radii on the two cut edges of a part.
    geometry = build(reference)
    for regions, start, _, _ in frames_of(geometry):
        radii = {start: set(), start + geometry.span_angle: set()}
        for region in regions:
            for edge in region.boundary:
                angle = math.atan2(edge.start[1], edge.start[0])
                for ray in radii:
                    if abs(angle - ray) < 1e-9:
                        radii[ray].add(round(math.hypot(*edge.start), 10))
        low, high = radii.values()
        assert low == high and len(low) >= 4


# Frames and units


def test_geometry_is_in_metres_and_radians(reference: Design) -> None:
    geometry = build(reference)
    assert geometry.outer_radius == pytest.approx(0.045)
    assert geometry.inner_radius == pytest.approx(0.026)
    assert geometry.stack_length == pytest.approx(0.012)
    longer = variant(reference, axial__stack_length_mm=15, axial__magnet_overhang_mm=1.5)
    assert build(longer).stack_length == pytest.approx(0.015)
    assert geometry.span_angle == pytest.approx(math.pi / 3)
    assert geometry.antiperiodic is True


def test_stator_is_cut_on_slot_centres_and_rotor_between_magnets(reference: Design) -> None:
    geometry = build(reference)
    assert geometry.stator_start_angle == pytest.approx(-math.pi / 36)
    assert geometry.rotor_start_angle == pytest.approx(-math.pi / 42)
    # Tooth 0 and magnet 0 are both centred on the x axis.
    for name in ("tooth_0", "magnet_0"):
        points = geometry.region(name).polygon()
        assert points[:, 1].max() == pytest.approx(-points[:, 1].min(), rel=1e-9)
        assert points[:, 0].min() > 0


def test_materials_come_from_the_design(reference: Design) -> None:
    design = variant(reference, steel__rotor="S235", winding__conductor="aluminium")
    geometry = build(design)
    materials = {region.role: region.material for region in geometry.regions}
    assert materials == {
        "stator_yoke": "M270-35A", "tooth": "M270-35A", "coil": "aluminium",
        "slot_air": "air", "air_gap": "air", "magnet": "N42SH", "magnet_gap": "air",
        "rotor_yoke": "S235",
    }


# Magnets


def test_magnets_alternate_starting_outwards(reference: Design) -> None:
    magnets = by_role(build(reference, full=True).rotor, "magnet")
    assert len(magnets) == 42
    assert [m.magnetisation.polarity for m in magnets] == [1, -1] * 21
    for j, magnet in enumerate(magnets):
        assert magnet.name == f"magnet_{j}"
        assert magnet.magnetisation.axis_angle == pytest.approx(2 * math.pi * j / 42)
        assert magnet.magnetisation.pattern == "parallel"


def test_magnet_axis_passes_through_the_middle_of_the_magnet(reference: Design) -> None:
    magnet = build(reference).region("magnet_3")
    centre = magnet.polygon().mean(axis=0)
    assert math.atan2(centre[1], centre[0]) == pytest.approx(
        magnet.magnetisation.axis_angle, abs=1e-6
    )


def test_magnetisation_pattern_follows_the_design(reference: Design) -> None:
    design = variant(reference, geometry__magnet__magnetisation="radial")
    assert build(design).region("magnet_0").magnetisation.pattern == "radial"


def test_only_magnets_are_magnetised_and_only_coils_carry_current(reference: Design) -> None:
    for region in build(reference).regions:
        assert (region.magnetisation is not None) == (region.kind == "magnet")
        assert (region.coil_side is not None) == (region.kind == "coil")


def test_the_next_slice_has_the_opposite_magnets_and_currents(reference: Design) -> None:
    whole = build(reference, full=True)
    for j in range(7):
        here = whole.region(f"magnet_{j}").magnetisation.polarity
        there = whole.region(f"magnet_{j + 7}").magnetisation.polarity
        assert there == -here
    for k in range(6):
        for side in ("cw", "ccw"):
            here = whole.region(f"coil_{k}_{side}").coil_side
            there = whole.region(f"coil_{k + 6}_{side}").coil_side
            assert there.phase == here.phase and there.sign == -here.sign


# Coil sides


def test_coil_sides_follow_the_winding(reference: Design) -> None:
    geometry = build(reference, full=True)
    for coil in concentrated_winding(36, 42):
        ccw = geometry.region(f"coil_{coil.tooth}_ccw").coil_side
        cw = geometry.region(f"coil_{coil.tooth}_cw").coil_side
        assert ccw.phase == cw.phase == coil.phase
        assert ccw.tooth == cw.tooth == coil.tooth
        # Out of the plane on the counter-clockwise side of a +1 coil's tooth.
        assert ccw.sign == coil.direction and cw.sign == -coil.direction
        assert ccw.turns == cw.turns == 20


def test_coil_sides_sit_either_side_of_their_tooth(reference: Design) -> None:
    geometry = build(reference)
    pitch = 2 * math.pi / 36
    for k in range(6):
        for side, sign in (("ccw", 1), ("cw", -1)):
            centre = geometry.region(f"coil_{k}_{side}").polygon().mean(axis=0)
            offset = math.atan2(centre[1], centre[0]) - k * pitch
            assert 0 < sign * offset < pitch / 2


def test_12_slot_10_pole_currents_for_positive_phase_a(reference: Design) -> None:
    # Layout AabBCcaABbcC: phase A is wound on teeth 0 and 7 (+) and 1 and 6 (-).
    geometry = build(variant(reference, slots=12, poles=10, **SQUARE_TIP), full=True)
    signs = {
        region.name: region.coil_side.sign
        for region in by_role(geometry.stator, "coil")
        if region.coil_side.phase == "A"
    }
    assert signs == {
        "coil_0_ccw": 1, "coil_0_cw": -1,
        "coil_1_ccw": -1, "coil_1_cw": 1,
        "coil_6_ccw": -1, "coil_6_cw": 1,
        "coil_7_ccw": 1, "coil_7_cw": -1,
    }


def test_neighbouring_coils_of_a_phase_group_share_current_direction_in_their_slot(
    reference: Design,
) -> None:
    # Teeth 0 (A) and 1 (a) are wound opposite ways, so in the slot between
    # them both coil sides carry current the same way.
    geometry = build(variant(reference, slots=12, poles=10, **SQUARE_TIP), full=True)
    assert geometry.region("coil_0_ccw").coil_side.sign == 1
    assert geometry.region("coil_1_cw").coil_side.sign == 1


def test_turns_come_from_the_design(reference: Design) -> None:
    geometry = build(variant(reference, winding__turns_per_coil=33))
    assert {r.coil_side.turns for r in by_role(geometry.stator, "coil")} == {33}


@pytest.mark.parametrize("taper", [0.6, 0])
def test_shared_corners_are_exactly_equal_inside_a_slice(reference: Design, taper: float) -> None:
    # Neighbouring regions must compute a shared corner to the last bit, so
    # that the mesher sees one point, not two a hair apart.
    design = variant(reference, geometry__stator__tooth_tip_taper_mm=taper)
    geometry = build(design)
    for regions in (geometry.stator, geometry.rotor):
        corners = {edge.start for region in regions for edge in region.boundary}
        rounded = {(round(x, 10), round(y, 10)) for x, y in corners}
        assert len(corners) == len(rounded)
