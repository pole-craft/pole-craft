"""Regions made of lines and arcs: exact areas, outlines and rotation."""

import math

import numpy as np
import pytest

from polecraft.core.geometry import (
    CoilSide,
    Edge,
    Geometry,
    Magnetisation,
    Region,
    closed_loop,
    polar,
    ring_sector,
)


def region(boundary: tuple[Edge, ...], **extra) -> Region:
    return Region(
        name="test", kind="air", role="test", material="air", boundary=boundary, **extra
    )


def polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# Edges


def test_line_has_no_radius_or_sweep() -> None:
    edge = Edge((0.0, 0.0), (1.0, 0.0))
    assert edge.radius == 0.0 and edge.sweep == 0.0
    assert edge.points(1e-6) == [(0.0, 0.0)]


def test_arc_radius_and_sweep() -> None:
    quarter = Edge((2.0, 0.0), (0.0, 2.0), centre=(0.0, 0.0), ccw=True)
    assert quarter.radius == pytest.approx(2.0)
    assert quarter.sweep == pytest.approx(math.pi / 2)
    back = Edge((0.0, 2.0), (2.0, 0.0), centre=(0.0, 0.0), ccw=False)
    assert back.sweep == pytest.approx(math.pi / 2)


def test_arc_points_stay_within_tolerance_of_the_arc() -> None:
    arc = Edge(polar(0.04, 0.0), polar(0.04, 0.5), centre=(0.0, 0.0), ccw=True)
    points = arc.points(1e-7) + [arc.end]
    assert points[0] == arc.start
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        assert math.hypot(x0, y0) == pytest.approx(0.04)
        # The middle of each chord is the point furthest inside the arc.
        middle = math.hypot((x0 + x1) / 2, (y0 + y1) / 2)
        assert 0 <= 0.04 - middle <= 1e-7


def test_rotated_edge() -> None:
    arc = Edge((1.0, 0.0), (0.0, 1.0), centre=(0.0, 0.0), ccw=True)
    turned = arc.rotated(math.pi / 2)
    assert turned.start == pytest.approx((0.0, 1.0))
    assert turned.end == pytest.approx((-1.0, 0.0))
    assert turned.centre == pytest.approx((0.0, 0.0))
    assert turned.ccw is True


def test_rotation_turns_the_centre_of_an_arc() -> None:
    arc = Edge((6.0, 5.0), (5.0, 6.0), centre=(5.0, 5.0), ccw=True)
    turned = arc.rotated(math.pi / 2)
    assert turned.centre == pytest.approx((-5.0, 5.0))
    assert turned.start == pytest.approx((-5.0, 6.0))
    assert turned.radius == pytest.approx(1.0)


def test_points_of_an_arc_off_the_origin_lie_on_the_arc() -> None:
    arc = Edge((6.0, 5.0), (5.0, 6.0), centre=(5.0, 5.0), ccw=True)
    points = arc.points(1e-6)
    assert len(points) > 10
    for x, y in points:
        assert math.hypot(x - 5.0, y - 5.0) == pytest.approx(1.0)


# Areas


def test_area_of_a_rectangle() -> None:
    corners = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]
    boundary = tuple(Edge(a, b) for a, b in zip(corners, corners[1:] + corners[:1]))
    assert region(boundary).area() == pytest.approx(6.0)


def test_area_is_negative_for_a_clockwise_loop() -> None:
    corners = [(0.0, 0.0), (0.0, 2.0), (3.0, 2.0), (3.0, 0.0)]
    boundary = tuple(Edge(a, b) for a, b in zip(corners, corners[1:] + corners[:1]))
    assert region(boundary).area() == pytest.approx(-6.0)


def test_area_of_a_half_disc() -> None:
    # Two quarter arcs and the diameter: pi r^2 / 2.
    boundary = (
        Edge((2.0, 0.0), (0.0, 2.0), centre=(0.0, 0.0)),
        Edge((0.0, 2.0), (-2.0, 0.0), centre=(0.0, 0.0)),
        Edge((-2.0, 0.0), (2.0, 0.0)),
    )
    assert region(boundary).area() == pytest.approx(math.pi * 2.0)


def test_area_of_an_arc_off_the_origin() -> None:
    # Quarter disc of radius 1 centred on (5, 5).
    boundary = (
        Edge((5.0, 5.0), (6.0, 5.0)),
        Edge((6.0, 5.0), (5.0, 6.0), centre=(5.0, 5.0)),
        Edge((5.0, 6.0), (5.0, 5.0)),
    )
    assert region(boundary).area() == pytest.approx(math.pi / 4)


@pytest.mark.parametrize(
    "inner, outer, start, end",
    [(0.026, 0.0292, -0.1, 0.1), (1.0, 2.0, 0.3, 1.9), (0.5, 3.0, -2.0, 1.0)],
)
def test_area_of_a_ring_sector(inner: float, outer: float, start: float, end: float) -> None:
    expected = 0.5 * (end - start) * (outer**2 - inner**2)
    assert region(ring_sector(inner, outer, start, end)).area() == pytest.approx(
        expected, rel=1e-12
    )


def test_breaks_add_corners_but_not_area() -> None:
    plain = ring_sector(1.0, 2.0, 0.0, 1.0)
    broken = ring_sector(1.0, 2.0, 0.0, 1.0, inner_breaks=(0.7, 0.2), outer_breaks=(0.5,))
    assert len(plain) == 4 and len(broken) == 7
    assert region(broken).area() == pytest.approx(region(plain).area(), rel=1e-12)
    # The inner arc runs clockwise, so its breaks come in falling order.
    inner = [edge for edge in broken if edge.centre and not edge.ccw]
    assert [edge.start for edge in inner] == [
        polar(1.0, 1.0), polar(1.0, 0.7), polar(1.0, 0.2)
    ]


def test_long_arcs_are_cut_in_two() -> None:
    # A half ring: each arc sweeps pi, which one edge could not describe.
    boundary = ring_sector(1.0, 2.0, 0.0, math.pi)
    assert len(boundary) == 6
    assert all(edge.sweep < 2.0 for edge in boundary)
    assert region(boundary).area() == pytest.approx(0.5 * math.pi * 3.0)


def test_coinciding_corners_are_merged() -> None:
    boundary = closed_loop(
        [(1.0, 0.0, "line"), (2.0, 0.0, "ccw"), (2.0, 0.0, "ccw"), (2.0, 1.0, "line"), (1.0, 1.0, "cw")]
    )
    assert len(boundary) == 4


def test_unknown_edge_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="'line', 'ccw' or 'cw'"):
        closed_loop([(1.0, 0.0, "line"), (2.0, 0.0, "arc"), (2.0, 1.0, "line")])


def test_arc_between_two_radii_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot join radius"):
        closed_loop([(1.0, 0.0, "ccw"), (2.0, 1.0, "line"), (1.5, 0.5, "line")])


def test_boundary_is_closed() -> None:
    boundary = ring_sector(1.0, 2.0, 0.2, 0.9, inner_breaks=(0.5,))
    for edge, following in zip(boundary, boundary[1:] + boundary[:1]):
        assert edge.end == following.start


# Outline as a polygon


def test_polygon_approaches_the_exact_area() -> None:
    sector = region(ring_sector(0.03, 0.045, 0.0, 0.5))
    exact = sector.area()
    coarse = polygon_area(sector.polygon(1e-5))
    fine = polygon_area(sector.polygon(1e-9))
    assert abs(fine - exact) < abs(coarse - exact)
    assert fine == pytest.approx(exact, rel=1e-6)


def test_polygon_starts_at_every_corner() -> None:
    sector = region(ring_sector(1.0, 2.0, 0.0, 1.0))
    points = [tuple(point) for point in sector.polygon(1e-3)]
    for edge in sector.boundary:
        assert edge.start in points


# Rotation


def test_rotation_keeps_area_and_moves_the_region() -> None:
    sector = region(ring_sector(1.0, 2.0, 0.0, 0.5))
    turned = sector.rotated(1.0)
    assert turned.area() == pytest.approx(sector.area(), rel=1e-12)
    assert turned.boundary[0].start == pytest.approx(polar(1.0, 1.0))


def test_rotation_turns_the_magnetisation_axis() -> None:
    magnet = region(
        ring_sector(1.0, 2.0, -0.1, 0.1), magnetisation=Magnetisation("parallel", 0.0, -1)
    )
    turned = magnet.rotated(0.4)
    assert turned.magnetisation == Magnetisation("parallel", 0.4, -1)
    assert magnet.magnetisation.axis_angle == 0.0


# Magnetisation


def test_parallel_magnetisation_has_one_direction_everywhere() -> None:
    outward = Magnetisation("parallel", math.pi / 2, 1)
    assert outward.direction(3.0, 4.0) == pytest.approx((0.0, 1.0))
    assert outward.direction(-1.0, 0.5) == pytest.approx((0.0, 1.0))
    inward = Magnetisation("parallel", math.pi / 2, -1)
    assert inward.direction(3.0, 4.0) == pytest.approx((0.0, -1.0))


def test_radial_magnetisation_follows_the_radius() -> None:
    outward = Magnetisation("radial", 0.0, 1)
    assert outward.direction(3.0, 4.0) == pytest.approx((0.6, 0.8))
    assert Magnetisation("radial", 0.0, -1).direction(0.0, 2.0) == pytest.approx((0.0, -1.0))


# Geometry container


def small_geometry() -> Geometry:
    stator = (
        Region(
            name="yoke", kind="steel", role="stator_yoke", material="M270-35A",
            boundary=ring_sector(1.0, 2.0, 0.0, 1.0),
        ),
        Region(
            name="coil", kind="coil", role="coil", material="copper",
            boundary=ring_sector(2.0, 2.4, 0.0, 1.0),
            coil_side=CoilSide(phase="A", tooth=0, sign=1, turns=20),
        ),
    )
    rotor = (
        Region(
            name="magnet", kind="magnet", role="magnet", material="N42SH",
            boundary=ring_sector(2.6, 3.0, 0.5, 1.5),
        ),
    )
    return Geometry(
        stator=stator,
        rotor=rotor,
        inner_radius=1.0,
        outer_radius=3.0,
        band_inner_radius=2.4,
        band_outer_radius=2.6,
        stator_start_angle=0.0,
        rotor_start_angle=0.5,
        span_angle=1.0,
        antiperiodic=True,
        stack_length=0.012,
    )


def test_geometry_lists_and_finds_regions() -> None:
    geometry = small_geometry()
    assert [r.name for r in geometry.regions] == ["yoke", "coil", "magnet"]
    assert geometry.region("magnet").material == "N42SH"
    with pytest.raises(KeyError):
        geometry.region("nothing")


def test_geometry_areas() -> None:
    geometry = small_geometry()
    assert geometry.band_area() == pytest.approx(0.5 * (2.6**2 - 2.4**2))
    assert geometry.area() == pytest.approx(0.5 * (3.0**2 - 1.0**2))
    parts = sum(r.area() for r in geometry.regions) + geometry.band_area()
    assert parts == pytest.approx(geometry.area())
