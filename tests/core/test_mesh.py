"""Meshing: labels, areas, triangle shape, periodic edges and the air-gap band."""

import math
import signal
import threading
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")

from polecraft.core.geometry import Geometry
from polecraft.core.mesh import BAND, Mesh, build_mesh
from polecraft.design import Design, design_from_dict, design_to_dict, load_design
from polecraft.machines import machine_for

SQUARE_TIP = {"geometry__stator__tooth_tip_taper_mm": 0}


@pytest.fixture(scope="module")
def reference() -> Design:
    root = Path(__file__).resolve().parents[2]
    return load_design(root / "designs" / "qdd_outer_36s42p.yaml")


@pytest.fixture(scope="module")
def geometry(reference: Design) -> Geometry:
    return machine_for(reference).build_geometry()


@pytest.fixture(scope="module")
def mesh(geometry: Geometry) -> Mesh:
    return build_mesh(geometry)


def variant(design: Design, **changes: object) -> Design:
    """The design with some values replaced; sections are joined with ``__``."""
    data = design_to_dict(design)
    for path, value in changes.items():
        *sections, key = path.split("__")
        block = data
        for name in sections:
            block = block[name]
        block[key] = value
    return design_from_dict(data)


def radius(mesh: Mesh) -> np.ndarray:
    return np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])


def triangles_of(mesh: Mesh, prefix: str) -> np.ndarray:
    """Triangles of every region whose name starts with a prefix."""
    wanted = [i for i, name in enumerate(mesh.region_names) if name.startswith(prefix)]
    return mesh.triangles[np.isin(mesh.triangle_region, wanted)]


def edges_used_once(mesh: Mesh) -> np.ndarray:
    """Edges that belong to one triangle only, as (m, 2) node numbers."""
    edges = np.sort(
        np.vstack([mesh.triangles[:, [0, 1]], mesh.triangles[:, [1, 2]], mesh.triangles[:, [2, 0]]]),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    assert counts.max() <= 2, "an edge belongs to more than two triangles"
    return unique[counts == 1]


def assert_conforming(mesh: Mesh, geometry: Geometry) -> None:
    """Triangles meet edge to edge; only the outline has edges used once.

    The outline is the inner and outer circle, the cut edges of the stator and
    of the rotor, and the stretches of the rotor's band circle where the band
    joins copies of rotor nodes.
    """
    r = radius(mesh)
    angle = np.arctan2(mesh.nodes[:, 1], mesh.nodes[:, 0])
    full = math.isclose(geometry.span_angle, 2 * math.pi)
    circles = [geometry.inner_radius, geometry.outer_radius]
    rays = []
    if not full:
        circles += [geometry.band_inner_radius, geometry.band_outer_radius]
        for start in (geometry.stator_start_angle, geometry.rotor_start_angle):
            rays += [start, start + geometry.span_angle]

    for a, b in edges_used_once(mesh):
        on_circle = any(abs(r[a] - c) < 1e-9 and abs(r[b] - c) < 1e-9 for c in circles)
        on_ray = any(
            all(abs(math.remainder(angle[n] - ray, 2 * math.pi)) * r[n] < 1e-9 for n in (a, b))
            for ray in rays
        )
        assert on_circle or on_ray, f"edge {mesh.nodes[a]} to {mesh.nodes[b]} has no neighbour"


# Plain arrays


def test_mesh_is_plain_arrays(mesh: Mesh) -> None:
    assert mesh.nodes.ndim == 2 and mesh.nodes.shape[1] == 2
    assert mesh.nodes.dtype == np.float64
    assert mesh.triangles.ndim == 2 and mesh.triangles.shape[1] == 3
    assert np.issubdtype(mesh.triangles.dtype, np.integer)
    assert mesh.triangle_region.shape == (len(mesh.triangles),)
    assert mesh.rotor_nodes.shape == (len(mesh.nodes),) and mesh.rotor_nodes.dtype == bool
    assert mesh.periodic.ndim == 2 and mesh.periodic.shape[1] == 2


def test_every_node_is_used_and_every_triangle_is_valid(mesh: Mesh) -> None:
    assert mesh.triangles.min() == 0
    assert set(np.unique(mesh.triangles)) == set(range(len(mesh.nodes)))
    for triangle in mesh.triangles[:: len(mesh.triangles) // 50]:
        assert len(set(triangle)) == 3


def test_triangles_are_counter_clockwise(mesh: Mesh) -> None:
    assert mesh.triangle_areas().min() > 0


def test_meshing_leaves_the_interrupt_handler_alone(geometry: Geometry) -> None:
    before = signal.getsignal(signal.SIGINT)
    build_mesh(geometry)
    assert signal.getsignal(signal.SIGINT) is before


def test_meshing_works_outside_the_main_thread(geometry: Geometry, mesh: Mesh) -> None:
    results = []
    worker = threading.Thread(target=lambda: results.append(build_mesh(geometry)))
    worker.start()
    worker.join()
    assert len(results) == 1 and np.array_equal(results[0].triangles, mesh.triangles)


def test_an_open_gmsh_session_is_not_taken_over(geometry: Geometry) -> None:
    import gmsh

    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            build_mesh(geometry)
        assert gmsh.isInitialized()
    finally:
        gmsh.finalize()


def test_meshing_twice_gives_the_same_mesh(geometry: Geometry, mesh: Mesh) -> None:
    again = build_mesh(geometry)
    assert np.array_equal(again.nodes, mesh.nodes)
    assert np.array_equal(again.triangles, mesh.triangles)


# Regions


def test_every_region_has_triangles_with_its_label(mesh: Mesh, geometry: Geometry) -> None:
    assert mesh.region_names == tuple(r.name for r in geometry.regions) + (BAND,)
    counts = Counter(mesh.triangle_region.tolist())
    assert set(counts) == set(range(len(mesh.region_names)))
    assert min(counts.values()) >= 2


def test_triangles_lie_inside_their_region(mesh: Mesh, geometry: Geometry) -> None:
    from matplotlib.path import Path as Outline

    centroids = mesh.nodes[mesh.triangles].mean(axis=1)
    for index, region in enumerate(geometry.regions):
        inside = Outline(region.polygon(1e-9)).contains_points(
            centroids[mesh.triangle_region == index]
        )
        assert inside.all(), region.name


def test_region_areas_match_the_geometry(mesh: Mesh, geometry: Geometry) -> None:
    # Straight triangle edges cut the corners of arcs, so the match is close
    # but not exact; the error is largest for the small regions.
    for region in geometry.regions:
        assert mesh.region_area(region.name) == pytest.approx(region.area(), rel=3e-3), region.name
    assert mesh.region_area(BAND) == pytest.approx(geometry.band_area(), rel=1e-4)
    assert mesh.triangle_areas().sum() == pytest.approx(geometry.area(), rel=1e-5)


# Triangle shape and size


def test_no_badly_shaped_triangles(mesh: Mesh) -> None:
    angles = mesh.smallest_angles()
    assert angles.min() > 20
    assert np.median(angles) > 40


def test_smallest_angle_of_a_known_triangle() -> None:
    mesh = Mesh(
        nodes=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, math.sqrt(3)]]),
        triangles=np.array([[0, 1, 2], [0, 1, 3]]),
        triangle_region=np.array([0, 0]),
        region_names=("a",),
        rotor_nodes=np.zeros(4, dtype=bool),
        periodic=np.zeros((0, 2), dtype=int),
        antiperiodic=False,
        band_divisions=0,
    )
    assert mesh.smallest_angles() == pytest.approx([45.0, 30.0])
    assert mesh.triangle_areas() == pytest.approx([0.5, math.sqrt(3) / 2])
    assert mesh.region_area("a") == pytest.approx(0.5 + math.sqrt(3) / 2)


def test_mesh_is_fine_at_the_air_gap_and_coarse_in_the_yokes(mesh: Mesh) -> None:
    areas = mesh.triangle_areas()

    def mean_area(prefix: str) -> float:
        wanted = [i for i, name in enumerate(mesh.region_names) if name.startswith(prefix)]
        return float(areas[np.isin(mesh.triangle_region, wanted)].mean())

    gap = mean_area("gap_stator")
    assert mean_area("stator_yoke") > 12 * gap
    assert mean_area("rotor_yoke") > 12 * gap
    assert mean_area("magnet_") > 3 * gap


def test_tooth_tips_are_meshed_finer_than_tooth_roots(mesh: Mesh, geometry: Geometry) -> None:
    index = mesh.region_names.index("tooth_0")
    chosen = mesh.triangle_region == index
    centre = np.hypot(*mesh.nodes[mesh.triangles[chosen]].mean(axis=1).T)
    areas = mesh.triangle_areas()[chosen]
    tip = areas[centre > 0.0385].mean()
    root = areas[centre < 0.032].mean()
    assert root > 8 * tip


def test_size_scales_the_triangles(geometry: Geometry, mesh: Mesh) -> None:
    finer = build_mesh(geometry, size=0.5)
    assert 2.5 < len(finer.triangles) / len(mesh.triangles) < 5
    assert finer.band_divisions > 1.5 * mesh.band_divisions
    assert finer.band_divisions % 84 == 0
    assert finer.smallest_angles().min() > 20
    assert_conforming(finer, geometry)
    assert finer.triangle_areas().sum() == pytest.approx(geometry.area(), rel=1e-5)


@pytest.mark.parametrize("size", [0, -1.0, math.inf, math.nan])
def test_size_must_be_positive(geometry: Geometry, size: float) -> None:
    with pytest.raises(ValueError, match="size must be a number greater than 0"):
        build_mesh(geometry, size=size)


# Conformity


def test_triangles_meet_edge_to_edge(mesh: Mesh, geometry: Geometry) -> None:
    assert_conforming(mesh, geometry)


def test_stator_and_rotor_nodes_are_told_apart(mesh: Mesh, geometry: Geometry) -> None:
    r = radius(mesh)
    assert r[mesh.rotor_nodes].min() == pytest.approx(geometry.band_outer_radius)
    assert r[~mesh.rotor_nodes].max() == pytest.approx(geometry.band_inner_radius)
    assert mesh.rotor_nodes.sum() > 100 and (~mesh.rotor_nodes).sum() > 100


# Periodic edges


def assert_periodic(mesh: Mesh, geometry: Geometry) -> None:
    """Every pair is one slice apart, nothing is missed and nothing chains."""
    span = geometry.span_angle
    r = radius(mesh)
    angle = np.arctan2(mesh.nodes[:, 1], mesh.nodes[:, 0])

    def off(ray: float) -> np.ndarray:
        """Distance of every node from a ray, measured round its own circle."""
        return np.abs(np.remainder(angle - ray + np.pi, 2 * np.pi) - np.pi) * r

    def turned(points: np.ndarray, by: float) -> np.ndarray:
        c, s = math.cos(by), math.sin(by)
        return points @ np.array([[c, -s], [s, c]]).T

    nodes, originals = mesh.periodic[:, 0], mesh.periodic[:, 1]
    assert len(set(nodes.tolist())) == len(nodes)
    assert not set(nodes.tolist()) & set(originals.tolist())
    assert np.array_equal(mesh.rotor_nodes[nodes], mesh.rotor_nodes[originals])

    paired = set(nodes.tolist())
    on_cut_edge = np.zeros(len(mesh.nodes), dtype=bool)
    for selected, start in (
        (~mesh.rotor_nodes, geometry.stator_start_angle),
        (mesh.rotor_nodes, geometry.rotor_start_angle),
    ):
        on_far = np.flatnonzero(selected & (off(start + span) < 1e-9))
        on_near = np.flatnonzero(selected & (off(start) < 1e-9))
        # Copies made for the band may also lie on these rays; they are
        # handled below, so count only nodes that some region's triangle uses.
        in_region = np.unique(mesh.triangles[mesh.triangle_region < len(mesh.region_names) - 1])
        on_far = np.intersect1d(on_far, in_region)
        on_near = np.intersect1d(on_near, in_region)
        assert len(on_far) == len(on_near) >= 4
        assert set(on_far.tolist()) <= paired
        assert not set(on_near.tolist()) & paired
        on_cut_edge[on_far] = True

    for node, original in mesh.periodic:
        if on_cut_edge[node]:
            # The far cut edge is the near one turned forwards by one slice.
            assert np.allclose(turned(mesh.nodes[original], span), mesh.nodes[node], atol=1e-10)
        else:
            # A copy for the band sits one slice before or after its original,
            # on the rotor's band circle, outside the rotor slice.
            forwards = np.allclose(turned(mesh.nodes[original], span), mesh.nodes[node], atol=1e-10)
            back = np.allclose(turned(mesh.nodes[original], -span), mesh.nodes[node], atol=1e-10)
            assert forwards or back
            assert mesh.rotor_nodes[node]
            assert r[node] == pytest.approx(geometry.band_outer_radius, abs=1e-10)
            past = (angle[node] - geometry.rotor_start_angle) / span
            assert past < -1e-9 or past > 1 + 1e-9


def test_periodic_nodes_are_paired_one_slice_apart(mesh: Mesh, geometry: Geometry) -> None:
    assert_periodic(mesh, geometry)
    assert len(mesh.periodic) > 20


def test_slice_of_the_reference_motor_is_antiperiodic(mesh: Mesh) -> None:
    assert mesh.antiperiodic is True


# Air gap and band


def test_air_gap_has_three_element_layers(mesh: Mesh, geometry: Geometry) -> None:
    r = radius(mesh)
    surface, magnet = 0.0394, 0.040
    layers = [
        (triangles_of(mesh, "gap_stator"), surface, geometry.band_inner_radius),
        (triangles_of(mesh, BAND), geometry.band_inner_radius, geometry.band_outer_radius),
        (triangles_of(mesh, "gap_rotor"), geometry.band_outer_radius, magnet),
    ]
    for triangles, inner, outer in layers:
        assert len(triangles) > 100
        # Every triangle of a layer lies within it, so no triangle crosses two.
        assert r[triangles].min() == pytest.approx(inner, abs=1e-9)
        assert r[triangles].max() == pytest.approx(outer, abs=1e-9)
    # No triangle reaches from the stator surface to the magnets.
    spans = r[mesh.triangles].max(axis=1) - r[mesh.triangles].min(axis=1)
    in_gap = (r[mesh.triangles].min(axis=1) >= surface - 1e-9) & (
        r[mesh.triangles].max(axis=1) <= magnet + 1e-9
    )
    assert spans[in_gap].max() == pytest.approx(0.0002, abs=1e-9)


def test_band_is_one_layer_joining_the_two_rings(mesh: Mesh, geometry: Geometry) -> None:
    band = triangles_of(mesh, BAND)
    assert len(band) == 2 * mesh.band_divisions
    r = radius(mesh)[band]
    on_inner = np.isclose(r, geometry.band_inner_radius, atol=1e-9)
    on_outer = np.isclose(r, geometry.band_outer_radius, atol=1e-9)
    assert np.all(on_inner | on_outer)
    assert np.all(on_inner.any(axis=1) & on_outer.any(axis=1))


def test_band_nodes_are_equally_spaced(mesh: Mesh, geometry: Geometry) -> None:
    # 6 slots and 7 poles in the slice: a multiple of 42 gives every slot pitch
    # and every pole pitch a whole number of spacings. It is doubled to 84 so
    # that the 0.714 degrees between the starts of the stator and rotor slices
    # is a whole number of spacings too.
    assert mesh.band_divisions % 84 == 0
    spacing = geometry.span_angle / mesh.band_divisions
    r = radius(mesh)
    for selected, ring_radius, start in (
        (~mesh.rotor_nodes, geometry.band_inner_radius, geometry.stator_start_angle),
        (mesh.rotor_nodes, geometry.band_outer_radius, geometry.rotor_start_angle),
    ):
        ring = np.flatnonzero(selected & np.isclose(r, ring_radius, atol=1e-9))
        angles = np.sort(np.arctan2(mesh.nodes[ring, 1], mesh.nodes[ring, 0]))
        steps = (angles - start) / spacing
        assert np.allclose(steps, np.round(steps), atol=1e-6)
        assert len(np.unique(np.round(steps))) == len(ring)


def test_band_spacing_is_close_to_its_thickness(mesh: Mesh, geometry: Geometry) -> None:
    thickness = geometry.band_outer_radius - geometry.band_inner_radius
    spacing = geometry.span_angle / mesh.band_divisions * geometry.band_inner_radius
    assert 0.6 < spacing / thickness < 1.6


def test_band_reaches_past_the_rotor_slice_with_copies(mesh: Mesh, geometry: Geometry) -> None:
    # The stator slice starts 5 degrees before the x axis and the rotor slice
    # 4.286, so the band needs copies of rotor nodes for the 0.714 degrees
    # between: one for every node spacing in that angle.
    spacing = geometry.span_angle / mesh.band_divisions
    needed = round(math.radians(5 - 180 / 42) / spacing)
    assert needed >= 1

    r = radius(mesh)
    angle = np.arctan2(mesh.nodes[:, 1], mesh.nodes[:, 0])
    before = np.flatnonzero(
        np.isclose(r, geometry.band_outer_radius, atol=1e-9)
        & (angle < geometry.rotor_start_angle - spacing / 2)
    )
    assert len(before) == needed
    assert mesh.rotor_nodes[before].all()
    assert set(before.tolist()) <= set(mesh.periodic[:, 0].tolist())
    # Stator pairs, rotor pairs and one pair per copy.
    cut_pairs = len(mesh.periodic) - needed
    assert cut_pairs > 10


# Other designs and the full machine


def test_full_machine_mesh_closes_on_itself(reference: Design) -> None:
    geometry = machine_for(reference).build_geometry(full=True)
    mesh = build_mesh(geometry)
    assert len(mesh.periodic) == 0 and mesh.antiperiodic is False
    assert mesh.region_names[-1] == BAND and len(mesh.region_names) == 6 * 77 + 1
    # 36 and 42 arcs on the rings need a multiple of 252, doubled as above.
    assert mesh.band_divisions % 504 == 0
    assert mesh.triangle_areas().sum() == pytest.approx(geometry.area(), rel=1e-5)
    assert mesh.smallest_angles().min() > 20
    # With no cut edges, only the bore and the outer surface are open.
    assert_conforming(mesh, geometry)


@pytest.mark.parametrize(
    "changes, antiperiodic",
    [
        ({"slots": 12, "poles": 10, "geometry__stator__tooth_tip_taper_mm": 1.2}, True),
        ({"slots": 24, "poles": 28}, True),
        ({"slots": 12, "poles": 8, **SQUARE_TIP}, False),
        (SQUARE_TIP, True),
        ({"geometry__magnet__arc_fraction": 1}, True),
        ({"geometry__air_gap_mm": 1.5}, True),
        # A cut edge of the rotor lies along a slot centre line of the stator.
        ({"slots": 12, "poles": 4, **SQUARE_TIP}, True),
        # A magnet edge lies along the cut edge of the stator.
        ({"slots": 12, "poles": 8, "geometry__magnet__arc_fraction": 2 / 3, **SQUARE_TIP}, False),
    ],
    ids=[
        "12s10p", "24s28p", "12s8p", "square_tip", "full_magnet_ring", "wide_gap",
        "12s4p", "magnet_edge_on_cut",
    ],
)
def test_other_designs_mesh_too(reference: Design, changes: dict, antiperiodic: bool) -> None:
    geometry = machine_for(variant(reference, **changes)).build_geometry()
    mesh = build_mesh(geometry)
    assert mesh.antiperiodic is antiperiodic
    assert set(np.unique(mesh.triangle_region)) == set(range(len(mesh.region_names)))
    assert mesh.triangle_areas().min() > 0
    assert mesh.triangle_areas().sum() == pytest.approx(geometry.area(), rel=1e-4)
    # Triangles are well shaped except at the fine point where the underside
    # of a tooth tip meets the slot, which is as sharp as the design makes it.
    angles = mesh.smallest_angles()
    sharp = {mesh.region_names[i] for i in mesh.triangle_region[angles < 20]}
    assert all(name.startswith("slot_air") for name in sharp)
    assert np.percentile(angles, 1) > 15
    assert_conforming(mesh, geometry)
    assert_periodic(mesh, geometry)
    for region in geometry.regions:
        assert mesh.region_area(region.name) == pytest.approx(region.area(), rel=1e-2)


def test_band_cells_are_cut_on_alternating_diagonals(mesh: Mesh, geometry: Geometry) -> None:
    # A diagonal joins a stator ring node to a rotor ring node at a different
    # angle. Going round the gap it must lean forwards and backwards in turn.
    band = triangles_of(mesh, BAND)
    r = radius(mesh)
    angle = np.arctan2(mesh.nodes[:, 1], mesh.nodes[:, 0])
    spacing = geometry.span_angle / mesh.band_divisions
    leans = {}
    for triangle in band:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            inner, outer = sorted((triangle[a], triangle[b]), key=lambda n: r[n])
            step = (angle[outer] - angle[inner]) / spacing
            if not math.isclose(r[inner], r[outer], abs_tol=1e-9) and abs(step) > 0.5:
                leans[round((angle[inner] + angle[outer]) / spacing)] = 1 if step > 0 else -1
    in_order = [leans[key] for key in sorted(leans)]
    assert len(in_order) == mesh.band_divisions
    assert all(a == -b for a, b in zip(in_order, in_order[1:]))


def test_region_too_thin_to_mesh_is_reported(reference: Design) -> None:
    # A magnet gap far thinner than an atom: its outline has no inside.
    design = variant(reference, geometry__magnet__arc_fraction=0.99999999999)
    geometry = machine_for(design).build_geometry()
    with pytest.raises(ValueError, match="magnet_gap_0.* too thin to mesh"):
        build_mesh(geometry)


def test_edge_too_short_to_mesh_is_reported(reference: Design) -> None:
    # Magnets 99.9999 % of the pole pitch leave a gap a few nanometres wide.
    design = variant(reference, geometry__magnet__arc_fraction=0.999999)
    geometry = machine_for(design).build_geometry()
    with pytest.raises(ValueError, match="magnet_gap_0.* too short to mesh"):
        build_mesh(geometry)


@pytest.mark.parametrize("size", [1.0, 2.0])
def test_band_of_a_slice_has_an_even_number_of_spacings(reference: Design, size: float) -> None:
    # A slice of three slots and one pole: its arcs alone would give an odd
    # number, and the alternating cut of the band cells would then not carry
    # on from one slice into the next.
    design = variant(reference, slots=18, poles=6)
    mesh = build_mesh(machine_for(design).build_geometry(), size)
    assert mesh.antiperiodic
    assert mesh.band_divisions % 2 == 0
    assert mesh.band.divisions == mesh.band_divisions
