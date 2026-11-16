"""Meshing: regions in, triangles out.

This is the only module that uses Gmsh, and it imports it only when a mesh is
asked for, so the rest of the package works without Gmsh installed. The mesh
comes back as plain arrays.

The stator and the rotor are meshed by Gmsh, each in its own frame. The band
of air between them is not: it is a single layer of triangles joining two
rings of equally spaced nodes, built here. Turning the rotor by one node
spacing only changes which nodes the band joins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from polecraft.core.geometry import Edge, Geometry, Region

BAND = "band"  # region name given to the triangles of the band

# Corners closer than this are one point (metres).
MERGE = 1e-9


@dataclass(frozen=True, eq=False)
class Mesh:
    """A triangle mesh of a machine cross-section, or of its repeating slice.

    Rotor nodes are at rotor angle 0. ``periodic`` lists node pairs
    ``(node, original)``: the field at ``node`` equals the field at
    ``original``, with the opposite sign if the mesh is antiperiodic. These
    are the nodes on the far cut edge of the stator and of the rotor, and the
    copies of rotor nodes that the band needs where the two slices do not
    line up.
    """

    nodes: np.ndarray  # (n, 2) positions in metres
    triangles: np.ndarray  # (m, 3) node numbers, counter-clockwise
    triangle_region: np.ndarray  # (m,) index into region_names
    region_names: tuple[str, ...]
    rotor_nodes: np.ndarray  # (n,) True for nodes that turn with the rotor
    periodic: np.ndarray  # (k, 2) node numbers: node, and the node it repeats
    antiperiodic: bool
    band_divisions: int  # node spacings around the band, over the meshed angle

    def triangle_areas(self) -> np.ndarray:
        """Area of every triangle in square metres."""
        a, b, c = (self.nodes[self.triangles[:, i]] for i in range(3))
        return 0.5 * (
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        )

    def region_area(self, name: str) -> float:
        """Summed triangle area of one region in square metres."""
        index = self.region_names.index(name)
        return float(self.triangle_areas()[self.triangle_region == index].sum())

    def smallest_angles(self) -> np.ndarray:
        """Smallest corner angle of every triangle, in degrees."""
        corners = self.nodes[self.triangles]  # (m, 3, 2)
        angles = []
        for i in range(3):
            u = corners[:, (i + 1) % 3] - corners[:, i]
            v = corners[:, (i + 2) % 3] - corners[:, i]
            cosine = (u * v).sum(axis=1) / (
                np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
            )
            angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        return np.min(angles, axis=0)


def build_mesh(geometry: Geometry, size: float = 1.0) -> Mesh:
    """Mesh a geometry with triangles.

    The mesh is finest in the air gap, where each layer is about as thick as
    its triangles are wide, and grows coarser with distance from it, so tooth
    tips and magnets get small triangles and the yokes large ones.

    ``size`` scales every triangle: 0.5 halves the edge lengths, for checking
    that results do not depend on the mesh. The useful range is about 0.5 to 2.

    The band is always one layer, a third of the air gap thick. ``size`` only
    changes its node spacing around the gap, so below about 0.5 its triangles
    become tall and thin. The spacing also has to put a node on every slot
    and every pole boundary, so slot and pole counts with little in common
    (39 slots with 38 poles, say) get a band finer than its thickness.

    A sharp corner in the design gives sharp triangles there, whatever the
    mesh size. The usual case is the air under a tooth tip whose taper is
    only just deep enough: the underside then meets the slot at a fine point.

    Raises ImportError if Gmsh is not installed or cannot be loaded, and ValueError if the
    geometry cannot be meshed, for example because a region has an edge too
    short to mesh.
    """
    if not math.isfinite(size) or size <= 0:
        raise ValueError(f"size must be a number greater than 0, got {size}")
    band_thickness = geometry.band_outer_radius - geometry.band_inner_radius
    fine = band_thickness * size
    coarse = (geometry.outer_radius - geometry.inner_radius) / 15 * size
    _check_edge_lengths(geometry, shortest=1e-3 * band_thickness)

    full = math.isclose(geometry.span_angle, 2 * math.pi)
    divisions = _band_divisions(geometry, fine)

    nodes, triangles, triangle_region = _mesh_regions(
        geometry, fine, coarse, growth=0.3 * size, divisions=divisions, full=full
    )
    names = [region.name for region in geometry.regions]
    stator_count = len(geometry.stator)

    rotor_nodes = np.zeros(len(nodes), dtype=bool)
    rotor_nodes[np.unique(triangles[triangle_region >= stator_count])] = True

    pairs: list[tuple[int, int]] = []
    if not full:
        pairs += _cut_edge_pairs(nodes, ~rotor_nodes, geometry.stator_start_angle, geometry.span_angle)
        pairs += _cut_edge_pairs(nodes, rotor_nodes, geometry.rotor_start_angle, geometry.span_angle)

    band = _build_band(geometry, nodes, rotor_nodes, divisions, full)
    nodes = np.vstack([nodes, band.new_nodes]) if len(band.new_nodes) else nodes
    rotor_nodes = np.concatenate([rotor_nodes, np.ones(len(band.new_nodes), dtype=bool)])
    pairs += band.pairs
    triangles = np.vstack([triangles, band.triangles])
    triangle_region = np.concatenate(
        [triangle_region, np.full(len(band.triangles), len(names), dtype=int)]
    )

    triangles = _counter_clockwise(nodes, triangles)
    return Mesh(
        nodes=nodes,
        triangles=triangles,
        triangle_region=triangle_region,
        region_names=tuple(names) + (BAND,),
        rotor_nodes=rotor_nodes,
        periodic=np.array(pairs, dtype=int).reshape(-1, 2),
        antiperiodic=geometry.antiperiodic,
        band_divisions=divisions,
    )


def _check_edge_lengths(geometry: Geometry, shortest: float) -> None:
    for region in geometry.regions:
        # A feature far thinner than any triangle leaves an outline with no
        # inside: two edges, there and back.
        if len(region.boundary) < 3:
            raise ValueError(f"region {region.name} is too thin to mesh")
        for edge in region.boundary:
            length = math.dist(edge.start, edge.end)
            if length < shortest:
                raise ValueError(
                    f"region {region.name} has an edge only {length:.3g} m long, "
                    f"too short to mesh"
                )


def _on_circle(edge: Edge, radius: float) -> bool:
    return (
        edge.centre is not None
        and abs(math.hypot(*edge.start) - radius) < MERGE
        and abs(math.hypot(*edge.end) - radius) < MERGE
    )


def _band_arcs(regions: tuple[Region, ...], geometry: Geometry) -> tuple[float, int]:
    """Radius of the band circle these regions touch, and how many arcs lie on it."""
    for radius in (geometry.band_inner_radius, geometry.band_outer_radius):
        count = sum(_on_circle(edge, radius) for region in regions for edge in region.boundary)
        if count:
            return radius, count
    raise ValueError("no region reaches the band")


def _band_divisions(geometry: Geometry, target: float) -> int:
    """Number of equal node spacings around the band, over the meshed angle.

    It must be a whole multiple of the number of arcs on each ring, so that
    every arc gets a whole number of spacings. The stator and rotor slices
    start at different angles, and the difference must be a whole number of
    spacings too, so that the nodes of the two rings face each other.
    """
    _, stator_arcs = _band_arcs(geometry.stator, geometry)
    _, rotor_arcs = _band_arcs(geometry.rotor, geometry)
    base = math.lcm(stator_arcs, rotor_arcs)
    offset = geometry.stator_start_angle - geometry.rotor_start_angle
    for multiple in range(1, 7):
        steps = offset / (geometry.span_angle / (base * multiple))
        if abs(steps - round(steps)) < 1e-6:
            base *= multiple
            break
    else:
        raise ValueError("stator and rotor slices cannot be given matching band nodes")

    middle = 0.5 * (geometry.band_inner_radius + geometry.band_outer_radius)
    wanted = geometry.span_angle * middle / target
    return base * max(1, round(wanted / base))


def _mesh_regions(
    geometry: Geometry, fine: float, coarse: float, growth: float, divisions: int, full: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesh every region with Gmsh. Returns nodes, triangles and region index.

    Triangle edges are ``fine`` long at the middle of the air gap and grow by
    ``growth`` metres per metre of radial distance from it, up to ``coarse``.
    """
    try:
        import gmsh
    except ImportError as error:
        raise ImportError(
            "meshing needs the optional Gmsh package, which is not installed; "
            'install it with: pip install "polecraft[mesh]"'
        ) from error
    except OSError as error:
        # Gmsh loads its compiled library on import; on Linux that fails if a
        # system library such as libGLU is absent.
        raise ImportError(
            f"Gmsh is installed but could not be loaded ({error}); on Linux it needs "
            "a few system libraries, listed in the install section of the README"
        ) from error
    if gmsh.isInitialized():
        raise RuntimeError("Gmsh is already in use in this process; close it before meshing")

    regions = geometry.regions
    edges = [edge for region in regions for edge in region.boundary]
    corners = np.array([point for edge in edges for point in (edge.start, edge.end)])
    tree = cKDTree(corners)
    # Every corner is given the number of the first corner at the same place.
    same = [min(tree.query_ball_point(corner, MERGE)) for corner in corners]

    # Not reading the user's Gmsh settings keeps the mesh the same on every
    # machine, and leaving Ctrl-C alone lets this run outside the main thread.
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        geo = gmsh.model.geo

        point_tag = {i: geo.addPoint(*corners[i], 0.0) for i in sorted(set(same))}
        origin = geo.addPoint(0.0, 0.0, 0.0)

        # One curve per distinct edge. A region that runs it the other way
        # refers to it with a minus sign.
        curve_tag: dict[tuple[int, int, bool], int] = {}
        curve_edge: dict[int, Edge] = {}
        rotor_curves: set[int] = set()
        loops = []
        position = 0
        for index, region in enumerate(regions):
            loop = []
            for edge in region.boundary:
                a, b = same[position], same[position + 1]
                position += 2
                arc = edge.centre is not None
                if arc and math.hypot(*edge.centre) > MERGE:
                    raise ValueError("only arcs about the origin can be meshed")
                if (a, b, arc) in curve_tag:
                    loop.append(curve_tag[(a, b, arc)])
                elif (b, a, arc) in curve_tag:
                    loop.append(-curve_tag[(b, a, arc)])
                else:
                    if arc:
                        tag = geo.addCircleArc(point_tag[a], origin, point_tag[b])
                    else:
                        tag = geo.addLine(point_tag[a], point_tag[b])
                    curve_tag[(a, b, arc)] = tag
                    curve_edge[tag] = edge
                    if index >= len(geometry.stator):
                        rotor_curves.add(tag)
                    loop.append(tag)
            loops.append(loop)

        surfaces = [geo.addPlaneSurface([geo.addCurveLoop(loop)]) for loop in loops]
        geo.synchronize()

        # Equal node spacing on the two band rings.
        for part in (geometry.stator, geometry.rotor):
            radius, arcs = _band_arcs(part, geometry)
            for tag, edge in curve_edge.items():
                if _on_circle(edge, radius):
                    gmsh.model.mesh.setTransfiniteCurve(tag, divisions // arcs + 1)

        # The far cut edge of each part gets the same nodes as the near one.
        if not full:
            turn = geometry.span_angle
            c, s = math.cos(turn), math.sin(turn)
            rotation = [c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            # Each part is searched on its own: a cut edge of one can lie
            # along a slot centre line or a magnet edge of the other.
            stator_edges = {t: e for t, e in curve_edge.items() if t not in rotor_curves}
            rotor_edges = {t: e for t, e in curve_edge.items() if t in rotor_curves}
            for start, own in (
                (geometry.stator_start_angle, stator_edges),
                (geometry.rotor_start_angle, rotor_edges),
            ):
                near = _curves_on_ray(own, start)
                far = _curves_on_ray(own, start + turn)
                near, far = _matching_cut_curves(near, far)
                if far:
                    gmsh.model.mesh.setPeriodic(1, far, near, rotation)

        # Triangle size grows with distance from the middle of the air gap.
        middle = 0.5 * (geometry.band_inner_radius + geometry.band_outer_radius)
        growing = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            growing, "F", f"{fine!r} + {growth!r} * abs(sqrt(x*x + y*y) - {middle!r})"
        )
        limit = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(limit, "F", repr(coarse))
        smallest = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(smallest, "FieldsList", [growing, limit])
        gmsh.model.mesh.field.setAsBackgroundMesh(smallest)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # frontal Delaunay: well-shaped triangles

        try:
            gmsh.model.mesh.generate(2)
        except Exception as error:
            raise ValueError(f"Gmsh could not mesh the geometry: {error}") from error

        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        positions = coordinates.reshape(-1, 3)[:, :2]
        number = {int(tag): i for i, tag in enumerate(tags)}

        triangles, triangle_region = [], []
        for index, surface in enumerate(surfaces):
            kinds, _, node_tags = gmsh.model.mesh.getElements(2, surface)
            for kind, element_nodes in zip(kinds, node_tags):
                if kind != 2:  # 2 is Gmsh's code for a three-node triangle
                    raise ValueError("Gmsh returned elements that are not triangles")
                found = np.array([number[int(tag)] for tag in element_nodes]).reshape(-1, 3)
                triangles.append(found)
                triangle_region.append(np.full(len(found), index, dtype=int))
    finally:
        gmsh.finalize()

    triangles = np.vstack(triangles)
    triangle_region = np.concatenate(triangle_region)

    # Drop nodes no triangle uses, such as the one Gmsh puts at the origin.
    used = np.unique(triangles)
    renumber = np.full(len(positions), -1, dtype=int)
    renumber[used] = np.arange(len(used))
    return positions[used], renumber[triangles], triangle_region


def _curves_on_ray(curve_edge: dict[int, Edge], angle: float) -> list[tuple[float, int]]:
    """Straight curves lying on the ray at an angle, as (inner radius, tag)."""
    found = []
    for tag, edge in curve_edge.items():
        if edge.centre is not None:
            continue
        on_ray = all(
            abs(math.remainder(math.atan2(y, x) - angle, 2 * math.pi)) < 1e-9
            for x, y in (edge.start, edge.end)
        )
        if on_ray:
            found.append((min(math.hypot(*edge.start), math.hypot(*edge.end)), tag))
    return sorted(found)


def _matching_cut_curves(
    near: list[tuple[float, int]], far: list[tuple[float, int]]
) -> tuple[list[int], list[int]]:
    """Pair the curves of the two cut edges by radius. Returns their tags."""
    if len(near) != len(far) or any(abs(a[0] - b[0]) > MERGE for a, b in zip(near, far)):
        raise ValueError("the two cut edges of a slice do not have matching corners")
    return [tag for _, tag in near], [tag for _, tag in far]


def _cut_edge_pairs(
    nodes: np.ndarray, selected: np.ndarray, start: float, span: float
) -> list[tuple[int, int]]:
    """Pair each node on the far cut edge with the node at the same radius on the near one."""
    angle = np.arctan2(nodes[:, 1], nodes[:, 0])
    radius = np.hypot(nodes[:, 0], nodes[:, 1])

    def on_ray(ray: float) -> np.ndarray:
        off = np.abs(np.remainder(angle - ray + np.pi, 2 * np.pi) - np.pi)
        chosen = np.flatnonzero(selected & (off * radius < MERGE))
        return chosen[np.argsort(radius[chosen])]

    near, far = on_ray(start), on_ray(start + span)
    if len(near) != len(far) or np.any(np.abs(radius[near] - radius[far]) > MERGE):
        raise ValueError("the nodes on the two cut edges of a slice do not match")
    return [(int(f), int(n)) for f, n in zip(far, near)]


@dataclass
class _Band:
    triangles: np.ndarray
    new_nodes: np.ndarray
    pairs: list[tuple[int, int]]


def _ring(nodes: np.ndarray, selected: np.ndarray, radius: float, start: float) -> np.ndarray:
    """Numbers of the selected nodes on a circle, in order of angle from ``start``."""
    distance = np.hypot(nodes[:, 0], nodes[:, 1])
    chosen = np.flatnonzero(selected & (np.abs(distance - radius) < MERGE))
    angle = np.arctan2(nodes[chosen, 1], nodes[chosen, 0]) - start
    # A small margin keeps the first node from wrapping to the end of the turn.
    angle = np.remainder(angle + 1e-9, 2 * np.pi)
    return chosen[np.argsort(angle)]


def _build_band(
    geometry: Geometry, nodes: np.ndarray, rotor_nodes: np.ndarray, divisions: int, full: bool
) -> _Band:
    """Join the stator ring and the rotor ring with one layer of triangles.

    Both rings have a node every ``span / divisions``. Stator node i faces the
    rotor node at the same angle. Where the rotor slice does not reach, the
    band uses a copy of the rotor node one slice away, turned into place and
    tied to its original as a periodic pair.
    """
    span = geometry.span_angle
    spacing = span / divisions
    stator_radius, _ = _band_arcs(geometry.stator, geometry)
    rotor_radius, _ = _band_arcs(geometry.rotor, geometry)
    stator_ring = _ring(nodes, ~rotor_nodes, stator_radius, geometry.stator_start_angle)
    rotor_ring = _ring(nodes, rotor_nodes, rotor_radius, geometry.rotor_start_angle)

    expected = divisions if full else divisions + 1
    if len(stator_ring) != expected or len(rotor_ring) != expected:
        raise ValueError(
            f"expected {expected} nodes on each band ring, found "
            f"{len(stator_ring)} and {len(rotor_ring)}"
        )

    # Rotor ring node facing stator ring node 0.
    shift = round((geometry.stator_start_angle - geometry.rotor_start_angle) / spacing)

    new_nodes: list[np.ndarray] = []
    pairs: list[tuple[int, int]] = []
    copies: dict[int, int] = {}

    def facing(i: int) -> int:
        """Number of the rotor-side node at the angle of stator ring node i."""
        j = i + shift
        if full:
            return int(rotor_ring[j % divisions])
        if 0 <= j <= divisions:
            return int(rotor_ring[j])
        if j not in copies:
            # One slice before the rotor's first node, or after its last.
            turns = -1 if j < 0 else 1
            original = int(rotor_ring[j - turns * divisions])
            c, s = math.cos(turns * span), math.sin(turns * span)
            x, y = nodes[original]
            copies[j] = len(nodes) + len(new_nodes)
            new_nodes.append(np.array([c * x - s * y, s * x + c * y]))
            pairs.append((copies[j], original))
        return copies[j]

    triangles = []
    for i in range(divisions):
        following = (i + 1) % divisions if full else i + 1
        s0, s1 = int(stator_ring[i]), int(stator_ring[following])
        r0, r1 = facing(i), facing(i + 1)
        # Each four-sided cell is cut along a diagonal, alternating from one
        # cell to the next. Cut all the same way, the triangles would all lean
        # one way round the gap and bias a torque worked out in the band.
        if i % 2 == 0:
            triangles += [(s0, s1, r1), (s0, r1, r0)]
        else:
            triangles += [(s0, s1, r0), (s1, r1, r0)]

    return _Band(
        triangles=np.array(triangles, dtype=int),
        new_nodes=np.array(new_nodes).reshape(-1, 2),
        pairs=pairs,
    )


def _counter_clockwise(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Reorder the corners of any clockwise triangle."""
    a, b, c = (nodes[triangles[:, i]] for i in range(3))
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    flipped = triangles.copy()
    clockwise = area < 0
    flipped[clockwise, 1], flipped[clockwise, 2] = triangles[clockwise, 2], triangles[clockwise, 1]
    return flipped
