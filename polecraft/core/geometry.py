"""Two-dimensional regions: what a machine hands to the mesher and the solver.

A machine is described as a set of regions, each a closed outline of straight
lines and circular arcs with a material and, where it applies, a
magnetisation or a coil side. Nothing here knows which machine the regions
came from.

Lengths are in metres and angles in radians. Angles are measured
counter-clockwise from the x axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

Point = tuple[float, float]

# Vertices closer than this are treated as the same point (metres).
SAME_POINT = 1e-12

# Arcs sweeping more than this are cut in two (radians, about 115 degrees).
LONGEST_ARC = 2.0


def polar(radius: float, angle: float) -> Point:
    """The point at a radius and angle from the origin."""
    return (radius * math.cos(angle), radius * math.sin(angle))


def _rotate(point: Point, angle: float) -> Point:
    c, s = math.cos(angle), math.sin(angle)
    return (c * point[0] - s * point[1], s * point[0] + c * point[1])


@dataclass(frozen=True)
class Edge:
    """A straight line, or a circular arc if ``centre`` is given.

    An arc runs from start to end counter-clockwise about its centre if
    ``ccw`` is true, clockwise otherwise, and always sweeps less than half a
    turn, so start, end and centre define it without ambiguity.
    """

    start: Point
    end: Point
    centre: Point | None = None
    ccw: bool = True

    @property
    def radius(self) -> float:
        """Arc radius; 0 for a straight line."""
        if self.centre is None:
            return 0.0
        return math.dist(self.start, self.centre)

    @property
    def sweep(self) -> float:
        """Angle the arc turns through, always positive; 0 for a straight line."""
        if self.centre is None:
            return 0.0
        a0 = math.atan2(self.start[1] - self.centre[1], self.start[0] - self.centre[0])
        a1 = math.atan2(self.end[1] - self.centre[1], self.end[0] - self.centre[0])
        turn = a1 - a0 if self.ccw else a0 - a1
        return turn % (2 * math.pi)

    def points(self, tolerance: float) -> list[Point]:
        """Points along the edge, start included and end excluded.

        An arc is cut into chords that stay within ``tolerance`` of it.
        """
        if self.centre is None:
            return [self.start]
        radius, sweep = self.radius, self.sweep
        # A chord spanning angle t lies r (1 - cos(t/2)) ~ r t^2 / 8 inside the arc.
        pieces = max(1, math.ceil(sweep / math.sqrt(8 * tolerance / radius)))
        a0 = math.atan2(self.start[1] - self.centre[1], self.start[0] - self.centre[0])
        step = (sweep if self.ccw else -sweep) / pieces
        return [self.start] + [
            (
                self.centre[0] + radius * math.cos(a0 + i * step),
                self.centre[1] + radius * math.sin(a0 + i * step),
            )
            for i in range(1, pieces)
        ]

    def rotated(self, angle: float) -> Edge:
        """The same edge turned about the origin."""
        centre = None if self.centre is None else _rotate(self.centre, angle)
        return Edge(_rotate(self.start, angle), _rotate(self.end, angle), centre, self.ccw)


@dataclass(frozen=True)
class Magnetisation:
    """How a magnet region is magnetised."""

    pattern: str  # "parallel": one direction throughout; "radial": along the radius
    axis_angle: float  # angle of the magnet's centre line
    polarity: int  # +1 points away from the machine axis, -1 towards it

    def direction(self, x: float, y: float) -> Point:
        """Unit vector of the magnetisation at a point of the magnet."""
        angle = self.axis_angle if self.pattern == "parallel" else math.atan2(y, x)
        return (self.polarity * math.cos(angle), self.polarity * math.sin(angle))


@dataclass(frozen=True)
class CoilSide:
    """One side of a coil, filling a region of a slot."""

    phase: str  # "A", "B" or "C"
    tooth: int  # the tooth this coil is wound on
    sign: int  # +1: positive phase current flows out of the plane here
    turns: int  # conductors of the coil passing through the region


@dataclass(frozen=True)
class Region:
    """A closed area of one material."""

    name: str  # unique within a geometry, e.g. "tooth_3"
    kind: str  # "steel", "air", "coil" or "magnet"
    part: str  # what it is in the machine, e.g. "tooth" or "rotor_yoke"
    material: str  # grade name, or "air"
    boundary: tuple[Edge, ...]  # one closed loop, counter-clockwise, no holes
    magnetisation: Magnetisation | None = None
    coil_side: CoilSide | None = None

    def area(self) -> float:
        """Exact enclosed area in square metres.

        Shoelace formula on the vertices, plus for each arc the sliver between
        the arc and its chord, r^2 (t - sin t) / 2 for a sweep t. An arc that
        turns the same way as the loop bulges outwards and adds area.
        """
        total = 0.0
        for edge in self.boundary:
            (x0, y0), (x1, y1) = edge.start, edge.end
            total += 0.5 * (x0 * y1 - x1 * y0)
            if edge.centre is not None:
                sweep = edge.sweep
                sliver = 0.5 * edge.radius**2 * (sweep - math.sin(sweep))
                total += sliver if edge.ccw else -sliver
        return total

    def polygon(self, tolerance: float = 1e-8) -> np.ndarray:
        """The outline as an (n, 2) array of points, arcs cut into short chords."""
        points = [p for edge in self.boundary for p in edge.points(tolerance)]
        return np.array(points)

    def rotated(self, angle: float) -> Region:
        """The same region turned about the origin."""
        magnetisation = self.magnetisation
        if magnetisation is not None:
            magnetisation = replace(
                magnetisation, axis_angle=magnetisation.axis_angle + angle
            )
        return replace(
            self,
            boundary=tuple(edge.rotated(angle) for edge in self.boundary),
            magnetisation=magnetisation,
        )


def closed_loop(corners: Sequence[tuple[float, float, str]]) -> tuple[Edge, ...]:
    """Build a boundary from corners given as (radius, angle, edge to the next).

    The edge to the next corner is "line", or "ccw" or "cw" for an arc about
    the origin in that direction. The last corner joins back to the first.
    Corners that coincide are merged, so a caller may list a corner that only
    exists for some dimensions. A long arc is cut in two at its middle, which
    keeps every arc well under half a turn.
    """
    edges = []
    count = len(corners)
    for i in range(count):
        radius, angle, kind = corners[i]
        next_radius, next_angle, _ = corners[(i + 1) % count]
        start, end = polar(radius, angle), polar(next_radius, next_angle)
        if math.dist(start, end) < SAME_POINT:
            continue
        if kind == "line":
            edges.append(Edge(start, end))
            continue
        ccw = kind == "ccw"
        turn = (next_angle - angle if ccw else angle - next_angle) % (2 * math.pi)
        if turn > LONGEST_ARC:
            middle = polar(radius, angle + (turn if ccw else -turn) / 2)
            edges.append(Edge(start, middle, centre=(0.0, 0.0), ccw=ccw))
            edges.append(Edge(middle, end, centre=(0.0, 0.0), ccw=ccw))
        else:
            edges.append(Edge(start, end, centre=(0.0, 0.0), ccw=ccw))
    return tuple(edges)


def ring_sector(
    inner_radius: float,
    outer_radius: float,
    start_angle: float,
    end_angle: float,
    inner_breaks: Sequence[float] = (),
    outer_breaks: Sequence[float] = (),
) -> tuple[Edge, ...]:
    """Boundary of the part of a ring between two angles.

    ``inner_breaks`` and ``outer_breaks`` are angles at which the inner and
    outer arcs get an extra vertex. A neighbouring region that has a corner on
    one of these arcs needs the arc split there, so that the two regions share
    whole edges.
    """
    corners = [(inner_radius, start_angle, "line"), (outer_radius, start_angle, "ccw")]
    corners += [(outer_radius, angle, "ccw") for angle in sorted(outer_breaks)]
    corners += [(outer_radius, end_angle, "line"), (inner_radius, end_angle, "cw")]
    corners += [(inner_radius, angle, "cw") for angle in sorted(inner_breaks, reverse=True)]
    return closed_loop(corners)


@dataclass(frozen=True)
class Geometry:
    """A machine cross-section, or the slice of it that repeats.

    The stator and the rotor are each described in their own frame, so the
    rotor regions never change as it turns. They are joined by a band of air
    between two circles, which is not a region: it is the layer that is
    reconnected when the rotor moves.

    Each part covers the same angle, starting from its own start angle. In an
    antiperiodic slice the field and currents of the next slice are these
    with the opposite sign.
    """

    stator: tuple[Region, ...]
    rotor: tuple[Region, ...]  # at rotor angle 0
    inner_radius: float
    outer_radius: float
    band_inner_radius: float
    band_outer_radius: float
    stator_start_angle: float
    rotor_start_angle: float
    angle: float  # angle covered; 2 pi for a full machine
    antiperiodic: bool
    stack_length: float  # along the shaft

    @property
    def regions(self) -> tuple[Region, ...]:
        return self.stator + self.rotor

    def region(self, name: str) -> Region:
        """Look a region up by name. Raises KeyError if there is none."""
        for candidate in self.regions:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def band_area(self) -> float:
        return 0.5 * self.angle * (self.band_outer_radius**2 - self.band_inner_radius**2)

    def area(self) -> float:
        """Area of the whole slice, from the inner radius to the outer radius."""
        return 0.5 * self.angle * (self.outer_radius**2 - self.inner_radius**2)
