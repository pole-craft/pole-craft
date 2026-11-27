"""Write the same model for FEMM, to check the solver against.

FEMM (https://www.femm.info) is a separate, free 2D field program. Pole-Craft
does not need it and never calls it. ``femm_script`` writes a script in
FEMM's Lua that draws a machine from the same regions the solver uses, with
the same materials, currents and boundaries, solves it and writes torque,
flux linkage and the air-gap flux density to a text file. ``read_results``
reads that file back.

Run a script with: femm.exe -lua-script=<file> -windowhide

The whole machine is drawn, not the repeating slice, so the comparison also
checks the coupling of the slice edges. Like the rest of the core, this
module knows regions, materials, coils and magnets only, not machine types.

Lengths are millimetres in the script, as FEMM is used, and metres here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from polecraft.core.materials import AIR, COPPER, MU0, BHCurve, LinearMaterial, Magnet, Material

if TYPE_CHECKING:
    from polecraft.core.geometry import Edge, Geometry, Region

# Points closer than this are one point (metres): the outlines of
# neighbouring regions are worked out separately.
WELD = 1e-9

# Triangle size in the air gap, as a fraction of the thickness of the band
# between stator and rotor; and in everything else.
GAP_SIZE = 0.5
BULK_SIZE = 4.0

STATOR_GROUP, ROTOR_GROUP = 1, 2


def femm_script(
    geometry: Geometry,
    materials: Mapping[str, Material],
    cases: Sequence[Mapping[str, float]],
    results: str | Path,
    model: str | Path | None = None,
    fineness: float = 1.0,
    gap_points: int = 3600,
    surround: float | None = None,
    rotor_angles: Sequence[float] | None = None,
    timing: bool = False,
) -> str:
    """The text of a FEMM Lua script that solves a machine and writes results.

    ``geometry`` is the whole machine, not a slice. ``materials`` maps the
    grade names of the regions to their models, as for ``build_problem``: a
    steel must be a ``BHCurve``, whose table goes into FEMM point for point.
    ``cases`` lists the currents to solve for, each a current in amperes for
    every phase. FEMM writes the file ``results`` and saves its model as
    ``model`` (next to the results if not given).

    ``fineness`` divides every triangle size: 2 halves them. ``gap_points``
    is the number of points round the middle of the air gap at which the
    flux density is written.

    The inner and outer surface are field lines, as in the solver. With
    ``surround`` they are not: the bore is filled with air and air is added
    outside, out to ``surround`` times the outer radius, which shows what
    the solver's boundary leaves out.

    ``rotor_angles`` gives the rotor angle of each case in radians,
    counter-clockwise from the position the geometry is built in. Everything
    on the rotor's side of the middle of the air gap is turned to it as one
    group and FEMM meshes again; each magnet's direction is set afresh, turned
    with the rotor. Without it every case is at angle 0.

    With ``timing`` the script also writes how long FEMM took, in seconds, for
    drawing the model (first case only) and, for each case, setting it up
    (turning the rotor, currents), meshing, solving, loading the solution and
    reading the results. In the first case the mesh is made by its own call,
    so that its time is separate from the solve. In later cases it is made
    inside the solve, and the mesh time written is about zero. For timing,
    keep ``gap_points`` small.
    """
    if not math.isclose(geometry.span_angle, 2.0 * math.pi, rel_tol=1e-9):
        raise ValueError("the FEMM model needs the whole machine: build_geometry(full=True)")
    if not cases:
        raise ValueError("at least one case of phase currents is needed")
    if not (math.isfinite(fineness) and fineness > 0.0):
        raise ValueError(f"fineness must be a positive number, not {fineness}")
    if gap_points < 8:
        raise ValueError(f"gap_points must be at least 8, not {gap_points}")
    if surround is not None and not surround > 1.0:
        raise ValueError(f"surround must be more than 1 (times the outer radius), not {surround}")
    if rotor_angles is not None:
        rotor_angles = [float(angle) for angle in rotor_angles]
        if len(rotor_angles) != len(cases) or not all(map(math.isfinite, rotor_angles)):
            raise ValueError("rotor_angles needs one finite angle in radians for each case")
    known = {"air": AIR, "copper": COPPER, **materials}
    phases = sorted({r.coil_side.phase for r in geometry.regions if r.coil_side is not None})
    for case in cases:
        if sorted(case) != phases:
            raise ValueError(
                f"every case needs a current for each phase ({', '.join(phases)}), "
                f"not {', '.join(sorted(case))}"
            )
    results = Path(results)
    model = results.with_suffix(".fem") if model is None else Path(model)

    band = geometry.band_outer_radius - geometry.band_inner_radius
    gap_size = GAP_SIZE * band / fineness
    bulk_size = BULK_SIZE * band / fineness
    lines = [
        "-- Written by Pole-Craft. Run with: femm.exe -lua-script=<this file> -windowhide",
        "newdocument(0)",
        *(["t_start = clock()"] if timing else []),
        f'mi_probdef(0, "millimeters", "planar", 1e-8, {_mm(geometry.stack_length)}, 30)',
        # Newer builds refine the mesh at every corner by themselves; the
        # sizes given here are to decide it, so that fineness means something.
        "if mi_smartmesh then mi_smartmesh(0) end",
    ]

    used = sorted({region.material for region in geometry.regions} | {"air"})
    for name in [*used, *phases]:
        if '"' in name or "\\" in name or "\n" in name:
            raise ValueError(
                f"{name!r}: a name written for FEMM cannot hold a double quote, "
                "a backslash or a line break"
            )
    for name in used:
        if name not in known:
            raise ValueError(f"no material model for '{name}'")
        lines += _material_lines(name, known[name])
    for phase in phases:
        lines.append(f'mi_addcircprop("{phase}", 0, 1)')
    lines.append('mi_addboundprop("field line", 0, 0, 0, 0, 0, 0, 0, 0, 0)')

    edges = _edges(geometry.regions)
    if surround is not None:
        far = surround * geometry.outer_radius
        quarter = [(far * math.cos(k * math.pi / 2), far * math.sin(k * math.pi / 2)) for k in range(4)]
        edges += [(quarter[k], quarter[(k + 1) % 4], (0.0, 0.0)) for k in range(4)]
        edge_radius = far
    for point in sorted({end for edge in edges for end in edge[:2]}):
        lines.append(f"mi_addnode({_mm(point[0])}, {_mm(point[1])})")
    for start, end, centre in edges:
        ends = f"{_mm(start[0])}, {_mm(start[1])}, {_mm(end[0])}, {_mm(end[1])}"
        if centre is None:
            lines.append(f"mi_addsegment({ends})")
            continue
        radius = math.dist(start, centre)
        sweep = _sweep(start, end, centre)
        on_surface = centre == (0.0, 0.0) and (
            math.isclose(radius, edge_radius, rel_tol=1e-9)
            if surround is not None
            else math.isclose(radius, geometry.inner_radius, rel_tol=1e-9)
            or math.isclose(radius, geometry.outer_radius, rel_tol=1e-9)
        )
        near_gap = (
            geometry.band_inner_radius - 2.0 * band
            <= radius
            <= geometry.band_outer_radius + 2.0 * band
        )
        # Chords about as long as the triangles beside them.
        size = gap_size if near_gap else bulk_size
        chord = min(10.0, max(0.02, math.degrees(size / radius)))
        middle = _on_arc(start, centre, 0.5 * sweep)
        lines += [
            f"mi_addarc({ends}, {math.degrees(sweep):.9f}, {chord:.4f})",
            f"mi_selectarcsegment({_mm(middle[0])}, {_mm(middle[1])})",
            f'mi_setarcsegmentprop({chord:.4f}, "{"field line" if on_surface else ""}", 0, 0)',
            "mi_clearselected()",
        ]

    def label(point: tuple[float, float], material: str, size: float, more: str) -> None:
        position = f"{_mm(point[0])}, {_mm(point[1])}"
        lines.extend(
            [
                f"mi_addblocklabel({position})",
                f"mi_selectlabel({position})",
                f'mi_setblockprop("{material}", 0, {size * 1e3:.6f}, {more})',
                "mi_clearselected()",
            ]
        )

    magnets = []  # rotor magnet labels, set again each time the rotor turns
    for region in geometry.regions:
        group = ROTOR_GROUP if region in geometry.rotor else STATOR_GROUP
        size = gap_size if region.kind == "air" else bulk_size
        circuit, direction, turns = "<None>", 0.0, 0
        if region.coil_side is not None:
            circuit = region.coil_side.phase
            turns = region.coil_side.sign * region.coil_side.turns
        point = inside_point(region)
        if region.magnetisation is not None:
            if region.magnetisation.pattern != "parallel":
                raise ValueError(
                    f"{region.name}: only parallel magnetisation can be written for FEMM so far"
                )
            x, y = region.magnetisation.direction(*point)
            direction = math.degrees(math.atan2(y, x))
            if group == ROTOR_GROUP:
                magnets.append((point, region.material, size, direction))
        label(point, region.material, size, f'"{circuit}", {direction:.9f}, {group}, {turns}')
    loose = []  # labels of no region (band, bore, outside air): in no group

    def loose_label(point: tuple[float, float], material: str, size: float) -> None:
        label(point, material, size, '"<None>", 0, 0, 0')
        loose.append(f"{_mm(point[0])}, {_mm(point[1])}")

    middle_radius = 0.5 * (geometry.band_inner_radius + geometry.band_outer_radius)
    # Between two sample points, so that no point falls on a label.
    loose_label(
        (middle_radius * math.cos(math.pi / gap_points), middle_radius * math.sin(math.pi / gap_points)),
        "air",
        gap_size,
    )
    if surround is not None:
        outside = 0.5 * (geometry.outer_radius + far)
        loose_label((outside, 0.0), "air", 2.0 * bulk_size)
        if geometry.inner_radius > 0.0:
            loose_label((0.0, 0.0), "air", 2.0 * bulk_size)
    elif geometry.inner_radius > 0.0:
        # The bore is outside the model, but FEMM wants every closed area named.
        loose_label((0.0, 0.0), "<No Mesh>", bulk_size)

    if rotor_angles is not None:
        # The band keeps stator and rotor apart, so the circle through its
        # middle has every rotor object on one side and every stator object
        # on the other. Selecting inside it picks one of the two.
        rotor_outside = any(
            math.hypot(*inside_point(region)) > middle_radius for region in geometry.rotor
        )
        inner, outer = (STATOR_GROUP, ROTOR_GROUP) if rotor_outside else (ROTOR_GROUP, STATOR_GROUP)
        everything = 2.0 * (far if surround is not None else geometry.outer_radius)
        lines += [
            f"mi_selectcircle(0, 0, {_mm(everything)}, 4)",
            f"mi_setgroup({outer})",
            "mi_clearselected()",
            f"mi_selectcircle(0, 0, {_mm(middle_radius)}, 4)",
            f"mi_setgroup({inner})",
            "mi_clearselected()",
        ]
        # The band's label lies on that circle and may go either way; torque
        # is taken over the rotor group, so it and the other labels of no
        # region go back to no group, as they are when nothing turns.
        for position in loose:
            lines += [f"mi_selectlabel({position})", "mi_setgroup(0)", "mi_clearselected()"]

    angles = 2.0 * math.pi * np.arange(gap_points) / gap_points
    lines += [
        f'mi_saveas("{_lua_path(model)}")',
        "gx = {" + ", ".join(_mm(middle_radius * math.cos(a)) for a in angles) + "}",
        "gy = {" + ", ".join(_mm(middle_radius * math.sin(a)) for a in angles) + "}",
        f'out = openfile("{_lua_path(results)}", "w")',
        f'write(out, "radius_mm {_mm(middle_radius)}\\n")',
    ]
    turned = 0.0
    for number, case in enumerate(cases):
        lines.append(f'write(out, "case {number}\\n")')
        if timing:
            lines.append("t0 = clock()")
            if number == 0:
                lines.append('write(out, format("time draw %.6f\\n", t0 - t_start))')
        if rotor_angles is not None:
            angle = rotor_angles[number]
            if angle != turned:
                lines += [
                    f"mi_selectgroup({ROTOR_GROUP})",
                    f"mi_moverotate(0, 0, {math.degrees(angle - turned)!r})",
                    "mi_clearselected()",
                ]
                for point, material, size, direction in magnets:
                    where = _on_arc(point, (0.0, 0.0), angle)
                    lines += [
                        f"mi_selectlabel({_mm(where[0])}, {_mm(where[1])})",
                        f'mi_setblockprop("{material}", 0, {size * 1e3:.6f}, "<None>", '
                        f"{direction + math.degrees(angle):.9f}, {ROTOR_GROUP}, 0)",
                        "mi_clearselected()",
                    ]
                turned = angle
            lines.append(f'write(out, "rotor_angle {angle!r}\\n")')
        for phase in phases:
            lines.append(f'mi_modifycircprop("{phase}", 1, {float(case[phase])!r})')
        if timing:
            lines += ["t1 = clock()", 'write(out, format("time setup %.6f\\n", t1 - t0))']
            if number == 0:
                # A separate mesh call on a model that has been turned made
                # FEMM 4.2 stop (a hidden message, as far as could be told),
                # so only the first case, which is not turned, is meshed apart.
                lines.append("mi_createmesh()")
            lines += [
                "t2 = clock()",
                'write(out, format("time mesh %.6f\\n", t2 - t1))',
                "mi_analyze(1)",
                "t3 = clock()",
                'write(out, format("time solve %.6f\\n", t3 - t2))',
                "mi_loadsolution()",
                "t4 = clock()",
                'write(out, format("time load %.6f\\n", t4 - t3))',
            ]
        else:
            lines += ["mi_analyze(1)", "mi_loadsolution()"]
        lines += [
            'write(out, format("nodes %d\\n", mo_numnodes()))',
            'write(out, format("elements %d\\n", mo_numelements()))',
            f"mo_groupselectblock({ROTOR_GROUP})",
            # Integral 22: torque about the origin from the weighted stress tensor.
            'write(out, format("torque %.12g\\n", mo_blockintegral(22)))',
            "mo_clearblock()",
        ]
        for phase in phases:
            lines += [
                f'amps, volts, linked = mo_getcircuitproperties("{phase}")',
                f'write(out, format("circuit {phase} %.12g %.12g\\n", amps, linked))',
            ]
        lines += [
            f"for k = 1, {gap_points} do",
            "  a, bx, by = mo_getpointvalues(gx[k], gy[k])",
            '  write(out, format("b %.9g %.9g\\n", bx, by))',
            "end",
        ]
        if timing:
            lines.append('write(out, format("time post %.6f\\n", clock() - t4))')
        lines.append("mo_close()")
    lines += ['write(out, "end\\n")', "closefile(out)", "mi_close()", "quit()"]
    return "\n".join(lines) + "\n"


def read_results(path: str | Path) -> list[dict]:
    """The results a FEMM script wrote, one dictionary for each case.

    Each has ``rotor_angle`` (radians; 0 if the script did not turn the
    rotor), ``torque`` (N m on the rotor, counter-clockwise positive),
    ``current`` and ``flux_linkage`` (amperes and webers, by phase),
    ``nodes`` and ``elements`` of FEMM's mesh, ``radius`` (metres) and
    ``angles`` (radians) of the points in the air gap, and ``radial`` and
    ``tangential``, the flux density there in tesla. A script written with
    ``timing`` also gives ``times``, FEMM's seconds by stage.
    """
    text = Path(path).read_text(encoding="utf-8")
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows or rows[-1] != ["end"]:
        raise ValueError(f"{path}: the results are not complete; FEMM did not finish")
    radius = None
    cases: list[dict] = []
    for row in rows[:-1]:
        key = row[0]
        if key == "radius_mm":
            radius = float(row[1]) * 1e-3
        elif key == "case":
            cases.append({"rotor_angle": 0.0, "current": {}, "flux_linkage": {}, "b": []})
        elif not cases:
            raise ValueError(f"{path}: '{key}' comes before any case")
        elif key in ("nodes", "elements"):
            cases[-1][key] = int(row[1])
        elif key == "torque":
            cases[-1]["torque"] = float(row[1])
        elif key == "rotor_angle":
            cases[-1]["rotor_angle"] = float(row[1])
        elif key == "time":
            cases[-1].setdefault("times", {})[row[1]] = float(row[2])
        elif key == "circuit":
            cases[-1]["current"][row[1]] = float(row[2])
            cases[-1]["flux_linkage"][row[1]] = float(row[3])
        elif key == "b":
            cases[-1]["b"].append((float(row[1]), float(row[2])))
        else:
            raise ValueError(f"{path}: unknown line '{' '.join(row)}'")
    for case in cases:
        b = np.array(case.pop("b"))
        if radius is None or "torque" not in case or len(b) == 0:
            raise ValueError(f"{path}: a case has no torque or no air-gap points")
        angles = 2.0 * math.pi * np.arange(len(b)) / len(b)
        case["radius"] = radius
        case["angles"] = angles
        case["radial"] = b[:, 0] * np.cos(angles) + b[:, 1] * np.sin(angles)
        case["tangential"] = b[:, 1] * np.cos(angles) - b[:, 0] * np.sin(angles)
    return cases


def inside_point(region: Region) -> tuple[float, float]:
    """A point well inside a region, for the label that names its material.

    Tried: points a little way in from the middle of every side, at several
    depths. The one farthest from the outline wins.
    """
    outline = region.polygon(tolerance=1e-7)
    following = np.roll(outline, -1, axis=0)
    middle = 0.5 * (outline + following)
    along = following - outline
    length = np.hypot(along[:, 0], along[:, 1])
    keep = length > 0.0
    middle, along, length = middle[keep], along[keep], length[keep]
    # The outline runs counter-clockwise, so the inside is on the left.
    inward = np.column_stack([-along[:, 1], along[:, 0]]) / length[:, None]
    size = math.sqrt(abs(region.area()))
    depths = size * np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5])
    candidates = (middle[:, None, :] + depths[None, :, None] * inward[:, None, :]).reshape(-1, 2)
    inside = _inside(candidates, outline)
    if not inside.any():
        raise ValueError(f"{region.name}: no point inside its outline was found")
    candidates = candidates[inside]
    clearance = _distance_to_outline(candidates, outline)
    best = candidates[int(np.argmax(clearance))]
    return (float(best[0]), float(best[1]))


def _inside(points: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """Which points lie inside a closed polygon (crossing count)."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x0, y0 = outline[:, 0][None, :], outline[:, 1][None, :]
    x1, y1 = np.roll(outline[:, 0], -1)[None, :], np.roll(outline[:, 1], -1)[None, :]
    spans = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (np.count_nonzero(spans & (x < crossing), axis=1) % 2) == 1


def _distance_to_outline(points: np.ndarray, outline: np.ndarray) -> np.ndarray:
    start = outline[None, :, :]
    along = (np.roll(outline, -1, axis=0) - outline)[None, :, :]
    offset = points[:, None, :] - start
    squared = (along**2).sum(axis=2)
    t = np.clip((offset * along).sum(axis=2) / np.where(squared > 0.0, squared, 1.0), 0.0, 1.0)
    return np.hypot(*np.moveaxis(offset - t[:, :, None] * along, 2, 0)).min(axis=1)


Point = tuple[float, float]
_Edge = tuple[Point, Point, "Point | None"]


def _edges(regions: Sequence[Region]) -> list[_Edge]:
    """Every edge of the regions once: (start, end, centre), arcs counter-clockwise.

    Neighbours share edges, each listing them in its own direction, and one
    may have a corner partway along the other's edge. Points are welded,
    every edge is cut at the points that lie on it, and what is left is
    listed once.
    """
    welded: dict[tuple[int, int], Point] = {}
    # Arc centres are welded too, apart from the corners: two arcs are the
    # same only if their centres are.
    centres: dict[tuple[int, int], Point] = {(0, 0): (0.0, 0.0)}

    def weld(point: Point, table: dict[tuple[int, int], Point]) -> Point:
        key = (round(point[0] / WELD), round(point[1] / WELD))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                found = table.get((key[0] + dx, key[1] + dy))
                if found is not None:
                    return found
        table[key] = (float(point[0]), float(point[1]))
        return table[key]

    raw: list[_Edge] = []
    for region in regions:
        for edge in region.boundary:
            start, end = weld(edge.start, welded), weld(edge.end, welded)
            if start == end:
                raise ValueError(
                    f"{region.name} has an edge shorter than {WELD} m, too small to draw for FEMM"
                )
            if edge.centre is None:
                raw.append((start, end, None))
            else:
                centre = weld(edge.centre, centres)
                raw.append((start, end, centre) if edge.ccw else (end, start, centre))
    points = np.array(sorted(welded.values()))

    pieces: set[_Edge] = set()
    for start, end, centre in raw:
        if centre is None:
            along = np.subtract(end, start)
            length = math.hypot(*along)
            offset = points - start
            position = offset @ along / length
            off_line = np.abs(offset[:, 0] * along[1] - offset[:, 1] * along[0]) / length
            on_edge = (off_line < WELD) & (position > WELD) & (position < length - WELD)
            stops = [start] + [tuple(p) for p in points[on_edge][np.argsort(position[on_edge])]] + [end]
            for a, b in zip(stops, stops[1:]):
                pieces.add((min(a, b), max(a, b), None))
        else:
            radius = math.dist(start, centre)
            sweep = _sweep(start, end, centre)
            offset = points - centre
            turned = (
                np.arctan2(offset[:, 1], offset[:, 0])
                - math.atan2(start[1] - centre[1], start[0] - centre[0])
            ) % (2.0 * math.pi)
            margin = WELD / radius
            on_edge = (
                (np.abs(np.hypot(offset[:, 0], offset[:, 1]) - radius) < WELD)
                & (turned > margin)
                & (turned < sweep - margin)
            )
            stops = [start] + [tuple(p) for p in points[on_edge][np.argsort(turned[on_edge])]] + [end]
            for a, b in zip(stops, stops[1:]):
                pieces.add((a, b, centre))
    return sorted(pieces, key=lambda piece: (piece[2] is not None, piece))


def _sweep(start: Point, end: Point, centre: Point) -> float:
    """Angle of the counter-clockwise arc from start to end, in radians."""
    a0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
    a1 = math.atan2(end[1] - centre[1], end[0] - centre[0])
    return (a1 - a0) % (2.0 * math.pi)


def _on_arc(start: Point, centre: Point, turn: float) -> Point:
    c, s = math.cos(turn), math.sin(turn)
    x, y = start[0] - centre[0], start[1] - centre[1]
    return (centre[0] + c * x - s * y, centre[1] + s * x + c * y)


def _material_lines(name: str, model: Material) -> list[str]:
    """FEMM commands that define one material."""
    if isinstance(model, Magnet):
        # FEMM's magnet: B = mu_r mu0 (H + Hc), so Hc = Br / (mu_r mu0).
        mu = model.relative_permeability
        coercivity = model.remanence / (mu * MU0)
        return [f'mi_addmaterial("{name}", {mu!r}, {mu!r}, {coercivity!r}, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)']
    if isinstance(model, LinearMaterial):
        mu = model.relative_permeability
        return [f'mi_addmaterial("{name}", {mu!r}, {mu!r}, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)']
    if isinstance(model, BHCurve):
        lines = [f'mi_addmaterial("{name}", 1000, 1000, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)']
        # The table itself, point for point: both programs start from the same numbers.
        return lines + [
            f'mi_addbhpoint("{name}", {float(b)!r}, {float(h)!r})' for b, h in zip(model.b, model.h)
        ]
    raise ValueError(
        f"'{name}': a material for FEMM must be linear, a magnet or a BH table, "
        f"not {type(model).__name__}"
    )


def _mm(metres: float) -> str:
    return f"{metres * 1e3:.9f}"


def _lua_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")
