"""Turning the rotor: the band joined again, nothing meshed again."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import AIR, LinearMaterial, Magnet, load_material
from polecraft.core.mesh import BAND, Band, Mesh, join_band
from polecraft.core.motion import band_steps, rotate, step_angle
from polecraft.core.post import flux_linkage, torque
from polecraft.core.problem import Gap, Problem, _directions, build_problem
from polecraft.core.solver import shape_gradients, solve
from polecraft.core.winding import phase_axis_angle

# A ring machine built by hand: no slots, so its answers are known


STATOR_RADII = 0.030 + 0.001 * np.arange(9)  # iron, a layer of current, air
ROTOR_RADII = 0.039 + 0.001 * np.arange(8)  # air, magnets, iron
IRON, COIL, MAGNET, OTHER = 1, 2, 3, 0


def _part(radii: np.ndarray, angles: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Nodes on circles at the given angles, and triangles cut on alternating diagonals."""
    rows = len(radii) - 1
    columns = len(angles) if closed else len(angles) - 1
    nodes = np.array([[r * math.cos(t), r * math.sin(t)] for t in angles for r in radii])
    triangles = []
    for i in range(columns):
        j = (i + 1) % len(angles)
        for k in range(rows):
            a, b = i * (rows + 1) + k, j * (rows + 1) + k
            c, d = b + 1, a + 1
            # Counter-clockwise: b is further round than a, and d further out.
            if (i + k) % 2 == 0:
                triangles += [(a, c, b), (a, d, c)]
            else:
                triangles += [(a, d, b), (b, d, c)]
    return nodes, np.array(triangles)


def ring_machine(
    poles: int = 8, model_poles: int | None = None, per_pole: int = 12, offset: int = 0
) -> dict:
    """Stator inside, rotor outside, each a structured mesh, joined by a band.

    The stator carries a layer of current that varies as the cosine of the
    electrical angle. The rotor carries magnets magnetised along the radius,
    outwards and inwards in turn. ``model_poles`` poles are meshed, all of
    them if it is not given. The rotor is meshed ``offset`` node spacings
    back from the stator.
    """
    model_poles = model_poles or poles
    closed = model_poles == poles
    columns = model_poles * per_pole
    spacing = 2.0 * math.pi / (poles * per_pole)
    span = columns * spacing
    around = columns if closed else columns + 1
    stator_nodes, stator_triangles = _part(STATOR_RADII, spacing * np.arange(around), closed)
    rotor_nodes, rotor_triangles = _part(
        ROTOR_RADII, spacing * (np.arange(around) - offset), closed
    )
    stator_count, per_stator, per_rotor = len(stator_nodes), len(STATOR_RADII), len(ROTOR_RADII)
    nodes = np.vstack([stator_nodes, rotor_nodes])
    triangles = np.vstack([stator_triangles, rotor_triangles + stator_count])
    is_rotor = np.arange(len(nodes)) >= stator_count

    stator_rows, rotor_rows = per_stator - 1, per_rotor - 1
    layer = np.concatenate(
        [np.repeat(np.tile(np.arange(stator_rows), columns), 2),
         np.repeat(np.tile(np.arange(rotor_rows), columns), 2)]
    )
    column = np.concatenate(
        [np.repeat(np.arange(columns), 2 * stator_rows), np.repeat(np.arange(columns), 2 * rotor_rows)]
    )
    on_rotor = np.arange(len(triangles)) >= len(stator_triangles)
    kind = np.where(
        on_rotor,
        np.where(layer == 0, OTHER, np.where(layer < 4, MAGNET, IRON)),
        np.where(layer < 3, IRON, np.where(layer < 6, COIL, OTHER)),
    )

    pairs = []
    if not closed:
        pairs += [(columns * per_stator + k, k) for k in range(per_stator)]
        pairs += [
            (stator_count + columns * per_rotor + k, stator_count + k) for k in range(per_rotor)
        ]
    bare = Mesh(
        nodes=nodes,
        triangles=triangles,
        triangle_region=on_rotor.astype(int),
        region_names=("stator", "rotor", BAND),
        rotor_nodes=is_rotor,
        periodic=np.array(pairs, dtype=int).reshape(-1, 2),
        antiperiodic=not closed and model_poles % 2 == 1,
        band_divisions=columns,
        band=Band(
            stator_ring=np.arange(around) * per_stator + stator_rows,
            rotor_ring=stator_count + np.arange(around) * per_rotor,
            span=span,
            facing=offset,
            rest=nodes,
            triangles=len(triangles),
            pairs=len(pairs),
        ),
    )
    return {
        "mesh": join_band(bare),
        "kind": kind,
        "pole": column // per_pole,
        "pole_pairs": poles // 2,
        "repeats": poles // model_poles,
    }


def ring_problem(model: dict, mesh: Mesh, current: float = 4e6) -> Problem:
    """The field problem of the ring machine on its mesh, as turned."""
    band = mesh.band
    meshed = band.triangles
    count = len(mesh.triangles)
    centre = mesh.nodes[mesh.triangles].mean(axis=1)
    angle = np.arctan2(centre[:, 1], centre[:, 0])
    outward = centre / np.linalg.norm(centre, axis=1)[:, None]
    kind = np.concatenate([model["kind"], np.full(count - meshed, OTHER)])
    polarity = np.concatenate([1 - 2 * (model["pole"] % 2), np.zeros(count - meshed)])
    p = model["pole_pairs"]
    in_coil = kind == COIL
    radius = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    return Problem(
        nodes=mesh.nodes,
        triangles=mesh.triangles,
        materials=(AIR, LinearMaterial(1000.0), Magnet(1.2)),
        triangle_material=np.where(kind == IRON, 1, np.where(kind == MAGNET, 2, 0)),
        current_density=np.where(in_coil, current * np.cos(p * angle), 0.0),
        remanence=np.where((kind == MAGNET)[:, None], 1.2 * polarity[:, None] * outward, 0.0),
        fixed_nodes=np.flatnonzero(
            np.isclose(radius, STATOR_RADII[0]) | np.isclose(radius, ROTOR_RADII[-1])
        ),
        periodic=mesh.periodic,
        antiperiodic=mesh.antiperiodic,
        depth=0.020,
        repeats=model["repeats"],
        windings={
            "d": np.where(in_coil, 1e6 * np.cos(p * angle), 0.0),
            "q": np.where(in_coil, 1e6 * np.sin(p * angle), 0.0),
        },
        gap=Gap(
            triangles=np.arange(meshed, count),
            inner_radius=STATOR_RADII[-1],
            outer_radius=ROTOR_RADII[0],
            rotor_outside=True,
        ),
    )


def ring_results(model: dict, angle: float, current: float = 4e6) -> tuple[float, dict]:
    problem = ring_problem(model, rotate(model["mesh"], angle), current)
    solution = solve(problem)
    return torque(problem, solution), flux_linkage(problem, solution)


def force_on_the_current(problem: Problem, solution) -> float:
    """Torque on the current layer in N m, from the force J x B on each triangle."""
    _, area = shape_gradients(problem.nodes, problem.triangles)
    centre = problem.nodes[problem.triangles].mean(axis=1)
    r_b_r = (solution.flux_density * centre).sum(axis=1)
    return problem.repeats * problem.depth * (problem.current_density * r_b_r * area).sum()


def assert_conforming(mesh: Mesh, surfaces: tuple[float, float], rings: tuple[float, float]) -> None:
    """Triangles meet side to side; only the outline has sides used once.

    The outline is the inner and outer surface, the cut edges of a slice,
    the stretches of the rotor's ring where the band joins copies, and the
    two ends of the band.
    """
    band = mesh.band
    edges = np.sort(
        np.vstack([mesh.triangles[:, [0, 1]], mesh.triangles[:, [1, 2]], mesh.triangles[:, [2, 0]]]),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    assert counts.max() == 2, "a side belongs to more than two triangles"
    alone = unique[counts == 1]
    radius = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])

    def on(nodes: np.ndarray, circle: float) -> np.ndarray:
        return np.abs(radius[nodes] - circle) < 1e-9

    a, b = alone[:, 0], alone[:, 1]
    on_surface = (on(a, surfaces[0]) & on(b, surfaces[0])) | (on(a, surfaces[1]) & on(b, surfaces[1]))
    if band.full:
        assert on_surface.all()
        return
    on_cut = np.zeros(len(mesh.nodes), dtype=bool)
    on_cut[mesh.periodic[: band.pairs].ravel()] = True
    along_rotor_ring = on(a, rings[1]) & on(b, rings[1])
    band_end = (on(a, rings[0]) & on(b, rings[1])) | (on(a, rings[1]) & on(b, rings[0]))
    assert (on_surface | (on_cut[a] & on_cut[b]) | along_rotor_ring | band_end).all()
    assert band_end.sum() == 2


def sorted_sides(mesh: Mesh, triangles: np.ndarray) -> np.ndarray:
    """The three side lengths of each triangle, smallest first."""
    corners = mesh.nodes[triangles]
    sides = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
    return np.sort(sides, axis=1)


@pytest.fixture(scope="module")
def ring() -> dict:
    return ring_machine()


@pytest.fixture(scope="module")
def pole() -> dict:
    """One pole of the ring machine: an antiperiodic slice."""
    return ring_machine(model_poles=1, per_pole=24, offset=-3)


ANGLES = [0.0, 0.0137, 0.2, -0.61, 1.9, 3.3, -7.7, 40.0]


# How the band is joined for an angle


@pytest.mark.parametrize("model_poles", [1, 2, 3, 8])
def test_steps_and_turn_are_the_angle_less_whole_repeats(model_poles: int) -> None:
    mesh = ring_machine(model_poles=model_poles, offset=2)["mesh"]
    band = mesh.band
    spacing = step_angle(mesh)
    assert spacing == pytest.approx(2 * math.pi / 96)
    assert spacing == pytest.approx(band.span / band.divisions)
    repeat = band.span * (2 if mesh.antiperiodic else 1)
    for angle in np.random.default_rng(1).uniform(-30.0, 30.0, 300):
        steps, turn = band_steps(mesh, angle)
        assert abs(turn - steps * spacing) <= 0.5 * spacing * (1 + 1e-9)
        whole = (angle - turn) / repeat
        assert whole == pytest.approx(round(whole), abs=1e-9)
        if not band.full:
            # The rotor slice stays within one slice of the stator's.
            assert -band.divisions <= band.facing - steps <= band.divisions


def test_a_small_angle_is_not_changed() -> None:
    mesh = ring_machine(model_poles=1)["mesh"]
    spacing = step_angle(mesh)
    assert band_steps(mesh, 0.0) == (0, 0.0)
    assert band_steps(mesh, 3 * spacing) == (3, 3 * spacing)
    steps, turn = band_steps(mesh, 2.4 * spacing)
    assert steps == 2 and turn == pytest.approx(2.4 * spacing, rel=1e-14)
    assert band_steps(mesh, -2.6 * spacing)[0] == -3


@pytest.mark.parametrize("model_poles", [1, 2, 8])
def test_very_large_angles_and_half_spacings_can_be_turned_to(model_poles: int) -> None:
    mesh = ring_machine(model_poles=model_poles)["mesh"]
    spacing = step_angle(mesh)
    rng = np.random.default_rng(3)
    angles = list(rng.uniform(-1e12, 1e12, 50))
    angles += [(k + 0.5) * spacing for k in rng.integers(-10**9, 10**9, 200)]
    for angle in angles:
        steps, turn = band_steps(mesh, angle)
        assert abs(turn - steps * spacing) <= 0.5 * spacing
        assert rotate(mesh, angle).triangle_areas().min() > 0


def test_band_of_a_slice_needs_an_even_number_of_spacings() -> None:
    # Cut on alternating diagonals, an odd number would put two cells cut
    # the same way side by side where one slice meets the next.
    with pytest.raises(ValueError, match="even number"):
        ring_machine(model_poles=1, per_pole=11)
    assert ring_machine(per_pole=11)["mesh"].band.divisions == 88  # a whole ring may


def test_joining_the_band_by_hand_sets_the_rotor_angle(pole: dict) -> None:
    mesh = pole["mesh"]
    spacing = step_angle(mesh)
    joined = join_band(rotate(mesh, 1.0), steps=2, turn=2.2 * spacing)
    assert joined.rotor_angle == joined.rotor_turn == 2.2 * spacing
    back = join_band(rotate(mesh, 1.0))
    assert back.rotor_angle == 0.0 and np.array_equal(back.nodes, mesh.nodes)


def test_turning_needs_a_band_and_a_finite_angle(ring: dict) -> None:
    mesh = ring["mesh"]
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            rotate(mesh, bad)
    bandless = dataclasses.replace(mesh, band=None)
    for call in (lambda: rotate(bandless, 0.1), lambda: step_angle(bandless),
                 lambda: band_steps(bandless, 0.1), lambda: join_band(bandless)):
        with pytest.raises(ValueError, match="no band"):
            call()


def test_join_band_refuses_what_it_cannot_join(pole: dict) -> None:
    mesh = pole["mesh"]
    spacing = step_angle(mesh)
    divisions = mesh.band.divisions
    with pytest.raises(ValueError, match="more than one slice"):
        join_band(mesh, steps=3 * divisions, turn=3 * divisions * spacing)
    with pytest.raises(ValueError, match="more than one slice"):
        join_band(mesh, steps=-3 * divisions, turn=-3 * divisions * spacing)
    with pytest.raises(ValueError, match="half a node spacing"):
        join_band(mesh, steps=1, turn=1.6 * spacing)
    with pytest.raises(ValueError, match="half a node spacing"):
        join_band(mesh, steps=1, turn=0.4 * spacing)
    # The furthest it can go: every rotor node a copy.
    edge = join_band(mesh, steps=mesh.band.facing + divisions, turn=(mesh.band.facing + divisions) * spacing)
    assert len(edge.nodes) == len(mesh.band.rest) + divisions


# The mesh after turning


def test_turning_to_zero_gives_the_mesh_back(pole: dict, ring: dict) -> None:
    for model in (pole, ring):
        mesh = model["mesh"]
        again = rotate(mesh, 0.0)
        for name in ("nodes", "triangles", "triangle_region", "rotor_nodes", "periodic"):
            assert np.array_equal(getattr(again, name), getattr(mesh, name)), name
        assert again.rotor_angle == 0.0 and again.rotor_turn == 0.0


@pytest.mark.parametrize("model_poles", [1, 2, 8])
@pytest.mark.parametrize("angle", ANGLES)
def test_only_the_rotor_moves_and_only_the_band_is_joined_again(model_poles: int, angle: float) -> None:
    mesh = ring_machine(model_poles=model_poles, offset=1)["mesh"]
    band = mesh.band
    turned = rotate(mesh, angle)
    steps, turn = band_steps(mesh, angle)
    count, meshed = len(band.rest), band.triangles
    assert turned.rotor_angle == angle and turned.rotor_turn == turn

    # Meshed triangles, their regions and the cut-edge pairs are untouched.
    assert np.array_equal(turned.triangles[:meshed], mesh.triangles[:meshed])
    assert np.array_equal(turned.triangle_region, mesh.triangle_region)
    assert np.array_equal(turned.periodic[: band.pairs], mesh.periodic[: band.pairs])
    assert len(turned.triangles) == len(mesh.triangles)
    assert np.array_equal(turned.rotor_nodes[:count], mesh.rotor_nodes[:count])
    assert turned.rotor_nodes[count:].all()

    # Stator nodes stay; rotor nodes turn as one body, the ring by whole steps.
    stator = ~mesh.rotor_nodes[:count]
    assert np.array_equal(turned.nodes[:count][stator], band.rest[stator])
    body = mesh.rotor_nodes[:count].copy()
    body[band.rotor_ring] = False
    c, s = math.cos(turn), math.sin(turn)
    expected = band.rest[body] @ np.array([[c, s], [-s, c]])
    assert np.allclose(turned.nodes[:count][body], expected, rtol=0, atol=1e-15)
    whole = steps * step_angle(mesh)
    c, s = math.cos(whole), math.sin(whole)
    expected = band.rest[band.rotor_ring] @ np.array([[c, s], [-s, c]])
    assert np.allclose(turned.nodes[band.rotor_ring], expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("model_poles", [1, 2, 8])
@pytest.mark.parametrize("angle", ANGLES)
def test_turned_mesh_is_valid(model_poles: int, angle: float) -> None:
    mesh = ring_machine(model_poles=model_poles, offset=1)["mesh"]
    turned = rotate(mesh, angle)
    band = mesh.band
    areas = turned.triangle_areas()
    assert areas.min() > 0
    # Nothing overlaps and nothing is left open: the area is what it was.
    assert areas.sum() == pytest.approx(mesh.triangle_areas().sum(), rel=1e-12)
    assert areas[band.triangles :].sum() == pytest.approx(
        mesh.triangle_areas()[band.triangles :].sum(), rel=1e-12
    )
    assert np.array_equal(np.unique(turned.triangles), np.arange(len(turned.nodes)))

    # Every band triangle has the shape it had: the band never leans.
    assert np.allclose(
        sorted_sides(turned, turned.triangles[band.triangles :]),
        sorted_sides(mesh, mesh.triangles[band.triangles :]),
        rtol=1e-12,
    )

    assert_conforming(turned, (STATOR_RADII[0], ROTOR_RADII[-1]), (STATOR_RADII[-1], ROTOR_RADII[0]))


@pytest.mark.parametrize("model_poles", [1, 2])
@pytest.mark.parametrize("angle", ANGLES)
def test_copies_stand_one_slice_from_their_originals(model_poles: int, angle: float) -> None:
    mesh = ring_machine(model_poles=model_poles, offset=1)["mesh"]
    band = mesh.band
    turned = rotate(mesh, angle)
    copies = turned.periodic[band.pairs :]
    count = len(band.rest)
    assert np.array_equal(copies[:, 0], np.arange(count, len(turned.nodes)))
    # One copy for every place the rotor slice is out of line with the stator's.
    assert len(copies) == abs(band.facing - band_steps(mesh, angle)[0])
    assert np.isin(copies[:, 1], band.rotor_ring).all()
    here, there = turned.nodes[copies[:, 0]], turned.nodes[copies[:, 1]]
    turn = np.arctan2(here[:, 1], here[:, 0]) - np.arctan2(there[:, 1], there[:, 0])
    slices = turn / band.span
    assert np.allclose(np.abs((slices + 0.5) % 1.0 - 0.5), 0.0, atol=1e-9)
    assert np.allclose(np.hypot(here[:, 0], here[:, 1]), ROTOR_RADII[0], rtol=1e-12)
    # Copies are only made for band nodes: each is a corner of a band triangle.
    assert np.isin(copies[:, 0], turned.triangles[band.triangles :]).all()


def test_no_copies_where_the_two_slices_line_up() -> None:
    mesh = ring_machine(model_poles=1, offset=1)["mesh"]
    lined_up = rotate(mesh, step_angle(mesh))
    assert len(lined_up.nodes) == len(mesh.band.rest)
    assert len(lined_up.periodic) == mesh.band.pairs
    assert len(mesh.nodes) == len(mesh.band.rest) + 1


def test_turning_does_not_add_up(pole: dict) -> None:
    mesh = pole["mesh"]
    once = rotate(mesh, 0.37)
    twice = rotate(rotate(mesh, 1.5), 0.37)
    assert np.array_equal(once.nodes, twice.nodes)
    assert np.array_equal(once.triangles, twice.triangles)
    assert np.array_equal(once.periodic, twice.periodic)


@pytest.mark.parametrize(("model_poles", "repeat_slices"), [(1, 2), (2, 1), (8, 1)])
def test_a_whole_repeat_gives_the_same_mesh(model_poles: int, repeat_slices: int) -> None:
    mesh = ring_machine(model_poles=model_poles)["mesh"]
    repeat = repeat_slices * mesh.band.span
    for angle in (0.0, 0.123, -0.4):
        near = rotate(mesh, angle)
        for whole in (1, -2, 5):
            far = rotate(mesh, angle + whole * repeat)
            assert np.array_equal(far.triangles, near.triangles)
            assert np.array_equal(far.periodic, near.periodic)
            assert np.allclose(far.nodes, near.nodes, rtol=0, atol=1e-13)


# The field after turning


@pytest.mark.parametrize(("model_poles", "offset"), [(1, 0), (1, -3), (2, 5), (2, -7), (4, 1)])
def test_slice_gives_what_the_whole_machine_gives_at_any_angle(model_poles: int, offset: int) -> None:
    # Both meshes are the same triangles pole after pole, so the two answers
    # differ by rounding only, however many slices the rotor has turned past.
    whole = ring_machine(offset=offset)
    part = ring_machine(model_poles=model_poles, offset=offset)
    for angle in ANGLES:
        whole_torque, whole_linked = ring_results(whole, angle)
        part_torque, part_linked = ring_results(part, angle)
        assert part_torque == pytest.approx(whole_torque, rel=1e-9, abs=1e-12)
        for name in whole_linked:
            assert part_linked[name] == pytest.approx(whole_linked[name], rel=1e-9, abs=1e-12)


def test_turning_an_antiperiodic_slice_by_one_slice_reverses_the_magnets(pole: dict) -> None:
    mesh = pole["mesh"]
    span = mesh.band.span
    for angle in (0.0, 0.05):
        near = ring_problem(pole, rotate(mesh, angle), current=0.0)
        far = ring_problem(pole, rotate(mesh, angle + span), current=0.0)
        a, b = solve(near), solve(far)
        stator = ~mesh.rotor_nodes[: len(mesh.band.rest)]
        scale = np.abs(a.potential).max()
        assert scale > 1e-4
        assert np.allclose(b.potential[: len(stator)][stator], -a.potential[: len(stator)][stator],
                           rtol=0, atol=1e-8 * scale)


def cycle_fit(angles: np.ndarray, values: np.ndarray, pole_pairs: int) -> tuple[float, float, float]:
    """Amplitude and phase of the part that varies once per electrical cycle, and what is left."""
    basis = np.column_stack(
        [np.ones(len(angles)), np.cos(pole_pairs * angles), np.sin(pole_pairs * angles)]
    )
    found, *_ = np.linalg.lstsq(basis, values, rcond=None)
    left = values - basis @ found
    return math.hypot(found[1], found[2]), math.atan2(found[2], found[1]), float(np.abs(left).max())


def cycle_of(model: dict, whole_steps: bool, count: int = 24) -> tuple[np.ndarray, np.ndarray, list]:
    """Torque and linkages at angles over one electrical cycle.

    Either at whole node spacings, or at angles that are none of them, each
    a different part of a spacing past the last whole one. (With the same
    part at every angle the error is itself a sine and hides in the fit.)
    """
    mesh = model["mesh"]
    spacing = step_angle(mesh)
    steps = round(2 * math.pi / model["pole_pairs"] / spacing)
    if whole_steps:
        angles = spacing * np.arange(0, steps, steps // count)
    else:
        angles = spacing * steps * (np.arange(count - 1) + 0.37) / (count - 1)
        part = np.remainder(angles / spacing, 1.0)
        assert np.minimum(part, 1.0 - part).min() > 0.005 and np.ptp(part) > 0.8
    found = [ring_results(model, angle) for angle in angles]
    return angles, np.array([t for t, _ in found]), [linked for _, linked in found]


@pytest.fixture(scope="module")
def on_steps(pole: dict) -> tuple:
    return cycle_of(pole, whole_steps=True)


@pytest.fixture(scope="module")
def off_steps(pole: dict) -> tuple:
    return cycle_of(pole, whole_steps=False)


# How far the torque may be from a pure sine, as a share of its peak. At
# whole node spacings no triangle changes shape and what is left is the
# error of the mesh itself. Between them the rotor's triangles next to the
# band lean, and the torque taken in the band beside them picks up some of
# their error; it falls as the mesh is refined.
ON_STEPS = 0.005
OFF_STEPS = 0.03


@pytest.mark.parametrize(("cycle", "limit"), [("on_steps", ON_STEPS), ("off_steps", OFF_STEPS)])
def test_torque_of_the_ring_machine_is_a_sine_of_the_angle(
    pole: dict, cycle: str, limit: float, request: pytest.FixtureRequest
) -> None:
    # A current layer that varies as cos(p phi) feels only the part of the
    # magnets' field that varies the same way, and that part turns with the
    # rotor: the torque goes as a sine of p times the angle, with no mean.
    angles, torques, _ = request.getfixturevalue(cycle)
    amplitude, _, left = cycle_fit(angles, torques, pole["pole_pairs"])
    assert amplitude > 0.2  # N m
    assert left < limit * amplitude
    assert abs(torques.mean()) < limit * amplitude


@pytest.mark.parametrize("cycle", ["on_steps", "off_steps"])
def test_linkage_of_the_ring_machine_is_a_sine_of_the_angle(
    pole: dict, cycle: str, request: pytest.FixtureRequest
) -> None:
    # Winding d follows cos(p phi) and winding q sin(p phi): the magnets'
    # flux in q is a quarter cycle behind that in d, and as large. Flux
    # linkage is read far from the band and is as good at any angle.
    angles, _, linked = request.getfixturevalue(cycle)
    p = pole["pole_pairs"]
    d_size, d_phase, d_left = cycle_fit(angles, np.array([x["d"] for x in linked]), p)
    q_size, q_phase, q_left = cycle_fit(angles, np.array([x["q"] for x in linked]), p)
    assert d_size > 1e-3
    assert d_left < 0.002 * d_size and q_left < 0.002 * q_size
    assert q_size == pytest.approx(d_size, rel=2e-3)
    lag = math.remainder(q_phase - d_phase, 2 * math.pi)
    assert lag == pytest.approx(math.pi / 2, abs=2e-3)


@pytest.mark.parametrize(("cycle", "limit"), [("on_steps", 0.01), ("off_steps", OFF_STEPS)])
def test_torque_is_the_rate_of_change_of_linked_flux_times_current(
    pole: dict, cycle: str, limit: float, request: pytest.FixtureRequest
) -> None:
    # With winding d carrying I, the torque is I d(psi_d)/d(angle) at fixed
    # current. The linkage found from the potential and the torque found
    # from the stress in the band are two separate calculations.
    angles, torques, linked = request.getfixturevalue(cycle)
    p = pole["pole_pairs"]
    size, phase, _ = cycle_fit(angles, np.array([x["d"] for x in linked]), p)
    current = 4e6 / 1e6  # the current layer is winding d with 4 A in it
    expected = -current * p * size * np.sin(p * angles - phase)
    assert np.abs(expected).max() > 0.2
    assert np.abs(torques - expected).max() < limit * np.abs(expected).max()


@pytest.mark.parametrize(
    ("spacings", "limit"), [((0, 1, 5, 12, -7, 100), 0.01), ((0.3, 0.5, 5.25, -7.6), OFF_STEPS)]
)
def test_band_torque_balances_the_force_on_the_current(pole: dict, spacings: tuple, limit: float) -> None:
    mesh = pole["mesh"]
    peak = 0.28  # N m, of this machine
    for count in spacings:
        problem = ring_problem(pole, rotate(mesh, count * step_angle(mesh)))
        solution = solve(problem)
        on_rotor = torque(problem, solution)
        on_stator = force_on_the_current(problem, solution)
        assert abs(on_rotor + on_stator) < limit * peak


def test_torque_between_node_spacings_improves_with_a_finer_mesh() -> None:
    left = []
    for per_pole in (12, 24):
        model = ring_machine(model_poles=1, per_pole=per_pole)
        angles, torques, _ = cycle_of(model, whole_steps=False, count=14)
        amplitude, _, worst = cycle_fit(angles, torques, model["pole_pairs"])
        left.append(worst / amplitude)
    assert left[0] < 0.05
    assert left[1] < 0.6 * left[0]


def test_torque_does_not_jump_where_the_band_is_joined_one_step_on(pole: dict) -> None:
    # Half a spacing either side of the change the rotor is in the same
    # place to within a hair, and so must the torque be.
    mesh = pole["mesh"]
    spacing = step_angle(mesh)
    before, _ = ring_results(pole, 2.5 * spacing - 1e-7)
    after, _ = ring_results(pole, 2.5 * spacing + 1e-7)
    assert band_steps(mesh, 2.5 * spacing - 1e-7)[0] == 2
    assert band_steps(mesh, 2.5 * spacing + 1e-7)[0] == 3
    assert after == pytest.approx(before, rel=OFF_STEPS)


def test_parallel_magnetisation_turns_with_its_region() -> None:
    class Parallel:
        pattern, axis_angle, polarity = "parallel", 0.2, -1

    points = np.array([[1.0, 0.0], [0.0, 2.0]])
    turned = _directions(Parallel, points, turn=0.5)
    assert np.allclose(turned, -np.array([[math.cos(0.7), math.sin(0.7)]] * 2))
    assert np.allclose(_directions(Parallel, points), -np.array([[math.cos(0.2), math.sin(0.2)]] * 2))


# A motor meshed by Gmsh

MATERIALS = {
    # Placeholders: not the designs' real steel or magnet grade.
    "M270-35A": lambda: load_material("generic_35A_placeholder"),
    "N42SH": lambda: Magnet(remanence=1.2, relative_permeability=1.05),
}


def _motor(name: str, full: bool = False, **changes: object) -> dict:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import design_from_dict, design_to_dict, load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    design = load_design(root / "designs" / f"{name}.yaml")
    if changes:
        data = design_to_dict(design)
        for path, value in changes.items():
            *sections, key = path.split("__")
            block = data
            for section in sections:
                block = block[section]
            block[key] = value
        design = design_from_dict(data)
    machine = machine_for(design)
    geometry = machine.build_geometry(full=full)
    return {
        "machine": machine,
        "geometry": geometry,
        "mesh": build_mesh(geometry),
        "materials": {grade: make() for grade, make in MATERIALS.items()},
    }


def motor_results(motor: dict, angle: float, current: float) -> tuple[float, dict, object, Mesh]:
    turned = rotate(motor["mesh"], angle)
    problem = build_problem(motor["machine"], turned, motor["materials"], current=current)
    solution = solve(problem)
    return torque(problem, solution), flux_linkage(problem, solution), solution, turned


@pytest.fixture(scope="module")
def motor() -> dict:
    return _motor("qdd_outer_36s42p")


@pytest.fixture(scope="module")
def small() -> dict:
    return _motor("qdd_outer_12s10p")


MOTOR_ANGLES = [0.0041, 0.0931, -0.35, 0.7, 1.0472, 2.9, -5.1, 12.0]


@pytest.mark.parametrize("angle", MOTOR_ANGLES)
def test_motor_is_not_meshed_again(motor: dict, angle: float, monkeypatch: pytest.MonkeyPatch) -> None:
    import polecraft.core.mesh as mesher

    def no_gmsh() -> None:
        raise AssertionError("turning the rotor must not mesh anything")

    monkeypatch.setattr(mesher, "require_gmsh", no_gmsh)
    mesh = motor["mesh"]
    band = mesh.band
    turned = rotate(mesh, angle)
    assert len(turned.triangles) == len(mesh.triangles)
    assert np.array_equal(turned.triangles[: band.triangles], mesh.triangles[: band.triangles])
    assert np.array_equal(turned.triangle_region, mesh.triangle_region)
    # The same nodes as before, and no more than a ring's worth of copies.
    count = len(band.rest)
    assert count <= len(turned.nodes) <= count + band.divisions
    for name in mesh.region_names:
        index = mesh.region_names.index(name)
        assert (turned.triangle_region == index).sum() == (mesh.triangle_region == index).sum()
    stator = ~mesh.rotor_nodes[:count]
    assert np.array_equal(turned.nodes[:count][stator], mesh.nodes[:count][stator])
    # The rotor is turned as one body: its nodes keep their distances.
    rotor = np.flatnonzero(mesh.rotor_nodes[:count] & ~np.isin(np.arange(count), band.rotor_ring))
    apart = np.linalg.norm(turned.nodes[rotor[1:]] - turned.nodes[rotor[:-1]], axis=1)
    assert np.allclose(apart, np.linalg.norm(band.rest[rotor[1:]] - band.rest[rotor[:-1]], axis=1),
                       rtol=0, atol=1e-15)


@pytest.mark.parametrize("angle", MOTOR_ANGLES)
def test_motor_mesh_is_valid_at_any_angle(motor: dict, angle: float) -> None:
    mesh = motor["mesh"]
    band = mesh.band
    geometry = motor["geometry"]
    turned = rotate(mesh, angle)
    areas, before = turned.triangle_areas(), mesh.triangle_areas()
    assert areas.min() > 0
    assert areas.sum() == pytest.approx(before.sum(), rel=1e-12)
    assert np.array_equal(np.unique(turned.triangles), np.arange(len(turned.nodes)))

    # Only the rotor's layer of air next to the band changes shape, and the
    # band and everything else keep theirs exactly.
    leaning = np.isin(turned.triangles, band.rotor_ring).any(axis=1)
    leaning[band.triangles :] = False
    names = {mesh.region_names[i] for i in np.unique(mesh.triangle_region[leaning])}
    assert names and all(name.startswith("gap_rotor") for name in names)
    assert np.allclose(sorted_sides(turned, turned.triangles[~leaning]),
                       sorted_sides(mesh, mesh.triangles[~leaning]), rtol=0, atol=1e-9)
    assert turned.smallest_angles()[~leaning].min() == pytest.approx(
        mesh.smallest_angles()[~leaning].min(), abs=1e-6
    )
    assert turned.smallest_angles().min() > 12

    assert_conforming(
        turned,
        (geometry.inner_radius, geometry.outer_radius),
        (geometry.band_inner_radius, geometry.band_outer_radius),
    )


def test_motor_turned_by_one_slice_has_the_field_reversed(motor: dict) -> None:
    # The slice is antiperiodic: one slice on, every magnet has taken the
    # place of one of the other polarity, and the currents that follow the
    # rotor are reversed too. The torque is the same.
    span = motor["mesh"].band.span
    for angle in (0.0, 0.0127):
        near_torque, near_linked, near, _ = motor_results(motor, angle, 10.0)
        far_torque, far_linked, far, turned = motor_results(motor, angle + span, 10.0)
        assert near_torque > 1.0
        assert far_torque == pytest.approx(near_torque, rel=1e-8)
        for phase in "ABC":
            assert far_linked[phase] == pytest.approx(-near_linked[phase], rel=1e-8, abs=1e-12)
        count = len(motor["mesh"].band.rest)
        stator = ~motor["mesh"].rotor_nodes[:count]
        assert np.allclose(far.potential[:count][stator], -near.potential[:count][stator],
                           rtol=0, atol=1e-8 * np.abs(near.potential).max())
        # The rotor's nodes are the same pieces of rotor, a slice further on.
        assert turned.rotor_turn == pytest.approx(angle + span) or turned.rotor_turn == pytest.approx(angle - span)


def test_motor_turned_by_two_slices_is_where_it_was(motor: dict) -> None:
    span = motor["mesh"].band.span
    near_torque, near_linked, near, mesh_near = motor_results(motor, 0.0127, 10.0)
    far_torque, far_linked, far, mesh_far = motor_results(motor, 0.0127 + 2 * span, 10.0)
    assert np.array_equal(mesh_far.triangles, mesh_near.triangles)
    assert np.allclose(mesh_far.nodes, mesh_near.nodes, rtol=0, atol=1e-13)
    assert far_torque == pytest.approx(near_torque, rel=1e-8)
    assert np.allclose(far.potential, near.potential, rtol=0, atol=1e-8 * np.abs(near.potential).max())
    for phase in "ABC":
        assert far_linked[phase] == pytest.approx(near_linked[phase], rel=1e-8, abs=1e-12)


def test_motor_with_parallel_magnets_turned_by_one_slice(motor: dict) -> None:
    # Catches a magnetisation axis that is not turned with its magnet, or
    # is turned by the angle asked for and not by the angle the nodes moved.
    design = motor["machine"].design
    other = "parallel" if design.geometry.magnet.magnetisation != "parallel" else "radial"
    changed = _motor("qdd_outer_36s42p", geometry__magnet__magnetisation=other)
    span = changed["mesh"].band.span
    for kind in (motor, changed):
        near_torque, near_linked, *_ = motor_results(kind, 0.021, 10.0)
        far_torque, far_linked, *_ = motor_results(kind, 0.021 + 3 * span, 10.0)
        assert far_torque == pytest.approx(near_torque, rel=1e-8)
        for phase in "ABC":
            assert far_linked[phase] == pytest.approx(-near_linked[phase], rel=1e-8, abs=1e-12)


@pytest.fixture(scope="module")
def small_whole() -> dict:
    return _motor("qdd_outer_12s10p", full=True)


@pytest.mark.parametrize("angle", [0.1234, 1.0, -0.7, 4.0])
def test_motor_slice_and_whole_machine_agree_at_any_angle(
    small: dict, small_whole: dict, angle: float
) -> None:
    whole = small_whole
    part_torque, part_linked, *_ = motor_results(small, angle, 10.0)
    whole_torque, whole_linked, *_ = motor_results(whole, angle, 10.0)
    assert part_torque == pytest.approx(whole_torque, rel=2e-3)
    for phase in "ABC":
        assert part_linked[phase] == pytest.approx(whole_linked[phase], rel=2e-3, abs=2e-6)


@pytest.fixture(scope="module")
def motor_cycle(motor: dict) -> dict:
    """One electrical cycle of the reference motor in 24 steps, without and with current."""
    pole_pairs = motor["machine"].design.poles // 2
    angles = 2 * math.pi / pole_pairs * np.arange(24) / 24
    idle = [motor_results(motor, angle, 0.0)[:2] for angle in angles]
    loaded = [motor_results(motor, angle, 10.0)[0] for angle in angles]
    return {"angles": angles, "idle": idle, "loaded": np.array(loaded)}


def test_motor_phase_linkages_follow_the_rotor_as_the_excitation_assumes(
    motor: dict, motor_cycle: dict
) -> None:
    # Machine.excitation takes the flux linked by phase A to vary as
    # cos(p angle - a), with B and C a third and two thirds of a cycle later.
    design = motor["machine"].design
    axis = phase_axis_angle(design.slots, design.poles)
    waves = {
        phase: np.fft.rfft([linked[phase] for _, linked in motor_cycle["idle"]]) / 12
        for phase in "ABC"
    }
    size = abs(waves["A"][1])
    assert size > 5e-3  # Wb
    for k, phase in enumerate("ABC"):
        assert abs(waves[phase][1]) == pytest.approx(size, rel=1e-3)
        expected = -axis - k * 2 * math.pi / 3
        assert math.remainder(np.angle(waves[phase][1]) - expected, 2 * math.pi) == pytest.approx(
            0.0, abs=math.radians(0.5)
        )
        # Nearly a pure sine: what else there is, is small.
        assert np.abs(waves[phase][2:]).max() < 0.05 * size


def test_motor_torque_holds_as_current_follows_the_rotor(motor_cycle: dict) -> None:
    loaded = motor_cycle["loaded"]
    assert loaded.min() > 1.0  # N m
    assert np.ptp(loaded) < 0.01 * loaded.mean()
    idle = np.array([t for t, _ in motor_cycle["idle"]])
    assert np.abs(idle).max() < 0.005 * loaded.mean()


def test_motor_torque_is_smooth_between_node_spacings(motor: dict) -> None:
    # The band is joined one step on at every half spacing. Nothing shows
    # there beyond the mesh noise of this coarse mesh.
    spacing = step_angle(motor["mesh"])
    torques = np.array(
        [motor_results(motor, f * spacing, 10.0)[0] for f in np.arange(0.0, 3.01, 0.25)]
    )
    assert np.ptp(torques) < 0.006 * torques.mean()
    assert np.abs(np.diff(torques)).max() < 0.002 * torques.mean()


def test_motor_linkage_changes_evenly_between_node_spacings(motor: dict) -> None:
    spacing = step_angle(motor["mesh"])
    linked = np.array(
        [motor_results(motor, f * spacing, 0.0)[1]["A"] for f in np.arange(0.0, 2.01, 0.25)]
    )
    # Over two spacings (15 electrical degrees) the curve is nearly a
    # parabola. What its third differences show beyond the curve's own is
    # mesh noise, a few parts in ten thousand of the linkage.
    assert np.abs(linked).max() > 5e-3  # Wb
    assert np.abs(np.diff(linked, 3)).max() < 5e-4 * np.abs(linked).max()
