"""Results read off a solved field: flux linkage, torque and core loss."""

import cmath
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import (
    AIR,
    MU0,
    CoreLossCoefficients,
    LinearMaterial,
    Magnet,
    load_material,
)
from polecraft.core.post import (
    core_loss_density,
    flux_linkage,
    gap_flux_density,
    linkage_per_metre,
    material_flux_density,
    ring_torque_per_metre,
    torque,
)
from polecraft.core.problem import Gap, Problem, build_problem
from polecraft.core.solver import shape_gradients, solve

NU0 = 1.0 / MU0


def grid(columns: int, rows: int, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """A rectangle from (0, 0) to (width, height), each cell cut on alternating diagonals."""
    x, y = np.meshgrid(np.linspace(0.0, width, columns + 1), np.linspace(0.0, height, rows + 1))
    nodes = np.column_stack([x.ravel(), y.ravel()])
    triangles = []
    for j in range(rows):
        for i in range(columns):
            a = j * (columns + 1) + i
            b, c, d = a + 1, a + columns + 2, a + columns + 1
            if (i + j) % 2 == 0:
                triangles += [(a, b, c), (a, c, d)]
            else:
                triangles += [(a, b, d), (b, c, d)]
    return nodes, np.array(triangles)


def centres(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    return nodes[triangles].mean(axis=1)


def turn_density(inside: np.ndarray, turns: float, area: np.ndarray) -> np.ndarray:
    """Turns per square metre of a coil side that fills the marked triangles."""
    return np.where(inside, turns / area[inside].sum(), 0.0)


# Flux linkage of a coil


def coil_in_a_uniform_field(field: float = 0.8) -> tuple[Problem, dict]:
    """A 25-turn coil, 12 mm between its sides, 30 mm deep, in a uniform field along y."""
    width, height, turns, depth = 0.020, 0.010, 25.0, 0.030
    nodes, triangles = grid(20, 10, width, height)
    centre = centres(nodes, triangles)
    _, area = shape_gradients(nodes, triangles)
    in_height = (centre[:, 1] > 0.003) & (centre[:, 1] < 0.007)
    go = in_height & (centre[:, 0] > 0.003) & (centre[:, 0] < 0.005)
    back = in_height & (centre[:, 0] > 0.015) & (centre[:, 0] < 0.017)
    edge = np.flatnonzero(
        (nodes[:, 0] == 0.0) | (nodes[:, 0] == width) | (nodes[:, 1] == 0.0) | (nodes[:, 1] == height)
    )
    problem = Problem(
        nodes=nodes,
        triangles=triangles,
        materials=(AIR,),
        fixed_nodes=edge,
        fixed_values=-field * nodes[edge, 0],  # B_y = -dA/dx
        depth=depth,
        windings={"coil": turn_density(go, turns, area) - turn_density(back, turns, area)},
    )
    return problem, {"turns": turns, "depth": depth, "spacing": 0.012, "field": field}


def test_coil_in_a_uniform_field_links_turns_times_flux() -> None:
    # Flux through the coil: field times the distance between the middles of
    # its sides times its depth. With current out of the plane in the left
    # side, the coil's own field between the sides is along +y, so a field
    # along +y is linked with a plus sign.
    problem, coil = coil_in_a_uniform_field()
    solution = solve(problem)
    assert solution.flux_density[:, 1] == pytest.approx(coil["field"])
    expected = coil["turns"] * coil["field"] * coil["spacing"] * coil["depth"]
    assert expected == pytest.approx(7.2e-3)
    assert flux_linkage(problem, solution) == pytest.approx({"coil": expected}, rel=1e-12)


def test_reversing_the_field_reverses_the_linkage() -> None:
    forward, _ = coil_in_a_uniform_field(0.8)
    backward, _ = coil_in_a_uniform_field(-0.8)
    linked = flux_linkage(forward, solve(forward))["coil"]
    assert flux_linkage(backward, solve(backward))["coil"] == pytest.approx(-linked, rel=1e-12)


def test_linkage_does_not_depend_on_the_level_of_the_potential() -> None:
    # A coil with both sides in the model links a difference of potential.
    problem, _ = coil_in_a_uniform_field()
    raised = Problem(
        nodes=problem.nodes, triangles=problem.triangles, materials=problem.materials,
        fixed_nodes=problem.fixed_nodes, fixed_values=problem.fixed_values + 5.0,
        depth=problem.depth, windings=problem.windings,
    )
    assert flux_linkage(raised, solve(raised))["coil"] == pytest.approx(
        flux_linkage(problem, solve(problem))["coil"], rel=1e-9
    )


def air_cored_coil(cells: int, current: float = 3.0) -> tuple[Problem, dict]:
    """A coil between two ideal iron plates, its two sides filling the full height.

    The top and bottom edges are left free (the plates). The left edge is a
    field line and the right edge is left free too, as iron that carries
    the flux back. The field is then along y, uniform between the sides and
    falling to zero through each of them.
    """
    side, spacing, height, turns, depth = 0.004, 0.012, 0.010, 40.0, 0.050
    width = spacing + 2.0 * side
    nodes, triangles = grid(cells * 5, cells, width, height)
    x = centres(nodes, triangles)[:, 0]
    _, area = shape_gradients(nodes, triangles)
    density = turn_density(x < side, turns, area) - turn_density(x > width - side, turns, area)
    problem = Problem(
        nodes=nodes,
        triangles=triangles,
        materials=(AIR,),
        current_density=density * current,
        fixed_nodes=np.flatnonzero(nodes[:, 0] == 0.0),
        depth=depth,
        windings={"coil": density},
    )
    # L = mu0 N^2 depth (spacing + 2/3 side) / height: the flux inside a side
    # links only the turns outside it, which makes a side count for a third
    # of its width at each end.
    inductance = MU0 * turns**2 * depth * (spacing + 2.0 * side / 3.0) / height
    return problem, {"current": current, "inductance": inductance, "turns": turns, "height": height}


def test_inductance_of_an_air_cored_coil() -> None:
    problem, coil = air_cored_coil(cells=8)
    solution = solve(problem)
    between = np.abs(centres(problem.nodes, problem.triangles)[:, 0] - 0.010) < 0.005
    field = MU0 * coil["turns"] * coil["current"] / coil["height"]
    assert solution.flux_density[between, 1] == pytest.approx(field, rel=1e-3)
    linked = flux_linkage(problem, solution)["coil"]
    assert linked == pytest.approx(coil["inductance"] * coil["current"], rel=2e-3)


def test_inductance_error_falls_with_a_finer_mesh() -> None:
    errors = []
    for cells in (4, 8, 16):
        problem, coil = air_cored_coil(cells)
        linked = flux_linkage(problem, solve(problem))["coil"]
        errors.append(abs(linked / (coil["inductance"] * coil["current"]) - 1.0))
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] > 3.0  # second order
    assert errors[2] < 3e-4


def test_linkage_times_current_is_twice_the_stored_energy() -> None:
    # For a coil that is the only source in linear materials, psi I = 2 W.
    nodes, triangles = grid(16, 12, 0.032, 0.024)
    centre = centres(nodes, triangles)
    _, area = shape_gradients(nodes, triangles)
    iron = (centre[:, 1] < 0.006) | (centre[:, 0] > 0.026)
    go = (centre[:, 0] > 0.006) & (centre[:, 0] < 0.010) & (centre[:, 1] > 0.010) & (centre[:, 1] < 0.018)
    back = (centre[:, 0] > 0.016) & (centre[:, 0] < 0.020) & (centre[:, 1] > 0.010) & (centre[:, 1] < 0.018)
    density = turn_density(go, 60.0, area) - turn_density(back, 60.0, area)
    edge = np.flatnonzero(
        (nodes[:, 0] == 0.0) | (nodes[:, 0] == 0.032) | (nodes[:, 1] == 0.0) | (nodes[:, 1] == 0.024)
    )
    problem = Problem(
        nodes=nodes, triangles=triangles, materials=(AIR, LinearMaterial(500.0)),
        triangle_material=iron.astype(int), current_density=density * 2.5,
        fixed_nodes=edge, depth=0.040, windings={"coil": density},
    )
    solution = solve(problem)
    b_squared = (solution.flux_density**2).sum(axis=1)
    energy = 0.040 * 0.5 * (solution.reluctivity * b_squared * area).sum()
    assert energy > 0.0
    assert flux_linkage(problem, solution)["coil"] * 2.5 == pytest.approx(2.0 * energy, rel=1e-10)


def test_each_winding_is_reported_under_its_name() -> None:
    problem, _ = coil_in_a_uniform_field()
    density = problem.windings["coil"]
    both = Problem(
        nodes=problem.nodes, triangles=problem.triangles, materials=problem.materials,
        fixed_nodes=problem.fixed_nodes, fixed_values=problem.fixed_values,
        depth=problem.depth, windings={"a": density, "b": -2.0 * density},
    )
    linked = flux_linkage(both, solve(both))
    assert list(linked) == ["a", "b"]
    assert linked["b"] == pytest.approx(-2.0 * linked["a"], rel=1e-12)
    assert all(isinstance(value, float) for value in linked.values())


def test_a_problem_without_windings_has_no_linkage() -> None:
    nodes, triangles = grid(2, 2, 1.0, 1.0)
    problem = Problem(nodes=nodes, triangles=triangles, materials=(AIR,), fixed_nodes=[0])
    assert flux_linkage(problem, solve(problem)) == {}


def test_linkage_per_metre_is_the_turn_weighted_mean_potential() -> None:
    nodes = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [2.0, 3.0]])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    potential = np.array([1.0, 2.0, 3.0, 6.0])
    # Triangle areas 3 and 3; mean potentials 2 and 11/3.
    density = np.array([4.0, -1.0])
    assert linkage_per_metre(nodes, triangles, potential, density) == pytest.approx(
        4.0 * 3.0 * 2.0 - 1.0 * 3.0 * 11.0 / 3.0
    )


def test_linkage_scales_with_depth_and_repeats() -> None:
    problem, _ = coil_in_a_uniform_field()
    solution = solve(problem)
    more = Problem(
        nodes=problem.nodes, triangles=problem.triangles, materials=problem.materials,
        fixed_nodes=problem.fixed_nodes, fixed_values=problem.fixed_values,
        depth=2.0 * problem.depth, repeats=3, windings=problem.windings,
    )
    assert flux_linkage(more, solution)["coil"] == pytest.approx(
        6.0 * flux_linkage(problem, solution)["coil"], rel=1e-12
    )


# Torque in a ring machine


def ring_machine(
    poles: int = 8,
    model_poles: int | None = None,
    current: float = 4e6,
    per_pole: int = 16,
    gap_rows: tuple[int, int] = (6, 9),
    rotor_outside: bool = False,
) -> dict:
    """Smooth iron, a ring of magnets, an air gap, a layer of current, smooth iron.

    Magnets are magnetised along the radius, outwards and inwards in turn.
    The current layer alternates the same way, each block facing a magnet,
    where it pushes hardest. With no slots there is no cogging.
    ``model_poles`` poles are meshed, all of them if it is not given.
    """
    model_poles = model_poles or poles
    closed = model_poles == poles
    rows = 16
    columns = model_poles * per_pole
    around = columns if closed else columns + 1
    radii = np.linspace(0.030, 0.046, rows + 1)
    angles = 2.0 * math.pi * np.arange(around) / (poles * per_pole)
    nodes = np.array([[r * math.cos(t), r * math.sin(t)] for t in angles for r in radii])
    triangles = []
    for i in range(columns):
        j = (i + 1) % around
        for k in range(rows):
            a, b = i * (rows + 1) + k, j * (rows + 1) + k
            c, d = b + 1, a + 1
            if (i + k) % 2 == 0:
                triangles += [(a, b, c), (a, c, d)]
            else:
                triangles += [(a, b, d), (b, c, d)]
    triangles = np.array(triangles)

    centre = centres(nodes, triangles)
    layer = np.repeat(np.tile(np.arange(rows), columns), 2)
    column = np.repeat(np.arange(columns), 2 * rows)
    pole = column // per_pole
    is_magnet = (layer >= 3) & (layer < 6)
    is_coil = (layer >= 9) & (layer < 12)
    is_iron = (layer < 3) | (layer >= 12)
    outward = centre / np.linalg.norm(centre, axis=1)[:, None]
    node_layer = np.tile(np.arange(rows + 1), around)
    in_gap = (layer >= gap_rows[0]) & (layer < gap_rows[1])
    problem = Problem(
        nodes=nodes,
        triangles=triangles,
        materials=(AIR, LinearMaterial(1000.0), Magnet(1.2)),
        triangle_material=np.where(is_iron, 1, np.where(is_magnet, 2, 0)),
        current_density=np.where(is_coil, current * (1 - 2 * (pole % 2)), 0.0),
        remanence=np.where(is_magnet[:, None], 1.2 * (1 - 2 * (pole % 2))[:, None] * outward, 0.0),
        fixed_nodes=np.flatnonzero((node_layer == 0) | (node_layer == rows)),
        periodic=None if closed else np.column_stack(
            [np.arange(columns * (rows + 1), (columns + 1) * (rows + 1)), np.arange(rows + 1)]
        ),
        antiperiodic=not closed and model_poles % 2 == 1,
        depth=0.020,
        repeats=poles // model_poles,
        gap=Gap(
            triangles=np.flatnonzero(in_gap),
            inner_radius=radii[gap_rows[0]],
            outer_radius=radii[gap_rows[1]],
            rotor_outside=rotor_outside,
        ),
    )
    return {"problem": problem, "is_coil": is_coil, "radii": radii, "centre": centre}


def force_on_the_current(model: dict, solution) -> float:
    """Torque on the current layer in N m, from the force J x B on each triangle.

    About the axis that is J r B_r per unit volume. The iron outside the
    layer is smooth, so this is the whole torque on the outer part.
    """
    problem = model["problem"]
    _, area = shape_gradients(problem.nodes, problem.triangles)
    centre = model["centre"]
    r_b_r = (solution.flux_density * centre).sum(axis=1)  # r times the radial part
    return problem.repeats * problem.depth * (problem.current_density * r_b_r * area).sum()


@pytest.fixture(scope="module")
def ring() -> dict:
    model = ring_machine()
    model["solution"] = solve(model["problem"])
    return model


def test_torque_on_the_inner_part_balances_the_force_on_the_current(ring: dict) -> None:
    # The magnets are on the inner part and the current on the outer one:
    # what the stress in the gap says about one must be the opposite of the
    # force on the other, worked out in a completely different way.
    on_magnets = torque(ring["problem"], ring["solution"])
    on_current = force_on_the_current(ring, ring["solution"])
    assert abs(on_magnets) > 0.5  # N m
    # Measured: 1.7 % apart on this mesh, 0.4 % with triangles half the size.
    assert on_magnets == pytest.approx(-on_current, rel=0.025)


def test_torque_agreement_improves_with_a_finer_mesh() -> None:
    errors = []
    for per_pole in (8, 16, 32):
        model = ring_machine(per_pole=per_pole)
        solution = solve(model["problem"])
        errors.append(
            abs(torque(model["problem"], solution) / -force_on_the_current(model, solution) - 1.0)
        )
    # Second order: a quarter of the difference for half the triangle size.
    assert errors[0] / errors[1] > 3.0
    assert errors[1] / errors[2] > 3.0
    assert errors[2] < 0.005


def test_no_current_means_no_torque() -> None:
    model = ring_machine(current=0.0)
    solution = solve(model["problem"])
    assert np.abs(solution.flux_density).max() > 0.5  # the magnets are still there
    loaded = abs(torque(ring_machine()["problem"], solve(ring_machine()["problem"])))
    assert abs(torque(model["problem"], solution)) < 1e-9 * loaded


def test_torque_changes_sign_with_the_current(ring: dict) -> None:
    reversed_ = ring_machine(current=-4e6)
    solution = solve(reversed_["problem"])
    forward = torque(ring["problem"], ring["solution"])
    assert torque(reversed_["problem"], solution) == pytest.approx(-forward, rel=1e-9)


def test_torque_is_proportional_to_the_current(ring: dict) -> None:
    half = ring_machine(current=2e6)
    forward = torque(ring["problem"], ring["solution"])
    assert torque(half["problem"], solve(half["problem"])) == pytest.approx(0.5 * forward, rel=1e-9)


@pytest.mark.parametrize("rows", [(6, 7), (7, 8), (8, 9), (6, 8), (7, 9)])
def test_torque_does_not_depend_on_the_radius_it_is_taken_at(ring: dict, rows: tuple) -> None:
    # Any ring of air between the magnets and the current will do.
    whole_gap = torque(ring["problem"], ring["solution"])
    model = ring_machine(gap_rows=rows)
    assert torque(model["problem"], ring["solution"]) == pytest.approx(whole_gap, rel=0.01)


def test_torque_on_an_outer_rotor_is_the_opposite(ring: dict) -> None:
    outer = ring_machine(rotor_outside=True)
    assert torque(outer["problem"], ring["solution"]) == -torque(ring["problem"], ring["solution"])


@pytest.mark.parametrize("model_poles", [1, 2, 4])
def test_torque_of_a_slice_is_that_of_the_whole_machine(ring: dict, model_poles: int) -> None:
    part = ring_machine(model_poles=model_poles)
    assert part["problem"].repeats == 8 // model_poles
    solution = solve(part["problem"])
    whole = torque(ring["problem"], ring["solution"])
    assert torque(part["problem"], solution) == pytest.approx(whole, rel=1e-9)


def test_ring_torque_per_metre_of_a_known_field() -> None:
    # B_r = a and B_t = b everywhere in a ring gives 2 pi r^2 a b / mu0 at
    # radius r; averaged over the ring's thickness, r^2 becomes
    # (ro^2 + ro ri + ri^2) / 3.
    model = ring_machine()
    problem = model["problem"]
    gap = problem.gap
    centre = model["centre"]
    radius = np.linalg.norm(centre, axis=1)
    outward = centre / radius[:, None]
    around = np.column_stack([-outward[:, 1], outward[:, 0]])
    field = 0.7 * outward + 0.2 * around
    found = ring_torque_per_metre(
        problem.nodes, problem.triangles, field, gap.triangles, gap.inner_radius, gap.outer_radius
    )
    ri, ro = gap.inner_radius, gap.outer_radius
    expected = 2.0 * math.pi * 0.7 * 0.2 / MU0 * (ro * ro + ro * ri + ri * ri) / 3.0
    # The triangles are flat-sided, so the ring is a little short of round.
    assert found == pytest.approx(expected, rel=2e-3)
    assert found > 0.0  # counter-clockwise on what the ring encloses


def test_a_problem_without_a_gap_has_no_torque_to_give() -> None:
    nodes, triangles = grid(2, 2, 1.0, 1.0)
    problem = Problem(nodes=nodes, triangles=triangles, materials=(AIR,), fixed_nodes=[0])
    with pytest.raises(ValueError, match="no air gap"):
        torque(problem, solve(problem))


def test_results_must_belong_to_the_problem(ring: dict) -> None:
    problem, _ = coil_in_a_uniform_field()
    other = ring_machine(per_pole=8)
    solution = solve(other["problem"])
    with pytest.raises(ValueError, match="solution does not belong to this problem"):
        flux_linkage(problem, solution)
    with pytest.raises(ValueError, match="solution does not belong to this problem"):
        torque(ring["problem"], solution)
    # Either part of the solution being of the wrong size is enough.
    good = ring["solution"]
    for wrong in (
        dataclasses.replace(good, potential=good.potential[:-1]),
        dataclasses.replace(good, flux_density=good.flux_density[:-1]),
    ):
        with pytest.raises(ValueError, match="solution does not belong to this problem"):
            flux_linkage(ring["problem"], wrong)
        with pytest.raises(ValueError, match="solution does not belong to this problem"):
            torque(ring["problem"], wrong)


def test_ring_must_be_triangle_numbers_between_two_sensible_radii(ring: dict) -> None:
    problem, solution = ring["problem"], ring["solution"]
    gap = problem.gap
    mask = np.zeros(len(problem.triangles), dtype=bool)
    mask[gap.triangles] = True
    arguments = (problem.nodes, problem.triangles, solution.flux_density)
    with pytest.raises(ValueError, match="triangle numbers"):
        ring_torque_per_metre(*arguments, mask, gap.inner_radius, gap.outer_radius)
    for inner, outer in ((0.039, 0.036), (0.036, 0.036), (0.0, 0.036), (0.036, math.nan)):
        with pytest.raises(ValueError, match="ring radii"):
            ring_torque_per_metre(*arguments, gap.triangles, inner, outer)
    with pytest.raises(ValueError, match="flux density"):
        ring_torque_per_metre(
            problem.nodes, problem.triangles, solution.flux_density[:, 0],
            gap.triangles, gap.inner_radius, gap.outer_radius,
        )


def test_linkage_needs_one_potential_per_node_and_one_density_per_triangle() -> None:
    nodes = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [2.0, 3.0]])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    with pytest.raises(ValueError, match="potential"):
        linkage_per_metre(nodes, triangles, np.ones((4, 1)), np.ones(2))
    with pytest.raises(ValueError, match="turn density"):
        linkage_per_metre(nodes, triangles, np.ones(4), np.ones(3))


# Flux density in the air gap


def test_gap_flux_density_of_a_uniform_field(ring: dict) -> None:
    # The same field in every triangle: its radial part at angle t is
    # bx cos t + by sin t, and its tangential part by cos t - bx sin t.
    problem = ring["problem"]
    uniform = np.tile([0.3, -0.5], (len(problem.triangles), 1))
    solution = dataclasses.replace(ring["solution"], flux_density=uniform)
    angles = np.linspace(0.0, 2.0 * math.pi, 97, endpoint=False) + 0.01
    radial, around = gap_flux_density(problem, solution, angles)
    assert radial == pytest.approx(0.3 * np.cos(angles) - 0.5 * np.sin(angles), abs=1e-12)
    assert around == pytest.approx(-0.5 * np.cos(angles) - 0.3 * np.sin(angles), abs=1e-12)


def test_gap_flux_density_points_out_of_a_magnet_magnetised_outwards(ring: dict) -> None:
    # Pole 0 is magnetised outwards and spans the first eighth of the circle.
    middles = (np.arange(8) + 0.5) * 2.0 * math.pi / 8 + 1e-3  # off the triangle sides
    radial, around = gap_flux_density(ring["problem"], ring["solution"], middles)
    assert np.all(radial[0::2] > 0.3) and np.all(radial[1::2] < -0.3)
    assert radial[1::2] == pytest.approx(-radial[0::2], rel=1e-6)
    assert np.all(np.abs(around) < 0.2 * np.abs(radial))


def test_gap_flux_density_is_the_value_of_the_triangle_at_mid_gap(ring: dict) -> None:
    problem, solution = ring["problem"], ring["solution"]
    gap = problem.gap
    radius = 0.5 * (gap.inner_radius + gap.outer_radius)
    corners = problem.nodes[problem.triangles[gap.triangles]]
    angles = np.linspace(0.0, 2.0 * math.pi, 50, endpoint=False) + 0.013
    radial, around = gap_flux_density(problem, solution, angles)
    for k, angle in enumerate(angles):
        point = radius * np.array([math.cos(angle), math.sin(angle)])

        def side(a: np.ndarray, b: np.ndarray) -> float:
            return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])

        # Inside a triangle the point is on the same side of all three edges.
        holds = [
            number
            for number, (a, b, c) in zip(gap.triangles, corners)
            if abs(np.sign([side(a, b), side(b, c), side(c, a)]).sum()) == 3
        ]
        assert len(holds) == 1
        bx, by = solution.flux_density[holds[0]]
        assert radial[k] == pytest.approx(bx * math.cos(angle) + by * math.sin(angle), rel=1e-12)
        assert around[k] == pytest.approx(by * math.cos(angle) - bx * math.sin(angle), rel=1e-12)


@pytest.mark.parametrize("model_poles", [1, 2, 4])
def test_gap_flux_density_of_a_slice_is_that_of_the_whole_machine(
    ring: dict, model_poles: int
) -> None:
    # One pole is an anti-periodic slice: the sign must turn over from each
    # part to the next. Two and four poles repeat as they are.
    part = ring_machine(model_poles=model_poles)
    assert part["problem"].antiperiodic == (model_poles == 1)
    solution = solve(part["problem"])
    # Off the triangle sides; also below zero and beyond a full turn.
    angles = np.linspace(-1.0, 7.0, 401) + 1e-3
    whole = gap_flux_density(ring["problem"], ring["solution"], angles)
    found = gap_flux_density(part["problem"], solution, angles)
    assert np.abs(whole[0]).max() > 0.3
    assert found[0] == pytest.approx(whole[0], abs=1e-8)
    assert found[1] == pytest.approx(whole[1], abs=1e-8)


def test_gap_flux_density_gives_back_the_shape_it_was_asked_for(ring: dict) -> None:
    radial, around = gap_flux_density(ring["problem"], ring["solution"], [])
    assert radial.shape == around.shape == (0,)
    radial, around = gap_flux_density(ring["problem"], ring["solution"], [0.1, 0.2, 0.3])
    assert radial.shape == around.shape == (3,)


def test_gap_flux_density_checks_what_it_is_given(ring: dict) -> None:
    problem, solution = ring["problem"], ring["solution"]
    nodes, triangles = grid(2, 2, 1.0, 1.0)
    plain = Problem(nodes=nodes, triangles=triangles, materials=(AIR,), fixed_nodes=[0])
    with pytest.raises(ValueError, match="no air gap"):
        gap_flux_density(plain, solve(plain), [0.0])
    for wrong in ([[0.0, 1.0]], [0.0, math.nan], 1.0):
        with pytest.raises(ValueError, match="angles"):
            gap_flux_density(problem, solution, wrong)
    with pytest.raises(ValueError, match="solution does not belong to this problem"):
        gap_flux_density(problem, dataclasses.replace(solution, potential=solution.potential[:-1]), [0.0])
    # A gap that is only part of the circle cannot answer for the rest.
    gap = problem.gap
    half = dataclasses.replace(
        problem, gap=Gap(gap.triangles[: len(gap.triangles) // 2], gap.inner_radius, gap.outer_radius)
    )
    with pytest.raises(ValueError, match="all the way round"):
        gap_flux_density(half, solution, np.linspace(0.0, 6.0, 50))


# The reference motor, with a placeholder steel curve


@pytest.fixture(scope="module")
def motor() -> dict:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / "qdd_outer_36s42p.yaml"))
    geometry = machine.build_geometry()
    mesh = build_mesh(geometry)
    # Placeholders: not the design's real steel or magnet grade.
    materials = {
        "M270-35A": load_material("generic_35A_placeholder"),
        "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
    }
    cases = {}
    for current in (0.0, 2.0, 10.0, -10.0, 40.0):
        problem = build_problem(machine, mesh, materials, current=current)
        cases[current] = (problem, solve(problem))
    return {"machine": machine, "geometry": geometry, "mesh": mesh, "cases": cases}


def motor_torque(motor: dict, current: float) -> float:
    return torque(*motor["cases"][current])


def test_motor_has_no_torque_without_current_at_a_symmetric_position(motor: dict) -> None:
    # At rotor angle 0 a tooth and a magnet are both centred on the x axis,
    # and the machine is its own mirror image in that axis: the pull of the
    # magnets on the teeth balances. What is left comes from the mesh, which
    # is not quite its own mirror image.
    rated = motor_torque(motor, 10.0)
    assert rated > 1.0  # N m
    assert abs(motor_torque(motor, 0.0)) < 1e-3 * rated


def test_motor_torque_changes_sign_with_the_current(motor: dict) -> None:
    # Exact only while the steel is not saturated: the current pattern has
    # no mirror axis in common with the magnets, so saturation leaves a small
    # difference, about 0.1 % at this current and 0.5 % at 40 A.
    forward, backward = motor_torque(motor, 10.0), motor_torque(motor, -10.0)
    assert forward > 0.0  # the current is set to drive the rotor counter-clockwise
    assert backward == pytest.approx(-forward, rel=0.01)


def gap_layer(motor: dict, prefix: str) -> Gap:
    """One of the two layers of air beside the band, as a ring to take torque in."""
    mesh, geometry = motor["mesh"], motor["geometry"]
    wanted = [k for k, region in enumerate(geometry.regions) if region.name.startswith(prefix)]
    inside = np.flatnonzero(np.isin(mesh.triangle_region, wanted))
    radius = np.linalg.norm(mesh.nodes[mesh.triangles[inside]], axis=2)
    return Gap(inside, float(radius.min()), float(radius.max()), rotor_outside=True)


@pytest.mark.parametrize("current", [10.0, 40.0])
def test_motor_torque_is_the_same_at_three_radii(motor: dict, current: float) -> None:
    problem, solution = motor["cases"][current]
    in_band = torque(problem, solution)
    for prefix in ("gap_stator", "gap_rotor"):
        layer = gap_layer(motor, prefix)
        assert layer.outer_radius - layer.inner_radius == pytest.approx(0.0002, rel=1e-6)
        found = -motor["geometry"].stack_length * problem.repeats * ring_torque_per_metre(
            problem.nodes, problem.triangles, solution.flux_density,
            layer.triangles, layer.inner_radius, layer.outer_radius,
        )
        # Measured: within 0.6 % on this mesh, 0.1 % with triangles half the size.
        assert found == pytest.approx(in_band, rel=0.015)


def test_motor_phase_linkages_nearly_add_up_to_nothing(motor: dict) -> None:
    # The fundamental parts of the three phases cancel. What is left is the
    # part common to all three, from the third harmonic of the magnet field;
    # it is a few percent here.
    for current in (0.0, 10.0):
        linked = flux_linkage(*motor["cases"][current])
        assert list(linked) == ["A", "B", "C"]
        largest = max(abs(value) for value in linked.values())
        assert 5e-3 < largest < 20e-3  # webers
        assert abs(sum(linked.values())) < 0.08 * largest


def space_vector(values: dict) -> complex:
    """Three phase values as one complex number: 2/3 (a + b e^(j 120) + c e^(j 240))."""
    turn = cmath.exp(2j * math.pi / 3.0)
    return 2.0 / 3.0 * (values["A"] + turn * values["B"] + turn * turn * values["C"])


def test_motor_torque_agrees_with_flux_linkage_times_current(motor: dict) -> None:
    # T = 3/2 p Im(conj(psi) i), with psi and i as space vectors. Exact for
    # a machine without saliency or saturation; close at a low current.
    problem, solution = motor["cases"][2.0]
    linked = space_vector(flux_linkage(problem, solution))
    current = space_vector(motor["machine"].excitation(0.0, 2.0))
    expected = 1.5 * 21 * (linked.conjugate() * current).imag
    driven = torque(problem, solution) - motor_torque(motor, 0.0)
    assert driven == pytest.approx(expected, rel=0.01)


def test_motor_current_is_a_quarter_turn_ahead_of_the_magnet_flux(motor: dict) -> None:
    # Current in phase with the back-EMF, which is what advance 0 promises.
    linked = space_vector(flux_linkage(*motor["cases"][0.0]))
    current = space_vector(motor["machine"].excitation(0.0, 2.0))
    lead = math.degrees(cmath.phase(current / linked))
    assert lead == pytest.approx(90.0, abs=0.5)


def test_motor_slice_and_whole_machine_give_the_same_results() -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / "qdd_outer_12s10p.yaml"))
    materials = {"M270-35A": load_material("generic_35A_placeholder"), "N42SH": Magnet(1.2)}
    found = []
    for full in (False, True):
        mesh = build_mesh(machine.build_geometry(full=full))
        problem = build_problem(machine, mesh, materials, current=10.0)
        assert problem.repeats == (1 if full else 2)
        solution = solve(problem)
        found.append((torque(problem, solution), flux_linkage(problem, solution)))
    (part_torque, part_linked), (whole_torque, whole_linked) = found
    assert part_torque == pytest.approx(whole_torque, rel=2e-3)
    for phase in "ABC":
        assert part_linked[phase] == pytest.approx(whole_linked[phase], rel=2e-3, abs=2e-6)


# Core loss from flux density waveforms

# Coefficients given by hand, of the size of a 0.35 mm sheet's.
SHEET = CoreLossCoefficients(hysteresis=0.017, eddy=5e-5, excess=4e-4)
# Mean of |cos|^1.5 over a turn, by a fine sum.
MEAN_COS_15 = float(
    np.mean(np.abs(np.cos(2 * math.pi * (np.arange(200000) + 0.5) / 200000)) ** 1.5)
)


def sine(peak: float, direction: float, samples: int = 48, phase: float = 0.0) -> np.ndarray:
    """One period of an alternating field along a direction, shape (samples, 1, 2)."""
    wave = peak * np.sin(2 * math.pi * np.arange(samples) / samples + phase)
    return wave[:, None, None] * np.array([math.cos(direction), math.sin(direction)])


def terms(coefficients: CoreLossCoefficients, frequency: float, peak: float) -> tuple:
    """The three terms of the loss formula for a sine wave, W/kg."""
    return (
        coefficients.hysteresis * frequency * peak**coefficients.hysteresis_exponent,
        coefficients.eddy * (frequency * peak) ** 2,
        coefficients.excess * (frequency * peak) ** 1.5,
    )


@pytest.mark.parametrize("frequency", [50.0, 400.0, 1000.0])
@pytest.mark.parametrize("peak", [0.3, 1.0, 1.6])
@pytest.mark.parametrize("direction", [0.0, 0.7, 2.5])
def test_sine_wave_gives_the_loss_formula(frequency, peak, direction) -> None:
    hysteresis, eddy, excess = core_loss_density(sine(peak, direction), 1 / frequency, SHEET)
    expected = terms(SHEET, frequency, peak)
    assert hysteresis[0] == pytest.approx(expected[0], rel=1e-12)
    assert eddy[0] == pytest.approx(expected[1], rel=1e-12)
    # The mean of |dB/dt|^1.5 is a sum over 48 samples of a curve with corners.
    assert excess[0] == pytest.approx(expected[2], rel=2e-4)
    assert hysteresis[0] + eddy[0] + excess[0] == pytest.approx(
        float(SHEET.loss(frequency, peak)), rel=1e-4
    )


def test_sine_wave_with_another_hysteresis_exponent() -> None:
    sheet = CoreLossCoefficients(0.02, 5e-5, 4e-4, hysteresis_exponent=1.8)
    hysteresis, _, _ = core_loss_density(sine(1.3, 0.4), 1 / 200.0, sheet)
    assert hysteresis[0] == pytest.approx(0.02 * 200.0 * 1.3**1.8, rel=1e-12)


def test_loss_terms_scale_with_frequency() -> None:
    wave = sine(1.2, 0.3) + 0.2 * sine(1.0, 1.9, phase=0.5) * np.array([1.0, -1.0])
    slow = core_loss_density(wave, 1 / 50.0, SHEET)
    fast = core_loss_density(wave, 1 / 400.0, SHEET)
    for power, low, high in zip((1.0, 2.0, 1.5), slow, fast):
        assert high[0] == pytest.approx(8.0**power * low[0], rel=1e-12)


def test_constant_field_has_no_loss() -> None:
    wave = np.broadcast_to(np.array([0.9, -1.4]), (48, 3, 2))
    for periodic in (True, False):
        for term in core_loss_density(wave, 0.01, SHEET, periodic=periodic):
            assert np.all(np.abs(term) < 1e-20)


def test_steady_field_under_the_swing_changes_nothing() -> None:
    plain = core_loss_density(sine(0.4, 0.6), 0.005, SHEET)
    offset = core_loss_density(sine(0.4, 0.6) + np.array([1.1, 0.3]), 0.005, SHEET)
    for a, b in zip(plain, offset):
        assert b[0] == pytest.approx(a[0], rel=1e-9)


def test_peak_between_samples_is_missed_by_no_more_than_the_spacing_allows() -> None:
    # 50 samples and a shifted phase: no sample falls on the peak.
    hysteresis, eddy, _ = core_loss_density(sine(1.0, 0.0, 50, phase=0.3), 0.01, SHEET)
    assert eddy[0] == pytest.approx(5e-5 * 100.0**2, rel=1e-12)
    full = 0.017 * 100.0
    assert full * math.cos(math.pi / 50) ** 2 <= hysteresis[0] <= full * (1 + 1e-12)


def test_harmonics_add_to_the_eddy_loss_by_their_order_squared() -> None:
    samples = 96
    turn = 2 * math.pi * np.arange(samples) / samples
    wave = np.zeros((samples, 1, 2))
    wave[:, 0, 0] = 1.0 * np.sin(turn) + 0.15 * np.sin(3 * turn + 0.4)
    wave[:, 0, 1] = 0.05 * np.cos(7 * turn)
    _, eddy, _ = core_loss_density(wave, 1 / 100.0, SHEET)
    expected = 5e-5 * 100.0**2 * (1.0**2 + 9 * 0.15**2 + 49 * 0.05**2)
    assert eddy[0] == pytest.approx(expected, rel=1e-12)


def test_turning_field_counts_as_two_alternating_fields() -> None:
    turn = 2 * math.pi * np.arange(48) / 48
    wave = 1.1 * np.stack([np.cos(turn), np.sin(turn)], axis=1)[:, None, :]
    hysteresis, eddy, excess = core_loss_density(wave, 1 / 100.0, SHEET)
    alternating = terms(SHEET, 100.0, 1.1)
    # The swing along any axis is read from 48 samples of the circle.
    assert hysteresis[0] == pytest.approx(2 * alternating[0], rel=2 * (1 - math.cos(math.pi / 48)))
    assert eddy[0] == pytest.approx(2 * alternating[1], rel=1e-12)
    # |dB/dt| is constant, not a rectified sine.
    assert excess[0] == pytest.approx(alternating[2] / MEAN_COS_15, rel=1e-9)


def test_elliptical_field_uses_its_two_axes_whatever_the_frame() -> None:
    turn = 2 * math.pi * np.arange(48) / 48
    ellipse = np.stack([1.2 * np.cos(turn), 0.5 * np.sin(turn)], axis=1)
    results = []
    for angle in (0.0, 0.9, 2.2):
        c, s = math.cos(angle), math.sin(angle)
        turned = ellipse @ np.array([[c, s], [-s, c]])
        results.append([term[0] for term in core_loss_density(turned[:, None, :], 0.01, SHEET)])
    assert results[0][0] == pytest.approx(0.017 * 100.0 * (1.2**2 + 0.5**2), rel=1e-12)
    assert results[0][1] == pytest.approx(5e-5 * 100.0**2 * (1.2**2 + 0.5**2), rel=1e-12)
    for other in results[1:]:
        assert other == pytest.approx(results[0], rel=1e-9)


def test_open_window_takes_rates_from_differences() -> None:
    intervals = 48
    turn = 2 * math.pi * np.arange(intervals + 1) / intervals
    wave = np.zeros((intervals + 1, 1, 2))
    wave[:, 0, 1] = 1.4 * np.sin(turn)
    hysteresis, eddy, excess = core_loss_density(wave, 1 / 200.0, SHEET, periodic=False)
    expected = terms(SHEET, 200.0, 1.4)
    x = math.pi / intervals
    assert hysteresis[0] == pytest.approx(expected[0], rel=1e-12)
    assert eddy[0] == pytest.approx(expected[1] * (math.sin(x) / x) ** 2, rel=1e-12)
    assert excess[0] == pytest.approx(expected[2], rel=3e-3)


def test_open_window_need_not_close() -> None:
    # A steady rise of 0.5 T in 2 ms: dB/dt = 250 T/s throughout.
    wave = np.zeros((11, 1, 2))
    wave[:, 0, 0] = np.linspace(0.2, 0.7, 11)
    hysteresis, eddy, excess = core_loss_density(wave, 0.002, SHEET, periodic=False)
    assert eddy[0] == pytest.approx(5e-5 / (2 * math.pi**2) * 250.0**2, rel=1e-12)
    assert excess[0] == pytest.approx(
        4e-4 / ((2 * math.pi) ** 1.5 * MEAN_COS_15) * 250.0**1.5, rel=1e-9
    )
    assert hysteresis[0] == pytest.approx(0.017 * 500.0 * 0.25**2, rel=1e-12)


def test_core_loss_density_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        core_loss_density(np.zeros((48, 2)), 0.01, SHEET)
    with pytest.raises(ValueError, match="duration"):
        core_loss_density(sine(1.0, 0.0), 0.0, SHEET)
    with pytest.raises(ValueError, match="too few"):
        core_loss_density(sine(1.0, 0.0, 3), 0.01, SHEET)
    with pytest.raises(ValueError, match="too few"):
        core_loss_density(sine(1.0, 0.0, 1), 0.01, SHEET, periodic=False)


def steel_block(field: tuple[float, float], turned: float = 0.0) -> tuple:
    """A 20 mm by 10 mm steel block in a uniform field, solved; turned about the origin."""
    flat, triangles = grid(8, 4, 0.02, 0.01)
    edge = np.flatnonzero(
        np.isclose(flat[:, 0], 0.0)
        | np.isclose(flat[:, 0], 0.02)
        | np.isclose(flat[:, 1], 0.0)
        | np.isclose(flat[:, 1], 0.01)
    )
    c, s = math.cos(turned), math.sin(turned)
    nodes = flat @ np.array([[c, s], [-s, c]])
    # B = (dA/dy, -dA/dx): this potential on the outline gives the field throughout.
    potential = field[0] * nodes[:, 1] - field[1] * nodes[:, 0]
    problem = Problem(
        nodes=nodes,
        triangles=triangles,
        materials=(LinearMaterial(2000.0),),
        fixed_nodes=edge,
        fixed_values=potential[edge],
    )
    return problem, solve(problem)


def test_block_in_uniform_sine_field_matches_the_loss_formula() -> None:
    frequency, peak, density = 400.0, 1.3, 7650.0
    wave = peak * np.sin(2 * math.pi * np.arange(48) / 48)
    steps = [steel_block((0.6 * b, 0.8 * b)) for b in wave]
    triangles = np.arange(len(steps[0][0].triangles))
    field, turn = material_flux_density(steps, triangles)
    assert np.array_equal(turn, np.zeros_like(turn))
    assert np.allclose(np.hypot(field[..., 0], field[..., 1]), np.abs(wave)[:, None], atol=1e-9)
    _, area = shape_gradients(steps[0][0].nodes, steps[0][0].triangles)
    mass = density * area * 0.05  # 50 mm deep
    assert mass.sum() == pytest.approx(7650.0 * 0.02 * 0.01 * 0.05)
    loss = sum(mass @ term for term in core_loss_density(field, 1 / frequency, SHEET))
    assert loss == pytest.approx(mass.sum() * float(SHEET.loss(frequency, peak)), rel=1e-4)
    slow = sum(mass @ term for term in core_loss_density(field, 1 / 50.0, SHEET))
    assert slow == pytest.approx(mass.sum() * float(SHEET.loss(50.0, peak)), rel=1e-4)


def test_block_in_constant_field_has_no_loss() -> None:
    steps = [steel_block((0.9, -0.4)) for _ in range(8)]
    field, _ = material_flux_density(steps, np.arange(len(steps[0][0].triangles)))
    for term in core_loss_density(field, 0.01, SHEET):
        assert np.all(term < 1e-12)


def test_block_turning_in_a_steady_field_sees_a_turning_field() -> None:
    angles = 2 * math.pi * np.arange(48) / 48
    steps = [steel_block((1.0, 0.0), turned=angle) for angle in angles]
    triangles = np.arange(len(steps[0][0].triangles))
    field, turn = material_flux_density(steps, triangles)
    assert np.allclose(turn, angles[:, None], atol=1e-9)
    # In the block's frame the field goes round backwards.
    assert np.allclose(field[..., 0], np.cos(angles)[:, None], atol=1e-9)
    assert np.allclose(field[..., 1], -np.sin(angles)[:, None], atol=1e-9)
    _, eddy, _ = core_loss_density(field, 0.01, SHEET)
    assert eddy == pytest.approx(2 * 5e-5 * 100.0**2, rel=1e-9)


def test_block_turning_with_its_field_has_no_loss() -> None:
    angles = 2 * math.pi * np.arange(48) / 48
    steps = [steel_block((math.cos(a), math.sin(a)), turned=a) for a in angles]
    field, _ = material_flux_density(steps, np.arange(len(steps[0][0].triangles)))
    assert np.allclose(field, [1.0, 0.0], atol=1e-9)
    for term in core_loss_density(field, 0.01, SHEET):
        assert np.all(term < 1e-12)


def test_material_flux_density_refuses_bad_input() -> None:
    step = steel_block((1.0, 0.0))
    with pytest.raises(ValueError, match="at least one"):
        material_flux_density([], np.arange(3))
    with pytest.raises(ValueError, match="triangle numbers"):
        material_flux_density([step], np.ones(len(step[0].triangles), dtype=bool))
    other = dataclasses.replace(step[0], triangles=step[0].triangles[:, [1, 2, 0]])
    with pytest.raises(ValueError, match="not those of the first"):
        material_flux_density([step, (other, step[1])], np.arange(3))
