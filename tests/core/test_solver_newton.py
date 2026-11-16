"""Nonlinear solve: Newton iteration on hand-built problems and on a motor."""

import inspect
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from polecraft.core.materials import (
    AIR,
    MU0,
    FrohlichSteel,
    LinearMaterial,
    Magnet,
    load_material,
)
from polecraft.core.problem import Problem, build_problem
from polecraft.core.solver import (
    ConvergenceError,
    Solution,
    assemble_stiffness,
    current_load,
    element_stiffness,
    magnet_load,
    shape_gradients,
    solve,
    solve_linear,
)

NU0 = 1.0 / MU0
LAW = FrohlichSteel(saturation=2.0, half_field=300.0)


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


# Linear problems


def linear_problem() -> Problem:
    """Iron, air, a magnet and a coil in a box, all with constant permeability."""
    nodes, triangles = grid(12, 10, 0.03, 0.02)
    centre = centres(nodes, triangles)
    kind = np.zeros(len(triangles), dtype=int)  # air
    kind[centre[:, 1] < 0.004] = 1  # iron along the bottom
    in_magnet = (centre[:, 1] > 0.004) & (centre[:, 1] < 0.008) & (centre[:, 0] < 0.015)
    kind[in_magnet] = 2
    in_coil = (centre[:, 1] > 0.012) & (centre[:, 0] > 0.02)
    edge = np.flatnonzero(
        (nodes[:, 0] == 0.0) | (nodes[:, 0] == 0.03) | (nodes[:, 1] == 0.0) | (nodes[:, 1] == 0.02)
    )
    return Problem(
        nodes=nodes,
        triangles=triangles,
        materials=(AIR, LinearMaterial(800.0), Magnet(remanence=1.1)),
        triangle_material=kind,
        current_density=np.where(in_coil, 3e6, 0.0),
        remanence=np.where(in_magnet[:, None], [0.0, 1.1], 0.0),
        fixed_nodes=edge,
        fixed_values=0.0,
    )


def by_hand(problem: Problem) -> np.ndarray:
    """The same problem put together from the linear building blocks."""
    nu = np.array([problem.materials[k].reluctivity(0.0) for k in problem.triangle_material])
    matrix = assemble_stiffness(problem.nodes, problem.triangles, nu)
    load = current_load(problem.nodes, problem.triangles, problem.current_density) + magnet_load(
        problem.nodes, problem.triangles, problem.remanence, nu
    )
    return solve_linear(
        matrix, load, problem.fixed_nodes, problem.fixed_values,
        problem.periodic, problem.antiperiodic,
    )


def test_linear_problem_takes_one_iteration() -> None:
    solution = solve(linear_problem())
    assert isinstance(solution, Solution)
    assert solution.iterations == 1


def test_linear_problem_gives_the_linear_answer() -> None:
    problem = linear_problem()
    expected = by_hand(problem)
    assert np.abs(expected).max() > 1e-4
    assert solve(problem).potential == pytest.approx(expected, rel=1e-12, abs=1e-18)


def test_solution_carries_the_field() -> None:
    problem = linear_problem()
    solution = solve(problem)
    count = len(problem.triangles)
    assert solution.potential.shape == (len(problem.nodes),)
    assert solution.flux_density.shape == (count, 2)
    assert solution.reluctivity.shape == (count,)
    gradients, _ = shape_gradients(problem.nodes, problem.triangles)
    slope = np.einsum("ei,eid->ed", solution.potential[problem.triangles], gradients)
    assert solution.flux_density == pytest.approx(np.column_stack([slope[:, 1], -slope[:, 0]]))
    nu = np.array([problem.materials[k].reluctivity(0.0) for k in problem.triangle_material])
    assert solution.reluctivity == pytest.approx(nu)
    assert np.abs(solution.flux_density).max() > 0.3


def test_residuals_start_large_and_end_at_rounding() -> None:
    solution = solve(linear_problem())
    assert len(solution.residuals) == 2  # before, and after the one iteration
    assert solution.residuals[0] > 1e-3
    assert solution.residuals[-1] < 1e-12
    assert solution.steps == (1.0,)


def test_a_problem_with_nothing_in_it_is_solved_at_once() -> None:
    nodes, triangles = grid(3, 3, 1.0, 1.0)
    problem = Problem(
        nodes=nodes, triangles=triangles, materials=(LAW,), fixed_nodes=[0], fixed_values=0.0
    )
    solution = solve(problem)
    assert solution.iterations == 0
    assert solution.residuals == (0.0,)
    assert np.all(solution.potential == 0.0)


def test_periodic_coupling_reaches_the_solve() -> None:
    nodes, triangles = grid(8, 4, 0.02, 0.01)
    centre = centres(nodes, triangles)
    left = np.flatnonzero(nodes[:, 0] == 0.0)
    right = np.flatnonzero(nodes[:, 0] == 0.02)
    ends = np.flatnonzero((nodes[:, 1] == 0.0) | (nodes[:, 1] == 0.01))
    problem = Problem(
        nodes=nodes, triangles=triangles, materials=(AIR,),
        current_density=np.where(centre[:, 0] < 0.01, 2e6, -2e6),
        fixed_nodes=ends, fixed_values=0.0,
        periodic=np.column_stack([right, left]), antiperiodic=True,
    )
    solution = solve(problem)
    assert np.abs(solution.potential[left]).max() > 0.0
    assert np.array_equal(solution.potential[right], -solution.potential[left])
    assert solution.potential == pytest.approx(by_hand(problem), rel=1e-12, abs=1e-18)


# A saturating core with an air gap


def core(ampere_turns: float, material: object = LAW) -> dict:
    """One leg of a C-core with an air gap, and the coil beside it.

    The leg is 20 mm of iron under a 0.5 mm gap, 10 mm wide; the flux runs
    along it, downwards for positive ampere-turns (current out of the
    plane). The top and bottom edges are left free, which makes them ideal
    iron: the yokes that close the circuit at no cost. The left edge is a
    field line. The coil is the 4 mm strip on the right, carrying
    ``ampere_turns``. The line between leg and coil is made a field line too,
    by coupling its nodes to one of them: no flux leaks from the leg into
    the coil, which is what the magnetic circuit formula assumes.
    """
    leg, coil_width, iron, gap = 0.010, 0.004, 0.020, 0.0005
    height = iron + gap
    nodes, triangles = grid(14, 41, leg + coil_width, height)
    centre = centres(nodes, triangles)
    in_leg = centre[:, 0] < leg
    in_iron = in_leg & (centre[:, 1] < iron)
    _, area = shape_gradients(nodes, triangles)
    wall = np.flatnonzero(np.isclose(nodes[:, 0], leg))
    problem = Problem(
        nodes=nodes,
        triangles=triangles,
        materials=(AIR, material),
        triangle_material=in_iron.astype(int),
        current_density=np.where(in_leg, 0.0, ampere_turns / area[~in_leg].sum()),
        fixed_nodes=np.flatnonzero(nodes[:, 0] == 0.0),
        fixed_values=0.0,
        periodic=np.column_stack([wall[1:], np.full(len(wall) - 1, wall[0])]),
    )
    return {"problem": problem, "in_leg": in_leg, "in_iron": in_iron, "iron": iron, "gap": gap}


def circuit_flux_density(ampere_turns: float, iron: float, gap: float) -> float:
    """B in the leg from the magnetic circuit: N I = H_iron(B) l_iron + B g / mu0."""
    return brentq(
        lambda b: float(LAW.field_strength(b)) * iron + b * gap / MU0 - ampere_turns, 0.0, 10.0,
        xtol=1e-14,
    )


# From almost linear to deep in saturation: the last one would give 4.5 T
# with the iron's starting permeability.
AMPERE_TURNS = [50.0, 400.0, 700.0, 1000.0, 1800.0]


@pytest.mark.parametrize("ampere_turns", AMPERE_TURNS)
def test_saturating_core_matches_the_magnetic_circuit(ampere_turns: float) -> None:
    case = core(ampere_turns)
    solution = solve(case["problem"])
    expected = circuit_flux_density(ampere_turns, case["iron"], case["gap"])
    b = solution.flux_density[case["in_leg"]]
    # The flux runs down the leg for current out of the plane on its right.
    assert b[:, 1] == pytest.approx(-expected, rel=1e-7)
    assert np.abs(b[:, 0]).max() < 1e-7 * expected


def test_core_cases_cover_the_whole_curve() -> None:
    fields = [circuit_flux_density(turns, 0.020, 0.0005) for turns in AMPERE_TURNS]
    assert fields[0] < 0.2
    assert 1.2 < fields[2] < 1.8
    assert fields[-1] > 2.0
    # Saturation matters: the starting permeability alone would give far more.
    linear = AMPERE_TURNS[-1] / (0.020 * float(LAW.reluctivity(0.0)) + 0.0005 / MU0)
    assert linear > 1.5 * fields[-1]


def test_core_reluctivity_is_that_of_the_solved_field() -> None:
    case = core(1000.0)
    solution = solve(case["problem"])
    b = np.linalg.norm(solution.flux_density, axis=1)
    iron = case["in_iron"]
    assert solution.reluctivity[iron] == pytest.approx(LAW.reluctivity(b[iron]), rel=1e-9)
    assert solution.reluctivity[~iron] == pytest.approx(NU0)
    assert solution.reluctivity[iron].min() > 5.0 * float(LAW.reluctivity(0.0))


def test_reversing_the_current_reverses_the_saturated_field() -> None:
    forward = solve(core(1000.0)["problem"])
    backward = solve(core(-1000.0)["problem"])
    assert backward.potential == pytest.approx(-forward.potential, rel=1e-9, abs=1e-15)
    assert backward.iterations == forward.iterations


def test_core_with_a_tabulated_curve_matches_its_own_circuit() -> None:
    # The same check with a BH table in place of the formula.
    steel = load_material("generic_35A_placeholder")
    case = core(1000.0, steel)
    solution = solve(case["problem"])
    expected = brentq(
        lambda b: float(steel.field_strength(b)) * case["iron"] + b * case["gap"] / MU0 - 1000.0,
        0.0, 10.0, xtol=1e-14,
    )
    assert solution.flux_density[case["in_leg"], 1] == pytest.approx(-expected, rel=1e-7)


# The iteration itself


@pytest.mark.parametrize("ampere_turns", AMPERE_TURNS)
def test_residual_falls_at_every_iteration(ampere_turns: float) -> None:
    solution = solve(core(ampere_turns)["problem"])
    residuals = np.array(solution.residuals)
    assert len(residuals) == solution.iterations + 1
    assert np.all(np.diff(residuals) < 0.0)
    assert residuals[-1] <= 1e-10


@pytest.mark.parametrize("ampere_turns", AMPERE_TURNS)
def test_iteration_count_stays_small(ampere_turns: float) -> None:
    solution = solve(core(ampere_turns)["problem"])
    assert 2 <= solution.iterations <= 15


def test_convergence_is_fast_at_the_end() -> None:
    # Newton's method: once close, each iteration about squares the error,
    # so from one part in ten thousand it is two iterations to rounding.
    residuals = np.array(solve(core(1000.0)["problem"], tolerance=1e-12).residuals)
    close = int(np.argmax(residuals < 1e-4))
    assert residuals[-1] < 1e-12
    assert len(residuals) - 1 - close <= 2


def test_relaxation_shortens_the_steps_that_overshoot() -> None:
    # Deep in saturation the first full steps overshoot and are cut back.
    solution = solve(core(1800.0)["problem"])
    assert len(solution.steps) == solution.iterations
    assert all(0.0 < step <= 1.0 for step in solution.steps)
    assert min(solution.steps) < 1.0
    assert solution.steps[-1] == 1.0  # the last iterations take the full step
    # Every step is a halving of the full one.
    assert all(math.log2(step) == round(math.log2(step)) for step in solution.steps)


def test_a_nearly_linear_case_needs_no_relaxation() -> None:
    steps = solve(core(50.0)["problem"]).steps
    assert len(steps) >= 2
    assert set(steps) == {1.0}


@pytest.mark.parametrize("field", [5.0, -5.0, 500.0, -500.0])
def test_a_start_far_from_the_answer_still_converges(field: float) -> None:
    # A uniform field of the wrong size, or the wrong way round, to begin with.
    problem = core(1000.0)["problem"]
    solution = solve(problem, start=field * problem.nodes[:, 0])
    assert np.all(np.diff(solution.residuals) < 0.0)
    assert solution.iterations <= 15
    assert solution.potential == pytest.approx(solve(problem).potential, rel=1e-6, abs=1e-12)


def test_a_rough_start_still_converges() -> None:
    problem = core(1000.0)["problem"]
    start = np.random.default_rng(41).uniform(-1.0, 1.0, len(problem.nodes))
    solution = solve(problem, start=start)
    assert np.all(np.diff(solution.residuals) < 0.0)
    assert solution.residuals[0] > 1e3  # far out of balance to begin with
    assert solution.potential == pytest.approx(solve(problem).potential, rel=1e-6, abs=1e-12)


def test_default_tolerance_is_one_part_in_ten_thousand_million() -> None:
    assert inspect.signature(solve).parameters["tolerance"].default == 1e-10
    solution = solve(core(1000.0)["problem"])
    assert solution.residuals[-1] <= 1e-10
    assert solution.residuals[-2] > 1e-10


def test_a_tolerance_below_rounding_is_reported_as_a_stall() -> None:
    with pytest.raises(ConvergenceError, match="stalled: no step lowers the residual") as caught:
        solve(core(1000.0)["problem"], tolerance=1e-17)
    assert caught.value.residuals[-1] > 1e-17
    assert caught.value.residuals[-1] < 1e-11  # it got as far as rounding allows
    assert len(caught.value.steps) == len(caught.value.residuals) - 1


def test_a_tighter_tolerance_takes_at_least_as_many_iterations() -> None:
    loose = solve(core(1000.0)["problem"], tolerance=1e-3)
    tight = solve(core(1000.0)["problem"], tolerance=1e-11)
    assert loose.residuals[-1] <= 1e-3
    assert tight.residuals[-1] <= 1e-11
    assert loose.iterations < tight.iterations
    assert loose.potential == pytest.approx(tight.potential, rel=1e-2)


def test_starting_from_the_solution_takes_no_iteration() -> None:
    problem = core(1000.0)["problem"]
    first = solve(problem)
    again = solve(problem, start=first.potential)
    assert again.iterations == 0
    assert np.array_equal(again.potential, first.potential)


def test_starting_close_takes_fewer_iterations() -> None:
    first = solve(core(1000.0)["problem"])
    cold = solve(core(1050.0)["problem"])
    warm = solve(core(1050.0)["problem"], start=first.potential)
    assert warm.iterations < cold.iterations
    assert warm.potential == pytest.approx(cold.potential, rel=1e-6, abs=1e-12)


def test_boundary_values_and_couplings_are_put_into_the_start() -> None:
    # Spoil the solution at the fixed nodes and at the copies: with those put
    # right first, it is the solution again and no iteration is needed.
    problem = two_way_problem(doubled=False)
    first = solve(problem)
    spoiled = first.potential.copy()
    spoiled[problem.fixed_nodes] = 0.7
    spoiled[problem.periodic[:, 0]] = -0.3
    again = solve(problem, start=spoiled)
    assert again.iterations == 0
    assert np.array_equal(again.potential, first.potential)


def test_a_start_that_breaks_the_boundary_values_is_put_right() -> None:
    problem = core(1000.0)["problem"]
    start = np.random.default_rng(40).uniform(-1e-3, 1e-3, len(problem.nodes))
    solution = solve(problem, start=start)
    assert np.all(solution.potential[problem.fixed_nodes] == 0.0)
    assert solution.potential == pytest.approx(solve(problem).potential, rel=1e-6, abs=1e-12)


def test_too_few_iterations_is_an_error_with_a_report() -> None:
    with pytest.raises(ConvergenceError, match="did not converge in 2 iterations") as caught:
        solve(core(1800.0)["problem"], max_iterations=2)
    assert len(caught.value.residuals) == 3
    assert len(caught.value.steps) == 2
    assert caught.value.residuals[-1] > 1e-8
    lines = str(caught.value).splitlines()
    assert lines[1] == "  iteration   residual    step"
    assert len(lines) == 1 + 1 + 3  # message, heading, start and two iterations
    assert f"{caught.value.residuals[-1]:.3e}" in lines[-1]
    assert lines[-1].endswith(f"{caught.value.steps[-1]:8.4f}")


def test_report_lists_every_iteration() -> None:
    solution = solve(core(1800.0)["problem"])  # with steps of several sizes
    lines = solution.report().splitlines()
    assert lines[0] == "Newton iteration"
    assert lines[1] == "  iteration   residual    step"
    assert len(lines) == 2 + (solution.iterations + 1) + 1
    assert lines[-1] == f"  converged in {solution.iterations} iterations"
    assert lines[2] == f"  {0:9d}  {solution.residuals[0]:.3e}"
    for number in range(1, solution.iterations + 1):
        assert lines[2 + number] == (
            f"  {number:9d}  {solution.residuals[number]:.3e}{solution.steps[number - 1]:8.4f}"
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"tolerance": 0.0}, "tolerance"),
        ({"tolerance": -1e-6}, "tolerance"),
        ({"tolerance": math.nan}, "tolerance"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"start": np.zeros(3)}, "start"),
        ({"start": np.full(15 * 42, math.nan)}, "start"),
        ({"start": np.full(15 * 42, 1e200)}, "start gives a field too large"),
    ],
)
def test_bad_options_are_rejected(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve(core(100.0)["problem"], **options)


def test_solve_does_not_change_the_problem_or_the_start() -> None:
    problem = core(1000.0)["problem"]
    start = np.zeros(len(problem.nodes))
    before = problem.nodes.copy(), problem.current_density.copy(), start.copy()
    solve(problem, start=start)
    assert np.array_equal(problem.nodes, before[0])
    assert np.array_equal(problem.current_density, before[1])
    assert np.array_equal(start, before[2])


# A field in two directions, with anti-periodic edges


def two_way_problem(doubled: bool) -> Problem:
    """A block of steel with a coil off-centre, repeating with its sign reversed.

    One period, with its left and right edges coupled anti-periodically; or,
    ``doubled``, two periods side by side, the second with the current
    reversed, coupled periodically. Node (i, j) of the first period is at the
    same place in both.
    """
    columns, rows, width, height = 12, 8, 0.024, 0.016
    periods = 2 if doubled else 1
    nodes, triangles = grid(periods * columns, rows, periods * width, height)
    centre = centres(nodes, triangles)
    x = centre[:, 0] % width
    in_coil = (x > 0.004) & (x < 0.010) & (centre[:, 1] > 0.004) & (centre[:, 1] < 0.010)
    sign = np.where(centre[:, 0] < width, 1.0, -1.0)
    left = np.flatnonzero(nodes[:, 0] == 0.0)
    right = np.flatnonzero(np.isclose(nodes[:, 0], periods * width))
    return Problem(
        nodes=nodes,
        triangles=triangles,
        materials=(LAW, AIR),
        triangle_material=in_coil.astype(int),
        current_density=np.where(in_coil, sign * 6e7, 0.0),
        fixed_nodes=np.flatnonzero((nodes[:, 1] == 0.0) | (nodes[:, 1] == height)),
        fixed_values=0.0,
        periodic=np.column_stack([right, left]),
        antiperiodic=not doubled,
    )


def test_antiperiodic_saturating_slice_matches_two_periods() -> None:
    one, two = two_way_problem(doubled=False), two_way_problem(doubled=True)
    half, whole = solve(one), solve(two)
    b = np.abs(half.flux_density)
    in_steel = one.triangle_material == 0
    # Both parts of the field matter, and the steel is well into saturation.
    assert b[in_steel, 0].max() > 1.0 and b[in_steel, 1].max() > 1.0
    assert half.reluctivity[in_steel].max() > 20.0 * float(LAW.reluctivity(0.0))
    assert half.iterations >= 4
    # Node (i, j) of the first period has number j * (columns + 1) + i.
    first = (np.arange(9)[:, None] * 25 + np.arange(13)[None, :]).ravel()
    second = first + 12
    scale = np.abs(whole.potential).max()
    assert whole.potential[first] == pytest.approx(half.potential, rel=1e-7, abs=1e-9 * scale)
    assert whole.potential[second] == pytest.approx(-half.potential, rel=1e-7, abs=1e-9 * scale)


def test_residual_is_the_unbalance_over_the_currents_being_balanced() -> None:
    # Rebuilt from the building blocks. At the start the potential is zero,
    # so the out-of-balance current is the load. It is measured against the
    # currents each triangle asks of its corners in the final field, added
    # without their signs, plus the load. A coupled pair counts as one node.
    problem = two_way_problem(doubled=False)
    solution = solve(problem)
    count = len(problem.nodes)
    local = element_stiffness(problem.nodes, problem.triangles, solution.reluctivity)
    asked = np.einsum("eij,ej->ei", local, solution.potential[problem.triangles])
    load = current_load(problem.nodes, problem.triangles, problem.current_density)
    size = np.bincount(problem.triangles.ravel(), weights=np.abs(asked).ravel(), minlength=count)
    size += np.abs(load)
    copy, original = problem.periodic.T
    start = load.copy()
    start[original] -= start[copy]  # the copy takes the opposite value
    size[original] += size[copy]
    free = np.ones(count, dtype=bool)
    free[problem.fixed_nodes] = False
    free[copy] = False
    expected = np.linalg.norm(start[free]) / np.linalg.norm(size[free])
    assert 1e-3 < expected < 1.0
    assert solution.residuals[0] == pytest.approx(expected, rel=1e-9)


def test_a_magnetised_triangle_needs_a_linear_material() -> None:
    nodes, triangles = grid(3, 3, 1.0, 1.0)
    problem = Problem(
        nodes=nodes, triangles=triangles, materials=(AIR, LAW),
        triangle_material=np.ones(len(triangles), dtype=int),
        remanence=[0.0, 0.5], fixed_nodes=[0],
    )
    with pytest.raises(ValueError, match="material 1 saturates.*cannot be magnetised"):
        solve(problem)


# A magnet driving a saturating core


def test_magnet_in_a_saturating_core_matches_the_magnetic_circuit() -> None:
    # A magnet, iron and an air gap in series, closed by ideal iron:
    # H_m l_m + H_iron(B) l_iron + B g / mu0 = 0, with H_m = (B - Br) / (mu_r mu0).
    # The right edge is a field line that is free to take any value, so the
    # flux is whatever the circuit allows. The iron here saturates at 1 T,
    # below the magnet's remanence, so it limits the flux.
    magnet, iron, gap = 0.004, 0.010, 0.0005
    remanence, mu_r = 1.3, 1.05
    weak = FrohlichSteel(saturation=1.0, half_field=300.0)
    nodes, triangles = grid(10, 29, 0.010, magnet + iron + gap)
    centre = centres(nodes, triangles)
    in_magnet = centre[:, 1] < magnet
    in_iron = (centre[:, 1] > magnet) & (centre[:, 1] < magnet + iron)
    kind = np.where(in_magnet, 2, np.where(in_iron, 1, 0))
    wall = np.flatnonzero(nodes[:, 0] == 0.010)
    problem = Problem(
        nodes=nodes,
        triangles=triangles,
        materials=(AIR, weak, Magnet(remanence, mu_r)),
        triangle_material=kind,
        remanence=np.where(in_magnet[:, None], [0.0, remanence], 0.0),
        fixed_nodes=np.flatnonzero(nodes[:, 0] == 0.0),
        fixed_values=0.0,
        periodic=np.column_stack([wall[1:], np.full(len(wall) - 1, wall[0])]),
    )
    solution = solve(problem)
    expected = brentq(
        lambda b: (b - remanence) / (mu_r * MU0) * magnet
        + float(weak.field_strength(b)) * iron
        + b * gap / MU0,
        0.0, remanence, xtol=1e-14,
    )
    # Without the iron the gap alone would leave 1.15 T.
    assert 1.0 < expected < 1.1
    assert solution.flux_density[:, 1] == pytest.approx(expected, rel=1e-7)
    assert solution.iterations <= 10


# The reference motor, with a placeholder steel curve


@pytest.fixture(scope="module")
def motor() -> dict:
    """The reference design meshed, with solves at several currents."""
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
    # Far past any current the winding could carry, to reach deep saturation.
    currents = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    solutions = {
        current: solve(build_problem(machine, mesh, materials, current=current))
        for current in currents
    }
    return {
        "machine": machine, "geometry": geometry, "mesh": mesh, "materials": materials,
        "currents": currents, "solutions": solutions,
    }


def region_kind(motor: dict) -> np.ndarray:
    """Kind of the region each triangle lies in; the band counts as air."""
    kinds = np.array([region.kind for region in motor["geometry"].regions] + ["air"])
    return kinds[motor["mesh"].triangle_region]


def band_torque(motor: dict, solution: Solution) -> float:
    """Torque on the rotor in N m, from the Maxwell stress averaged over the band.

    T = L / (mu0 t) * integral of r B_r B_t over the band, t being its
    thickness, times the number of slices in the machine. That is the torque
    on what the band encloses, the stator; the rotor of this design is the
    outer part, so its torque is the opposite. Only a rough figure: the
    proper torque calculation is a later step.
    """
    mesh, geometry = motor["mesh"], motor["geometry"]
    band = mesh.triangle_region == len(geometry.regions)
    centre = centres(mesh.nodes, mesh.triangles)[band]
    radius = np.linalg.norm(centre, axis=1)
    b = solution.flux_density[band]
    radial = (b * centre).sum(axis=1) / radius
    around = (b[:, 1] * centre[:, 0] - b[:, 0] * centre[:, 1]) / radius
    area = mesh.triangle_areas()[band]
    thickness = geometry.band_outer_radius - geometry.band_inner_radius
    slices = round(2.0 * math.pi / geometry.span_angle)
    return -slices * geometry.stack_length / (MU0 * thickness) * (
        radius * radial * around * area
    ).sum()


def test_motor_solve_converges_at_every_current(motor: dict) -> None:
    for current, solution in motor["solutions"].items():
        assert solution.residuals[-1] <= 1e-10, current
        assert np.all(np.diff(solution.residuals) < 0.0), current
        assert solution.iterations <= 20, current


def test_motor_air_gap_field_is_sensible_without_current(motor: dict) -> None:
    mesh, geometry = motor["mesh"], motor["geometry"]
    solution = motor["solutions"][0.0]
    band = mesh.triangle_region == len(geometry.regions)
    centre = centres(mesh.nodes, mesh.triangles)[band]
    angle = np.arctan2(centre[:, 1], centre[:, 0])
    radial = (solution.flux_density[band] * centre).sum(axis=1) / np.linalg.norm(centre, axis=1)
    pitch = 2.0 * math.pi / 42
    for j in range(7):
        under = np.abs(angle - j * pitch) < 0.25 * pitch
        mean = radial[under].mean()
        # Outwards under magnet 0, inwards under magnet 1, and so on; the
        # size is that of a surface-magnet motor with 1.2 T magnets.
        assert np.sign(mean) == (-1) ** j
        assert 0.6 < abs(mean) < 1.1


def test_motor_steel_is_not_driven_past_saturation(motor: dict) -> None:
    kind = region_kind(motor)
    steel = kind == "steel"
    for current, solution in motor["solutions"].items():
        b = np.linalg.norm(solution.flux_density, axis=1)
        assert np.all(np.isfinite(b))
        assert b.max() < 5.0, current
        # The placeholder steel saturates at 2 T of polarisation, J = B - mu0 H.
        # B itself has no such limit, and its peak at a corner grows with
        # a finer mesh.
        polarisation = b[steel] * (1.0 - MU0 * solution.reluctivity[steel])
        assert polarisation.max() < 2.01, current
    quiet = np.linalg.norm(motor["solutions"][0.0].flux_density, axis=1)
    assert 1.0 < np.percentile(quiet[kind == "steel"], 99) < 2.1


def test_motor_steel_saturates_as_the_current_rises(motor: dict) -> None:
    kind = region_kind(motor)
    stiffest = [
        motor["solutions"][current].reluctivity[kind == "steel"].max()
        for current in motor["currents"]
    ]
    assert stiffest[-1] > 10.0 * stiffest[0]
    low, high = motor["solutions"][5.0], motor["solutions"][80.0]
    assert np.median(high.reluctivity[kind == "steel"]) > np.median(low.reluctivity[kind == "steel"])
    assert np.all(high.reluctivity[kind != "steel"] == low.reluctivity[kind != "steel"])


def test_motor_torque_rises_less_than_linearly_with_current(motor: dict) -> None:
    currents = motor["currents"]
    torque = np.array([band_torque(motor, motor["solutions"][c]) for c in currents])
    drive = torque[1:] - torque[0]  # without the cogging torque at this position
    per_ampere = drive / np.array(currents[1:])
    # The current is set to push the rotor counter-clockwise.
    assert np.all(drive > 0.0)
    assert np.all(np.diff(drive) > 0.0)
    # Nearly linear at low current, then falling away as the steel saturates.
    assert per_ampere[1] == pytest.approx(per_ampere[0], rel=0.03)
    assert np.all(np.diff(per_ampere) < 0.0)
    assert per_ampere[-1] < 0.8 * per_ampere[0]


def test_motor_warm_start_saves_iterations(motor: dict) -> None:
    problem = build_problem(motor["machine"], motor["mesh"], motor["materials"], current=42.0)
    cold = solve(problem)
    warm = solve(problem, start=motor["solutions"][40.0].potential)
    assert warm.iterations < cold.iterations
    assert warm.potential == pytest.approx(cold.potential, rel=1e-6, abs=1e-10)
