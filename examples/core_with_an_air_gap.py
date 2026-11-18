"""A saturating iron core with an air gap, built by hand and checked against a formula.

Run from any folder: python examples/core_with_an_air_gap.py
Needs no mesher: the mesh is a plain grid made here.

This shows that the solver takes any problem, not only a motor. The iron
follows a saturation law given by a formula, so the answer is known: with N I
ampere-turns round a core of iron length l and gap g,

    N I = H(B) * l + B * g / mu0

where H(B) is the iron's curve. The solver should give the same B.
"""

import numpy as np
from scipy.optimize import brentq

from polecraft.core.materials import AIR, MU0, FrohlichSteel
from polecraft.core.problem import Problem
from polecraft.core.solver import shape_gradients, solve

IRON_LENGTH = 0.020  # metres of iron along the flux path
GAP = 0.0005  # metres of air
LEG_WIDTH = 0.010
COIL_WIDTH = 0.004
STEEL = FrohlichSteel(saturation=2.0, half_field=300.0)  # B = mu0 H + 2 H / (300 + H)


def grid(columns: int, rows: int, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and triangles of a rectangle, each cell cut on alternating diagonals."""
    x, y = np.meshgrid(np.linspace(0.0, width, columns + 1), np.linspace(0.0, height, rows + 1))
    nodes = np.column_stack([x.ravel(), y.ravel()])
    triangles = []
    for j in range(rows):
        for i in range(columns):
            a = j * (columns + 1) + i
            b, c, d = a + 1, a + columns + 2, a + columns + 1
            triangles += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    return nodes, np.array(triangles)


def core_problem(ampere_turns: float) -> tuple[Problem, np.ndarray]:
    """One leg of the core with its gap, and the coil beside it.

    The flux runs along the leg. The top and bottom edges are left free,
    which makes them ideal iron: the yokes that close the circuit. The left
    edge is a field line. So is the line between leg and coil, made one by
    coupling its nodes to a single node, so that no flux leaks into the
    coil: that is what the formula assumes.
    """
    nodes, triangles = grid(14, 41, LEG_WIDTH + COIL_WIDTH, IRON_LENGTH + GAP)
    centre = nodes[triangles].mean(axis=1)
    in_leg = centre[:, 0] < LEG_WIDTH
    in_iron = in_leg & (centre[:, 1] < IRON_LENGTH)
    _, area = shape_gradients(nodes, triangles)
    wall = np.flatnonzero(np.isclose(nodes[:, 0], LEG_WIDTH))
    problem = Problem(
        nodes=nodes,
        triangles=triangles,
        materials=(AIR, STEEL),
        triangle_material=in_iron.astype(int),  # 1 in the iron, 0 elsewhere
        current_density=np.where(in_leg, 0.0, ampere_turns / area[~in_leg].sum()),
        fixed_nodes=np.flatnonzero(nodes[:, 0] == 0.0),
        periodic=np.column_stack([wall[1:], np.full(len(wall) - 1, wall[0])]),
    )
    return problem, in_leg


def by_formula(ampere_turns: float) -> float:
    """B in the leg from the magnetic circuit, in tesla."""
    return brentq(
        lambda b: float(STEEL.field_strength(b)) * IRON_LENGTH + b * GAP / MU0 - ampere_turns,
        0.0,
        10.0,
        xtol=1e-14,
    )


def main() -> None:
    print("ampere-turns   B by formula   B from the solver   iterations")
    for ampere_turns in (50.0, 400.0, 700.0, 1000.0, 1800.0):
        problem, in_leg = core_problem(ampere_turns)
        solution = solve(problem)
        # The field in the leg is uniform; take its size in any triangle there.
        solved = float(np.abs(solution.flux_density[in_leg, 1]).mean())
        print(
            f"{ampere_turns:10.0f}   {by_formula(ampere_turns):10.6f} T   {solved:13.6f} T"
            f"   {solution.iterations:8d}"
        )
    unsaturated = 1800.0 / (IRON_LENGTH * float(STEEL.reluctivity(0.0)) + GAP / MU0)
    print(f"\nWithout saturation the last case would give {unsaturated:.1f} T.")


if __name__ == "__main__":
    main()
