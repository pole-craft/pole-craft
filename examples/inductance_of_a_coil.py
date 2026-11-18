"""Inductance of a coil from its flux linkage, checked against a formula.

Run from any folder: python examples/inductance_of_a_coil.py
Needs no mesher: the mesh is a plain grid made here.

A coil of N turns between two iron plates a distance h apart, its two sides
a distance d apart and each of width t, in a model of depth L, has

    inductance = mu0 * N^2 * L * (d + 2 t / 3) / h

The flux inside a coil side links only the turns outside it, which is where
the two thirds comes from. The solver gives the flux linked by the coil; the
inductance is that divided by the current.
"""

import numpy as np

from polecraft.core.materials import AIR, MU0
from polecraft.core.post import flux_linkage
from polecraft.core.problem import Problem
from polecraft.core.solver import shape_gradients, solve

TURNS = 40.0
CURRENT = 3.0  # amperes
SIDE = 0.004  # width of each coil side, metres
SPACING = 0.012  # between the two sides
HEIGHT = 0.010  # between the plates
DEPTH = 0.050  # along the axis


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


def coil_problem(cells: int) -> Problem:
    """The coil between its plates, on a grid of ``cells`` rows.

    The top and bottom edges are left free, which makes them the iron
    plates. The left edge is a field line; the right edge is left free too,
    as iron that carries the flux back.
    """
    width = SPACING + 2.0 * SIDE
    nodes, triangles = grid(5 * cells, cells, width, HEIGHT)
    x = nodes[triangles].mean(axis=1)[:, 0]
    _, area = shape_gradients(nodes, triangles)
    going = x < SIDE  # current out of the plane
    returning = x > width - SIDE  # and back in
    # Turns per square metre, with the sign of the current's direction.
    turns = np.where(going, TURNS / area[going].sum(), 0.0)
    turns -= np.where(returning, TURNS / area[returning].sum(), 0.0)
    return Problem(
        nodes=nodes,
        triangles=triangles,
        materials=(AIR,),
        current_density=turns * CURRENT,
        fixed_nodes=np.flatnonzero(nodes[:, 0] == 0.0),
        depth=DEPTH,
        windings={"coil": turns},
    )


def main() -> None:
    expected = MU0 * TURNS**2 * DEPTH * (SPACING + 2.0 * SIDE / 3.0) / HEIGHT
    print(f"by formula: {expected * 1e6:.4f} microhenries")
    print("rows of triangles   inductance      difference")
    for cells in (4, 8, 16, 32):
        problem = coil_problem(cells)
        solution = solve(problem)
        inductance = flux_linkage(problem, solution)["coil"] / CURRENT
        print(
            f"{cells:12d}       {inductance * 1e6:9.4f} uH   {100.0 * (inductance / expected - 1.0):+8.4f} %"
        )
    print("\nHalving the triangles cuts the difference to a quarter.")


if __name__ == "__main__":
    main()
