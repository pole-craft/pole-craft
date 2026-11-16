"""Linear assembly: element matrices, the assembled matrix and fixed-value boundaries."""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from polecraft.core.materials import MU0
from polecraft.core.solver import (
    assemble_stiffness,
    element_stiffness,
    flux_density,
    shape_gradients,
    solve_linear,
)

NU0 = 1.0 / MU0


def grid(
    columns: int,
    rows: int,
    width: float = 1.0,
    height: float = 1.0,
    jitter: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A rectangle from (0, 0) to (width, height) cut into triangles.

    Each cell is split on alternating diagonals. ``jitter`` moves the inside
    nodes by up to that fraction of a cell, to make the triangles irregular.
    Below a quarter of a cell no triangle can fold over.
    """
    assert jitter < 0.25
    x, y = np.meshgrid(np.linspace(0.0, width, columns + 1), np.linspace(0.0, height, rows + 1))
    nodes = np.column_stack([x.ravel(), y.ravel()])
    if jitter:
        rng = np.random.default_rng(seed)
        inside = ~on_boundary(nodes, width, height)
        cell = np.array([width / columns, height / rows])
        nodes[inside] += rng.uniform(-jitter, jitter, (int(inside.sum()), 2)) * cell
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


def on_boundary(nodes: np.ndarray, width: float = 1.0, height: float = 1.0) -> np.ndarray:
    x, y = nodes[:, 0], nodes[:, 1]
    return (x == 0.0) | (x == width) | (y == 0.0) | (y == height)


def random_triangles(count: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Separate triangles of varied shape and size, some of them clockwise."""
    rng = np.random.default_rng(seed)
    nodes = rng.uniform(-1.0, 1.0, (3 * count, 2)) * rng.uniform(1e-3, 10.0, (count, 1)).repeat(
        3, axis=0
    )
    return nodes, np.arange(3 * count).reshape(count, 3)


def corner_angles(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Corner angle at each node of each triangle, in radians, shape (m, 3)."""
    p = nodes[triangles]
    angles = np.empty(triangles.shape)
    for i in range(3):
        u = p[:, (i + 1) % 3] - p[:, i]
        v = p[:, (i + 2) % 3] - p[:, i]
        cosine = (u * v).sum(axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        angles[:, i] = np.arccos(cosine)
    return angles


def assemble_one_by_one(nodes: np.ndarray, triangles: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """A plain loop over elements and corners, as a reference for the assembly."""
    matrix = np.zeros((len(nodes), len(nodes)))
    local = element_stiffness(nodes, triangles, nu)
    for e, triangle in enumerate(triangles):
        for i in range(3):
            for j in range(3):
                matrix[triangle[i], triangle[j]] += local[e, i, j]
    return matrix


# Shape function gradients


def test_gradients_of_the_unit_right_triangle() -> None:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    gradients, area = shape_gradients(nodes, np.array([[0, 1, 2]]))
    assert area == pytest.approx([0.5])
    # N0 = 1 - x - y, N1 = x, N2 = y.
    assert gradients[0] == pytest.approx(np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]))


def test_shape_functions_are_one_at_their_node_and_zero_at_the_others() -> None:
    nodes, triangles = random_triangles(50)
    gradients, _ = shape_gradients(nodes, triangles)
    p = nodes[triangles]
    for i in range(3):
        for j in range(3):
            # N_i is linear, so N_i(p_j) - N_i(p_i) = grad N_i . (p_j - p_i).
            change = (gradients[:, i] * (p[:, j] - p[:, i])).sum(axis=1)
            assert change == pytest.approx(0.0 if i == j else -1.0, abs=1e-9)


def test_gradients_sum_to_zero() -> None:
    nodes, triangles = random_triangles(50)
    gradients, _ = shape_gradients(nodes, triangles)
    scale = np.abs(gradients).max(axis=(1, 2))
    assert np.all(np.abs(gradients.sum(axis=1)).max(axis=1) <= 1e-12 * scale)


def test_area_is_positive_whichever_way_the_triangle_is_numbered() -> None:
    nodes = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    _, area = shape_gradients(nodes, np.array([[0, 1, 2], [0, 2, 1]]))
    assert area == pytest.approx([3.0, 3.0])


def test_areas_add_up_to_the_rectangle() -> None:
    nodes, triangles = grid(7, 5, width=0.04, height=0.01, jitter=0.2)
    _, area = shape_gradients(nodes, triangles)
    assert area.sum() == pytest.approx(0.04 * 0.01, rel=1e-12)


# Element matrix


def test_element_matrix_of_the_unit_right_triangle() -> None:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    matrix = element_stiffness(nodes, np.array([[0, 1, 2]]), 1.0)
    expected = 0.5 * np.array([[2.0, -1.0, -1.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    assert matrix[0] == pytest.approx(expected)


def test_element_matrix_of_an_equilateral_triangle() -> None:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
    matrix = element_stiffness(nodes, np.array([[0, 1, 2]]), 6.0)
    # Off-diagonal -nu cot(60 deg) / 2, diagonal twice that with the sign changed.
    off = -6.0 / (2.0 * math.sqrt(3.0))
    expected = np.full((3, 3), off) + np.diag([-3.0 * off] * 3)
    assert matrix[0] == pytest.approx(expected)


def test_element_matrices_are_symmetric() -> None:
    nodes, triangles = random_triangles(200)
    matrices = element_stiffness(nodes, triangles, NU0)
    assert np.array_equal(matrices, matrices.transpose(0, 2, 1))


def test_element_matrix_rows_sum_to_zero() -> None:
    nodes, triangles = random_triangles(200)
    matrices = element_stiffness(nodes, triangles, NU0)
    scale = np.abs(matrices).max(axis=(1, 2))
    assert np.all(np.abs(matrices.sum(axis=2)).max(axis=1) <= 1e-12 * scale)
    assert np.all(np.abs(matrices.sum(axis=1)).max(axis=1) <= 1e-12 * scale)


def test_off_diagonal_terms_follow_the_cotangent_formula() -> None:
    # K_ij = -nu cot(angle at the third corner) / 2: an independent formula.
    nodes, triangles = random_triangles(200)
    matrices = element_stiffness(nodes, triangles, 3.0)
    angles = corner_angles(nodes, triangles)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        expected = -3.0 / (2.0 * np.tan(angles[:, k]))
        scale = np.abs(matrices).max(axis=(1, 2))
        assert np.all(np.abs(matrices[:, i, j] - expected) <= 1e-9 * scale)


def test_element_matrix_does_not_depend_on_the_numbering_direction() -> None:
    nodes, triangles = random_triangles(20)
    forward = element_stiffness(nodes, triangles, 2.0)
    backward = element_stiffness(nodes, triangles[:, [0, 2, 1]], 2.0)
    assert backward == pytest.approx(forward[:, [0, 2, 1]][:, :, [0, 2, 1]], rel=1e-12)


def test_element_matrix_does_not_depend_on_size_position_or_rotation() -> None:
    nodes, triangles = random_triangles(20)
    turn = np.array([[math.cos(0.7), -math.sin(0.7)], [math.sin(0.7), math.cos(0.7)]])
    moved = 250.0 * nodes @ turn.T + np.array([3.0, -8.0])
    original = element_stiffness(nodes, triangles, 2.0)
    assert element_stiffness(moved, triangles, 2.0) == pytest.approx(
        original, rel=1e-9, abs=1e-9 * np.abs(original).max()
    )


def test_element_matrix_scales_with_reluctivity() -> None:
    nodes, triangles = random_triangles(20)
    nu = np.linspace(100.0, NU0, 20)
    unit = element_stiffness(nodes, triangles, 1.0)
    assert element_stiffness(nodes, triangles, nu) == pytest.approx(
        unit * nu[:, None, None], rel=1e-12
    )


def test_element_matrix_is_positive_semidefinite() -> None:
    nodes, triangles = random_triangles(200)
    matrices = element_stiffness(nodes, triangles, 1.0)
    values = np.linalg.eigvalsh(matrices)
    assert np.all(values[:, 0] >= -1e-9 * values[:, 2])
    assert np.all(values[:, 1] > 0.0)


def test_element_energy_of_a_linear_field() -> None:
    # a^T K a = nu |grad A|^2 area for A = p x + q y.
    nodes, triangles = random_triangles(50)
    p, q, nu = 1.5, -0.4, 7.0
    a = (p * nodes[:, 0] + q * nodes[:, 1])[triangles]
    matrices = element_stiffness(nodes, triangles, nu)
    _, area = shape_gradients(nodes, triangles)
    energy = np.einsum("ei,eij,ej->e", a, matrices, a)
    assert energy == pytest.approx(nu * (p * p + q * q) * area, rel=1e-9)


@pytest.mark.parametrize(
    "points",
    [
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],  # in a line
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],  # two corners together
    ],
)
def test_a_triangle_without_area_is_rejected(points: list) -> None:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]] + points)
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    with pytest.raises(ValueError, match="triangle 1 has no area"):
        element_stiffness(nodes, triangles, 1.0)


def test_a_sliver_is_rejected_whatever_its_size() -> None:
    for scale in (1.0, 1e-6):
        nodes = scale * np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1e-13]])
        with pytest.raises(ValueError, match="triangle 0 has no area"):
            element_stiffness(nodes, np.array([[0, 1, 2]]), 1.0)


def test_a_very_small_triangle_is_accepted() -> None:
    nodes = 1e-7 * np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    _, area = shape_gradients(nodes, np.array([[0, 1, 2]]))
    assert area == pytest.approx([5e-15], rel=1e-12)


def test_a_small_triangle_far_from_the_origin_keeps_its_accuracy() -> None:
    corners = 1e-5 * np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    triangle = np.array([[0, 1, 2]])
    near = element_stiffness(corners, triangle, 1.0)
    far = element_stiffness(corners + np.array([1.0, -1.0]), triangle, 1.0)
    # Only the rounding of the shifted positions themselves is left.
    assert far == pytest.approx(near, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("nu", [0.0, -5.0, math.nan, math.inf])
def test_impossible_reluctivity_is_rejected(nu: float) -> None:
    nodes, triangles = grid(2, 2)
    values = np.full(len(triangles), 10.0)
    values[3] = nu
    with pytest.raises(ValueError, match="reluctivity"):
        element_stiffness(nodes, triangles, values)
    with pytest.raises(ValueError, match="reluctivity"):
        element_stiffness(nodes, triangles, nu)


def test_reluctivity_must_be_one_value_or_one_per_triangle() -> None:
    nodes, triangles = grid(2, 2)
    with pytest.raises(ValueError, match="one value per triangle"):
        element_stiffness(nodes, triangles, np.ones(len(triangles) - 1))


# Assembled matrix


@pytest.fixture(scope="module")
def rough() -> tuple[np.ndarray, np.ndarray]:
    """An irregular mesh of the unit square."""
    return grid(9, 7, jitter=0.2, seed=3)


def test_assembled_matrix_is_sparse_and_square(rough: tuple) -> None:
    nodes, triangles = rough
    matrix = assemble_stiffness(nodes, triangles, NU0)
    assert sparse.issparse(matrix)
    assert matrix.format == "csr"
    assert matrix.shape == (len(nodes), len(nodes))
    assert matrix.dtype == np.float64


def test_assembly_matches_a_plain_loop(rough: tuple) -> None:
    nodes, triangles = rough
    nu = np.random.default_rng(5).uniform(100.0, NU0, len(triangles))
    matrix = assemble_stiffness(nodes, triangles, nu)
    expected = assemble_one_by_one(nodes, triangles, nu)
    assert matrix.toarray() == pytest.approx(expected, rel=1e-12, abs=1e-12 * NU0)


def test_assembled_matrix_has_one_entry_per_node_pair(rough: tuple) -> None:
    nodes, triangles = rough
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    pairs = {(int(a), int(b)) for t in triangles for a in t for b in t}
    assert matrix.nnz == len(pairs)
    assert matrix.has_canonical_format


def test_entries_that_are_zero_are_kept() -> None:
    # Across a right angle the coupling is zero (cot 90 deg), but the entry
    # stays, so the pattern is the same for every reluctivity.
    nodes, triangles = grid(3, 3)
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    pairs = {(int(a), int(b)) for t in triangles for a in t for b in t}
    assert np.any(matrix.data == 0.0)
    assert matrix.nnz == len(pairs)


def test_assembled_matrix_is_symmetric(rough: tuple) -> None:
    nodes, triangles = rough
    nu = np.random.default_rng(5).uniform(100.0, NU0, len(triangles))
    matrix = assemble_stiffness(nodes, triangles, nu)
    assert abs(matrix - matrix.T).max() <= 1e-12 * abs(matrix).max()


def test_assembled_rows_sum_to_zero(rough: tuple) -> None:
    nodes, triangles = rough
    nu = np.random.default_rng(5).uniform(100.0, NU0, len(triangles))
    matrix = assemble_stiffness(nodes, triangles, nu)
    assert np.abs(matrix @ np.ones(len(nodes))).max() <= 1e-12 * abs(matrix).max()


def test_a_linear_field_leaves_no_residual_inside(rough: tuple) -> None:
    nodes, triangles = rough
    matrix = assemble_stiffness(nodes, triangles, NU0)
    a = 2.0 * nodes[:, 0] - 3.0 * nodes[:, 1] + 0.5
    residual = matrix @ a
    inside = ~on_boundary(nodes)
    assert np.abs(residual[inside]).max() <= 1e-12 * abs(matrix).max() * np.abs(a).max()
    # On the boundary the residual is the edge current that holds the field up.
    assert np.abs(residual[~inside]).max() > 1e-3 * abs(matrix).max()


def test_assembled_matrix_is_positive_semidefinite(rough: tuple) -> None:
    nodes, triangles = rough
    values = np.linalg.eigvalsh(assemble_stiffness(nodes, triangles, 1.0).toarray())
    assert values[0] == pytest.approx(0.0, abs=1e-10 * values[-1])
    assert values[1] > 1e-6 * values[-1]  # only the constant field costs no energy


def test_a_node_in_no_triangle_gets_an_empty_row() -> None:
    nodes, triangles = grid(2, 2)
    nodes = np.vstack([nodes, [[5.0, 5.0]]])
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    assert matrix.shape == (10, 10)
    assert matrix[9].nnz == 0


@pytest.mark.parametrize("number", [9, -1])
def test_triangle_naming_a_missing_node_is_rejected(number: int) -> None:
    nodes, triangles = grid(2, 2)
    triangles = triangles.copy()
    triangles[2, 1] = number
    with pytest.raises(ValueError, match=f"names node {number},"):
        assemble_stiffness(nodes, triangles, 1.0)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_node_positions_must_be_finite(value: float) -> None:
    nodes, triangles = grid(2, 2)
    nodes[4, 1] = value
    with pytest.raises(ValueError, match="node positions must be finite"):
        assemble_stiffness(nodes, triangles, 1.0)


@pytest.mark.parametrize(
    ("nodes", "triangles", "message"),
    [
        (np.zeros((4, 3)), np.array([[0, 1, 2]]), "nodes"),
        (np.zeros((4, 2)), np.array([[0, 1, 2, 3]]), "triangles"),
        (np.zeros((4, 2)), np.array([[0.0, 1.0, 2.0]]), "whole numbers"),
    ],
)
def test_wrongly_shaped_mesh_arrays_are_rejected(
    nodes: np.ndarray, triangles: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        assemble_stiffness(nodes, triangles, 1.0)


def test_inputs_are_not_changed(rough: tuple) -> None:
    nodes, triangles = rough
    nu = np.full(len(triangles), 3.0)
    before = nodes.copy(), triangles.copy(), nu.copy()
    assemble_stiffness(nodes, triangles, nu)
    assert np.array_equal(nodes, before[0])
    assert np.array_equal(triangles, before[1])
    assert np.array_equal(nu, before[2])


# Flux density


def test_flux_density_is_the_curl_of_the_potential() -> None:
    # B = (dA/dy, -dA/dx).
    nodes, triangles = grid(4, 3, jitter=0.2)
    a = 2.0 * nodes[:, 0] - 3.0 * nodes[:, 1] + 0.5
    b = flux_density(nodes, triangles, a)
    assert b.shape == (len(triangles), 2)
    assert b == pytest.approx(np.tile([-3.0, -2.0], (len(triangles), 1)), rel=1e-9)


def test_flux_density_does_not_depend_on_the_numbering_direction() -> None:
    nodes, triangles = grid(4, 3, jitter=0.2)
    a = np.sin(3.0 * nodes[:, 0]) * np.cos(2.0 * nodes[:, 1])
    forward = flux_density(nodes, triangles, a)
    backward = flux_density(nodes, triangles[:, [0, 2, 1]], a)
    assert backward == pytest.approx(forward, rel=1e-12, abs=1e-12)


def test_flux_between_two_nodes_is_the_potential_difference() -> None:
    # Field lines are lines of constant A: around a wire on the axis they are
    # circles, counter-clockwise for current out of the plane, and A falls
    # outwards.
    nodes = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 0.5]])
    a = np.array([3.0, 1.0, 3.0])  # depends on x only, falling outwards
    b = flux_density(nodes, np.array([[0, 1, 2]]), a)
    assert b[0] == pytest.approx([0.0, 2.0])  # +y at a point on the +x axis


def test_potential_must_have_one_value_per_node() -> None:
    nodes, triangles = grid(2, 2)
    with pytest.raises(ValueError, match="one value per node"):
        flux_density(nodes, triangles, np.zeros(len(nodes) + 1))


# Fixed-value boundaries


def test_patch_test_reproduces_a_linear_field(rough: tuple) -> None:
    nodes, triangles = rough
    exact = 2.0 * nodes[:, 0] - 3.0 * nodes[:, 1] + 0.5
    edge = np.flatnonzero(on_boundary(nodes))
    matrix = assemble_stiffness(nodes, triangles, NU0)
    a = solve_linear(matrix, np.zeros(len(nodes)), edge, exact[edge])
    assert a == pytest.approx(exact, rel=1e-12, abs=1e-12)
    b = flux_density(nodes, triangles, a)
    assert b == pytest.approx(np.tile([-3.0, -2.0], (len(triangles), 1)), rel=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_patch_test_on_other_meshes_and_fields(seed: int) -> None:
    rng = np.random.default_rng(seed)
    width, height = rng.uniform(1e-3, 0.1, 2)
    nodes, triangles = grid(
        int(rng.integers(3, 12)), int(rng.integers(3, 12)), width, height, jitter=0.2, seed=seed
    )
    p, q, r = rng.uniform(-5.0, 5.0, 3)
    exact = p * nodes[:, 0] + q * nodes[:, 1] + r * width
    edge = np.flatnonzero(on_boundary(nodes, width, height))
    matrix = assemble_stiffness(nodes, triangles, rng.uniform(100.0, NU0))
    a = solve_linear(matrix, np.zeros(len(nodes)), edge, exact[edge])
    assert a == pytest.approx(exact, rel=1e-11, abs=1e-11 * np.abs(exact).max())


def test_uniform_field_between_two_fixed_edges() -> None:
    # A = 0 on the bottom edge and A = flux on the top edge, so both are field
    # lines. The sides are left free, and the field crosses them at right
    # angles. The flux per metre of depth between the fixed edges is their
    # potential difference, so B = flux / height everywhere, along +x.
    width, height, flux = 0.03, 0.008, 4e-3
    nodes, triangles = grid(12, 5, width, height, jitter=0.2, seed=2)
    bottom = np.flatnonzero(nodes[:, 1] == 0.0)
    top = np.flatnonzero(nodes[:, 1] == height)
    fixed = np.concatenate([bottom, top])
    values = np.concatenate([np.zeros(len(bottom)), np.full(len(top), flux)])
    matrix = assemble_stiffness(nodes, triangles, NU0)
    a = solve_linear(matrix, np.zeros(len(nodes)), fixed, values)

    assert a == pytest.approx(flux * nodes[:, 1] / height, rel=1e-11, abs=1e-14)
    b = flux_density(nodes, triangles, a)
    # B_x = dA/dy = flux / height = 0.5 T, B_y = 0.
    assert b[:, 0] == pytest.approx(0.5, rel=1e-10)
    assert b[:, 1] == pytest.approx(0.0, abs=1e-10)
    # Stored energy per metre of depth: B^2 / (2 mu0) times the area.
    energy = 0.5 * a @ (matrix @ a)
    assert energy == pytest.approx(0.5**2 / (2.0 * MU0) * width * height, rel=1e-10)


def test_uniform_field_does_not_depend_on_reluctivity_across_the_flux() -> None:
    # Two materials side by side, the boundary between them crossed by the
    # flux: B is the same in both (its normal part is continuous).
    width, height = 2.0, 1.0
    nodes, triangles = grid(8, 4, width, height)
    centre_x = nodes[triangles].mean(axis=1)[:, 0]
    nu = np.where(centre_x < 1.0, NU0, NU0 / 2000.0)
    fixed = np.flatnonzero((nodes[:, 1] == 0.0) | (nodes[:, 1] == height))
    values = np.where(nodes[fixed, 1] == 0.0, 0.0, 1.2)
    a = solve_linear(
        assemble_stiffness(nodes, triangles, nu), np.zeros(len(nodes)), fixed, values
    )
    b = flux_density(nodes, triangles, a)
    assert b[:, 0] == pytest.approx(1.2, rel=1e-9)
    assert b[:, 1] == pytest.approx(0.0, abs=1e-9)


def test_flux_divides_between_two_layers_by_their_permeability() -> None:
    # Two layers stacked between the fixed edges, flux running along both:
    # H is the same in both (its part along the boundary is continuous), so
    # B is in proportion to permeability, and the two fluxes add up to the
    # potential difference.
    width, height, flux = 3.0, 1.0, 0.9
    thick_1, mu_1, mu_2 = 0.25, 500.0, 1.0  # layer 1 is the bottom quarter
    nodes, triangles = grid(6, 8, width, height)
    centre_y = nodes[triangles].mean(axis=1)[:, 1]
    lower = centre_y < thick_1
    nu = np.where(lower, NU0 / mu_1, NU0 / mu_2)
    fixed = np.flatnonzero((nodes[:, 1] == 0.0) | (nodes[:, 1] == height))
    values = np.where(nodes[fixed, 1] == 0.0, 0.0, flux)
    a = solve_linear(
        assemble_stiffness(nodes, triangles, nu), np.zeros(len(nodes)), fixed, values
    )
    b = flux_density(nodes, triangles, a)

    h = flux / (mu_1 * MU0 * thick_1 + mu_2 * MU0 * (height - thick_1))
    assert b[lower, 0] == pytest.approx(mu_1 * MU0 * h, rel=1e-9)
    assert b[~lower, 0] == pytest.approx(mu_2 * MU0 * h, rel=1e-9)
    assert b[:, 1] == pytest.approx(0.0, abs=1e-9 * mu_1 * MU0 * h)


def test_fixed_nodes_keep_their_values_exactly(rough: tuple) -> None:
    nodes, triangles = rough
    edge = np.flatnonzero(on_boundary(nodes))
    values = np.random.default_rng(8).uniform(-1.0, 1.0, len(edge))
    a = solve_linear(assemble_stiffness(nodes, triangles, NU0), np.zeros(len(nodes)), edge, values)
    assert np.array_equal(a[edge], values)


def test_free_nodes_balance_the_load(rough: tuple) -> None:
    nodes, triangles = rough
    rng = np.random.default_rng(9)
    load = rng.uniform(-1.0, 1.0, len(nodes))
    edge = np.flatnonzero(on_boundary(nodes))
    matrix = assemble_stiffness(nodes, triangles, 50.0)
    a = solve_linear(matrix, load, edge, rng.uniform(-1.0, 1.0, len(edge)))
    free = ~on_boundary(nodes)
    assert (matrix @ a)[free] == pytest.approx(load[free], rel=1e-9, abs=1e-9)


def test_solution_is_linear_in_load_and_fixed_values(rough: tuple) -> None:
    nodes, triangles = rough
    rng = np.random.default_rng(10)
    load = rng.uniform(-1.0, 1.0, len(nodes))
    edge = np.flatnonzero(on_boundary(nodes))
    values = rng.uniform(-1.0, 1.0, len(edge))
    matrix = assemble_stiffness(nodes, triangles, 50.0)
    from_load = solve_linear(matrix, load, edge, 0.0)
    from_edge = solve_linear(matrix, np.zeros(len(nodes)), edge, values)
    both = solve_linear(matrix, 3.0 * load, edge, -2.0 * values)
    assert both == pytest.approx(3.0 * from_load - 2.0 * from_edge, rel=1e-9, abs=1e-12)


def test_one_fixed_value_can_be_given_for_all_fixed_nodes(rough: tuple) -> None:
    nodes, triangles = rough
    edge = np.flatnonzero(on_boundary(nodes))
    matrix = assemble_stiffness(nodes, triangles, 50.0)
    a = solve_linear(matrix, np.zeros(len(nodes)), edge, 0.25)
    assert a == pytest.approx(0.25, rel=1e-12)


def test_fixed_nodes_can_be_given_in_any_order(rough: tuple) -> None:
    nodes, triangles = rough
    edge = np.flatnonzero(on_boundary(nodes))
    values = nodes[edge, 0] ** 2
    order = np.random.default_rng(11).permutation(len(edge))
    matrix = assemble_stiffness(nodes, triangles, 50.0)
    load = np.zeros(len(nodes))
    assert np.array_equal(
        solve_linear(matrix, load, edge[order], values[order]),
        solve_linear(matrix, load, edge, values),
    )


def test_every_node_may_be_fixed(rough: tuple) -> None:
    nodes, triangles = rough
    values = np.arange(len(nodes), dtype=float)
    matrix = assemble_stiffness(nodes, triangles, 50.0)
    a = solve_linear(matrix, np.zeros(len(nodes)), np.arange(len(nodes)), values)
    assert np.array_equal(a, values)


def test_matrix_and_load_are_not_changed(rough: tuple) -> None:
    nodes, triangles = rough
    matrix = assemble_stiffness(nodes, triangles, 50.0)
    load = np.random.default_rng(12).uniform(-1.0, 1.0, len(nodes))
    before = matrix.copy(), load.copy()
    solve_linear(matrix, load, np.flatnonzero(on_boundary(nodes)), 1.0)
    assert (matrix != before[0]).nnz == 0
    assert np.array_equal(load, before[1])


def test_a_problem_with_no_fixed_node_is_rejected(rough: tuple) -> None:
    nodes, triangles = rough
    matrix = assemble_stiffness(nodes, triangles, 50.0)
    with pytest.raises(ValueError, match="at least one node must be fixed"):
        solve_linear(matrix, np.zeros(len(nodes)), np.array([], dtype=int), np.array([]))


def two_squares() -> tuple[np.ndarray, np.ndarray]:
    """Two separate meshes in one: nodes 0 to 15 and 16 to 31."""
    nodes, triangles = grid(3, 3)
    return np.vstack([nodes, nodes + [5.0, 0.0]]), np.vstack([triangles, triangles + len(nodes)])


def test_a_part_of_the_mesh_with_no_fixed_node_is_rejected() -> None:
    nodes, triangles = two_squares()
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    with pytest.raises(ValueError, match="node 16 .* no fixed node"):
        solve_linear(matrix, np.zeros(len(nodes)), np.array([0]), 0.0)


def test_separate_parts_each_with_a_fixed_node_are_solved() -> None:
    nodes, triangles = two_squares()
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    a = solve_linear(matrix, np.zeros(len(nodes)), np.array([0, 20]), np.array([1.0, -2.0]))
    assert a[:16] == pytest.approx(1.0)
    assert a[16:] == pytest.approx(-2.0)


def test_a_node_in_no_triangle_may_be_fixed() -> None:
    nodes, triangles = grid(2, 2)
    nodes = np.vstack([nodes, [[5.0, 5.0]]])
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    a = solve_linear(matrix, np.zeros(10), np.array([0, 9]), np.array([1.0, 7.0]))
    assert a[9] == 7.0
    assert a[:9] == pytest.approx(1.0)


def test_a_free_node_in_no_triangle_is_rejected() -> None:
    nodes, triangles = grid(2, 2)
    nodes = np.vstack([nodes, [[5.0, 5.0]]])
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    with pytest.raises(ValueError, match="node 9 .* no triangle"):
        solve_linear(matrix, np.zeros(10), np.array([0]), 0.0)


@pytest.mark.parametrize(
    ("fixed", "values", "message"),
    [
        ([0, 1, 99], [0.0, 0.0, 0.0], "node 99"),
        ([0, 16], [0.0, 0.0], "node 16"),
        ([0, -1], [0.0, 0.0], "node -1"),
        ([0, 1, 2], [0.0, 0.0], "one value, or one per fixed node"),
        ([0, 1, 1], [0.0, 1.0, 2.0], "node 1 is fixed twice"),
        ([0, 1], [0.0, math.nan], "finite"),
        ([0.5, 1.0], [0.0, 0.0], "whole numbers"),
    ],
)
def test_bad_fixed_nodes_are_rejected(fixed: list, values: list, message: str) -> None:
    nodes, triangles = grid(3, 3)
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    with pytest.raises(ValueError, match=message):
        solve_linear(matrix, np.zeros(len(nodes)), np.array(fixed), np.array(values))


def test_a_node_fixed_twice_to_the_same_value_is_accepted() -> None:
    nodes, triangles = grid(3, 3)
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    edge = np.flatnonzero(on_boundary(nodes))
    twice = np.concatenate([edge, edge[:3]])
    a = solve_linear(matrix, np.zeros(len(nodes)), twice, 2.0)
    assert a == pytest.approx(2.0)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_load_must_be_finite(value: float) -> None:
    nodes, triangles = grid(3, 3)
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    load = np.zeros(len(nodes))
    load[5] = value
    with pytest.raises(ValueError, match="load must be a finite number"):
        solve_linear(matrix, load, np.array([0]), 0.0)


def test_stiffness_matrix_must_be_square() -> None:
    with pytest.raises(ValueError, match="must be square"):
        solve_linear(sparse.csr_matrix((3, 4)), np.zeros(3), np.array([0]), 0.0)


def test_load_must_have_one_value_per_node() -> None:
    nodes, triangles = grid(3, 3)
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    with pytest.raises(ValueError, match="one value per node"):
        solve_linear(matrix, np.zeros(len(nodes) - 1), np.array([0]), 0.0)


# The reference machine


@pytest.fixture(scope="module")
def machine_mesh():
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    design = load_design(root / "designs" / "qdd_outer_36s42p.yaml")
    return build_mesh(machine_for(design).build_geometry())


def test_machine_mesh_assembles(machine_mesh) -> None:
    mesh = machine_mesh
    steel = np.array([name.startswith(("tooth", "stator_yoke", "rotor_yoke")) for name in mesh.region_names])
    nu = np.where(steel[mesh.triangle_region], NU0 / 4000.0, NU0)
    matrix = assemble_stiffness(mesh.nodes, mesh.triangles, nu)
    count = len(mesh.nodes)
    assert matrix.shape == (count, count)
    assert abs(matrix - matrix.T).max() <= 1e-12 * abs(matrix).max()
    assert np.abs(matrix @ np.ones(count)).max() <= 1e-10 * abs(matrix).max()


def test_machine_mesh_passes_the_patch_test(machine_mesh) -> None:
    # Fix every node on an edge that only one triangle uses; the rest must
    # take up the linear field. This runs across the air-gap band as well.
    mesh = machine_mesh
    t = mesh.triangles
    edges = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    edge = np.unique(unique[counts == 1])
    assert 0 < len(edge) < len(mesh.nodes) / 2
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    exact = 30.0 * x - 20.0 * y + 0.1
    matrix = assemble_stiffness(mesh.nodes, mesh.triangles, NU0)
    a = solve_linear(matrix, np.zeros(len(x)), edge, exact[edge])
    assert a == pytest.approx(exact, rel=1e-9, abs=1e-9 * np.abs(exact).max())
