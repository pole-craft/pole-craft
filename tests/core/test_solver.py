"""Linear assembly: element matrices, the assembled matrix and fixed-value boundaries."""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from polecraft.core.materials import MU0
from polecraft.core.solver import (
    assemble_stiffness,
    current_load,
    element_stiffness,
    flux_density,
    magnet_load,
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


# Current sources


def centres(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    return nodes[triangles].mean(axis=1)


def disc(sectors: int, wire_radius: float = 1e-3, reach: float = 20.0) -> tuple:
    """A disc of rings about the origin, with a ring exactly on the wire surface.

    Ring radii grow in proportion to the radius, which keeps the triangles
    well shaped from the middle of the wire out to about ``reach`` wire
    radii. Returns nodes, triangles and the outer radius.
    """
    growth = 1.0 + 2.0 * math.pi / sectors
    inner = round(math.log(4.0) / math.log(growth))
    outer = round(math.log(reach) / math.log(growth))
    radii = wire_radius * growth ** np.arange(-inner, outer + 1)
    angles = 2.0 * math.pi * np.arange(sectors) / sectors
    rings = [np.column_stack([r * np.cos(angles), r * np.sin(angles)]) for r in radii]
    nodes = np.vstack([[[0.0, 0.0]]] + rings)
    triangles = []
    for i in range(sectors):
        j = (i + 1) % sectors
        triangles.append((0, 1 + i, 1 + j))
        for k in range(len(radii) - 1):
            a, b = 1 + k * sectors + i, 1 + k * sectors + j
            c, d = b + sectors, a + sectors
            if (i + k) % 2 == 0:
                triangles += [(a, d, b), (b, d, c)]
            else:
                triangles += [(a, d, c), (a, c, b)]
    return nodes, np.array(triangles), radii[-1]


def wire_solution(sectors: int, current: float = 10.0, wire_radius: float = 1e-3) -> dict:
    """Field of a round wire on the axis of a disc whose rim is held at A = 0."""
    nodes, triangles, rim = disc(sectors, wire_radius)
    centre = centres(nodes, triangles)
    radius = np.hypot(centre[:, 0], centre[:, 1])
    inside = radius < wire_radius
    _, area = shape_gradients(nodes, triangles)
    # The meshed wire is a polygon; spread the current over its true area.
    density = np.where(inside, current / area[inside].sum(), 0.0)
    load = current_load(nodes, triangles, density)
    node_radius = np.hypot(nodes[:, 0], nodes[:, 1])
    fixed = np.flatnonzero(np.isclose(node_radius, rim))
    matrix = assemble_stiffness(nodes, triangles, NU0)
    a = solve_linear(matrix, load, fixed, 0.0)
    b = flux_density(nodes, triangles, a)
    around = np.column_stack([-centre[:, 1], centre[:, 0]]) / radius[:, None]
    outward = centre / radius[:, None]
    return {
        "nodes": nodes, "node_radius": node_radius, "radius": radius, "inside": inside,
        "a": a, "load": load, "rim": rim, "area": area,
        "b_around": (b * around).sum(axis=1), "b_outward": (b * outward).sum(axis=1),
    }


def test_current_load_of_one_triangle_is_shared_equally() -> None:
    nodes = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    load = current_load(nodes, np.array([[0, 1, 2]]), 5.0)
    assert load == pytest.approx([5.0, 5.0, 5.0])  # 5 A/m2 on 3 m2, a third each


def test_current_load_adds_up_to_the_total_current(rough: tuple) -> None:
    nodes, triangles = rough
    density = np.random.default_rng(20).uniform(-4e6, 4e6, len(triangles))
    _, area = shape_gradients(nodes, triangles)
    load = current_load(nodes, triangles, density)
    assert load.shape == (len(nodes),)
    assert load.sum() == pytest.approx((density * area).sum(), rel=1e-12)


def test_current_load_is_zero_away_from_the_conductor() -> None:
    nodes, triangles = grid(6, 6)
    in_coil = centres(nodes, triangles)[:, 0] < 1.0 / 3.0
    load = current_load(nodes, triangles, np.where(in_coil, 2.0, 0.0))
    assert np.all(load[nodes[:, 0] > 0.34] == 0.0)
    assert np.all(load[nodes[:, 0] < 0.34] > 0.0)
    assert load.sum() == pytest.approx(2.0 / 3.0)


def test_loads_have_one_value_per_node_even_for_unused_nodes() -> None:
    nodes, triangles = grid(2, 2)
    nodes = np.vstack([nodes, [[5.0, 5.0]]])
    for load in (
        current_load(nodes, triangles, 3.0),
        magnet_load(nodes, triangles, [0.0, 1.0], NU0),
    ):
        assert load.shape == (10,)
        assert load[9] == 0.0


def test_loads_of_a_mesh_without_triangles_are_zero() -> None:
    nodes = np.zeros((4, 2))
    none = np.zeros((0, 3), dtype=int)
    for load in (current_load(nodes, none, 3.0), magnet_load(nodes, none, [0.0, 1.0], NU0)):
        assert load.dtype == np.float64
        assert np.array_equal(load, np.zeros(4))


def test_current_load_does_not_depend_on_the_numbering_direction(rough: tuple) -> None:
    nodes, triangles = rough
    density = np.linspace(-1.0, 2.0, len(triangles))
    assert current_load(nodes, triangles[:, [0, 2, 1]], density) == pytest.approx(
        current_load(nodes, triangles, density), rel=1e-12
    )


@pytest.mark.parametrize(
    ("density", "message"),
    [
        (np.ones(5), "one value per triangle"),
        (np.ones((8, 1)), "one value per triangle"),
        (math.nan, "finite"),
        (np.array([1.0] * 7 + [math.inf]), "finite"),
    ],
)
def test_bad_current_density_is_rejected(density: object, message: str) -> None:
    nodes, triangles = grid(2, 2)
    with pytest.raises(ValueError, match=f"current density.*{message}"):
        current_load(nodes, triangles, density)


def test_uniform_current_between_two_field_lines() -> None:
    # A slab carrying J between walls held at A = 0: A = mu0 J x (w - x) / 2,
    # and B_y = -dA/dx = mu0 J (x - w/2), zero in the middle.
    width, height, density = 0.02, 0.005, 3e6
    nodes, triangles = grid(40, 10, width, height)
    walls = np.flatnonzero((nodes[:, 0] == 0.0) | (nodes[:, 0] == width))
    matrix = assemble_stiffness(nodes, triangles, NU0)
    a = solve_linear(matrix, current_load(nodes, triangles, density), walls, 0.0)
    x = nodes[:, 0]
    peak = MU0 * density * width**2 / 8.0
    assert a == pytest.approx(MU0 * density * x * (width - x) / 2.0, abs=2e-3 * peak)
    b = flux_density(nodes, triangles, a)
    centre_x = centres(nodes, triangles)[:, 0]
    edge_field = MU0 * density * width / 2.0
    assert b[:, 1] == pytest.approx(MU0 * density * (centre_x - width / 2.0), abs=0.03 * edge_field)
    assert b[:, 0] == pytest.approx(0.0, abs=0.03 * edge_field)


@pytest.fixture(scope="module")
def wire() -> dict:
    return wire_solution(sectors=64)


def test_field_outside_a_round_conductor(wire: dict) -> None:
    # B = mu0 I / (2 pi r), running counter-clockwise for current out of the plane.
    # B is constant over a triangle, so it is compared at the triangle centre
    # and is only first-order accurate: about 1.7 % on this mesh.
    outside = (wire["radius"] > 1.1e-3) & (wire["radius"] < 0.9 * wire["rim"])
    expected = MU0 * 10.0 / (2.0 * math.pi * wire["radius"][outside])
    assert wire["b_around"][outside] == pytest.approx(expected, rel=0.025)
    assert np.all(wire["b_around"][outside] > 0.0)
    assert np.abs(wire["b_outward"][outside] / expected).max() < 0.025


def test_field_inside_a_round_conductor(wire: dict) -> None:
    # B = mu0 I r / (2 pi a^2): it rises in proportion to the radius.
    inside = wire["inside"]
    surface = MU0 * 10.0 / (2.0 * math.pi * 1e-3)
    expected = surface * wire["radius"][inside] / 1e-3
    assert wire["b_around"][inside] == pytest.approx(expected, abs=0.02 * surface)
    assert np.abs(wire["b_outward"][inside]).max() < 0.02 * surface


def test_potential_around_a_round_conductor(wire: dict) -> None:
    # A = mu0 I / (2 pi) ln(R / r) outside the wire, zero on the rim R.
    r = wire["node_radius"]
    outside = r > 1.05e-3
    expected = MU0 * 10.0 / (2.0 * math.pi) * np.log(wire["rim"] / r[outside])
    assert wire["a"][outside] == pytest.approx(expected, rel=0.01, abs=1e-3 * expected.max())


def test_energy_of_a_round_conductor(wire: dict) -> None:
    # Per metre: mu0 I^2 / (4 pi) (1/4 + ln(R / a)), inside plus outside the wire.
    energy = 0.5 * wire["a"] @ wire["load"]
    expected = MU0 * 10.0**2 / (4.0 * math.pi) * (0.25 + math.log(wire["rim"] / 1e-3))
    assert energy == pytest.approx(expected, rel=0.005)


def test_round_conductor_error_falls_with_a_finer_mesh() -> None:
    errors, field_errors = [], []
    for sectors in (24, 48, 96):
        s = wire_solution(sectors)
        outside = s["node_radius"] > 1.05e-3
        exact = MU0 * 10.0 / (2.0 * math.pi) * np.log(s["rim"] / s["node_radius"][outside])
        errors.append(np.abs(s["a"][outside] - exact).max() / exact.max())
        ring = (s["radius"] > 1.1e-3) & (s["radius"] < 0.9 * s["rim"])
        field = MU0 * 10.0 / (2.0 * math.pi * s["radius"][ring])
        field_errors.append(np.abs(s["b_around"][ring] / field - 1.0).max())
    # The potential is second order: half the triangle size, a quarter the error.
    assert errors[0] / errors[1] > 3.0
    assert errors[1] / errors[2] > 3.0
    assert errors[2] < 1e-3
    # The flux density is first order: half the triangle size, half the error.
    assert 1.8 < field_errors[0] / field_errors[1] < 2.2
    assert 1.8 < field_errors[1] / field_errors[2] < 2.2
    assert field_errors[2] < 0.012


def test_reversing_the_current_reverses_the_field(rough: tuple) -> None:
    nodes, triangles = rough
    density = np.where(centres(nodes, triangles)[:, 0] < 0.4, 2e6, 0.0)
    edge = np.flatnonzero(on_boundary(nodes))
    matrix = assemble_stiffness(nodes, triangles, NU0)
    forward = solve_linear(matrix, current_load(nodes, triangles, density), edge, 0.0)
    backward = solve_linear(matrix, current_load(nodes, triangles, -density), edge, 0.0)
    assert np.abs(forward).max() > 0.0
    assert np.array_equal(backward, -forward)
    assert np.array_equal(
        flux_density(nodes, triangles, backward), -flux_density(nodes, triangles, forward)
    )


# Magnet sources


def test_uniform_magnet_loads_only_its_surface() -> None:
    # A uniformly magnetised block acts like a current sheet on its sides.
    nodes, triangles = grid(8, 8, jitter=0.2, seed=4)
    load = magnet_load(nodes, triangles, np.array([0.3, -1.1]), NU0 / 1.05)
    inside = ~on_boundary(nodes)
    assert np.abs(load[inside]).max() <= 1e-12 * np.abs(load).max()
    assert np.abs(load[~inside]).max() > 0.0
    assert load.sum() == pytest.approx(0.0, abs=1e-12 * np.abs(load).max())


def test_magnet_load_is_the_equivalent_surface_current() -> None:
    # Magnetised along +y with M = Br / (mu_r mu0): the sheet current M x n
    # is M amperes per metre out of the plane on the left face (n = -x), the
    # same into the plane on the right face, and nothing on the other two.
    remanence, mu_r, step = 1.2, 1.05, 0.25
    nodes, triangles = grid(4, 4)
    load = magnet_load(nodes, triangles, np.array([0.0, remanence]), NU0 / mu_r)
    m = remanence / (mu_r * MU0)
    x, y = nodes[:, 0], nodes[:, 1]
    corner = ((y == 0.0) | (y == 1.0))
    assert load[(x == 0.0) & ~corner] == pytest.approx(m * step)
    assert load[(x == 0.0) & corner] == pytest.approx(m * step / 2.0)
    assert load[(x == 1.0) & ~corner] == pytest.approx(-m * step)
    assert load[(x == 1.0) & corner] == pytest.approx(-m * step / 2.0)
    assert load[(x > 0.0) & (x < 1.0)] == pytest.approx(0.0, abs=1e-9 * m)


def test_magnet_load_is_zero_outside_the_magnet() -> None:
    nodes, triangles = grid(6, 6)
    in_magnet = centres(nodes, triangles)[:, 0] < 1.0 / 3.0
    remanence = np.where(in_magnet[:, None], [0.0, 1.0], 0.0)
    load = magnet_load(nodes, triangles, remanence, NU0)
    assert np.all(load[nodes[:, 0] > 0.34] == 0.0)


@pytest.mark.parametrize("angle_deg", [0.0, 90.0, 37.0, 200.0])
def test_magnet_closed_by_ideal_iron_carries_its_remanence(angle_deg: float) -> None:
    # Free outside edges behave as ideal iron. A magnet filling the space
    # inside it is short-circuited: H = 0, so B = Br, whatever its direction.
    angle = math.radians(angle_deg)
    remanence = 1.3 * np.array([math.cos(angle), math.sin(angle)])
    nodes, triangles = grid(7, 5, width=0.03, height=0.01, jitter=0.2, seed=6)
    nu = NU0 / 1.05
    matrix = assemble_stiffness(nodes, triangles, nu)
    load = magnet_load(nodes, triangles, remanence, nu)
    a = solve_linear(matrix, load, np.array([0]), 0.0)
    b = flux_density(nodes, triangles, a)
    assert b == pytest.approx(np.tile(remanence, (len(triangles), 1)), rel=1e-9, abs=1e-9)


def test_magnet_closed_in_by_field_lines_carries_no_flux() -> None:
    # With A = 0 all round, no flux can leave or circulate: B = 0, H = -Br / mu.
    nodes, triangles = grid(7, 5, jitter=0.2, seed=6)
    nu = NU0 / 1.05
    load = magnet_load(nodes, triangles, np.array([0.4, 1.1]), nu)
    edge = np.flatnonzero(on_boundary(nodes))
    a = solve_linear(assemble_stiffness(nodes, triangles, nu), load, edge, 0.0)
    assert np.abs(flux_density(nodes, triangles, a)).max() < 1e-9


def magnet_circuit(yoke: float, iron_mu_r: float = 5000.0) -> dict:
    """Two magnets side by side, one magnetised up and one down, under an air gap.

    The flux goes up through the left magnet and its gap, across the top,
    down through the right gap and magnet, and back along the bottom. With
    ``yoke`` = 0 the top and bottom edges are left free, which makes them
    ideal iron; otherwise iron plates of that thickness are meshed and the
    whole outside is a field line.
    """
    half, magnet, gap, cell = 0.05, 0.004, 0.001, 0.001
    remanence, mu_r = 1.2, 1.05
    height = 2.0 * yoke + magnet + gap
    nodes, triangles = grid(
        round(2.0 * half / (2.0 * cell)), round(height / cell), 2.0 * half, height
    )
    centre = centres(nodes, triangles)
    y = centre[:, 1] - yoke
    in_magnet = (y > 0.0) & (y < magnet)
    in_gap = (y > magnet) & (y < magnet + gap)
    left = centre[:, 0] < half
    nu = np.full(len(triangles), NU0 / iron_mu_r)
    nu[in_magnet] = NU0 / mu_r
    nu[in_gap] = NU0
    br = np.zeros((len(triangles), 2))
    br[in_magnet & left, 1] = remanence
    br[in_magnet & ~left, 1] = -remanence
    if yoke:
        fixed = np.flatnonzero(on_boundary(nodes, 2.0 * half, height))
    else:
        fixed = np.flatnonzero((nodes[:, 0] == 0.0) | (nodes[:, 0] == 2.0 * half))
    matrix = assemble_stiffness(nodes, triangles, nu)
    a = solve_linear(matrix, magnet_load(nodes, triangles, br, nu), fixed, 0.0)
    b = flux_density(nodes, triangles, a)
    # Away from the middle, where the two magnets leak into each other.
    far = np.abs(centre[:, 0] - half) > 0.5 * half
    return {
        "left_gap": b[in_gap & left & far], "right_gap": b[in_gap & ~left & far],
        "left_magnet": b[in_magnet & left & far],
        "ideal": remanence / (1.0 + mu_r * gap / magnet),
    }


def test_magnet_with_an_air_gap_in_ideal_iron() -> None:
    # Magnetic circuit: H_m l_m + H_g g = 0 and the same B in magnet and gap
    # give B = Br / (1 + mu_r g / l_m) = 1.2 / 1.2625 = 0.9505 T.
    circuit = magnet_circuit(yoke=0.0)
    assert circuit["ideal"] == pytest.approx(0.950495, rel=1e-6)
    assert circuit["left_gap"][:, 1] == pytest.approx(circuit["ideal"], rel=1e-6)
    assert circuit["left_magnet"][:, 1] == pytest.approx(circuit["ideal"], rel=1e-6)
    assert circuit["right_gap"][:, 1] == pytest.approx(-circuit["ideal"], rel=1e-6)
    assert np.abs(circuit["left_gap"][:, 0]).max() < 1e-6


def test_magnet_with_an_air_gap_in_real_iron() -> None:
    # Iron plates of finite permeability add their reluctance to the circuit.
    # Each plate carries the flux of one 50 mm magnet through 20 mm of iron
    # over about 50 mm: 0.05 / (mu_r 0.02) per plate, against
    # (0.001 + 0.004 / 1.05) / 0.05 for each gap and magnet. The gap field
    # falls by that ratio: 0.52 % for mu_r = 5000 and 5.2 % for mu_r = 500.
    # The estimate is rough (the flux enters the plate all along its length),
    # so it is held to a fifth of the drop.
    for iron_mu_r in (5000.0, 500.0):
        circuit = magnet_circuit(yoke=0.02, iron_mu_r=iron_mu_r)
        gap = circuit["left_gap"][:, 1]
        ratio = (0.05 / (iron_mu_r * 0.02)) / ((0.001 + 0.004 / 1.05) / 0.05)
        drop = 1.0 - gap.mean() / circuit["ideal"]
        assert np.all(gap < circuit["ideal"])
        assert drop == pytest.approx(ratio / (1.0 + ratio), rel=0.2)


def test_reversing_the_magnetisation_reverses_the_field(rough: tuple) -> None:
    nodes, triangles = rough
    in_magnet = centres(nodes, triangles)[:, 1] < 0.5
    remanence = np.where(in_magnet[:, None], [0.2, 1.1], 0.0)
    nu = np.where(in_magnet, NU0 / 1.05, NU0)
    edge = np.flatnonzero(on_boundary(nodes))
    matrix = assemble_stiffness(nodes, triangles, nu)
    forward = solve_linear(matrix, magnet_load(nodes, triangles, remanence, nu), edge, 0.0)
    backward = solve_linear(matrix, magnet_load(nodes, triangles, -remanence, nu), edge, 0.0)
    assert np.abs(forward).max() > 0.0
    assert np.array_equal(backward, -forward)


def test_current_and_magnet_fields_add_up(rough: tuple) -> None:
    nodes, triangles = rough
    centre = centres(nodes, triangles)
    in_magnet = centre[:, 1] < 0.4
    remanence = np.where(in_magnet[:, None], [0.0, 1.1], 0.0)
    density = np.where(centre[:, 1] > 0.7, 3e6, 0.0)
    nu = np.where(in_magnet, NU0 / 1.05, NU0)
    edge = np.flatnonzero(on_boundary(nodes))
    matrix = assemble_stiffness(nodes, triangles, nu)
    from_current = current_load(nodes, triangles, density)
    from_magnet = magnet_load(nodes, triangles, remanence, nu)
    both = solve_linear(matrix, from_current + from_magnet, edge, 0.0)
    apart = solve_linear(matrix, from_current, edge, 0.0) + solve_linear(
        matrix, from_magnet, edge, 0.0
    )
    assert both == pytest.approx(apart, rel=1e-9, abs=1e-12 * np.abs(both).max())


def test_one_remanence_can_be_given_for_all_triangles() -> None:
    nodes, triangles = grid(3, 3, jitter=0.2)
    one = magnet_load(nodes, triangles, [0.5, -0.2], 4.0)
    each = magnet_load(nodes, triangles, np.tile([0.5, -0.2], (len(triangles), 1)), 4.0)
    assert np.array_equal(one, each)


def test_magnet_load_does_not_depend_on_the_numbering_direction(rough: tuple) -> None:
    nodes, triangles = rough
    remanence = np.random.default_rng(21).uniform(-1.0, 1.0, (len(triangles), 2))
    forward = magnet_load(nodes, triangles, remanence, NU0)
    backward = magnet_load(nodes, triangles[:, [0, 2, 1]], remanence, NU0)
    assert backward == pytest.approx(forward, rel=1e-12, abs=1e-12 * np.abs(forward).max())


@pytest.mark.parametrize(
    ("remanence", "message"),
    [
        (np.ones(8), "x and y"),
        (np.ones((5, 2)), "x and y"),
        (np.ones((8, 3)), "x and y"),
        ([1.0, math.nan], "finite"),
    ],
)
def test_bad_remanence_is_rejected(remanence: object, message: str) -> None:
    nodes, triangles = grid(2, 2)
    with pytest.raises(ValueError, match=f"remanence.*{message}"):
        magnet_load(nodes, triangles, remanence, NU0)


def test_magnet_load_checks_the_reluctivity() -> None:
    nodes, triangles = grid(2, 2)
    with pytest.raises(ValueError, match="reluctivity"):
        magnet_load(nodes, triangles, [0.0, 1.0], -1.0)


def test_sources_do_not_change_their_inputs(rough: tuple) -> None:
    nodes, triangles = rough
    density = np.full(len(triangles), 2.0)
    remanence = np.full((len(triangles), 2), 0.5)
    before = nodes.copy(), triangles.copy(), density.copy(), remanence.copy()
    current_load(nodes, triangles, density)
    magnet_load(nodes, triangles, remanence, NU0)
    for now, then in zip((nodes, triangles, density, remanence), before):
        assert np.array_equal(now, then)


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
