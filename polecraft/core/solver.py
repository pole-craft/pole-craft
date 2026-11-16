"""Two-dimensional magnetic field solve on first-order triangles.

The unknown is the out-of-plane vector potential A at every node, in Wb/m
(the same as T m). The flux density lies in the plane, B = (dA/dy, -dA/dx),
so field lines are lines of constant A, and the flux crossing between two
points, per metre of depth, is the difference of their potentials.

The field equation is div(nu grad A) = -J, with nu the reluctivity. Its
discrete form is K a = f: K is the stiffness matrix built here, in m/H, and
f the current fed in at each node, in amperes. Everything is per metre of
depth.

Boundaries: an outside edge whose nodes are fixed to one value is a field
line, so no flux crosses it. An outside edge whose nodes are left free gets
the natural condition instead: the field crosses it at right angles, as at
the surface of ideal iron.

This module takes plain arrays (node positions, triangles, one reluctivity
per triangle), so it does not depend on the mesher.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

# A triangle whose doubled area is below this fraction of its longest edge
# squared is treated as having no area.
NO_AREA = 1e-12


def _mesh_arrays(nodes: ArrayLike, triangles: ArrayLike) -> tuple[NDArray, NDArray]:
    """The mesh as arrays, checked for shape and node numbers."""
    nodes = np.asarray(nodes, dtype=float)
    triangles = np.asarray(triangles)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes must be an (n, 2) array of positions, not {nodes.shape}")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("node positions must be finite numbers")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("triangles must hold node numbers as whole numbers")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"triangles must be an (m, 3) array of node numbers, not {triangles.shape}")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(nodes)):
        bad = triangles[(triangles < 0) | (triangles >= len(nodes))][0]
        raise ValueError(f"a triangle names node {bad}, but the mesh has nodes 0 to {len(nodes) - 1}")
    return nodes, triangles


def shape_gradients(
    nodes: ArrayLike, triangles: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gradients of the three shape functions of every triangle, and its area.

    On a first-order triangle the potential is A = sum of N_i a_i, where N_i
    is 1 at corner i and falls linearly to 0 at the other two. Its gradient
    is constant over the triangle.

    Returns the gradients, shape (m, 3, 2) in 1/m, and the areas, shape (m,)
    in square metres, positive whichever way the triangle is numbered.
    """
    nodes, triangles = _mesh_arrays(nodes, triangles)
    x, y = nodes[triangles, 0], nodes[triangles, 1]  # (m, 3)
    after, before = [1, 2, 0], [2, 0, 1]
    # grad N_i = (y_j - y_k, x_k - x_j) / (2 area), with i, j, k in order
    # round the triangle. The signed area makes it right for either direction.
    b = y[:, after] - y[:, before]
    c = x[:, before] - x[:, after]
    # Doubled signed area from edge vectors, not from positions, so that a
    # small triangle far from the origin loses no digits.
    doubled = b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]

    longest = np.max(b * b + c * c, axis=1)
    flat = np.flatnonzero(np.abs(doubled) <= NO_AREA * longest)
    if flat.size:
        raise ValueError(f"triangle {flat[0]} has no area")

    gradients = np.stack([b, c], axis=2) / doubled[:, None, None]
    return gradients, 0.5 * np.abs(doubled)


def _reluctivity_per_triangle(reluctivity: ArrayLike, count: int) -> NDArray[np.float64]:
    nu = np.asarray(reluctivity, dtype=float)
    if nu.ndim == 0:
        nu = np.full(count, float(nu))
    elif nu.shape != (count,):
        raise ValueError(
            f"reluctivity must be one value, or one value per triangle ({count}), not {nu.shape}"
        )
    if not (np.all(np.isfinite(nu)) and np.all(nu > 0.0)):
        raise ValueError("reluctivity must be a positive, finite number in every triangle")
    return nu


def element_stiffness(
    nodes: ArrayLike, triangles: ArrayLike, reluctivity: ArrayLike
) -> NDArray[np.float64]:
    """Stiffness matrix of every triangle, shape (m, 3, 3), in m/H.

    K_ij = nu * area * (grad N_i . grad N_j), the integral of
    nu grad N_i . grad N_j over the triangle. ``reluctivity`` is one value
    in m/H, or one per triangle.
    """
    gradients, area = shape_gradients(nodes, triangles)
    nu = _reluctivity_per_triangle(reluctivity, len(area))
    products = np.einsum("eid,ejd->eij", gradients, gradients)
    return (nu * area)[:, None, None] * products


def assemble_stiffness(
    nodes: ArrayLike, triangles: ArrayLike, reluctivity: ArrayLike
) -> sparse.csr_matrix:
    """Stiffness matrix of the whole mesh, shape (n, n), in m/H.

    Every node pair that shares a triangle gets one stored entry, also where
    the value happens to be zero, so the pattern depends on the mesh only.
    """
    nodes, triangles = _mesh_arrays(nodes, triangles)
    local = element_stiffness(nodes, triangles, reluctivity)
    rows = np.repeat(triangles, 3, axis=1)  # node i of each of the 9 entries
    columns = np.tile(triangles, (1, 3))  # node j
    count = len(nodes)
    # Converting to CSR adds up the entries that fall on the same node pair.
    matrix = sparse.coo_matrix(
        (local.ravel(), (rows.ravel(), columns.ravel())), shape=(count, count)
    ).tocsr()
    matrix.sort_indices()
    return matrix


def flux_density(
    nodes: ArrayLike, triangles: ArrayLike, potential: ArrayLike
) -> NDArray[np.float64]:
    """Flux density (Bx, By) in every triangle, shape (m, 2), in tesla.

    B = (dA/dy, -dA/dx), constant over a first-order triangle. ``potential``
    is A at every node in Wb/m.
    """
    nodes, triangles = _mesh_arrays(nodes, triangles)
    a = np.asarray(potential, dtype=float)
    if a.shape != (len(nodes),):
        raise ValueError(f"potential must have one value per node ({len(nodes)}), not {a.shape}")
    gradients, _ = shape_gradients(nodes, triangles)
    slope = np.einsum("ei,eid->ed", a[triangles], gradients)  # (dA/dx, dA/dy)
    return np.column_stack([slope[:, 1], -slope[:, 0]])


def _fixed_nodes(
    fixed_nodes: ArrayLike, fixed_values: ArrayLike, count: int
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Fixed node numbers without repeats, and the value of each."""
    fixed = np.asarray(fixed_nodes)
    if fixed.size and not np.issubdtype(fixed.dtype, np.integer):
        raise ValueError("fixed nodes must be node numbers as whole numbers")
    fixed = fixed.astype(np.intp).ravel()
    values = np.asarray(fixed_values, dtype=float)
    if values.ndim == 0:
        values = np.full(fixed.size, float(values))
    elif values.shape != (fixed.size,):
        raise ValueError(
            f"fixed values must be one value, or one per fixed node ({fixed.size}), "
            f"not {values.shape}"
        )
    if fixed.size == 0:
        raise ValueError("at least one node must be fixed, or the potential has no reference")
    outside = fixed[(fixed < 0) | (fixed >= count)]
    if outside.size:
        raise ValueError(f"fixed node {outside[0]} is not in the mesh (nodes 0 to {count - 1})")
    if not np.all(np.isfinite(values)):
        raise ValueError("fixed values must be finite numbers")

    unique, first, inverse = np.unique(fixed, return_index=True, return_inverse=True)
    clash = np.flatnonzero(values != values[first][inverse])
    if clash.size:
        node = fixed[clash[0]]
        raise ValueError(f"node {node} is fixed twice, to different values")
    return unique, values[first]


def solve_linear(
    stiffness: sparse.spmatrix,
    load: ArrayLike,
    fixed_nodes: ArrayLike,
    fixed_values: ArrayLike,
) -> NDArray[np.float64]:
    """Solve K a = f with the potential fixed at some nodes.

    ``load`` is the current fed in at every node, in amperes (zeros where
    there is no source). ``fixed_nodes`` are node numbers and
    ``fixed_values`` their potential in Wb/m: one value for all, or one per
    node. Every separate part of the mesh needs at least one fixed node.

    Returns the potential at every node, in Wb/m.
    """
    matrix = sparse.csr_matrix(stiffness)
    count = matrix.shape[0]
    if matrix.shape != (count, count):
        raise ValueError(f"the stiffness matrix must be square, not {matrix.shape}")
    f = np.asarray(load, dtype=float)
    if f.shape != (count,):
        raise ValueError(f"load must have one value per node ({count}), not {f.shape}")
    if not np.all(np.isfinite(f)):
        raise ValueError("load must be a finite number at every node")
    fixed, values = _fixed_nodes(fixed_nodes, fixed_values, count)

    potential = np.empty(count)
    potential[fixed] = values
    is_free = np.ones(count, dtype=bool)
    is_free[fixed] = False
    free = np.flatnonzero(is_free)
    if free.size == 0:
        return potential

    _check_supported(matrix, is_free)
    rows = matrix[free]
    # Move the known potentials to the right-hand side: K_ff a_f = f_f - K_fc a_c.
    right = f[free] - rows[:, fixed] @ values
    potential[free] = spsolve(rows[:, free].tocsc(), right)
    return potential


def _check_supported(matrix: sparse.csr_matrix, is_free: NDArray[np.bool_]) -> None:
    """Raise ValueError if some free node is not tied to a fixed one.

    Such a node makes the system singular: its potential could be anything.
    """
    alone = np.flatnonzero(is_free & (matrix.diagonal() == 0.0))
    if alone.size:
        raise ValueError(f"node {alone[0]} is not fixed and belongs to no triangle")
    pieces, piece = connected_components(matrix, directed=False)
    held = np.zeros(pieces, dtype=bool)
    held[piece[~is_free]] = True
    adrift = np.flatnonzero(~held[piece])
    if adrift.size:
        raise ValueError(
            f"node {adrift[0]} is in a part of the mesh with no fixed node, "
            "so its potential has no reference"
        )
