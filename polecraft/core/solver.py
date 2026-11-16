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

The building blocks take plain arrays (node positions, triangles, one
reluctivity per triangle). ``solve`` takes a ``Problem``, which holds the
same things together with the material models, and iterates when a material
saturates. Nothing here knows what kind of machine, if any, the problem
describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    from polecraft.core.problem import Problem

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
    return _assemble(local, triangles, len(nodes))


def _assemble(local: NDArray[np.float64], triangles: NDArray, count: int) -> sparse.csr_matrix:
    """Add up 3 by 3 matrices of the triangles, shape (m, 3, 3), into one matrix."""
    rows = np.repeat(triangles, 3, axis=1)  # node i of each of the 9 entries
    columns = np.tile(triangles, (1, 3))  # node j
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


def _to_nodes(local: NDArray[np.float64], triangles: NDArray, count: int) -> NDArray[np.float64]:
    """Add up per-corner values, shape (m, 3), at the nodes they belong to."""
    total = np.bincount(triangles.ravel(), weights=local.ravel(), minlength=count)
    return total.astype(float, copy=False)  # bincount gives integers for no triangles


def current_load(
    nodes: ArrayLike, triangles: ArrayLike, current_density: ArrayLike
) -> NDArray[np.float64]:
    """Load vector of the currents in the mesh, shape (n,), in amperes.

    ``current_density`` is J in A/m2, positive out of the plane: one value,
    or one per triangle (zero where there is no conductor). The current of a
    triangle, J times its area, is shared equally between its three nodes.
    """
    nodes, triangles = _mesh_arrays(nodes, triangles)
    _, area = shape_gradients(nodes, triangles)
    density = np.asarray(current_density, dtype=float)
    if density.ndim == 0:
        density = np.full(len(area), float(density))
    elif density.shape != (len(area),):
        raise ValueError(
            f"current density must be one value, or one value per triangle ({len(area)}), "
            f"not {density.shape}"
        )
    if not np.all(np.isfinite(density)):
        raise ValueError("current density must be a finite number in every triangle")
    share = density * area / 3.0
    return _to_nodes(np.repeat(share[:, None], 3, axis=1), triangles, len(nodes))


def magnet_load(
    nodes: ArrayLike, triangles: ArrayLike, remanence: ArrayLike, reluctivity: ArrayLike
) -> NDArray[np.float64]:
    """Load vector of the permanent magnets in the mesh, shape (n,), in amperes.

    A magnet follows B = mu H + Br, that is H = nu (B - Br), with nu its
    recoil reluctivity. Put into the field equation, the Br term becomes the
    load f_i = integral of nu (Br_x dN_i/dy - Br_y dN_i/dx). For a uniformly
    magnetised block this is zero inside and leaves a current sheet of
    nu (Br x n) amperes per metre on its faces, n being the outward normal:
    nu Br on a face that runs along the magnetisation, nothing on a face
    square to it.

    ``remanence`` is (Br_x, Br_y) in tesla: one vector, or one per triangle,
    shape (m, 2), zero outside the magnets. ``reluctivity`` is the same one
    the stiffness matrix is built with, one value or one per triangle.
    """
    nodes, triangles = _mesh_arrays(nodes, triangles)
    gradients, area = shape_gradients(nodes, triangles)
    nu = _reluctivity_per_triangle(reluctivity, len(area))
    br = np.asarray(remanence, dtype=float)
    if br.shape == (2,):
        br = np.tile(br, (len(area), 1))
    elif br.shape != (len(area), 2):
        raise ValueError(
            "remanence must be one vector with x and y parts, or one per triangle "
            f"({len(area)}, 2), not {br.shape}"
        )
    if not np.all(np.isfinite(br)):
        raise ValueError("remanence must be finite numbers")
    strength = (nu * area)[:, None] * br  # nu Br times area, (m, 2)
    local = strength[:, None, 0] * gradients[:, :, 1] - strength[:, None, 1] * gradients[:, :, 0]
    return _to_nodes(local, triangles, len(nodes))


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


def _couplings(
    periodic: ArrayLike | None, antiperiodic: bool, count: int
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """For every node, the node it takes its value from, and the sign.

    A node that is nobody's copy takes its own value, with sign +1. A copy
    of a copy is followed back to the first node, the signs multiplied.
    """
    origin = np.arange(count)
    sign = np.ones(count)
    if periodic is None or np.size(periodic) == 0:
        return origin, sign
    pairs = np.asarray(periodic)
    if not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("periodic pairs must hold node numbers as whole numbers")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            f"periodic must be pairs of node numbers, shape (k, 2), not {pairs.shape}"
        )
    outside = pairs[(pairs < 0) | (pairs >= count)]
    if outside.size:
        raise ValueError(
            f"a periodic pair names node {outside[0]}, but the mesh has nodes 0 to {count - 1}"
        )
    same = pairs[pairs[:, 0] == pairs[:, 1], 0]
    if same.size:
        raise ValueError(f"periodic pairs: node {same[0]} is coupled to itself")
    pairs = np.unique(pairs, axis=0)
    copies, times = np.unique(pairs[:, 0], return_counts=True)
    if np.any(times > 1):
        raise ValueError(f"periodic pairs: node {copies[times > 1][0]} is the copy of two nodes")

    direct = origin.copy()
    direct[pairs[:, 0]] = pairs[:, 1]
    origin = direct.copy()
    if antiperiodic:
        sign[pairs[:, 0]] = -1.0
    # Each pass doubles the number of steps followed, so chains resolve fast.
    for _ in range(64):
        further = origin[origin]
        if np.array_equal(further, origin):
            break
        sign = sign * sign[origin]
        origin = further
    # A chain that ends does so at a node that is nobody's copy.
    if not np.array_equal(direct[origin], origin):
        raise ValueError("periodic pairs form a loop, so no node gives the others their value")
    return origin, sign


class _Reduction:
    """Fixed values and couplings of a problem, applied to its equations.

    Built once for a set of nodes; then used for every matrix and load on
    them, which is what the Newton iteration needs.
    """

    def __init__(
        self,
        count: int,
        fixed_nodes: ArrayLike,
        fixed_values: ArrayLike,
        periodic: ArrayLike | None,
        antiperiodic: bool,
    ) -> None:
        self.count = count
        fixed, values = _fixed_nodes(fixed_nodes, fixed_values, count)
        self.origin, self.sign = _couplings(periodic, antiperiodic, count)
        self.own = self.origin == np.arange(count)  # nodes with an equation of their own
        self.coupled = not np.all(self.own)
        self.link = None
        if self.coupled:
            # a = L a_own, with one entry per row of L: +1 or -1 in the column
            # of the node's origin. The equations become L^T K L a_own = L^T f.
            self.link = sparse.csr_matrix(
                (self.sign, (np.arange(count), self.origin)), shape=(count, count)
            )
            fixed, values = _fixed_origins(fixed, values, self.origin, self.sign)
        self.fixed, self.values = fixed, values
        self.is_free = self.own.copy()
        self.is_free[fixed] = False
        self.free = np.flatnonzero(self.is_free)
        self._checked = False

    def conform(self, potential: NDArray[np.float64]) -> NDArray[np.float64]:
        """A potential with the fixed values and couplings put in."""
        potential = potential.copy()
        potential[self.fixed] = self.values
        return self.sign * potential[self.origin] if self.coupled else potential

    def free_part(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """A nodal vector of currents as seen by the equations that are solved."""
        if self.coupled:
            vector = self.link.T @ vector
        return vector[self.free]

    def free_sum(self, sizes: NDArray[np.float64]) -> NDArray[np.float64]:
        """As ``free_part``, for sizes that add up whatever the sign of a coupling."""
        if self.coupled:
            sizes = abs(self.link).T @ sizes
        return sizes[self.free]

    def solve(self, matrix: sparse.csr_matrix, f: NDArray[np.float64]) -> NDArray[np.float64]:
        """The potential at every node for a matrix and a load."""
        joined = None if self._checked else abs(matrix)  # which nodes are tied, signs aside
        if self.coupled:
            if joined is not None:
                # Two entries of opposite sign can cancel in L^T K L; the
                # nodes are tied all the same, so ties are counted without signs.
                joined = abs(self.link).T @ joined @ abs(self.link)
            matrix = (self.link.T @ matrix @ self.link).tocsr()
            f = self.link.T @ f

        potential = np.empty(self.count)
        potential[self.fixed] = self.values
        if self.free.size:
            if joined is not None:
                _check_supported(matrix, joined, self.is_free, self.own)
                self._checked = True
            rows = matrix[self.free]
            # Move the known potentials to the right-hand side: K_ff a_f = f_f - K_fc a_c.
            right = f[self.free] - rows[:, self.fixed] @ self.values
            potential[self.free] = spsolve(rows[:, self.free].tocsc(), right)
        if self.coupled:
            potential = self.sign * potential[self.origin]
        return potential


def solve_linear(
    stiffness: sparse.spmatrix,
    load: ArrayLike,
    fixed_nodes: ArrayLike,
    fixed_values: ArrayLike,
    periodic: ArrayLike | None = None,
    antiperiodic: bool = False,
) -> NDArray[np.float64]:
    """Solve K a = f with the potential fixed at some nodes.

    ``load`` is the current fed in at every node, in amperes (zeros where
    there is no source). ``fixed_nodes`` are node numbers and
    ``fixed_values`` their potential in Wb/m: one value for all, or one per
    node. Every separate part of the mesh needs at least one fixed node.

    ``periodic`` couples the two cut edges of a slice of a machine. It is a
    list of node pairs (copy, original), shape (k, 2): the potential at the
    copy equals that at the original, or its opposite if ``antiperiodic``
    is true, as when the slice holds an odd number of poles. The pair then
    shares one equation, so the slice behaves as part of the whole machine.
    Coupling does not stand in for a fixed node: one is still needed.

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
    reduction = _Reduction(count, fixed_nodes, fixed_values, periodic, antiperiodic)
    return reduction.solve(matrix, f)


def _fixed_origins(
    fixed: NDArray[np.intp],
    values: NDArray[np.float64],
    origin: NDArray[np.intp],
    sign: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Fixed values moved from copies to the nodes they take their value from."""
    at_origin = sign[fixed] * values
    unique, first, inverse = np.unique(origin[fixed], return_index=True, return_inverse=True)
    # Values worked out separately for a node and its copy may differ in the
    # last digits; the first one given is used.
    allowed = 1e-12 * np.abs(at_origin).max()
    clash = np.flatnonzero(np.abs(at_origin - at_origin[first][inverse]) > allowed)
    if clash.size:
        one, other = fixed[first][inverse][clash[0]], fixed[clash[0]]
        raise ValueError(
            f"node {one} and node {other} are coupled but fixed to values that do not agree"
        )
    return unique, at_origin[first]


def _check_supported(
    matrix: sparse.csr_matrix,
    joined: sparse.spmatrix,
    is_free: NDArray[np.bool_],
    own: NDArray[np.bool_],
) -> None:
    """Raise ValueError if some free node is not tied to a fixed one.

    Such a node makes the system singular: its potential could be anything.
    ``joined`` has an entry for every two nodes that are tied, and ``own``
    marks the nodes that are not copies; copies have no equation.
    """
    alone = np.flatnonzero(is_free & (matrix.diagonal() == 0.0))
    if alone.size:
        raise ValueError(f"node {alone[0]} is not fixed and belongs to no triangle")
    pieces, piece = connected_components(joined, directed=False)
    held = np.zeros(pieces, dtype=bool)
    held[piece[own & ~is_free]] = True
    adrift = np.flatnonzero(own & ~held[piece])
    if adrift.size:
        raise ValueError(
            f"node {adrift[0]} is in a part of the mesh with no fixed node, "
            "so its potential has no reference"
        )


class ConvergenceError(RuntimeError):
    """The Newton iteration did not reach its tolerance.

    ``residuals`` and ``steps`` are those of ``Solution``, up to where the
    iteration stopped.
    """

    def __init__(
        self, message: str, residuals: tuple[float, ...], steps: tuple[float, ...]
    ) -> None:
        self.residuals = residuals
        self.steps = steps
        super().__init__(f"{message}\n{_iteration_table(residuals, steps)}")


@dataclass(frozen=True, eq=False)
class Solution:
    """A solved field and how the iteration got there."""

    potential: NDArray[np.float64]  # (n,) A at the nodes, Wb/m
    flux_density: NDArray[np.float64]  # (m, 2) B in every triangle, tesla
    reluctivity: NDArray[np.float64]  # (m,) nu in every triangle at that B, m/H
    # Out-of-balance current at the start and after each iteration, relative
    # to the size of the currents being balanced in the final field.
    residuals: tuple[float, ...]
    steps: tuple[float, ...]  # relaxation factor of each iteration; 1 is a full Newton step

    @property
    def iterations(self) -> int:
        """Number of Newton iterations taken."""
        return len(self.steps)

    def report(self) -> str:
        """The residual and relaxation factor of every iteration, as text."""
        return "\n".join(
            [
                "Newton iteration",
                _iteration_table(self.residuals, self.steps),
                f"  converged in {self.iterations} iterations",
            ]
        )


def _iteration_table(residuals: tuple[float, ...], steps: tuple[float, ...]) -> str:
    lines = ["  iteration   residual    step"]
    for number, residual in enumerate(residuals):
        step = f"{steps[number - 1]:8.4f}" if number else ""
        lines.append(f"  {number:9d}  {residual:.3e}{step}")
    return "\n".join(lines)


# The relaxation factor is halved until the residual falls; below this the
# iteration gives up.
SMALLEST_STEP = 1.0 / 1024.0


def solve(
    problem: Problem,
    start: ArrayLike | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 30,
) -> Solution:
    """Solve a field problem, with steel that saturates, by Newton iteration.

    Each iteration solves the problem linearised about the present field,
    using the slope of every material's BH curve there. The full step is
    taken if it lowers the out-of-balance current at the free nodes; if
    not, the step is halved until it does (relaxation), so the residual
    falls at every iteration. A problem with only linear materials is
    solved in one iteration.

    The iteration stops when the out-of-balance current is at or below
    ``tolerance`` times the size of the currents being balanced. The
    potential is then usually good to a few thousand times ``tolerance``,
    because steel and air differ so much in reluctivity.

    ``start`` is a potential to begin from, for example the solution of a
    nearby problem; it defaults to zero. Fixed values and couplings are put
    into it first.

    A magnetised triangle must have a linear material, as a magnet on its
    recoil line has.

    Raises ConvergenceError if ``max_iterations`` is not enough, or if no
    step lowers the residual, which also happens when ``tolerance`` is
    below what rounding allows (about 1e-13).
    """
    if not (np.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError(f"tolerance must be a positive number, not {tolerance}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1, not {max_iterations}")
    nodes, triangles = _mesh_arrays(problem.nodes, problem.triangles)
    count = len(nodes)
    if start is None:
        potential = np.zeros(count)
    else:
        potential = np.array(start, dtype=float)
        if potential.shape != (count,) or not np.all(np.isfinite(potential)):
            raise ValueError(f"start must be a finite potential at every node ({count})")

    gradients, area = shape_gradients(nodes, triangles)
    members = [np.flatnonzero(problem.triangle_material == k) for k in range(len(problem.materials))]
    magnetised = np.any(problem.remanence != 0.0, axis=1)
    for number, (material, inside) in enumerate(zip(problem.materials, members)):
        if not material.linear and np.any(magnetised[inside]):
            raise ValueError(
                f"material {number} saturates, so its triangles cannot be magnetised: "
                "give magnets a linear material"
            )
    from_current = current_load(nodes, triangles, problem.current_density)
    # Per unit reluctivity; the magnets' own reluctivity is put in below.
    magnet_local = area[:, None] * (
        problem.remanence[:, None, 0] * gradients[:, :, 1]
        - problem.remanence[:, None, 1] * gradients[:, :, 0]
    )
    reduction = _Reduction(
        count, problem.fixed_nodes, problem.fixed_values, problem.periodic, problem.antiperiodic
    )

    def state(a: NDArray[np.float64]) -> dict:
        """Field, material values and residual for a potential."""
        slope = np.einsum("ei,eid->ed", a[triangles], gradients)  # grad A
        b = np.hypot(slope[:, 0], slope[:, 1])
        nu = np.empty(len(triangles))
        change = np.empty(len(triangles))
        for material, inside in zip(problem.materials, members):
            nu[inside] = material.reluctivity(b[inside])
            change[inside] = material.reluctivity_derivative(b[inside])
        # Currents the field asks for at each corner: nu area (grad N . grad A).
        pull = np.einsum("eid,ed->ei", gradients, slope)
        wanted = (nu * area)[:, None] * pull
        load = from_current + _to_nodes(nu[:, None] * magnet_local, triangles, count)
        out_of_balance = reduction.free_part(_to_nodes(wanted, triangles, count) - load)
        size = np.linalg.norm(
            reduction.free_sum(_to_nodes(np.abs(wanted), triangles, count) + np.abs(load))
        )
        return {
            "slope": slope, "b": b, "nu": nu, "change": change, "pull": pull, "load": load,
            "unbalance": float(np.linalg.norm(out_of_balance)), "size": float(size),
        }

    potential = reduction.conform(potential)
    with np.errstate(over="ignore", invalid="ignore"):  # checked just below
        now = state(potential)
    if not (np.isfinite(now["unbalance"]) and np.isfinite(now["size"])):
        raise ValueError("start gives a field too large to work with; use a smaller one")
    unbalance, steps = [now["unbalance"]], []

    def relative() -> tuple[float, ...]:
        """The out-of-balance history against the present size of the currents."""
        return tuple(value / now["size"] if now["size"] > 0.0 else 0.0 for value in unbalance)

    while now["unbalance"] > tolerance * now["size"]:
        if len(steps) == max_iterations:
            raise ConvergenceError(
                f"the field solve did not converge in {max_iterations} iterations: "
                f"residual {relative()[-1]:.3e}, tolerance {tolerance:.1e}",
                relative(), tuple(steps),
            )
        # Tangent matrix: nu K, plus the change of nu with B. With
        # q = area (dnu/dB) / B and v = grad N . grad A, the extra term of a
        # triangle is q v v^T; it vanishes with B.
        b, pull = now["b"], now["pull"]
        q = np.divide(area * now["change"], b, out=np.zeros_like(b), where=b > 0.0)
        local = (now["nu"] * area)[:, None, None] * np.einsum(
            "eid,ejd->eij", gradients, gradients
        ) + q[:, None, None] * pull[:, :, None] * pull[:, None, :]
        tangent = _assemble(local, triangles, count)
        # Newton step written for the new potential: T a_new = f + (T - nu K) a,
        # and (T - nu K) a = q B^2 v at each corner.
        right = now["load"] + _to_nodes((q * b * b)[:, None] * pull, triangles, count)
        target = reduction.solve(tangent, right)

        step = 1.0
        while True:
            trial = target if step == 1.0 else potential + step * (target - potential)
            new = state(trial)
            if new["unbalance"] < now["unbalance"]:
                break
            step *= 0.5
            if step < SMALLEST_STEP:
                raise ConvergenceError(
                    "the field solve stalled: no step lowers the residual "
                    f"(residual {relative()[-1]:.3e}, tolerance {tolerance:.1e})",
                    relative(), tuple(steps),
                )
        potential, now = trial, new
        unbalance.append(now["unbalance"])
        steps.append(step)

    slope = now["slope"]
    return Solution(
        potential=potential,
        flux_density=np.column_stack([slope[:, 1], -slope[:, 0]]),
        reluctivity=now["nu"],
        residuals=relative(),
        steps=tuple(steps),
    )
