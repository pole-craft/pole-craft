"""The heavy arithmetic of a solve: sparse linear systems and matrix assembly.

The field solver hands its linear systems to ``solve_sparse`` and names the
solver to use, so a faster one can be put in without touching the solver's
logic. All of them run on the processor. Three need nothing beyond SciPy.
PARDISO is somewhat quicker but comes in a package of its own, which is
optional. SciPy's SuperLU is used unless another solver is named; PARDISO is
used only when asked for by name, and if it is not installed then, SuperLU
does the work. It does not exist for Macs or ARM processors; there it is
never used and never suggested.
``available_solvers`` lists what can be chosen on this machine.

Two things to know about PARDISO. The first solve in a process takes about
a third of a second longer, while its library loads; the timing report of a
field solve lists that as its own line. And it uses several processor cores
by itself, so when many solves run side by side in separate processes, give
each one core (set the environment variable MKL_NUM_THREADS to 1).

The matrices of a field problem are symmetric and positive definite: the
stiffness matrix with its fixed nodes taken out.
"""

from __future__ import annotations

import platform
import sys
import threading
import warnings
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.linalg import LinAlgError, solveh_banded
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import splu

# Always there, and what is used when no solver is named.
DEFAULT_SOLVER = "superlu"

# Optional solvers, used only when named: the package extra that installs each.
_EXTRAS = {"pardiso": "pardiso"}

# Optional packages once tried: the module, or the error that loading it gave.
_LOADED: dict[str, object] = {}


class SolverUnavailableWarning(UserWarning):
    """A solver was asked for whose package is not installed; another was used."""

# A solution is refused if it leaves more than this fraction of the
# right-hand side unbalanced. Rounding alone leaves far less, even with steel
# beside air (about 1e-12 on motor models); a singular matrix usually leaves
# about all of it.
WORST_RESIDUAL = 1e-4

SINGULAR = (
    "the linear system is singular or too badly conditioned to solve: "
    "some unknown is not tied to a fixed value"
)


def _superlu(matrix: sparse.csc_matrix, right: NDArray[np.float64]) -> NDArray[np.float64]:
    return splu(matrix).solve(right)


def _superlu_symmetric(
    matrix: sparse.csc_matrix, right: NDArray[np.float64]
) -> NDArray[np.float64]:
    factors = splu(
        matrix,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True},
    )
    return factors.solve(right)


def _banded(matrix: sparse.csc_matrix, right: NDArray[np.float64]) -> NDArray[np.float64]:
    size = matrix.shape[0]
    difference = abs(matrix - matrix.T)
    if difference.nnz and difference.max() > 1e-10 * abs(matrix).max():
        raise ValueError("the banded solver needs a symmetric matrix")
    # Renumber the unknowns so that tied ones are close together: the matrix
    # then has all its entries in a narrow band along the diagonal.
    order = reverse_cuthill_mckee(matrix.tocsr(), symmetric_mode=True)
    entries = matrix[order][:, order].tocoo()
    below = entries.row >= entries.col
    row, column, value = entries.row[below], entries.col[below], entries.data[below]
    offset = row - column
    width = int(offset.max()) + 1 if offset.size else 1
    # Lower band storage: entry (i, j) is kept at [i - j, j].
    band = np.bincount(offset * size + column, weights=value, minlength=width * size)
    try:
        x = solveh_banded(band.reshape(width, size), right[order], lower=True)
    except LinAlgError:
        raise ValueError(
            "the banded solver needs a symmetric positive definite matrix; "
            "this one is singular or indefinite"
        ) from None
    result = np.empty(size)
    result[order] = x
    return result


def _load(name: str) -> object:
    """Import the package of an optional solver."""
    if name == "pardiso":
        import pypardiso

        return pypardiso
    raise ImportError(name)


def _supported(name: str) -> bool:
    """Whether a solver's package exists for this kind of machine at all.

    PARDISO comes with Intel's MKL, which there is for Intel and AMD
    processors on Windows and Linux, not for Macs or ARM processors.
    """
    if name == "pardiso":
        system, processor = _machine()
        return system != "darwin" and processor in ("x86_64", "AMD64")
    return True


def _machine() -> tuple[str, str]:
    """The operating system and the processor type, as Python names them."""
    return sys.platform, platform.machine()


def _usable(name: str) -> bool:
    """Whether a solver can be used here: built in, or its package loads."""
    if name not in _EXTRAS:
        return True
    if not _supported(name):
        return False
    if name not in _LOADED:
        try:
            _LOADED[name] = _load(name)
        except (ImportError, OSError) as error:  # not installed, or its library will not load
            _LOADED[name] = error
    return not isinstance(_LOADED[name], Exception)


def _load_failure(name: str) -> str | None:
    """Why an installed optional solver could not be loaded; None if it is just not installed."""
    if not _supported(name) or _usable(name):
        return None
    error = _LOADED.get(name)
    if not isinstance(error, Exception) or isinstance(error, ModuleNotFoundError):
        return None
    return str(error)


# PARDISO solver objects by matrix type, kept so that each is set up once.
# One solve at a time: the objects hold the library's working memory, and two
# threads using them together crash the process.
_PARDISO: dict[int, object] = {}
_PARDISO_LOCK = threading.Lock()

# The answer of PARDISO's symmetric mode is taken only if it balances the
# equations to rounding. A matrix that is not quite symmetric gives a wrong
# answer there that a looser check would let through.
SYMMETRIC_RESIDUAL = 1e-10


def _pardiso_solver(matrix_type: int) -> object:
    if matrix_type not in _PARDISO:
        _PARDISO[matrix_type] = _LOADED["pardiso"].PyPardisoSolver(mtype=matrix_type)
    return _PARDISO[matrix_type]


def _pardiso(matrix: sparse.csc_matrix, right: NDArray[np.float64]) -> NDArray[np.float64]:
    # A field problem's matrix is symmetric and positive definite. PARDISO
    # then needs only the upper triangle and works about a third faster, so
    # that is tried first. If the answer does not balance the equations, the
    # matrix was not of that kind, and it is solved again as a general one.
    failure = _LOADED["pardiso"].pardiso_wrapper.PyPardisoError
    with _PARDISO_LOCK:
        symmetric = _pardiso_solver(2)
        try:
            x = symmetric.solve(sparse.triu(matrix, format="csr"), right)
            if np.abs(matrix @ x - right).max() <= SYMMETRIC_RESIDUAL * np.abs(right).max():
                return x
        except (failure, ValueError):  # not positive definite, or an empty row
            pass
        finally:
            symmetric.free_memory()  # the factors are not used again
        general = _pardiso_solver(11)
        try:
            return general.solve(matrix.tocsr(), right)
        except failure:
            raise ValueError(SINGULAR) from None
        finally:
            general.free_memory()


_READY: set[str] = set()


def prepare(solver: str) -> None:
    """Get a solver ready, so that its first real solve is not slowed by loading it.

    PARDISO loads its library on first use, which takes a fraction of a
    second, once in a process. The other solvers need nothing.
    """
    if solver == "pardiso" and solver not in _READY and _usable(solver):
        _pardiso(sparse.identity(2, format="csc"), np.ones(2))
        _READY.add(solver)


_SOLVERS: dict[str, tuple[Callable, str]] = {
    "superlu": (
        _superlu,
        "SciPy's SuperLU, a sparse direct solver; steady across model sizes",
    ),
    "superlu_symmetric": (
        _superlu_symmetric,
        "SuperLU set up for symmetric matrices; often a little quicker, much slower on some meshes",
    ),
    "banded": (
        _banded,
        "renumbering to a narrow band, then LAPACK's banded Cholesky; slower, a separate check "
        "that needs a symmetric positive definite matrix",
    ),
    "pardiso": (
        _pardiso,
        "PARDISO from Intel's MKL, through the optional pypardiso package; quickest on large "
        "models, about level with SuperLU on small ones",
    ),
}


def known_solvers() -> list[str]:
    """Names of all the solvers there are, installed here or not."""
    return list(_SOLVERS)


def available_solvers() -> list[str]:
    """Names of the solvers that can be used here; first the one used by default."""
    others = sorted(name for name in _SOLVERS if name != DEFAULT_SOLVER and _usable(name))
    return [DEFAULT_SOLVER] + others


def solver_descriptions() -> dict[str, str]:
    """One line about each available solver, by name."""
    return {name: _SOLVERS[name][1] for name in available_solvers()}


def install_hint(solver: str) -> str | None:
    """The command that installs an optional solver.

    None for a solver that is built in, and for one that does not exist for
    this kind of machine.
    """
    if solver not in _EXTRAS or not _supported(solver):
        return None
    return f'pip install "polecraft[{_EXTRAS[solver]}]"'


def faster_solver_hint() -> str | None:
    """The command that would install a faster solver than there is.

    None if the fastest is already here, cannot be had on this machine, or is
    installed but will not load. Once installed it is used when named.
    """
    for name in _EXTRAS:
        if not _usable(name) and not _load_failure(name):
            return install_hint(name)
    return None


def solver_name(solver: str | None) -> str:
    """The name of the solver to use: the one given, or the default.

    A solver whose package is not installed is not an error: a
    SolverUnavailableWarning says so, with the command that installs it, and
    the default solver is used. Raises ValueError for a name that is no
    solver at all.
    """
    if solver is None:
        return DEFAULT_SOLVER
    if not isinstance(solver, str) or solver not in _SOLVERS:
        missing = [name for name in _SOLVERS if not _usable(name)]
        note = f"; not installed: {', '.join(missing)}" if missing else ""
        raise ValueError(
            f"unknown solver '{solver}'; the solvers here are: "
            f"{', '.join(available_solvers())}{note}"
        )
    if not _usable(solver):
        instead = f"so '{DEFAULT_SOLVER}' is used instead"
        failure = _load_failure(solver)
        if not _supported(solver):
            reason = f"does not exist for this kind of machine, {instead}"
        elif failure:
            reason = f"is installed but could not be loaded ({failure}), {instead}"
        else:
            reason = f"is not installed, {instead}; to install it: {install_hint(solver)}"
        warnings.warn(f"solver '{solver}' {reason}", SolverUnavailableWarning, stacklevel=3)
        return DEFAULT_SOLVER
    return solver


def solve_sparse(
    matrix: sparse.spmatrix | ArrayLike, right: ArrayLike, solver: str | None = None
) -> NDArray[np.float64]:
    """Solve ``matrix @ x = right`` for a sparse, square matrix.

    ``solver`` is one of ``known_solvers()``; None means the default,
    SuperLU. If the one named is not installed, a warning says so and the
    default is used. The inputs are not changed.

    The answer is checked: it must balance the equations. Raises ValueError
    if it does not, or if the solver cannot factor the matrix, which is what
    a singular matrix normally leads to. A singular system whose right-hand
    side happens to allow a solution can still return one of its many
    solutions; the field solver rules that case out before it gets here.
    """
    name = solver_name(solver)
    if np.iscomplexobj(matrix.data if sparse.issparse(matrix) else matrix) or np.iscomplexobj(
        right
    ):
        raise ValueError("the matrix and the right-hand side must be real numbers")
    matrix = sparse.csc_matrix(matrix, dtype=float)
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        raise ValueError(f"the matrix must be square, not {matrix.shape}")
    right = np.asarray(right, dtype=float)
    if right.shape != (size,):
        raise ValueError(
            f"the right-hand side must have one value per row ({size}), not {right.shape}"
        )
    if not (np.all(np.isfinite(matrix.data)) and np.all(np.isfinite(right))):
        raise ValueError("the matrix and the right-hand side must be finite numbers")
    if size == 0:
        return np.zeros(0)

    try:
        x = _SOLVERS[name][0](matrix, right)
    except RuntimeError:  # SuperLU: "Factor is exactly singular"
        raise ValueError(SINGULAR) from None
    unbalanced = np.abs(matrix @ x - right).max()
    if not np.all(np.isfinite(x)) or unbalanced > WORST_RESIDUAL * np.abs(right).max():
        raise ValueError(SINGULAR)
    return x


class Assembler:
    """Adds the 3 by 3 matrices of the triangles into one sparse matrix.

    Which entries the matrix has depends on the mesh only, so that is worked
    out once here. Each call to ``assemble`` then only adds up values, which
    is what the iteration for saturating steel does again and again. Every
    node pair that shares a triangle gets an entry, zero or not.
    """

    def __init__(self, triangles: ArrayLike, count: int) -> None:
        """``triangles`` is (m, 3) node numbers; ``count`` the number of nodes."""
        triangles = np.asarray(triangles)
        if triangles.size == 0:
            triangles = np.zeros((0, 3), dtype=np.intp)
        if not np.issubdtype(triangles.dtype, np.integer):
            raise ValueError("triangles must hold node numbers as whole numbers")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(
                f"triangles must be an (m, 3) array of node numbers, not {triangles.shape}"
            )
        if triangles.size and (triangles.min() < 0 or triangles.max() >= count):
            bad = triangles[(triangles < 0) | (triangles >= count)][0]
            raise ValueError(f"a triangle names node {bad}, but there are nodes 0 to {count - 1}")
        self.count = count
        self.triangles = len(triangles)
        rows = np.repeat(triangles, 3, axis=1).ravel()  # node i of each of the 9 entries
        columns = np.tile(triangles, (1, 3)).ravel()  # node j
        # Entries in row order, then column order, each node pair once.
        pairs, self._place = np.unique(rows.astype(np.int64) * count + columns, return_inverse=True)
        self._indices = (pairs % count).astype(np.int32)
        self._indptr = np.searchsorted(pairs // count, np.arange(count + 1)).astype(np.int32)

    def assemble(self, local: NDArray[np.float64]) -> sparse.csr_matrix:
        """The matrix for element matrices ``local``, shape (m, 3, 3)."""
        local = np.asarray(local, dtype=float)
        if local.shape != (self.triangles, 3, 3):
            raise ValueError(
                f"there must be one 3 by 3 matrix per triangle ({self.triangles}), "
                f"not shape {local.shape}"
            )
        data = np.bincount(self._place, weights=local.ravel(), minlength=len(self._indices))
        # Each matrix gets its own copy of the pattern, so that changing one
        # in place cannot spoil the next.
        return sparse.csr_matrix(
            (data, self._indices.copy(), self._indptr.copy()), shape=(self.count, self.count)
        )
