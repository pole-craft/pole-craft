"""Compute layer: selectable sparse solvers, assembly with a fixed pattern, timing."""

import math
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from polecraft.core import compute
from polecraft.core.compute import (
    DEFAULT_SOLVER,
    Assembler,
    SolverUnavailableWarning,
    available_solvers,
    install_hint,
    known_solvers,
    solve_sparse,
    solver_descriptions,
)
from polecraft.core.materials import AIR, MU0, FrohlichSteel, Magnet, load_material
from polecraft.core.problem import Problem, build_problem
from polecraft.core.solver import (
    assemble_stiffness,
    element_stiffness,
    shape_gradients,
    solve,
    solve_linear,
)

SOLVERS = available_solvers()


def grid(columns: int, rows: int, width: float = 1.0, height: float = 1.0) -> tuple:
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


def stiff_system(columns: int = 30, rows: int = 20, seed: int = 0) -> tuple:
    """A symmetric positive definite matrix like the solver's, and a right-hand side.

    The stiffness matrix of a rectangle with reluctivities spread over four
    decades, with the rows and columns of the boundary nodes taken out.
    """
    rng = np.random.default_rng(seed)
    nodes, triangles = grid(columns, rows)
    nu = 10.0 ** rng.uniform(2.0, 6.0, len(triangles))
    matrix = assemble_stiffness(nodes, triangles, nu)
    x, y = nodes.T
    inside = np.flatnonzero((x > 0.0) & (x < 1.0) & (y > 0.0) & (y < 1.0))
    return matrix[inside][:, inside].tocsc(), rng.normal(size=len(inside))


# Which solvers there are


BUILT_IN = ["superlu", "superlu_symmetric", "banded"]
OPTIONAL = ["pardiso"]


def test_superlu_is_always_there_to_fall_back_on() -> None:
    assert DEFAULT_SOLVER == "superlu"
    assert DEFAULT_SOLVER in SOLVERS


def test_the_solver_used_by_default_is_listed_first() -> None:
    assert SOLVERS[0] == DEFAULT_SOLVER
    assert SOLVERS[1:] == sorted(SOLVERS[1:])


def test_known_solvers_are_the_built_in_ones_and_the_optional_ones() -> None:
    assert sorted(known_solvers()) == sorted(BUILT_IN + OPTIONAL)
    assert set(BUILT_IN) <= set(SOLVERS) <= set(known_solvers())


def test_pardiso_can_be_chosen_exactly_when_it_is_installed() -> None:
    try:
        import pypardiso  # noqa: F401

        installed = compute._supported("pardiso")
    except (ImportError, OSError):
        installed = False
    assert ("pardiso" in SOLVERS) == installed


def test_pardiso_is_used_only_when_named(monkeypatch: pytest.MonkeyPatch) -> None:
    # As with PARDISO installed, whatever this machine has: it must not be
    # called, or even loaded, unless it is asked for.
    called = []
    monkeypatch.setattr(compute, "_supported", lambda name: True)
    monkeypatch.setattr(compute, "_usable", lambda name: True)
    monkeypatch.setattr(compute, "_pardiso", lambda matrix, right: called.append("pardiso"))
    description = compute._SOLVERS["pardiso"][1]
    monkeypatch.setitem(compute._SOLVERS, "pardiso", (compute._pardiso, description))
    monkeypatch.setattr(compute, "_READY", set())
    assert available_solvers()[0] == "superlu" and "pardiso" in available_solvers()
    assert compute.solver_name(None) == "superlu"
    matrix, right = stiff_system(columns=8, rows=6)
    assert np.array_equal(solve_sparse(matrix, right), solve_sparse(matrix, right, "superlu"))
    solution = solve(saturating_problem())
    assert solution.solver == "superlu"
    nodes, triangles = grid(4, 3)
    solve_linear(assemble_stiffness(nodes, triangles, 1.0), np.ones(len(nodes)), [0], 0.0)
    assert called == []
    assert compute.solver_name("pardiso") == "pardiso"


def test_optional_solvers_say_how_to_install_them(no_optional_solvers: None) -> None:
    assert install_hint("pardiso") == 'pip install "polecraft[pardiso]"'
    for name in BUILT_IN:
        assert install_hint(name) is None


@pytest.fixture
def no_optional_solvers(monkeypatch: pytest.MonkeyPatch) -> None:
    """As on a Windows or Linux machine with none of the optional solver packages installed."""
    monkeypatch.setattr(compute, "_supported", lambda name: True)
    monkeypatch.setattr(compute, "_usable", lambda name: name not in compute._EXTRAS)


@pytest.fixture
def on_a_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    """As on a Mac, whatever this machine is: MKL does not exist there."""
    monkeypatch.setattr(compute, "_machine", lambda: ("darwin", "arm64"))
    monkeypatch.setattr(compute, "_LOADED", {})


def test_pardiso_is_never_used_or_suggested_on_a_mac(on_a_mac: None) -> None:
    assert "pardiso" not in available_solvers()
    assert "pardiso" in known_solvers()
    assert install_hint("pardiso") is None
    assert compute.faster_solver_hint() is None
    problem = saturating_problem()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to be told about by default
        solution = solve(problem)
    assert solution.solver == "superlu"
    lines = solution.timing_report().splitlines()
    assert lines[0] == "Timing (solver: superlu)"
    assert "install" not in solution.timing_report()
    assert lines[-1].split()[0] == "total"


def test_asking_for_pardiso_on_a_mac_says_it_does_not_exist_there(on_a_mac: None) -> None:
    matrix, right = stiff_system(columns=8, rows=6)
    with pytest.warns(SolverUnavailableWarning) as caught:
        x = solve_sparse(matrix, right, "pardiso")
    assert str(caught[0].message) == (
        "solver 'pardiso' does not exist for this kind of machine, so 'superlu' is used instead"
    )
    assert "pip install" not in str(caught[0].message)
    assert np.array_equal(x, solve_sparse(matrix, right, "superlu"))


def test_pardiso_is_not_looked_for_on_a_mac_even_if_a_package_of_that_name_is_there(
    on_a_mac: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_load(name: str) -> object:
        raise AssertionError("tried to load an optional solver on a machine it does not exist for")

    monkeypatch.setattr(compute, "_load", must_not_load)
    assert available_solvers() == ["superlu", "banded", "superlu_symmetric"]


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("win32", "AMD64", True),
        ("linux", "x86_64", True),
        ("linux", "aarch64", False),
        ("darwin", "arm64", False),
        ("darwin", "x86_64", False),
        ("win32", "ARM64", False),
    ],
)
def test_pardiso_exists_for_intel_and_amd_processors_on_windows_and_linux_only(
    system: str, machine: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compute, "_machine", lambda: (system, machine))
    assert compute._supported("pardiso") is expected
    assert compute._supported("superlu")


def test_without_optional_solvers_superlu_is_the_default(no_optional_solvers: None) -> None:
    assert available_solvers() == ["superlu", "banded", "superlu_symmetric"]
    assert list(solver_descriptions()) == available_solvers()
    matrix, right = stiff_system(columns=8, rows=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the default must not warn
        x = solve_sparse(matrix, right)
    assert x == pytest.approx(np.linalg.solve(matrix.toarray(), right), rel=1e-9)


def test_a_missing_optional_solver_is_reported_and_superlu_is_used(
    no_optional_solvers: None,
) -> None:
    matrix, right = stiff_system(columns=8, rows=6)
    with pytest.warns(SolverUnavailableWarning) as caught:
        x = solve_sparse(matrix, right, "pardiso")
    assert len(caught) == 1
    assert str(caught[0].message) == (
        "solver 'pardiso' is not installed, so 'superlu' is used instead; "
        'to install it: pip install "polecraft[pardiso]"'
    )
    # The work goes on, with the same answer.
    assert np.array_equal(x, solve_sparse(matrix, right, "superlu"))


def test_a_missing_optional_solver_does_not_stop_a_field_solve(
    no_optional_solvers: None,
) -> None:
    problem = saturating_problem()
    with pytest.warns(SolverUnavailableWarning, match="'superlu' is used instead"):
        solution = solve(problem, solver="pardiso")
    assert solution.solver == "superlu"  # what was used, not what was asked for
    reference = solve(problem, solver="superlu")
    assert np.array_equal(solution.potential, reference.potential)
    nodes, triangles = grid(4, 3)
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    with pytest.warns(SolverUnavailableWarning):
        found = solve_linear(matrix, np.ones(len(nodes)), [0], 0.0, solver="pardiso")
    assert np.array_equal(found, solve_linear(matrix, np.ones(len(nodes)), [0], 0.0))


def test_an_optional_solver_that_cannot_be_loaded_counts_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Installed, but its library fails to load: the import raises OSError.
    import builtins

    real_import = builtins.__import__

    def broken_import(name: str, *arguments, **keywords):
        if name == "pypardiso":
            raise OSError("mkl_rt: cannot open shared object file")
        return real_import(name, *arguments, **keywords)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    monkeypatch.delitem(__import__("sys").modules, "pypardiso", raising=False)
    monkeypatch.setattr(compute, "_LOADED", {})
    monkeypatch.setattr(compute, "_supported", lambda name: True)
    assert "pardiso" not in available_solvers()
    matrix, right = stiff_system(columns=8, rows=6)
    with pytest.warns(SolverUnavailableWarning) as caught:
        solve_sparse(matrix, right, "pardiso")
    # The real cause is given, and no advice to install what is already there.
    assert str(caught[0].message) == (
        "solver 'pardiso' is installed but could not be loaded "
        "(mkl_rt: cannot open shared object file), so 'superlu' is used instead"
    )
    assert compute.faster_solver_hint() is None


def test_the_warning_points_at_the_line_that_asked_for_the_solver(
    no_optional_solvers: None,
) -> None:
    matrix, right = stiff_system(columns=8, rows=6)
    with pytest.warns(SolverUnavailableWarning) as caught:
        solve_sparse(matrix, right, "pardiso")
    assert Path(caught[0].filename).name == "test_compute.py"
    with pytest.warns(SolverUnavailableWarning) as caught:
        solve(saturating_problem(), solver="pardiso")
    assert Path(caught[0].filename).name == "test_compute.py"


def test_an_unknown_name_is_still_an_error_not_a_fallback(no_optional_solvers: None) -> None:
    matrix, right = stiff_system(columns=8, rows=6)
    with pytest.raises(ValueError, match="unknown solver 'cholmod'.*superlu.*not installed: pardiso"):
        solve_sparse(matrix, right, "cholmod")


def test_solvers_are_listed_once_each_with_a_description() -> None:
    assert len(SOLVERS) >= 3
    assert len(set(SOLVERS)) == len(SOLVERS)
    described = solver_descriptions()
    assert list(described) == SOLVERS
    for name, text in described.items():
        assert name == name.lower() and " " not in name
        assert len(text) > 20


@pytest.fixture
def pardiso_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """The matrix types PARDISO is called with: 2 symmetric positive definite, 11 general."""
    if "pardiso" not in SOLVERS:
        pytest.skip("the optional PARDISO solver is not installed")
    calls: list[int] = []
    real = compute._pardiso_solver

    def recording(matrix_type: int) -> object:
        calls.append(matrix_type)
        return real(matrix_type)

    monkeypatch.setattr(compute, "_pardiso_solver", recording)
    return calls


def test_pardiso_solves_a_field_matrix_in_its_symmetric_mode_only(pardiso_calls: list) -> None:
    matrix, right = stiff_system()
    x = solve_sparse(matrix, right, "pardiso")
    assert pardiso_calls == [2]
    assert x == pytest.approx(solve_sparse(matrix, right, "superlu"), rel=1e-9, abs=1e-18)


def test_pardiso_solves_other_matrices_as_general_ones(pardiso_calls: list) -> None:
    matrix = sparse.csc_matrix(np.array([[4.0, 1.0, 0.0], [0.5, 4.0, 1.0], [0.0, 1.0, 4.0]]))
    solve_sparse(matrix, np.ones(3), "pardiso")
    assert pardiso_calls == [2, 11]


@pytest.mark.parametrize("skew", [1e-12, 1e-8, 1e-6, 1e-5, 1e-3])
def test_pardiso_is_right_for_a_matrix_that_is_not_quite_symmetric(skew: float) -> None:
    # The symmetric mode reads the upper triangle only. Its answer for such a
    # matrix is a little wrong, and must not be let through.
    if "pardiso" not in SOLVERS:
        pytest.skip("the optional PARDISO solver is not installed")
    size = 200
    upper = np.full(size - 1, -1.0)
    matrix = sparse.diags([upper * (1.0 + skew), np.full(size, 2.0), upper], [-1, 0, 1]).tocsc()
    right = np.random.default_rng(7).normal(size=size)
    expected = np.linalg.solve(matrix.toarray(), right)
    assert solve_sparse(matrix, right, "pardiso") == pytest.approx(expected, rel=1e-8, abs=1e-12)


def test_pardiso_can_be_called_from_several_threads() -> None:
    if "pardiso" not in SOLVERS:
        pytest.skip("the optional PARDISO solver is not installed")
    systems = [stiff_system(columns=10 + k, rows=8, seed=k) for k in range(4)]
    expected = [solve_sparse(matrix, right, "superlu") for matrix, right in systems]
    wrong: list[str] = []

    def work(number: int) -> None:
        matrix, right = systems[number]
        for _ in range(30):
            x = solve_sparse(matrix, right, "pardiso")
            if np.abs(x - expected[number]).max() > 1e-9 * np.abs(expected[number]).max():
                wrong.append(f"thread {number}")

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert wrong == []


def test_pardiso_takes_a_matrix_that_is_not_symmetric() -> None:
    if "pardiso" not in SOLVERS:
        pytest.skip("the optional PARDISO solver is not installed")
    matrix = sparse.csc_matrix(np.array([[4.0, 1.0, 0.0], [0.5, 4.0, 1.0], [0.0, 1.0, 4.0]]))
    expected = np.linalg.solve(matrix.toarray(), np.ones(3))
    assert solve_sparse(matrix, np.ones(3), "pardiso") == pytest.approx(expected)
    # And one that is symmetric but has no Cholesky factor.
    indefinite = sparse.csc_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert solve_sparse(indefinite, np.ones(2), "pardiso") == pytest.approx([1.0 / 3.0, 1.0 / 3.0])


def test_the_scipy_solvers_need_nothing_extra() -> None:
    assert {"superlu", "superlu_symmetric", "banded"} <= set(SOLVERS)


# Solving


@pytest.mark.parametrize("solver", SOLVERS)
def test_each_solver_solves_the_system(solver: str) -> None:
    matrix, right = stiff_system()
    x = solve_sparse(matrix, right, solver)
    assert x.shape == right.shape and x.dtype == np.float64
    residual = matrix @ x - right
    assert np.abs(residual).max() <= 1e-9 * np.abs(right).max()


def test_all_solvers_give_the_same_result() -> None:
    matrix, right = stiff_system(seed=1)
    reference = solve_sparse(matrix, right, DEFAULT_SOLVER)
    scale = np.abs(reference).max()
    assert scale > 0.0
    for solver in SOLVERS:
        x = solve_sparse(matrix, right, solver)
        assert np.abs(x - reference).max() <= 1e-9 * scale, solver


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_matches_a_dense_solve(solver: str) -> None:
    matrix, right = stiff_system(columns=8, rows=6, seed=2)
    expected = np.linalg.solve(matrix.toarray(), right)
    assert solve_sparse(matrix, right, solver) == pytest.approx(expected, rel=1e-9, abs=1e-18)


def test_no_solver_named_means_the_default_one() -> None:
    matrix, right = stiff_system(columns=8, rows=6)
    assert np.array_equal(
        solve_sparse(matrix, right), solve_sparse(matrix, right, DEFAULT_SOLVER)
    )
    assert np.array_equal(solve_sparse(matrix, right, None), solve_sparse(matrix, right))


@pytest.mark.parametrize("form", ["csr", "csc", "coo", "dense"])
def test_matrix_can_be_in_any_form(form: str) -> None:
    matrix, right = stiff_system(columns=8, rows=6)
    expected = solve_sparse(matrix, right)
    given = matrix.toarray() if form == "dense" else matrix.asformat(form)
    for solver in SOLVERS:
        assert solve_sparse(given, right, solver) == pytest.approx(expected, rel=1e-9, abs=1e-18)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_does_not_change_its_inputs(solver: str) -> None:
    matrix, right = stiff_system(columns=8, rows=6)
    before = matrix.copy(), right.copy()
    solve_sparse(matrix, right, solver)
    assert (matrix != before[0]).nnz == 0
    assert np.array_equal(right, before[1])


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_system_of_one_unknown(solver: str) -> None:
    assert solve_sparse(sparse.csc_matrix([[4.0]]), np.array([2.0]), solver) == pytest.approx([0.5])


def test_unknown_solver_is_rejected_with_the_names_there_are() -> None:
    matrix, right = stiff_system(columns=8, rows=6)
    with pytest.raises(ValueError, match="unknown solver 'fastest'.*superlu"):
        solve_sparse(matrix, right, "fastest")


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_singular_system_is_an_error_not_a_wrong_answer(solver: str) -> None:
    # A stiffness matrix with nothing held: any constant can be added.
    nodes, triangles = grid(6, 5)
    matrix = assemble_stiffness(nodes, triangles, 1.0)
    right = np.random.default_rng(3).normal(size=len(nodes))
    with pytest.raises(ValueError, match="singular"):
        solve_sparse(matrix, right, solver)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_matrix_of_zeros_is_an_error(solver: str) -> None:
    with pytest.raises(ValueError, match="singular"):
        solve_sparse(sparse.csc_matrix((5, 5)), np.ones(5), solver)


@pytest.mark.parametrize(
    ("matrix", "right", "message"),
    [
        (sparse.csc_matrix((3, 4)), np.ones(3), "square"),
        (sparse.identity(3, format="csc"), np.ones(4), "one value per row"),
        (sparse.identity(3, format="csc"), np.ones((3, 1)), "one value per row"),
        (sparse.identity(3, format="csc"), np.array([1.0, math.nan, 1.0]), "finite"),
        (sparse.csc_matrix(np.diag([1.0, math.inf, 1.0])), np.ones(3), "finite"),
    ],
)
def test_bad_systems_are_rejected(matrix: object, right: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve_sparse(matrix, right)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_the_check_on_the_answer_does_not_depend_on_the_size_of_the_numbers(
    solver: str, scale: float
) -> None:
    matrix, right = stiff_system(columns=8, rows=6)
    expected = solve_sparse(matrix, right, solver)
    assert solve_sparse(matrix, scale * right, solver) == pytest.approx(scale * expected, rel=1e-9)
    nodes, triangles = grid(6, 5)
    singular = assemble_stiffness(nodes, triangles, 1.0)
    with pytest.raises(ValueError, match="singular"):
        solve_sparse(singular, scale * np.random.default_rng(3).normal(size=len(nodes)), solver)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_system_with_no_unknowns_has_an_empty_answer(solver: str) -> None:
    x = solve_sparse(sparse.csc_matrix((0, 0)), np.zeros(0), solver)
    assert x.shape == (0,)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_right_hand_side_of_zeros_gives_zeros(solver: str) -> None:
    matrix, right = stiff_system(columns=8, rows=6)
    assert np.all(solve_sparse(matrix, np.zeros_like(right), solver) == 0.0)


def test_complex_numbers_are_refused() -> None:
    with pytest.raises(ValueError, match="real numbers"):
        solve_sparse(sparse.csc_matrix(np.array([[1.0 + 2.0j]])), np.ones(1))
    with pytest.raises(ValueError, match="real numbers"):
        solve_sparse(sparse.identity(2, format="csc"), np.array([1.0, 1.0j]))


@pytest.mark.parametrize("solver", [3, ["superlu"], "SuperLU", "superlu "])
def test_a_solver_name_must_be_one_of_the_names_exactly(solver: object) -> None:
    with pytest.raises(ValueError, match="unknown solver"):
        solve_sparse(sparse.identity(2, format="csc"), np.ones(2), solver)


def test_banded_solver_says_what_it_needs_of_an_indefinite_matrix() -> None:
    # Not singular, and SuperLU solves it; but it has no Cholesky factor.
    matrix = sparse.csc_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="symmetric positive definite.*singular or indefinite"):
        solve_sparse(matrix, np.ones(2), "banded")
    assert solve_sparse(matrix, np.ones(2), "superlu") == pytest.approx([1.0 / 3.0, 1.0 / 3.0])


def test_banded_solver_refuses_a_matrix_that_is_not_symmetric() -> None:
    matrix = sparse.csc_matrix(np.array([[4.0, 1.0, 0.0], [0.5, 4.0, 1.0], [0.0, 1.0, 4.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        solve_sparse(matrix, np.ones(3), "banded")
    # SuperLU takes any matrix.
    expected = np.linalg.solve(matrix.toarray(), np.ones(3))
    assert solve_sparse(matrix, np.ones(3), "superlu") == pytest.approx(expected)


# Assembly with the pattern worked out once


def test_assembler_gives_the_same_matrix_as_plain_assembly() -> None:
    nodes, triangles = grid(9, 7)
    nu = np.random.default_rng(4).uniform(100.0, 1.0 / MU0, len(triangles))
    local = element_stiffness(nodes, triangles, nu)
    expected = assemble_stiffness(nodes, triangles, nu)
    matrix = Assembler(triangles, len(nodes)).assemble(local)
    assert matrix.format == "csr" and matrix.shape == expected.shape
    assert np.array_equal(matrix.indptr, expected.indptr)
    assert np.array_equal(matrix.indices, expected.indices)
    assert matrix.data == pytest.approx(expected.data, rel=1e-12, abs=1e-12 * np.abs(expected.data).max())
    assert matrix.has_canonical_format


def test_assembler_puts_each_entry_in_its_own_row_and_column() -> None:
    # Element matrices that are not symmetric, against a plain loop.
    nodes, triangles = grid(5, 4)
    local = np.random.default_rng(6).normal(size=(len(triangles), 3, 3))
    expected = np.zeros((len(nodes), len(nodes)))
    for number, triangle in enumerate(triangles):
        for i in range(3):
            for j in range(3):
                expected[triangle[i], triangle[j]] += local[number, i, j]
    matrix = Assembler(triangles, len(nodes)).assemble(local)
    assert matrix.toarray() == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert np.abs(expected - expected.T).max() > 0.1  # the check could tell a transpose apart


def test_changing_one_assembled_matrix_does_not_spoil_the_next() -> None:
    nodes, triangles = grid(3, 3)
    assembler = Assembler(triangles, len(nodes))
    local = element_stiffness(nodes, triangles, 1.0)
    first = assembler.assemble(local)
    entries = first.nnz
    first.eliminate_zeros()  # changes the first matrix's pattern in place
    assert first.nnz < entries
    second = assembler.assemble(local)
    assert second.nnz == entries
    assert second.toarray() == pytest.approx(assemble_stiffness(nodes, triangles, 1.0).toarray())


def test_assembler_can_be_used_again_with_other_values() -> None:
    nodes, triangles = grid(5, 4)
    assembler = Assembler(triangles, len(nodes))
    first = assembler.assemble(element_stiffness(nodes, triangles, 3.0))
    second = assembler.assemble(element_stiffness(nodes, triangles, 7.0))
    assert second.data == pytest.approx(first.data * 7.0 / 3.0, rel=1e-12, abs=1e-12)
    # The first matrix is not changed by making the second.
    assert first.data == pytest.approx(assemble_stiffness(nodes, triangles, 3.0).data, abs=1e-12)
    assert np.array_equal(first.indices, second.indices)


def test_assembler_keeps_entries_that_are_zero() -> None:
    # Across a right angle the coupling is zero, but the entry stays.
    nodes, triangles = grid(3, 3)
    matrix = Assembler(triangles, len(nodes)).assemble(element_stiffness(nodes, triangles, 1.0))
    pairs = {(int(a), int(b)) for t in triangles for a in t for b in t}
    assert np.any(matrix.data == 0.0)
    assert matrix.nnz == len(pairs)


def test_assembler_leaves_an_empty_row_for_a_node_in_no_triangle() -> None:
    nodes, triangles = grid(2, 2)
    matrix = Assembler(triangles, len(nodes) + 1).assemble(element_stiffness(nodes, triangles, 1.0))
    assert matrix.shape == (10, 10)
    assert matrix[9].nnz == 0


def test_assembler_checks_what_it_is_given() -> None:
    nodes, triangles = grid(2, 2)
    assembler = Assembler(triangles, len(nodes))
    with pytest.raises(ValueError, match="one 3 by 3 matrix per triangle"):
        assembler.assemble(np.zeros((len(triangles) - 1, 3, 3)))
    with pytest.raises(ValueError, match="node 9"):
        Assembler(np.array([[0, 1, 9]]), 9)
    with pytest.raises(ValueError, match="whole numbers"):
        Assembler(np.array([[0.0, 1.0, 2.5]]), 9)
    with pytest.raises(ValueError, match=r"\(m, 3\)"):
        Assembler(np.array([[0, 1, 2, 3]]), 9)


def test_assembler_of_no_triangles_gives_an_empty_matrix() -> None:
    for none in ([], np.zeros((0, 3), dtype=int)):
        matrix = Assembler(none, 4).assemble(np.zeros((0, 3, 3)))
        assert matrix.shape == (4, 4) and matrix.nnz == 0


# The solver through the compute layer


def saturating_problem() -> Problem:
    """Steel with a coil off-centre and anti-periodic edges: two-way field, saturated."""
    columns, rows, width, height = 12, 8, 0.024, 0.016
    nodes, triangles = grid(columns, rows, width, height)
    centre = nodes[triangles].mean(axis=1)
    in_coil = (
        (centre[:, 0] > 0.004) & (centre[:, 0] < 0.010)
        & (centre[:, 1] > 0.004) & (centre[:, 1] < 0.010)
    )
    left = np.flatnonzero(nodes[:, 0] == 0.0)
    right = np.flatnonzero(np.isclose(nodes[:, 0], width))
    return Problem(
        nodes=nodes,
        triangles=triangles,
        materials=(FrohlichSteel(saturation=2.0, half_field=300.0), AIR),
        triangle_material=in_coil.astype(int),
        current_density=np.where(in_coil, 6e7, 0.0),
        fixed_nodes=np.flatnonzero((nodes[:, 1] == 0.0) | (nodes[:, 1] == height)),
        periodic=np.column_stack([right, left]),
        antiperiodic=True,
    )


def test_every_solver_gives_the_same_field() -> None:
    problem = saturating_problem()
    reference = solve(problem)
    assert reference.solver == DEFAULT_SOLVER
    assert reference.iterations >= 4
    scale = np.abs(reference.potential).max()
    for solver in SOLVERS:
        solution = solve(problem, solver=solver)
        assert solution.solver == solver
        assert solution.iterations == reference.iterations, solver
        assert np.abs(solution.potential - reference.potential).max() <= 1e-9 * scale, solver


def test_linear_solve_takes_a_solver_too() -> None:
    nodes, triangles = grid(8, 6)
    matrix = assemble_stiffness(nodes, triangles, 50.0)
    load = np.random.default_rng(5).normal(size=len(nodes))
    edge = np.flatnonzero((nodes[:, 0] == 0.0) | (nodes[:, 0] == 1.0))
    reference = solve_linear(matrix, load, edge, 0.25)
    for solver in SOLVERS:
        found = solve_linear(matrix, load, edge, 0.25, solver=solver)
        assert found == pytest.approx(reference, rel=1e-9, abs=1e-12)


def test_unknown_solver_is_rejected_before_any_work() -> None:
    with pytest.raises(ValueError, match="unknown solver 'pardiso2'"):
        solve(saturating_problem(), solver="pardiso2")
    nodes, triangles = grid(2, 2)
    with pytest.raises(ValueError, match="unknown solver 'x'"):
        solve_linear(assemble_stiffness(nodes, triangles, 1.0), np.zeros(9), [0], 0.0, solver="x")


# Timing


SECTIONS = [
    "solver start-up",
    "assemble",
    "boundary conditions",
    "linear solve",
    "materials and residual",
    "other",
]


def test_timing_sections_add_up_to_the_total() -> None:
    solution = solve(saturating_problem())
    timings = solution.timings
    assert [name for name, _ in timings.stages] == SECTIONS
    seconds = [value for _, value in timings.stages]
    assert all(value >= 0.0 for value in seconds)
    assert sum(seconds) == pytest.approx(timings.total, rel=1e-12)
    assert timings.total > 0.0


class FakeClock:
    """Stands in for the time module: the time is whatever the test says."""

    def __init__(self) -> None:
        self.now = 100.0

    def perf_counter(self) -> float:
        return self.now


@pytest.fixture
def timed_solver(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A clock that only moves inside the sparse solver, by one second a call.

    Nothing depends on how fast or how busy the machine is.
    """
    import polecraft.core.solver as solver_module

    clock = FakeClock()
    calls = {"count": 0, "clock": clock}
    function, description = compute._SOLVERS["superlu"]

    def one_second(matrix: object, right: object) -> np.ndarray:
        calls["count"] += 1
        clock.now += 1.0
        return function(matrix, right)

    monkeypatch.setitem(compute._SOLVERS, "superlu", (one_second, description))
    monkeypatch.setattr(solver_module, "time", clock)
    return calls


def test_time_in_the_solver_is_counted_as_linear_solve_and_nowhere_else(timed_solver: dict) -> None:
    solution = solve(saturating_problem(), solver="superlu")
    assert timed_solver["count"] == solution.iterations >= 4
    named = dict(solution.timings.stages)
    # One second for every call of the sparse solver, and nothing anywhere else.
    assert named["linear solve"] == pytest.approx(solution.iterations)
    for name in SECTIONS:
        if name != "linear solve":
            assert named[name] == pytest.approx(0.0, abs=1e-9), name
    assert solution.timings.total == pytest.approx(solution.iterations)


def test_time_outside_the_solver_goes_to_the_other_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    # A clock that moves by one second every time it is read: each section
    # then gets the number of readings it spans, and the total is the time
    # from the first reading to the last.
    import polecraft.core.solver as solver_module

    class Ticking:
        now = 0.0

        def perf_counter(self) -> float:
            self.now += 1.0
            return self.now

    clock = Ticking()
    monkeypatch.setattr(solver_module, "time", clock)
    solution = solve(saturating_problem(), solver="superlu")
    named = dict(solution.timings.stages)
    assert all(seconds >= 0.0 for seconds in named.values())
    assert named["assemble"] >= solution.iterations
    assert named["materials and residual"] >= solution.iterations + 1
    # The first reading was 1; the total runs to the last reading made.
    assert solution.timings.total == pytest.approx(clock.now - 1.0)


def test_a_solve_that_needs_no_iteration_still_reports_its_time() -> None:
    problem = saturating_problem()
    first = solve(problem)
    again = solve(problem, start=first.potential)
    assert again.iterations == 0
    timings = dict(again.timings.stages)
    assert list(timings) == SECTIONS
    assert timings["linear solve"] == 0.0 and timings["assemble"] == 0.0
    assert timings["boundary conditions"] == 0.0
    assert timings["solver start-up"] + timings["materials and residual"] + timings[
        "other"
    ] == pytest.approx(again.timings.total, rel=1e-12)


def test_loading_the_solver_is_timed_on_its_own_line(monkeypatch: pytest.MonkeyPatch) -> None:
    # Getting the solver ready takes three seconds on this clock, and the
    # solver itself none: the three seconds must not be booked as solve time.
    import polecraft.core.solver as solver_module

    clock = FakeClock()

    def slow_start(solver: str) -> None:
        clock.now += 3.0

    monkeypatch.setattr(compute, "prepare", slow_start)
    monkeypatch.setattr(solver_module, "time", clock)
    solution = solve(saturating_problem(), solver="superlu")
    named = dict(solution.timings.stages)
    assert named["solver start-up"] == pytest.approx(3.0)
    assert named["linear solve"] == 0.0 and named["other"] == pytest.approx(0.0, abs=1e-9)
    assert solution.timings.total == pytest.approx(3.0)


def test_only_pardiso_needs_getting_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(compute, "_pardiso", lambda matrix, right: called.append("pardiso"))
    monkeypatch.setattr(compute, "_usable", lambda name: True)
    monkeypatch.setattr(compute, "_READY", set())
    for name in BUILT_IN:
        compute.prepare(name)
    assert called == []
    compute.prepare("pardiso")
    compute.prepare("pardiso")
    assert called == ["pardiso"]  # once in a process


def test_timing_report_names_the_solver_and_every_section() -> None:
    solution = solve(saturating_problem(), solver="banded")
    lines = solution.timing_report().splitlines()
    assert lines[0] == "Timing (solver: banded)"
    timed = lines[1 : len(SECTIONS) + 2]
    assert [line.split("  ")[1].strip() for line in timed] == SECTIONS + ["total"]
    assert all(line.endswith(" s") for line in timed)


def test_timing_report_says_when_a_faster_solver_could_be_installed(
    no_optional_solvers: None,
) -> None:
    lines = solve(saturating_problem()).timing_report().splitlines()
    assert lines[0] == "Timing (solver: superlu)"
    assert lines[-1] == (
        '  a faster solver can be installed and then named: pip install "polecraft[pardiso]"'
    )
    assert len(lines) == 1 + len(SECTIONS) + 1 + 1


def test_timing_report_has_no_such_note_when_the_faster_solver_is_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compute, "_supported", lambda name: True)
    monkeypatch.setattr(compute, "_usable", lambda name: True)
    lines = solve(saturating_problem(), solver="superlu").timing_report().splitlines()
    assert len(lines) == 1 + len(SECTIONS) + 1
    assert "installed" not in lines[-1]


# The solver that is asked for is the one that runs


@pytest.mark.parametrize("solver", SOLVERS)
def test_the_named_solver_is_the_one_that_is_called(
    solver: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = []
    for name in SOLVERS:
        function, description = compute._SOLVERS[name]

        def recording(matrix: object, right: object, name: str = name, function=function):
            called.append(name)
            return function(matrix, right)

        monkeypatch.setitem(compute._SOLVERS, name, (recording, description))

    solution = solve(saturating_problem(), solver=solver)
    assert called == [solver] * solution.iterations
    called.clear()
    nodes, triangles = grid(4, 3)
    solve_linear(assemble_stiffness(nodes, triangles, 1.0), np.ones(len(nodes)), [0], 0.0, solver=solver)
    assert called == [solver]
    called.clear()
    solve_sparse(sparse.identity(3, format="csc"), np.ones(3), solver)
    assert called == [solver]


# The reference motor


@pytest.fixture(scope="module")
def motor_problem() -> Problem:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / "qdd_outer_36s42p.yaml"))
    mesh = build_mesh(machine.build_geometry())
    # Placeholders: not the design's real steel or magnet grade.
    materials = {
        "M270-35A": load_material("generic_35A_placeholder"),
        "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
    }
    return build_problem(machine, mesh, materials, current=20.0)


def test_every_solver_gives_the_same_motor_field(motor_problem: Problem) -> None:
    reference = solve(motor_problem)
    scale = np.abs(reference.potential).max()
    for solver in SOLVERS:
        solution = solve(motor_problem, solver=solver)
        # The solvers differ by rounding, which the steel's stiffness
        # magnifies; the fields agree far more closely than they are known.
        assert np.abs(solution.potential - reference.potential).max() <= 1e-7 * scale, solver
        assert solution.residuals[-1] <= 1e-10


def test_motor_timing_adds_up_and_is_mostly_accounted_for(motor_problem: Problem) -> None:
    timings = solve(motor_problem).timings
    named = dict(timings.stages)
    assert list(named) == SECTIONS
    assert sum(named.values()) == pytest.approx(timings.total, rel=1e-12)
    # On a model of this size the sparse solver takes most of the time.
    assert named["linear solve"] == max(named.values())
