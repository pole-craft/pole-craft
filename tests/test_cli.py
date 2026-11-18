"""The polecraft command."""

import importlib
import re
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import numpy as np
import pytest

import polecraft
from polecraft.cli import _parser, main

INSTALL = 'pip install "polecraft[mesh]"'
NO_GMSH = (
    "polecraft: meshing needs the optional Gmsh package, which is not installed; "
    f"install it with: {INSTALL}\n"
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def shipped_designs() -> list[Path]:
    return sorted((Path(__file__).resolve().parents[1] / "designs").glob("*.yaml"))


@pytest.fixture
def reference(repo_root: Path) -> Path:
    return repo_root / "designs" / "qdd_outer_36s42p.yaml"


@pytest.fixture
def needs_gmsh() -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")


@pytest.fixture
def gmsh_or_stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """For tests of what happens before any meshing: they run without Gmsh too.

    The command checks for Gmsh first, so where it is not installed a
    stand-in takes its place. It could not mesh, and these tests never do.
    """
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError):
        stand_in = types.ModuleType("gmsh")
        stand_in.initialize = lambda: None
        monkeypatch.setitem(sys.modules, "gmsh", stand_in)


# The mesh command


def test_the_three_example_designs_are_shipped() -> None:
    assert [path.name for path in shipped_designs()] == [
        "qdd_outer_12s10p.yaml",
        "qdd_outer_24s28p.yaml",
        "qdd_outer_36s42p.yaml",
    ]


@pytest.mark.parametrize("design", shipped_designs(), ids=lambda path: path.stem)
def test_mesh_command_writes_a_picture(
    needs_gmsh: None, design: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    picture = tmp_path / "motor.png"
    assert main(["mesh", str(design), "-o", str(picture)]) == 0
    data = picture.read_bytes()
    assert data.startswith(PNG_SIGNATURE) and len(data) > 50_000

    output = capsys.readouterr()
    assert output.err == ""
    assert design.name in output.out and str(picture) in output.out
    assert "repeating slice" in output.out
    assert "nodes" in output.out and "triangles" in output.out


def test_mesh_command_reports_the_design_and_mesh(
    needs_gmsh: None, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    main(["mesh", str(reference), "-o", str(tmp_path / "motor.png")])
    out = capsys.readouterr().out
    assert "36 slots, 42 poles" in out
    assert "repeating slice: 77 regions" in out

    mesh = build_mesh(machine_for(load_design(reference)).build_geometry())
    assert f"mesh: {len(mesh.nodes)} nodes, {len(mesh.triangles)} triangles" in out
    angle = float(re.search(r"smallest triangle angle: (\d+\.\d) degrees", out).group(1))
    assert angle == pytest.approx(mesh.smallest_angles().min(), abs=0.05)
    assert 20 < angle < 60


def test_mesh_command_prints_a_timing_report(
    needs_gmsh: None, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    main(["mesh", str(reference), "-o", str(tmp_path / "motor.png")])
    lines = capsys.readouterr().out.splitlines()
    report = lines[lines.index("Timing") + 1 :]
    names = [line.split("  ")[1].strip() for line in report]
    assert names == ["load libraries", "load design", "geometry", "mesh", "picture", "total"]
    seconds = [float(line.split()[-2]) for line in report]
    assert all(line.endswith(" s") for line in report)
    assert sum(seconds[:-1]) == pytest.approx(seconds[-1], abs=0.003)


def test_picture_is_named_after_the_design_by_default(
    needs_gmsh: None, reference: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["mesh", str(reference)]) == 0
    assert (tmp_path / "qdd_outer_36s42p_mesh.png").is_file()


def test_full_machine_and_size_options(
    needs_gmsh: None, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    def triangles(*options: str) -> int:
        assert main(["mesh", str(reference), "-o", str(tmp_path / "motor.png"), *options]) == 0
        line = next(l for l in capsys.readouterr().out.splitlines() if "triangles" in l)
        return int(line.split(",")[1].split()[0])

    slice_count = triangles()
    assert triangles("--size", "2") < slice_count

    assert main(["mesh", str(reference), "--full", "-o", str(tmp_path / "motor.png")]) == 0
    out = capsys.readouterr().out
    assert "whole machine: 462 regions" in out
    whole = int(re.search(r"(\d+) triangles", out).group(1))
    assert whole == pytest.approx(6 * slice_count, rel=0.05)


def test_picture_may_be_svg_and_carries_the_title(
    needs_gmsh: None, reference: Path, tmp_path: Path
) -> None:
    assert main(["mesh", str(reference), "-o", str(tmp_path / "motor.svg")]) == 0
    text = (tmp_path / "motor.svg").read_text(encoding="utf-8")
    assert "<svg" in text
    assert "qdd_outer_36s42p.yaml: 36 slots, 42 poles, repeating slice" in text


@pytest.mark.parametrize(
    "name, reason",
    [
        ("motor", "must end in .png, .svg or .pdf"),
        ("motor.xyz", "must end in .png, .svg or .pdf"),
        ("motor.jpg", "must end in .png, .svg or .pdf"),
        ("missing/motor.png", "does not exist"),
        ("", "is a folder"),
    ],
)
def test_bad_picture_path_is_refused_before_any_work(
    gmsh_or_stand_in: None,
    reference: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    name: str,
    reason: str,
) -> None:
    assert main(["mesh", str(reference), "-o", str(tmp_path / name)]) == 1
    output = capsys.readouterr()
    assert output.out == "" and reason in output.err
    assert output.err.startswith("polecraft: ") and "Traceback" not in output.err
    assert list(tmp_path.iterdir()) == []


def test_upper_case_extension_is_accepted(
    needs_gmsh: None, reference: Path, tmp_path: Path
) -> None:
    assert main(["mesh", str(reference), "-o", str(tmp_path / "MOTOR.PNG")]) == 0
    assert (tmp_path / "MOTOR.PNG").read_bytes().startswith(PNG_SIGNATURE)


# The solve command


STAND_INS = "materials: stand-ins, not the grades the design names"


def solve_output(capsys: pytest.CaptureFixture, *arguments: str) -> str:
    assert main(["solve", *arguments]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    return output.out


def value(out: str, start: str) -> float:
    """The first number on the line that starts with some words."""
    (line,) = [line for line in out.splitlines() if line.strip().startswith(start)]
    return float(re.search(r"[-+]?\d+\.\d+", line.split(start)[1]).group())


@pytest.mark.parametrize("design", shipped_designs(), ids=lambda path: path.stem)
def test_solve_command_writes_a_picture_of_the_field(
    needs_gmsh: None, design: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    picture = tmp_path / "field.png"
    out = solve_output(capsys, str(design), "--current", "5", "-o", str(picture))
    data = picture.read_bytes()
    assert data.startswith(PNG_SIGNATURE) and len(data) > 50_000
    assert design.name in out and f"picture: {picture}" in out
    assert "repeating slice" in out and "converged in" in out
    assert value(out, "torque:") > 0.05
    assert STAND_INS in out


def test_solve_command_prints_what_the_python_interface_gives(
    needs_gmsh: None, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from polecraft.core.materials import Magnet, load_material
    from polecraft.core.mesh import build_mesh
    from polecraft.core.post import flux_linkage, torque
    from polecraft.core.problem import build_problem
    from polecraft.core.solver import solve
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    out = solve_output(capsys, str(reference), "--current", "10", "-o", str(tmp_path / "f.png"))
    machine = machine_for(load_design(reference))
    mesh = build_mesh(machine.build_geometry())
    # The stand-ins the command says it uses.
    materials = {
        "M270-35A": load_material("generic_35A_placeholder"),
        "N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
    }
    problem = build_problem(machine, mesh, materials, current=10.0)
    solution = solve(problem)
    assert "36 slots, 42 poles" in out
    assert f"mesh: {len(mesh.nodes)} nodes, {len(mesh.triangles)} triangles" in out
    assert "current: 10 A peak, advance 0 electrical degrees, rotor angle 0" in out
    assert value(out, "torque:") == pytest.approx(torque(problem, solution), abs=6e-4)
    for phase, linked in flux_linkage(problem, solution).items():
        assert value(out, f"flux linkage of phase {phase}:") == pytest.approx(linked * 1e3, abs=6e-4)
    assert solution.report() in out
    steel = value(out, "flux density in the steel:")
    b = np.linalg.norm(solution.flux_density, axis=1)
    in_steel = np.array([not m.linear for m in problem.materials])[problem.triangle_material]
    assert steel == pytest.approx(np.percentile(b[in_steel], 99), abs=6e-3)
    assert 1.2 < steel < 2.5


def test_solve_command_names_the_stand_in_materials(
    needs_gmsh: None, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = solve_output(capsys, str(reference), "-o", str(tmp_path / "f.png"))
    lines = out.splitlines()
    start = lines.index(f"  {STAND_INS}")
    assert lines[start + 1] == "    steel M270-35A: generic_35A_placeholder, a curve from a formula"
    assert lines[start + 2] == "    magnet N42SH: remanence 1.2 T, relative permeability 1.05"
    assert "show how the tool behaves, not how this motor performs" in out
    assert "2D, no end effects" in out


def test_solve_command_without_current_says_why_there_is_no_torque(
    needs_gmsh: None, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = solve_output(capsys, str(reference), "-o", str(tmp_path / "f.png"))
    assert "current: 0 A peak (give --current for torque)" in out
    assert "  torque: 0.000 N m" in out.splitlines()  # no minus sign on a rounded zero


def test_solve_command_prints_a_timing_report(
    needs_gmsh: None, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = solve_output(capsys, str(reference), "-o", str(tmp_path / "f.png"))
    lines = out.splitlines()
    report = lines[lines.index("Timing") + 1 :]
    names = [line.split("  ")[1].strip() for line in report]
    assert names == [
        "load libraries", "load design", "geometry", "mesh", "solve", "results", "picture", "total",
    ]
    seconds = [float(line.split()[-2]) for line in report]
    assert sum(seconds[:-1]) == pytest.approx(seconds[-1], abs=0.004)
    assert "sparse solver: superlu" in out


def test_field_picture_is_named_after_the_design_by_default(
    needs_gmsh: None, reference: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["solve", str(reference)]) == 0
    assert [path.name for path in tmp_path.iterdir()] == ["qdd_outer_36s42p_field.png"]


def test_solve_command_options(
    needs_gmsh: None, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    picture = str(tmp_path / "f.svg")
    base = value(solve_output(capsys, str(reference), "--current", "10", "-o", picture), "torque:")
    text = (tmp_path / "f.svg").read_text(encoding="utf-8")
    assert "<svg" in text
    assert "qdd_outer_36s42p.yaml: flux density at 10 A, stand-in steel and magnet" in text

    out = solve_output(capsys, str(reference), "--current", "-10", "-o", picture)
    assert "current: -10 A peak" in out
    assert value(out, "torque:") < -0.9 * base

    # A quarter turn of advance puts the current where it makes no torque.
    out = solve_output(capsys, str(reference), "--current", "10", "--advance", "90", "-o", picture)
    assert "advance 90 electrical degrees" in out
    assert abs(value(out, "torque:")) < 0.05 * base

    out = solve_output(capsys, str(reference), "--current", "10", "--full", "-o", picture)
    assert "whole machine: 462 regions" in out
    assert value(out, "torque:") == pytest.approx(base, rel=0.005)

    def nodes(text: str) -> int:
        return int(re.search(r"mesh: (\d+) nodes", text).group(1))

    usual = nodes(solve_output(capsys, str(reference), "--current", "10", "-o", picture))
    out = solve_output(capsys, str(reference), "--current", "10", "--size", "2", "-o", picture)
    assert value(out, "torque:") == pytest.approx(base, rel=0.03)
    # Fewer nodes; how many exactly depends on the mesher's build.
    assert 0.2 * usual < nodes(out) < 0.6 * usual

    out = solve_output(capsys, str(reference), "--current", "10", "--solver", "banded", "-o", picture)
    assert "sparse solver: banded" in out
    assert value(out, "torque:") == pytest.approx(base, abs=2e-3)

    weaker = solve_output(capsys, str(reference), "--current", "10", "--remanence", "0.6", "-o", picture)
    assert "magnet N42SH: remanence 0.6 T" in weaker
    assert value(weaker, "torque:") < 0.7 * base


def test_solve_command_refuses_what_it_cannot_do(
    needs_gmsh: None,
    reference: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def meshed(*arguments: object, **keywords: object) -> None:
        pytest.fail("meshed before checking the options")

    picture = str(tmp_path / "f.png")
    for options, message in (
        (["--solver", "fastest"], "polecraft: unknown solver 'fastest'"),
        (["--size", "0"], "polecraft: size must be a number greater than 0"),
        (["--size", "100"], "polecraft: the mesh is too coarse to solve at --size 100"),
        (["--remanence", "-1"], "polecraft: remanence must be zero or more"),
        (["--current", "nan"], "polecraft: current must be a finite number"),
        (["--advance", "inf"], "polecraft: advance must be a finite number"),
    ):
        with monkeypatch.context() as patch:
            if options[0] != "--size":  # a bad size is found by the mesher itself
                patch.setattr("polecraft.core.mesh.build_mesh", meshed)
            assert main(["solve", str(reference), "-o", picture, *options]) == 1
        output = capsys.readouterr()
        assert output.err.startswith(message), options
        assert "Traceback" not in output.err
    assert list(tmp_path.iterdir()) == []


def test_solve_command_checks_the_picture_path_before_any_work(
    gmsh_or_stand_in: None,
    reference: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(
            "polecraft.cli.load_design", lambda path: pytest.fail("worked before checking the path")
        )
        assert main(["solve", str(reference), "-o", str(tmp_path / "field.jpg")]) == 1
    output = capsys.readouterr()
    assert output.out == "" and "must end in .png, .svg or .pdf" in output.err
    assert main(["solve", str(tmp_path / "nowhere.yaml")]) == 1
    assert "design file not found" in capsys.readouterr().err


def test_a_solve_that_does_not_converge_is_reported_in_one_line(
    needs_gmsh: None,
    reference: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polecraft.core import solver

    def fails(problem, **options):
        raise solver.ConvergenceError("no convergence in 30 iterations", (1.0,), (1.0,))

    monkeypatch.setattr(solver, "solve", fails)
    assert main(["solve", str(reference), "-o", str(tmp_path / "f.png")]) == 1
    error = capsys.readouterr().err
    assert error.startswith("polecraft: no convergence in 30 iterations\n")
    assert "Traceback" not in error
    assert list(tmp_path.iterdir()) == []


def test_a_solver_that_is_not_installed_is_reported_in_one_plain_line(
    needs_gmsh: None,
    reference: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polecraft.core import compute

    # As on a machine PARDISO exists for, but without its package.
    monkeypatch.setattr(compute, "_supported", lambda name: True)
    monkeypatch.setattr(compute, "_usable", lambda name: name not in compute._EXTRAS)
    assert main(["solve", str(reference), "--solver", "pardiso", "-o", str(tmp_path / "f.png")]) == 0
    output = capsys.readouterr()
    assert "sparse solver: superlu" in output.out
    assert output.err == (
        "polecraft: solver 'pardiso' is not installed, so 'superlu' is used instead; "
        'to install it: pip install "polecraft[pardiso]"\n'
    )


def test_stator_and_rotor_may_be_of_different_steels(
    needs_gmsh: None, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    text = reference.read_text(encoding="utf-8")
    assert "  rotor: M270-35A" in text
    design = tmp_path / "two_steels.yaml"
    design.write_text(text.replace("  rotor: M270-35A", "  rotor: OtherSteel"), encoding="utf-8")
    out = solve_output(capsys, str(design), "--current", "10", "-o", str(tmp_path / "f.png"))
    assert "    steel M270-35A: generic_35A_placeholder, a curve from a formula" in out
    assert "    steel OtherSteel: generic_35A_placeholder, a curve from a formula" in out
    assert value(out, "torque:") == pytest.approx(3.187, abs=2e-3)


def test_a_magnet_named_like_a_steel_is_refused(
    needs_gmsh: None, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    text = reference.read_text(encoding="utf-8")
    assert "  material: N42SH" in text
    design = tmp_path / "same_name.yaml"
    design.write_text(text.replace("  material: N42SH", "  material: M270-35A"), encoding="utf-8")
    assert main(["solve", str(design), "-o", str(tmp_path / "f.png")]) == 1
    error = capsys.readouterr().err
    assert error == (
        "polecraft: M270-35A is named as the magnet and as a steel; give them different names\n"
    )
    assert [path.name for path in tmp_path.iterdir()] == ["same_name.yaml"]


def test_the_solve_command_draws_the_outlines_of_the_regions_it_solved(
    needs_gmsh: None,
    reference: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polecraft.plot as plot

    seen = {}
    real = plot.plot_field

    def recording(problem, solution, path, **keywords):
        seen.update(keywords)
        return real(problem, solution, path, **keywords)

    monkeypatch.setattr(plot, "plot_field", recording)
    assert main(["solve", str(reference), "--full", "-o", str(tmp_path / "f.png")]) == 0
    assert len(seen["outlines"].regions) == 462  # the whole machine, as solved
    assert "stand-in steel and magnet" in seen["title"]


def test_a_fault_in_the_program_is_not_dressed_up_as_a_message(
    gmsh_or_stand_in: None, reference: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path) -> None:
        raise NotImplementedError

    monkeypatch.setattr("polecraft.cli.load_design", broken)
    with pytest.raises(NotImplementedError):
        main(["solve", str(reference), "-o", str(tmp_path / "f.png")])


def test_help_describes_the_solve_command(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["solve", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    for option in ("--output", "--current", "--advance", "--full", "--size", "--solver", "--remanence"):
        assert option in text
    assert f"Needs Gmsh: {INSTALL}" in text
    assert "the steel and the magnet are stand-ins" in text  # said before anyone runs it
    assert "<design name>_field.png" in text
    assert "rotor angle 0" in text


def test_solve_command_without_gmsh_never_shows_a_traceback(reference: Path, tmp_path: Path) -> None:
    done = run_without_gmsh("solve", str(reference), "-o", str(tmp_path / "f.png"))
    assert done.returncode == 1
    assert done.stdout == ""
    assert done.stderr == NO_GMSH
    assert list(tmp_path.iterdir()) == []
    # Gmsh comes before everything, a bad picture name included.
    bad = run_without_gmsh("solve", str(reference), "-o", str(tmp_path / "f.jpg"))
    assert bad.returncode == 1 and bad.stderr == NO_GMSH
    shown = run_without_gmsh("solve", "--help")
    assert shown.returncode == 0 and "usage: polecraft solve" in shown.stdout


# Failures


def test_design_problems_are_printed_without_a_traceback(
    gmsh_or_stand_in: None, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    design = tmp_path / "bad.yaml"
    design.write_text("type: qdd_outer\nslots: 36\npoles: 41\n", encoding="utf-8")
    assert main(["mesh", str(design)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith(f"{design}: ")
    assert "poles: must be even" in output.err
    assert "Traceback" not in output.err
    assert not (tmp_path / "bad_mesh.png").exists()


def test_missing_design_file_is_reported(
    gmsh_or_stand_in: None, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["mesh", str(tmp_path / "nowhere.yaml")]) == 1
    assert "design file not found" in capsys.readouterr().err


def test_bad_size_is_reported(
    needs_gmsh: None, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["mesh", str(reference), "--size", "0", "-o", str(tmp_path / "m.png")]) == 1
    assert "polecraft: size must be a number greater than 0" in capsys.readouterr().err


def test_missing_gmsh_is_reported(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "gmsh", None)
    assert main(["mesh", str(reference), "-o", str(tmp_path / "m.png")]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == NO_GMSH
    assert list(tmp_path.iterdir()) == []


def test_missing_gmsh_is_reported_before_any_work_is_done(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Even a design file that does not exist comes second: without Gmsh the
    # command cannot do anything, so that is what the user is told.
    monkeypatch.setitem(sys.modules, "gmsh", None)
    assert main(["mesh", str(tmp_path / "nowhere.yaml")]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == NO_GMSH


def test_gmsh_that_cannot_load_is_reported_in_one_line(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def broken_import(name: str, *arguments, **keywords):
        if name == "gmsh":
            raise OSError("libGLU.so.1: cannot open shared object file")
        return real_import(name, *arguments, **keywords)

    monkeypatch.delitem(sys.modules, "gmsh", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken_import)
    assert main(["mesh", str(reference), "-o", str(tmp_path / "m.png")]) == 1
    error = capsys.readouterr().err
    assert error.count("\n") == 1
    assert error.startswith("polecraft: Gmsh is installed but could not be loaded (libGLU.so.1")
    assert "README" in error
    assert list(tmp_path.iterdir()) == []


def run_without_gmsh(*arguments: str) -> subprocess.CompletedProcess:
    """Run the real command in a fresh interpreter in which Gmsh cannot be imported."""
    code = (
        "import sys; sys.modules['gmsh'] = None; "
        "from polecraft.cli import main; sys.exit(main(sys.argv[1:]))"
    )
    return subprocess.run(
        [sys.executable, "-c", code, *arguments], capture_output=True, text=True
    )


def test_mesh_command_without_gmsh_never_shows_a_traceback(
    reference: Path, tmp_path: Path
) -> None:
    done = run_without_gmsh("mesh", str(reference), "-o", str(tmp_path / "m.png"))
    assert done.returncode == 1
    assert done.stdout == ""
    assert done.stderr == NO_GMSH
    assert "Traceback" not in done.stderr
    assert list(tmp_path.iterdir()) == []


def test_version_and_help_work_without_gmsh() -> None:
    version = run_without_gmsh("--version")
    assert version.returncode == 0
    assert version.stdout.strip() == f"polecraft {polecraft.__version__}"
    for arguments in (["--help"], ["mesh", "--help"], ["solve", "--help"]):
        shown = run_without_gmsh(*arguments)
        assert shown.returncode == 0, shown.stderr
        assert "usage: polecraft" in shown.stdout
        assert shown.stderr == ""


# Every command, with arguments to run it by, and whether it needs Gmsh.
COMMANDS = {
    "mesh": (["mesh", "nowhere.yaml"], True),
    "solve": (["solve", "nowhere.yaml"], True),
}


def test_every_command_is_listed_with_its_need_for_gmsh() -> None:
    # A new command must be added to the table above, so that the test below
    # holds it to the same message.
    actions = [a for a in _parser()._actions if isinstance(a.choices, dict)]
    assert set(actions[0].choices) == set(COMMANDS)


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_every_command_that_needs_gmsh_says_so_when_it_is_missing(
    command: str, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, needs_gmsh = COMMANDS[command]
    monkeypatch.setitem(sys.modules, "gmsh", None)
    code = main(arguments)
    error = capsys.readouterr().err
    assert (error == NO_GMSH) == needs_gmsh
    if needs_gmsh:
        assert code == 1
        with pytest.raises(SystemExit):
            main([command, "--help"])
        assert f"Needs Gmsh: {INSTALL}" in " ".join(capsys.readouterr().out.split())


def test_a_folder_named_gmsh_is_not_taken_for_gmsh(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Python imports an empty folder called gmsh as a module with nothing in it.
    monkeypatch.setitem(sys.modules, "gmsh", types.ModuleType("gmsh"))
    assert main(["mesh", str(reference), "-o", str(tmp_path / "m.png")]) == 1
    assert capsys.readouterr().err == NO_GMSH
    assert list(tmp_path.iterdir()) == []


def test_ctrl_c_stops_quietly(
    gmsh_or_stand_in: None,
    reference: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(path: Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("polecraft.cli.load_design", interrupted)
    assert main(["mesh", str(reference), "-o", str(tmp_path / "m.png")]) == 130
    assert capsys.readouterr().err == "polecraft: interrupted\n"


# Command-line basics


def test_a_command_is_required(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 2
    assert "usage: polecraft" in capsys.readouterr().err


def test_version_option(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.strip() == f"polecraft {polecraft.__version__}"


def test_help_describes_the_mesh_command(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["mesh", "--help"])
    text = capsys.readouterr().out
    for option in ("--output", "--full", "--size"):
        assert option in text
    # Line breaks in the help text fall where the terminal width puts them.
    assert f"Needs Gmsh: {INSTALL}" in " ".join(text.split())


def test_top_level_help_says_which_commands_need_gmsh(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "needs Gmsh" in " ".join(capsys.readouterr().out.split())


def test_module_can_be_run_directly(needs_gmsh: None, tmp_path: Path) -> None:
    def run(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "polecraft", *arguments], capture_output=True, text=True
        )

    version = run("--version")
    assert version.returncode == 0
    assert version.stdout.strip() == f"polecraft {polecraft.__version__}"

    failed = run("mesh", str(tmp_path / "nowhere.yaml"))
    assert failed.returncode == 1
    assert "design file not found" in failed.stderr


def test_command_is_declared_in_pyproject(repo_root: Path) -> None:
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {"polecraft": "polecraft.cli:main"}
    module, function = project["project"]["scripts"]["polecraft"].split(":")
    assert getattr(importlib.import_module(module), function) is main
