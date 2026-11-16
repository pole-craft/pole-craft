"""The polecraft command."""

import importlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import polecraft
from polecraft.cli import main

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def shipped_designs() -> list[Path]:
    return sorted((Path(__file__).resolve().parents[1] / "designs").glob("*.yaml"))


@pytest.fixture
def reference(repo_root: Path) -> Path:
    return repo_root / "designs" / "qdd_outer_36s42p.yaml"


@pytest.fixture
def needs_gmsh() -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")


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
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture, name: str, reason: str
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


# Failures


def test_design_problems_are_printed_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture
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


def test_missing_design_file_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
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
    assert output.err == (
        "polecraft: meshing needs the optional Gmsh package, which is not installed; "
        'install it with: pip install "polecraft[mesh]"\n'
    )
    assert list(tmp_path.iterdir()) == []


def test_ctrl_c_stops_quietly(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
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


def test_module_can_be_run_directly(tmp_path: Path) -> None:
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
