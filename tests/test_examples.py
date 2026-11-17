"""The example scripts run as written."""

import subprocess
import sys
from pathlib import Path

import pytest

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_mesh_a_design_example_runs(repo_root: Path, tmp_path: Path) -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    script = repo_root / "examples" / "mesh_a_design.py"
    done = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    lines = done.stdout.splitlines()
    assert lines[0] == "36 slots, 42 poles"
    assert lines[1] == "repeats 6 times, every 60 degrees"
    assert lines[2] == "winding layout: " + "ACcbBAacCBba" * 3
    assert lines[3] == "winding factor: 0.9330"
    assert lines[4].startswith("77 regions, magnet area ")
    assert lines[5] == "2480 nodes, 4838 triangles"
    assert (tmp_path / "qdd_outer_36s42p_mesh.png").read_bytes().startswith(PNG_SIGNATURE)


def test_readme_shows_the_calls_the_example_uses(repo_root: Path) -> None:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    for call in ("load_design(", "machine_for(", "build_geometry(", "build_mesh(", "plot_mesh("):
        assert call in readme, call
        assert call in (repo_root / "examples" / "mesh_a_design.py").read_text(encoding="utf-8")


def test_mesh_a_design_example_explains_a_missing_gmsh(repo_root: Path, tmp_path: Path) -> None:
    script = repo_root / "examples" / "mesh_a_design.py"
    code = (
        "import runpy, sys; sys.modules['gmsh'] = None; "
        "runpy.run_path(sys.argv[1], run_name='__main__')"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert done.stderr.strip().splitlines() == [
        "meshing needs the optional Gmsh package, which is not installed; "
        'install it with: pip install "polecraft[mesh]"'
    ]
    # What does not need a mesh is still printed.
    assert done.stdout.splitlines()[0] == "36 slots, 42 poles"
    assert list(tmp_path.iterdir()) == []
