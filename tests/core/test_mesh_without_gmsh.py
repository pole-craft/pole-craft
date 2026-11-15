"""What the mesher does when the optional Gmsh package is not there."""

import subprocess
import sys
from pathlib import Path

import pytest

from polecraft.core.mesh import build_mesh
from polecraft.design import load_design
from polecraft.machines import machine_for


def test_the_package_does_not_load_gmsh_until_a_mesh_is_built() -> None:
    code = (
        "import sys, polecraft, polecraft.core.mesh;"
        "from polecraft.machines import machine_for;"
        "assert 'gmsh' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_gmsh_is_explained(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    design = load_design(repo_root / "designs" / "qdd_outer_36s42p.yaml")
    geometry = machine_for(design).build_geometry()
    # A None entry makes "import gmsh" fail as if it were not installed.
    monkeypatch.setitem(sys.modules, "gmsh", None)
    with pytest.raises(ImportError, match="pip install gmsh"):
        build_mesh(geometry)
