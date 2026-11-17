"""What the mesher does when the optional Gmsh package is not there."""

import subprocess
import sys
from pathlib import Path

import pytest

from polecraft.core.mesh import build_mesh, require_gmsh
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
    with pytest.raises(ImportError, match=r'not installed; install it with: pip install "polecraft\[mesh\]"'):
        build_mesh(geometry)


def test_gmsh_that_cannot_load_its_library_is_explained(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Linux, importing Gmsh fails with OSError when a system library is absent."""
    import builtins

    design = load_design(repo_root / "designs" / "qdd_outer_36s42p.yaml")
    geometry = machine_for(design).build_geometry()
    real_import = builtins.__import__

    def broken_import(name: str, *arguments, **keywords):
        if name == "gmsh":
            raise OSError("libGLU.so.1: cannot open shared object file")
        return real_import(name, *arguments, **keywords)

    monkeypatch.delitem(sys.modules, "gmsh", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(ImportError, match="installed but could not be loaded") as caught:
        build_mesh(geometry)
    assert "libGLU.so.1" in str(caught.value) and "README" in str(caught.value)


def test_require_gmsh_gives_the_module_when_it_is_there() -> None:
    gmsh = pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    assert require_gmsh() is gmsh


def test_require_gmsh_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gmsh", None)
    with pytest.raises(ImportError) as caught:
        require_gmsh()
    assert str(caught.value) == (
        "meshing needs the optional Gmsh package, which is not installed; "
        'install it with: pip install "polecraft[mesh]"'
    )
