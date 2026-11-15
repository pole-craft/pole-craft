"""The package installs, imports and reports a version."""

import importlib
import re
from importlib import metadata

import pytest

import polecraft

SUBPACKAGES = ["core", "machines", "studies", "data"]


def test_version_is_a_version_string() -> None:
    assert isinstance(polecraft.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+([.\-+a-z0-9]*)?", polecraft.__version__)


def test_version_matches_installed_metadata() -> None:
    """The version in the code and the one pip installed must be the same."""
    assert metadata.version("polecraft") == polecraft.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"polecraft.{name}")


def test_main_entry_points_are_exported() -> None:
    for name in ("load_design", "design_from_dict", "machine_for", "DesignError"):
        assert name in polecraft.__all__
        assert callable(getattr(polecraft, name))
