"""Shared fixtures for the design-file tests."""

import copy
from pathlib import Path
from typing import Any, Callable

import pytest

import polecraft
from polecraft.design import design_from_dict
from polecraft.validation import DesignError

REPO_ROOT = Path(__file__).resolve().parents[1]

# A complete, valid design. Values differ from the shipped reference design on
# purpose, so tests cannot pass by reading the wrong file.
VALID_DESIGN: dict[str, Any] = {
    "format_version": 1,
    "type": "qdd_outer",
    "slots": 12,
    "poles": 14,
    "geometry": {
        "air_gap_mm": 0.5,
        "rotor": {"outer_diameter_mm": 60.0, "yoke_thickness_mm": 2.0},
        "magnet": {
            "thickness_mm": 2.0,
            "arc_fraction": 0.8,
            "magnetisation": "radial",
        },
        "stator": {
            "tooth_width_mm": 5.0,
            "slot_opening_mm": 2.0,
            "tooth_tip_height_mm": 0.75,
            "tooth_tip_taper_mm": 0.25,
            "slot_depth_mm": 8.0,
            "yoke_thickness_mm": 3.0,
        },
    },
    "axial": {
        "stack_length_mm": 10.0,
        "magnet_overhang_mm": 1.0,
        "skew_deg": 2.0,
        "end_winding_overhang_mm": 4.0,
    },
    "magnet": {"material": "N35", "temperature_C": 60.0},
    "winding": {
        "wire_diameter_mm": 0.4,
        "insulated_diameter_mm": 0.45,
        "turns_per_coil": 30,
        "parallel_strands": 2,
        "conductor": "copper",
        "connection": "star",
        "temperature_C": 90.0,
    },
    "steel": {"stator": "M250-35A", "rotor": "S235", "stacking_factor": 0.97},
}


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def package_dir() -> Path:
    """The folder of the polecraft package the tests import.

    In a checkout that is the source tree. When the tests run from a source
    distribution against an installed wheel, which has no source tree beside
    it, it is the installed copy, and that is the one to check.
    """
    return Path(polecraft.__file__).resolve().parent


@pytest.fixture
def data_dir() -> Path:
    return REPO_ROOT / "tests" / "data"


@pytest.fixture
def valid_data() -> dict[str, Any]:
    """A fresh copy of a valid design that a test may modify."""
    return copy.deepcopy(VALID_DESIGN)


def _parent(data: dict[str, Any], path: str) -> tuple[dict[str, Any], str]:
    *sections, key = path.split(".")
    for name in sections:
        data = data[name]
    return data, key


@pytest.fixture
def changed(valid_data: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    """Return the valid design with one or more dotted fields replaced."""

    def _changed(path: str, value: Any, *more: Any) -> dict[str, Any]:
        pairs = [(path, value)] + list(zip(more[::2], more[1::2]))
        for field, new in pairs:
            parent, key = _parent(valid_data, field)
            parent[key] = new
        return valid_data

    return _changed


@pytest.fixture
def without(valid_data: dict[str, Any]) -> Callable[[str], dict[str, Any]]:
    """Return the valid design with one dotted field or section removed."""

    def _without(path: str) -> dict[str, Any]:
        parent, key = _parent(valid_data, path)
        del parent[key]
        return valid_data

    return _without


@pytest.fixture
def problems_of() -> Callable[[Any], dict[str, str]]:
    """Load a design that must fail and return {field path: message}."""

    def _problems_of(data: Any) -> dict[str, str]:
        with pytest.raises(DesignError) as caught:
            design_from_dict(data)
        return {p.path: p.message for p in caught.value.problems}

    return _problems_of
