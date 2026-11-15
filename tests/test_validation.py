"""Field reading and error reporting, independent of any motor."""

import math

import numpy as np
import pytest

from polecraft.validation import DesignError, Problem, Section


def read(data: dict) -> tuple[Section, list[Problem]]:
    problems: list[Problem] = []
    return Section(data, "", problems), problems


# Numbers


def test_number_accepts_int_and_returns_float() -> None:
    section, problems = read({"x": 3})
    value = section.number("x")
    assert value == 3.0 and type(value) is float
    assert problems == []


def test_number_accepts_numpy_scalars() -> None:
    section, problems = read({"x": np.float64(0.5), "n": np.int64(4)})
    assert section.number("x") == 0.5
    assert section.integer("n") == 4 and type(section.integer("n")) is int
    assert problems == []


@pytest.mark.parametrize("bad", [True, False])
def test_number_rejects_booleans(bad: bool) -> None:
    section, problems = read({"x": bad})
    assert section.number("x") is None
    assert "must be a number" in problems[0].message


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_number_rejects_nan_and_infinity(bad: float) -> None:
    section, problems = read({"x": bad})
    assert section.number("x") is None
    assert "finite" in problems[0].message


def test_number_as_text_is_rejected_with_a_hint() -> None:
    # PyYAML reads an unquoted 1e-3 as text, so this is easy to hit.
    section, problems = read({"x": "1e-3"})
    assert section.number("x") is None
    assert "0.001" in problems[0].message


def test_text_that_is_not_a_number_is_rejected() -> None:
    section, problems = read({"x": "thick"})
    assert section.number("x") is None
    assert "must be a number" in problems[0].message


@pytest.mark.parametrize(
    "bounds, value, ok",
    [
        ({"gt": 0}, 0.0, False),
        ({"gt": 0}, -0.0, False),
        ({"gt": 0}, 1e-9, True),
        ({"ge": 0}, 0.0, True),
        ({"ge": 0}, -1e-9, False),
        ({"gt": 0, "le": 1}, 1.0, True),
        ({"gt": 0, "le": 1}, 1.0001, False),
        ({"lt": 5}, 5.0, False),
    ],
)
def test_number_bounds(bounds: dict, value: float, ok: bool) -> None:
    section, problems = read({"x": value})
    result = section.number("x", **bounds)
    assert (result is not None) == ok
    assert (problems == []) == ok


def test_bound_message_states_the_limit_and_the_value() -> None:
    section, problems = read({"x": 1.2})
    section.number("x", gt=0, le=1)
    assert problems[0].message == "must be greater than 0 and at most 1, got 1.2"


def test_negative_zero_is_stored_as_plain_zero() -> None:
    section, _ = read({"x": -0.0})
    assert math.copysign(1.0, section.number("x", ge=0)) == 1.0


# Whole numbers


def test_integer_accepts_whole_float() -> None:
    section, problems = read({"n": 42.0})
    value = section.integer("n")
    assert value == 42 and type(value) is int
    assert problems == []


@pytest.mark.parametrize("bad", [36.5, True, "36", math.nan, [36]])
def test_integer_rejects_non_whole_values(bad: object) -> None:
    section, problems = read({"n": bad})
    assert section.integer("n") is None
    assert "whole number" in problems[0].message


def test_integer_lower_bound() -> None:
    section, problems = read({"n": 0})
    assert section.integer("n", ge=1) is None
    assert problems[0].message == "must be at least 1, got 0"


# Text


def test_text_is_stripped() -> None:
    section, _ = read({"name": "  N42SH "})
    assert section.text("name") == "N42SH"


@pytest.mark.parametrize("bad", [1010, 3.5, True, "", "   ", ["a"]])
def test_text_rejects_non_text_and_blank(bad: object) -> None:
    section, problems = read({"name": bad})
    assert section.text("name") is None
    assert len(problems) == 1


def test_text_choices_are_listed() -> None:
    section, problems = read({"connection": "wye"})
    assert section.text("connection", choices=("star", "delta")) is None
    assert problems[0].message == "must be one of star, delta; got 'wye'"


# Missing, empty, defaults


def test_missing_required_field_is_named() -> None:
    section, problems = read({})
    assert section.number("air_gap_mm") is None
    assert problems == [Problem("air_gap_mm", "required field is missing")]


def test_missing_optional_field_uses_default() -> None:
    section, problems = read({})
    assert section.number("skew_deg", default=0.0) == 0.0
    assert section.integer("strands", default=1) == 1
    assert section.text("mode", default="radial") == "radial"
    assert section.number("overhang_mm", default=None) is None
    assert problems == []


def test_empty_value_is_reported_as_empty_not_missing() -> None:
    section, problems = read({"x": None})
    assert section.number("x") is None
    assert "empty" in problems[0].message


def test_empty_value_is_allowed_when_the_default_is_none() -> None:
    section, problems = read({"x": None})
    assert section.number("x", default=None) is None
    assert problems == []


# Sections and unknown fields


def test_nested_paths_are_dotted() -> None:
    section, problems = read({"geometry": {"rotor": {}}})
    section.section("geometry").section("rotor").number("outer_diameter_mm")
    assert problems[0].path == "geometry.rotor.outer_diameter_mm"


def test_missing_section_is_reported_once() -> None:
    section, problems = read({})
    child = section.section("winding")
    assert child.number("wire_diameter_mm") is None
    assert child.integer("turns_per_coil") is None
    section.finish()
    assert problems == [Problem("winding", "required section is missing")]


@pytest.mark.parametrize("bad", [2.5, "text", [1, 2], None])
def test_section_given_as_a_value_is_rejected(bad: object) -> None:
    section, problems = read({"magnet": bad})
    child = section.section("magnet")
    assert child.number("thickness_mm") is None
    assert len(problems) == 1 and problems[0].path == "magnet"


def test_unknown_field_suggests_the_closest_name() -> None:
    section, problems = read({"turns_per_col": 20})
    section.integer("turns_per_coil", default=1)
    section.finish()
    assert problems == [
        Problem("turns_per_col", "unknown field (did you mean 'turns_per_coil'?)")
    ]


def test_field_names_are_case_sensitive() -> None:
    section, problems = read({"Poles": 42})
    section.integer("poles")
    section.finish()
    paths = [p.path for p in problems]
    assert paths == ["poles", "Poles"]
    assert "did you mean 'poles'" in problems[1].message


def test_finish_checks_nested_sections() -> None:
    section, problems = read({"steel": {"stator": "M270-35A", "rottor": "x"}})
    section.section("steel").text("stator")
    section.finish()
    assert [p.path for p in problems] == ["steel.rottor"]


def test_ignored_section_is_not_checked() -> None:
    section, problems = read({"geometry": {"anything": 1}})
    section.ignore("geometry")
    section.finish()
    assert problems == []


# The error itself


def test_error_with_one_problem_is_a_single_line() -> None:
    error = DesignError([Problem("poles", "must be even, got 41")], "bad.yaml")
    assert str(error) == "bad.yaml: poles: must be even, got 41"


def test_error_lists_every_problem() -> None:
    error = DesignError(
        [Problem("poles", "must be even, got 41"), Problem("", "file is odd")],
        "bad.yaml",
    )
    assert str(error) == (
        "bad.yaml: 2 problems\n  poles: must be even, got 41\n  file is odd"
    )
    assert isinstance(error, ValueError)


def test_error_without_a_source() -> None:
    error = DesignError([Problem("slots", "required field is missing")])
    assert str(error) == "design: slots: required field is missing"
