"""Loading design files into typed design objects."""

import copy
import dataclasses
import pickle
from pathlib import Path

import pytest

import polecraft
from polecraft.design import Design, design_from_dict, design_to_dict, load_design
from polecraft.machines.qdd_outer import QddOuterParameters
from polecraft.validation import DesignError

REQUIRED_FIELDS = [
    "type",
    "slots",
    "poles",
    "geometry.air_gap_mm",
    "geometry.rotor.outer_diameter_mm",
    "geometry.rotor.yoke_thickness_mm",
    "geometry.magnet.thickness_mm",
    "geometry.magnet.arc_fraction",
    "geometry.stator.tooth_width_mm",
    "geometry.stator.slot_opening_mm",
    "geometry.stator.tooth_tip_height_mm",
    "geometry.stator.tooth_tip_taper_mm",
    "geometry.stator.slot_depth_mm",
    "geometry.stator.yoke_thickness_mm",
    "axial.stack_length_mm",
    "magnet.material",
    "magnet.temperature_C",
    "winding.wire_diameter_mm",
    "winding.turns_per_coil",
    "winding.conductor",
    "winding.connection",
    "winding.temperature_C",
    "steel.stator",
    "steel.rotor",
]

SECTIONS = [
    "geometry",
    "geometry.rotor",
    "geometry.magnet",
    "geometry.stator",
    "axial",
    "magnet",
    "winding",
    "steel",
]


def write(tmp_path: Path, text: str | bytes) -> Path:
    path = tmp_path / "design.yaml"
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def load_error(path: Path) -> DesignError:
    with pytest.raises(DesignError) as caught:
        load_design(path)
    return caught.value


# Valid designs


def test_valid_file_loads_expected_values(data_dir: Path) -> None:
    design = load_design(data_dir / "valid_design.yaml")

    assert design.type == "qdd_outer"
    assert (design.slots, design.poles) == (12, 14)
    assert design.format_version == 1

    geometry = design.geometry
    assert isinstance(geometry, QddOuterParameters)
    assert geometry.air_gap_mm == 0.5
    assert geometry.rotor.outer_diameter_mm == 60.0
    assert geometry.rotor.yoke_thickness_mm == 2.0
    assert geometry.magnet.thickness_mm == 2.0
    assert geometry.magnet.arc_fraction == 0.8
    assert geometry.magnet.magnetisation == "radial"
    assert geometry.stator.tooth_width_mm == 5.0
    assert geometry.stator.slot_opening_mm == 2.0
    assert geometry.stator.tooth_tip_height_mm == 0.75
    assert geometry.stator.tooth_tip_taper_mm == 0.25
    assert geometry.stator.slot_depth_mm == 8.0
    assert geometry.stator.yoke_thickness_mm == 3.0

    assert design.axial.stack_length_mm == 10.0
    assert design.axial.magnet_overhang_mm == 1.0
    assert design.axial.skew_deg == 2.0
    assert design.axial.end_winding_overhang_mm == 4.0

    assert design.magnet.material == "N35"
    assert design.magnet.temperature_C == 60.0

    assert design.winding.wire_diameter_mm == 0.4
    assert design.winding.turns_per_coil == 30
    assert design.winding.parallel_strands == 2
    assert design.winding.conductor == "copper"
    assert design.winding.connection == "delta"
    assert design.winding.temperature_C == 90.0

    assert design.steel.stator == "M250-35A"
    assert design.steel.rotor == "S235"
    assert design.steel.stacking_factor == 0.97


def test_file_and_dict_give_the_same_design(data_dir: Path, valid_data: dict) -> None:
    assert load_design(data_dir / "valid_design.yaml") == design_from_dict(valid_data)


def test_values_have_the_expected_types(data_dir: Path) -> None:
    # The file writes "outer_diameter_mm: 60" and "temperature_C: 60" as integers.
    design = load_design(data_dir / "valid_design.yaml")
    assert type(design.geometry.rotor.outer_diameter_mm) is float
    assert type(design.magnet.temperature_C) is float
    assert type(design.slots) is int
    assert type(design.winding.turns_per_coil) is int


def test_design_cannot_be_modified(valid_data: dict) -> None:
    design = design_from_dict(valid_data)
    with pytest.raises(dataclasses.FrozenInstanceError):
        design.slots = 24  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        design.geometry.air_gap_mm = 1.0


def test_optional_fields_get_defaults(without) -> None:
    for path in [
        "format_version",
        "geometry.magnet.magnetisation",
        "axial.magnet_overhang_mm",
        "axial.skew_deg",
        "axial.end_winding_overhang_mm",
        "winding.parallel_strands",
        "steel.stacking_factor",
    ]:
        data = without(path)
    design = design_from_dict(data)

    assert design.format_version == 1
    assert design.geometry.magnet.magnetisation == "parallel"
    assert design.axial.magnet_overhang_mm == 0.0
    assert design.axial.skew_deg == 0.0
    assert design.axial.end_winding_overhang_mm is None
    assert design.winding.parallel_strands == 1
    assert design.steel.stacking_factor == 0.95


def test_design_survives_a_round_trip_through_a_dict(valid_data: dict) -> None:
    design = design_from_dict(valid_data)
    as_dict = design_to_dict(design)
    assert as_dict == valid_data
    assert design_from_dict(as_dict) == design


def test_loading_does_not_change_the_input_dict(valid_data: dict, without) -> None:
    for path in ["winding.parallel_strands", "axial.skew_deg", "format_version"]:
        data = without(path)
    before = copy.deepcopy(data)
    design_from_dict(data)
    assert data == before


def test_loader_is_available_from_the_package(data_dir: Path) -> None:
    design = polecraft.load_design(data_dir / "valid_design.yaml")
    assert isinstance(design, Design)
    assert polecraft.DesignError is DesignError


def test_path_may_be_a_string(data_dir: Path) -> None:
    assert load_design(str(data_dir / "valid_design.yaml")).slots == 12


# Shipped designs


def shipped_designs() -> list[Path]:
    return sorted((Path(__file__).resolve().parents[1] / "designs").glob("*.yaml"))


def test_there_is_at_least_one_shipped_design() -> None:
    assert shipped_designs()


@pytest.mark.parametrize("path", shipped_designs(), ids=lambda p: p.name)
def test_shipped_design_loads(path: Path) -> None:
    assert isinstance(load_design(path), Design)


def test_reference_design_values(repo_root: Path) -> None:
    design = load_design(repo_root / "designs" / "qdd_outer_36s42p.yaml")
    assert design.type == "qdd_outer"
    assert (design.slots, design.poles) == (36, 42)
    assert design.geometry.rotor.outer_diameter_mm == 90.0
    assert design.geometry.air_gap_mm == 0.6
    assert design.geometry.magnet.thickness_mm == 2.5
    assert design.geometry.magnet.arc_fraction == 0.85
    assert design.axial.stack_length_mm == 12.0
    assert design.magnet.material == "N42SH"
    assert design.winding.turns_per_coil == 20
    assert design.winding.connection == "star"
    assert design.steel.stator == "M270-35A"


# Missing fields


@pytest.mark.parametrize("path", REQUIRED_FIELDS)
def test_missing_required_field_is_named(path: str, without, problems_of) -> None:
    assert problems_of(without(path)) == {path: "required field is missing"}


@pytest.mark.parametrize("path", SECTIONS)
def test_missing_section_is_named(path: str, without, problems_of) -> None:
    assert problems_of(without(path)) == {path: "required section is missing"}


# Impossible values


@pytest.mark.parametrize("gap", [0, 0.0, -0.0, -0.6])
def test_zero_or_negative_air_gap_is_rejected(gap, changed, problems_of) -> None:
    problems = problems_of(changed("geometry.air_gap_mm", gap))
    assert list(problems) == ["geometry.air_gap_mm"]
    assert "greater than 0" in problems["geometry.air_gap_mm"]


@pytest.mark.parametrize("poles", [41, 13, 1])
def test_odd_pole_count_is_rejected(poles, changed, problems_of) -> None:
    problems = problems_of(changed("poles", poles))
    assert list(problems) == ["poles"]
    assert "even" in problems["poles"] or "at least" in problems["poles"]


@pytest.mark.parametrize("poles", [0, -2])
def test_pole_count_must_be_positive(poles, changed, problems_of) -> None:
    assert list(problems_of(changed("poles", poles))) == ["poles"]


@pytest.mark.parametrize("slots", [0, 2, -12, 12.5])
def test_bad_slot_count_is_rejected(slots, changed, problems_of) -> None:
    assert list(problems_of(changed("slots", slots))) == ["slots"]


def test_smallest_slot_and_pole_counts_are_accepted(changed) -> None:
    design = design_from_dict(changed("slots", 3, "poles", 2))
    assert (design.slots, design.poles) == (3, 2)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.01, 2])
def test_arc_fraction_outside_0_to_1_is_rejected(fraction, changed, problems_of) -> None:
    problems = problems_of(changed("geometry.magnet.arc_fraction", fraction))
    assert list(problems) == ["geometry.magnet.arc_fraction"]


def test_arc_fraction_of_one_is_a_full_ring_and_allowed(changed) -> None:
    design = design_from_dict(changed("geometry.magnet.arc_fraction", 1))
    assert design.geometry.magnet.arc_fraction == 1.0


def test_magnet_thicker_than_the_rotor_allows_is_rejected(changed, problems_of) -> None:
    # Outer radius 30 mm, rotor yoke 2 mm: nothing thicker than 28 mm can fit.
    problems = problems_of(changed("geometry.magnet.thickness_mm", 40))
    assert list(problems) == ["geometry.magnet.thickness_mm"]
    message = problems["geometry.magnet.thickness_mm"]
    assert "40" in message and "28" in message


def test_magnet_that_fills_the_rotor_completely_is_rejected(changed, problems_of) -> None:
    problems = problems_of(changed("geometry.magnet.thickness_mm", 28))
    assert list(problems) == ["geometry.magnet.thickness_mm"]


def test_unknown_machine_type_lists_the_known_ones(changed, problems_of) -> None:
    problems = problems_of(changed("type", "qdd_outter"))
    assert list(problems) == ["type"]
    assert "unknown machine type 'qdd_outter'" in problems["type"]
    assert "did you mean 'qdd_outer'" in problems["type"]
    assert "qdd_outer" in problems["type"].split("known types:")[1]


@pytest.mark.parametrize(
    "path",
    [
        "geometry.rotor.outer_diameter_mm",
        "geometry.rotor.yoke_thickness_mm",
        "geometry.magnet.thickness_mm",
        "geometry.stator.tooth_width_mm",
        "geometry.stator.slot_opening_mm",
        "geometry.stator.tooth_tip_height_mm",
        "geometry.stator.slot_depth_mm",
        "geometry.stator.yoke_thickness_mm",
        "axial.stack_length_mm",
        "winding.wire_diameter_mm",
        "winding.turns_per_coil",
        "winding.parallel_strands",
        "steel.stacking_factor",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_sizes_and_counts_must_be_positive(path, value, changed, problems_of) -> None:
    assert list(problems_of(changed(path, value))) == [path]


@pytest.mark.parametrize(
    "path",
    [
        "geometry.stator.tooth_tip_taper_mm",
        "axial.magnet_overhang_mm",
        "axial.skew_deg",
        "axial.end_winding_overhang_mm",
    ],
)
def test_fields_that_may_be_zero_but_not_negative(path, changed, problems_of) -> None:
    design_from_dict(changed(path, 0))
    assert list(problems_of(changed(path, -0.1))) == [path]


def test_stacking_factor_cannot_exceed_one(changed, problems_of) -> None:
    assert list(problems_of(changed("steel.stacking_factor", 1.01))) == [
        "steel.stacking_factor"
    ]


@pytest.mark.parametrize("path", ["magnet.temperature_C", "winding.temperature_C"])
def test_temperature_below_absolute_zero_is_rejected(path, changed, problems_of) -> None:
    design_from_dict(changed(path, -273))
    assert list(problems_of(changed(path, -273.15))) == [path]


def test_unknown_connection_lists_the_choices(changed, problems_of) -> None:
    problems = problems_of(changed("winding.connection", "wye"))
    assert problems == {"winding.connection": "must be one of star, delta; got 'wye'"}


def test_unknown_magnetisation_is_rejected(changed, problems_of) -> None:
    problems = problems_of(changed("geometry.magnet.magnetisation", "halbach"))
    assert list(problems) == ["geometry.magnet.magnetisation"]


def test_unsupported_format_version_is_rejected(changed, problems_of) -> None:
    assert list(problems_of(changed("format_version", 2))) == ["format_version"]


def test_material_name_that_parses_as_a_number_is_rejected(changed, problems_of) -> None:
    problems = problems_of(changed("steel.rotor", 1010))
    assert list(problems) == ["steel.rotor"]
    assert "quotes" in problems["steel.rotor"]


# Reporting


def test_all_problems_are_reported_together(changed, problems_of) -> None:
    data = changed(
        "poles", 41, "geometry.air_gap_mm", -0.5, "winding.connection", "wye"
    )
    del data["steel"]["rotor"]
    assert set(problems_of(data)) == {
        "poles",
        "geometry.air_gap_mm",
        "winding.connection",
        "steel.rotor",
    }


def test_fit_checks_are_skipped_while_single_fields_are_wrong(
    changed, problems_of
) -> None:
    # The magnet would not fit, but the air gap must be fixed first; reporting
    # a fit problem computed from a bad value would only confuse.
    data = changed("geometry.magnet.thickness_mm", 40, "geometry.air_gap_mm", -1)
    assert list(problems_of(data)) == ["geometry.air_gap_mm"]


def test_unknown_field_is_rejected_at_every_level(valid_data, problems_of) -> None:
    valid_data["pole"] = 14
    valid_data["winding"]["turns_per_col"] = 20
    valid_data["geometry"]["stator"]["tooth_widht_mm"] = 5.0
    problems = problems_of(valid_data)
    assert set(problems) == {
        "pole",
        "winding.turns_per_col",
        "geometry.stator.tooth_widht_mm",
    }
    assert "did you mean 'poles'" in problems["pole"]


def test_unknown_type_does_not_flag_the_geometry_fields(changed, problems_of) -> None:
    assert list(problems_of(changed("type", "induction"))) == ["type"]


def test_error_from_a_dict_names_the_source_given(changed) -> None:
    with pytest.raises(DesignError) as caught:
        design_from_dict(changed("poles", 41), source="sweep point 7")
    assert str(caught.value).startswith("sweep point 7: poles: ")
    assert caught.value.source == "sweep point 7"


@pytest.mark.parametrize("data", [None, [], [1, 2], "text", 42])
def test_top_level_must_be_named_fields(data, problems_of) -> None:
    assert list(problems_of(data)) == [""]


# File problems


def test_error_message_starts_with_the_file_name(tmp_path: Path) -> None:
    path = write(tmp_path, "type: qdd_outer\n")
    error = load_error(path)
    assert str(error).startswith(f"{path}: ")
    assert error.source == str(path)
    assert "slots" in [p.path for p in error.problems]


def test_missing_file_is_reported(tmp_path: Path) -> None:
    error = load_error(tmp_path / "nowhere.yaml")
    assert "not found" in str(error)


def test_directory_instead_of_file_is_reported(tmp_path: Path) -> None:
    assert "folder" in str(load_error(tmp_path))


@pytest.mark.parametrize("text", ["", "\n\n", "# only a comment\n"])
def test_empty_file_is_reported(tmp_path: Path, text: str) -> None:
    assert "empty" in str(load_error(write(tmp_path, text)))


@pytest.mark.parametrize("text", ["- 1\n- 2\n", "just some text\n", "42\n"])
def test_file_that_is_not_a_set_of_fields_is_reported(tmp_path: Path, text: str) -> None:
    assert "top level" in str(load_error(write(tmp_path, text)))


def test_duplicate_field_is_rejected_with_its_line(data_dir: Path) -> None:
    message = str(load_error(data_dir / "duplicate_key.yaml"))
    assert "'slots' appears twice" in message
    assert "line 5" in message


def test_duplicate_field_in_a_section_is_rejected(tmp_path: Path) -> None:
    text = "winding:\n  turns_per_coil: 20\n  turns_per_coil: 30\n"
    assert "'turns_per_coil' appears twice" in str(load_error(write(tmp_path, text)))


def test_tab_indentation_reports_the_line(tmp_path: Path) -> None:
    message = str(load_error(write(tmp_path, "type: qdd_outer\nmagnet:\n\tmaterial: N35\n")))
    assert "not valid YAML" in message and "line 3" in message


def test_file_that_is_not_utf8_is_reported(tmp_path: Path) -> None:
    error = load_error(write(tmp_path, "name: caf\xe9\n".encode("latin-1")))
    assert "UTF-8" in str(error)


def test_file_with_a_byte_order_mark_loads(tmp_path: Path, data_dir: Path) -> None:
    text = (data_dir / "valid_design.yaml").read_text(encoding="utf-8")
    path = write(tmp_path, b"\xef\xbb\xbf" + text.encode("utf-8"))
    assert load_design(path).slots == 12


def test_yaml_tags_cannot_build_python_objects(tmp_path: Path) -> None:
    text = "type: !!python/object/apply:os.getcwd []\n"
    assert "not valid YAML" in str(load_error(write(tmp_path, text)))


def test_yes_and_no_are_not_accepted_as_numbers(tmp_path: Path, data_dir: Path) -> None:
    # YAML 1.1 reads a bare "yes" as true.
    text = (data_dir / "valid_design.yaml").read_text(encoding="utf-8")
    path = write(tmp_path, text.replace("skew_deg: 2.0", "skew_deg: yes"))
    error = load_error(path)
    assert [p.path for p in error.problems] == ["axial.skew_deg"]


@pytest.mark.parametrize(
    "line, shown",
    [
        ("slots: 036", "'036'"),  # octal 30 in YAML 1.1
        ("slots: 1:30", "'1:30'"),  # base 60, i.e. 90
        ("slots: 0x24", "'0x24'"),
        ("slots: 3_6", "'3_6'"),
    ],
)
def test_only_plain_decimal_numbers_are_read_as_numbers(
    tmp_path: Path, data_dir: Path, line: str, shown: str
) -> None:
    text = (data_dir / "valid_design.yaml").read_text(encoding="utf-8")
    error = load_error(write(tmp_path, text.replace("slots: 12", line)))
    assert [p.path for p in error.problems] == ["slots"]
    assert shown in error.problems[0].message


def test_exponent_without_a_decimal_point_is_a_number(
    tmp_path: Path, data_dir: Path
) -> None:
    text = (data_dir / "valid_design.yaml").read_text(encoding="utf-8")
    path = write(tmp_path, text.replace("air_gap_mm: 0.5", "air_gap_mm: 5e-1"))
    assert load_design(path).geometry.air_gap_mm == 0.5


@pytest.mark.parametrize(
    "line",
    ["material: 2024-13-45", "material: !!int abc", "material: " + "[" * 2000],
)
def test_values_pyyaml_chokes_on_are_reported_as_design_errors(
    tmp_path: Path, data_dir: Path, line: str
) -> None:
    text = (data_dir / "valid_design.yaml").read_text(encoding="utf-8")
    error = load_error(write(tmp_path, text.replace("material: N35", line)))
    assert "not valid YAML" in str(error)


def test_number_too_large_for_a_float_is_rejected(changed, problems_of) -> None:
    assert list(problems_of(changed("slots", 10**400))) == ["slots"]
    assert list(problems_of(changed("slots", 12, "geometry.air_gap_mm", 10**400))) == [
        "geometry.air_gap_mm"
    ]
    assert list(problems_of(changed("geometry.air_gap_mm", 0.5, "magnet.material", 10**400))) == [
        "magnet.material"
    ]


def test_error_survives_pickling(changed) -> None:
    with pytest.raises(DesignError) as caught:
        design_from_dict(changed("poles", 41, "slots", 2), source="point 3")
    restored = pickle.loads(pickle.dumps(caught.value))
    assert str(restored) == str(caught.value)
    assert restored.problems == caught.value.problems
    assert restored.source == "point 3"
