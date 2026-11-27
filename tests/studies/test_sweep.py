"""Sweeps over design values: cases, failures, the result table and resuming."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from polecraft.design import load_design
from polecraft.studies.sweep import design_for, run_sweep, sweep_cases

ROOT = Path(__file__).resolve().parents[2]
WIRE = "winding.wire_diameter_mm"
GRADE = "magnet.material"
TURNS = "winding.turns_per_coil"


@pytest.fixture(scope="module")
def base():
    return load_design(ROOT / "designs" / "qdd_outer_12s10p.yaml")


class Counting:
    """An evaluation that needs no field solve and counts its calls."""

    def __init__(self, fail_when=None, stop_at=None):
        self.calls = []
        self.fail_when = fail_when or (lambda design: False)
        self.stop_at = stop_at

    def __call__(self, design):
        self.calls.append(design)
        if self.stop_at is not None and len(self.calls) == self.stop_at:
            raise KeyboardInterrupt
        if self.fail_when(design):
            raise RuntimeError("this case does not solve")
        winding = design.winding
        area = winding.parallel_strands * math.pi * winding.wire_diameter_mm**2 / 4.0
        return {"copper_mm2": area, "turns": winding.turns_per_coil, "grade": design.magnet.material}


# The cases


@pytest.mark.parametrize(
    "lists",
    [
        {WIRE: [0.3, 0.35, 0.4], GRADE: ["N42SH", "N52"]},
        {WIRE: [0.3], GRADE: ["N42SH"], TURNS: [10, 15, 20, 25]},
        {TURNS: [12, 14, 16, 18, 20]},
    ],
)
def test_the_case_count_is_the_product_of_the_lists(base, lists) -> None:
    expected = math.prod(len(values) for values in lists.values())
    assert len(sweep_cases(lists)) == expected
    sweep = run_sweep(base, lists, Counting())
    assert len(sweep.cases) == expected and len(sweep.ok) == expected


def test_no_swept_values_is_one_case_the_base_design(base) -> None:
    assert sweep_cases({}) == [{}]
    evaluate = Counting()
    sweep = run_sweep(base, {}, evaluate)
    assert len(sweep.cases) == 1 and evaluate.calls == [base]


def test_cases_come_in_order_with_the_last_value_changing_fastest(base) -> None:
    cases = sweep_cases({WIRE: [0.3, 0.4], GRADE: ["A", "B", "C"]})
    assert cases == [
        {WIRE: 0.3, GRADE: "A"}, {WIRE: 0.3, GRADE: "B"}, {WIRE: 0.3, GRADE: "C"},
        {WIRE: 0.4, GRADE: "A"}, {WIRE: 0.4, GRADE: "B"}, {WIRE: 0.4, GRADE: "C"},
    ]


def test_each_case_evaluates_the_base_design_with_its_values_put_in(base) -> None:
    evaluate = Counting()
    sweep = run_sweep(base, {WIRE: [0.3, 0.35], GRADE: ["N42SH", "N52"]}, evaluate)
    seen = [(design.winding.wire_diameter_mm, design.magnet.material) for design in evaluate.calls]
    assert seen == [(0.3, "N42SH"), (0.3, "N52"), (0.35, "N42SH"), (0.35, "N52")]
    for design in evaluate.calls:  # everything not swept is the base design's
        assert design.geometry == base.geometry and design.winding.turns_per_coil == 20
    assert [case.results["grade"] for case in sweep.cases] == ["N42SH", "N52", "N42SH", "N52"]


def test_design_for_checks_the_design_as_a_file_would_be(base) -> None:
    assert design_for(base, {WIRE: 0.35}).winding.wire_diameter_mm == 0.35
    with pytest.raises(ValueError, match="must be greater than 0"):
        design_for(base, {WIRE: -1.0})


# Failures


def test_a_failed_case_does_not_stop_the_sweep(base) -> None:
    evaluate = Counting(fail_when=lambda design: design.winding.wire_diameter_mm == 0.35)
    sweep = run_sweep(base, {WIRE: [0.3, 0.35, 0.4]}, evaluate)
    assert len(evaluate.calls) == 3
    assert [case.status for case in sweep.cases] == ["ok", "failed", "ok"]
    failed = sweep.failed[0]
    assert failed.values == {WIRE: 0.35}
    assert failed.error == "RuntimeError: this case does not solve" and failed.results == {}


def test_a_design_the_checks_refuse_is_a_failed_case_not_an_error(base) -> None:
    evaluate = Counting()
    # 0.8 mm wire over-fills the slot; 60 turns too.
    sweep = run_sweep(base, {WIRE: [0.4, 0.8], TURNS: [20, 60]}, evaluate)
    assert [case.ok for case in sweep.cases] == [True, False, False, False]
    assert len(evaluate.calls) == 1
    assert all("DesignError" in case.error and "over-filled" in case.error for case in sweep.failed)


def test_results_that_are_not_single_values_fail_the_case(base) -> None:
    def evaluate(design):
        if design.winding.wire_diameter_mm == 0.3:
            return {"torque": np.float64(1.5), "turns": np.int64(20)}  # NumPy numbers are fine
        if design.winding.wire_diameter_mm == 0.35:
            return {"torque": np.array([1.0, 2.0])}
        return {WIRE: 1.0}  # a swept value's name is taken

    sweep = run_sweep(base, {WIRE: [0.3, 0.35, 0.4]}, evaluate)
    first, second, third = sweep.cases
    assert first.results == {"torque": 1.5, "turns": 20}
    assert type(first.results["torque"]) is float and type(first.results["turns"]) is int
    assert not second.ok and "result 'torque' must be a single number" in second.error
    assert not third.ok and "is taken by a swept value" in third.error


@pytest.mark.parametrize(
    "lists, message",
    [
        ({"winding.wire_size": [0.3]}, "the design has no such value"),
        ({"rotor.outer_diameter_mm": [50]}, "the design has no 'rotor'"),
        ({"geometry.rotor": [1]}, "is a group of values"),
        ({WIRE: []}, "the list of values is empty"),
        ({WIRE: 0.3}, "give a list of values"),
        ({GRADE: "N52"}, "give a list of values"),
        ({TURNS: [20, 20.0]}, "in the list twice"),
        ({WIRE: [[0.3, 0.4]]}, "must be a single number"),
    ],
)
def test_bad_sweeps_are_refused_before_any_case_runs(base, lists, message, tmp_path) -> None:
    evaluate = Counting()
    store = tmp_path / "sweep.jsonl"
    with pytest.raises(ValueError, match=message):
        run_sweep(base, lists, evaluate, store=store)
    assert evaluate.calls == [] and not store.exists()


# The result table


def test_the_table_has_the_swept_values_the_results_and_the_status(base, tmp_path) -> None:
    evaluate = Counting(fail_when=lambda design: design.magnet.material == "N52")
    sweep = run_sweep(base, {WIRE: [0.3, 0.4], GRADE: ["N42SH", "N52"], TURNS: [40]}, evaluate)
    assert sweep.columns() == [WIRE, GRADE, TURNS, "copper_mm2", "turns", "grade",
                               "status", "error", "warnings", "seconds"]
    rows = sweep.table()
    assert len(rows) == 4
    assert rows[0][WIRE] == 0.3 and rows[0]["copper_mm2"] == pytest.approx(math.pi * 0.09 / 4)
    assert rows[1]["status"] == "failed" and rows[1]["copper_mm2"] is None
    # 40 turns of 0.4 mm wire is a fill the checks warn about but accept.
    assert rows[0]["warnings"] is None and rows[0]["error"] is None
    assert "fill factor" in rows[2]["warnings"] and rows[2]["status"] == "ok"
    assert all(row["seconds"] >= 0.0 for row in rows)

    path = tmp_path / "table.csv"
    sweep.write_csv(path)
    with open(path, newline="", encoding="utf-8") as file:
        read = list(csv.DictReader(file))
    assert list(read[0]) == sweep.columns()
    assert [row["status"] for row in read] == ["ok", "failed", "ok", "failed"]
    assert read[1]["copper_mm2"] == "" and float(read[2]["copper_mm2"]) == pytest.approx(math.pi * 0.16 / 4)


def test_the_report_counts_the_cases(base, tmp_path) -> None:
    store = tmp_path / "sweep.jsonl"
    evaluate = Counting(fail_when=lambda design: design.winding.wire_diameter_mm == 0.4)
    run_sweep(base, {WIRE: [0.3, 0.4]}, evaluate, store=store)
    sweep = run_sweep(base, {WIRE: [0.3, 0.35, 0.4]}, evaluate, store=store)
    lines = sweep.report().splitlines()
    assert lines[0] == f"Sweep over {WIRE}: 3 cases, 2 ok, 1 failed"
    assert lines[1].split()[:3] == ["run", "now", "2"]  # 0.35 is new, 0.4 failed before and is retried
    assert lines[2].split() == ["from", "the", "store", "1"]


def test_progress_reports_one_line_per_case(base) -> None:
    lines = []
    run_sweep(base, {WIRE: [0.3, 0.4]}, Counting(), progress=lines.append)
    assert len(lines) == 2
    assert lines[0].startswith("case 1/2 ok in ") and lines[0].endswith(f"{WIRE}=0.3")


# Resuming


def test_a_resumed_sweep_skips_finished_cases(base, tmp_path) -> None:
    store = tmp_path / "sweep.jsonl"
    lists = {WIRE: [0.3, 0.35], GRADE: ["N42SH", "N52"]}
    first = run_sweep(base, lists, Counting(), store=store)
    evaluate = Counting()
    again = run_sweep(base, lists, evaluate, store=store)
    assert evaluate.calls == []
    assert all(case.resumed for case in again.cases)
    assert [case.results for case in again.cases] == [case.results for case in first.cases]
    assert [case.seconds for case in again.cases] == [case.seconds for case in first.cases]


def test_a_stopped_sweep_resumes_where_it_stopped(base, tmp_path) -> None:
    store = tmp_path / "sweep.jsonl"
    lists = {WIRE: [0.3, 0.35, 0.4], GRADE: ["N42SH", "N52"]}
    with pytest.raises(KeyboardInterrupt):
        run_sweep(base, lists, Counting(stop_at=3), store=store)
    evaluate = Counting()
    sweep = run_sweep(base, lists, evaluate, store=store)
    assert len(evaluate.calls) == 4
    assert [case.resumed for case in sweep.cases] == [True, True, False, False, False, False]
    assert len(sweep.ok) == 6


def test_failed_cases_are_run_again_unless_asked_not_to(base, tmp_path) -> None:
    store = tmp_path / "sweep.jsonl"
    lists = {WIRE: [0.3, 0.35]}
    run_sweep(base, lists, Counting(fail_when=lambda d: d.winding.wire_diameter_mm == 0.35), store=store)
    kept = Counting()
    sweep = run_sweep(base, lists, kept, store=store, retry_failed=False)
    assert kept.calls == [] and [case.ok for case in sweep.cases] == [True, False]
    retried = Counting()
    sweep = run_sweep(base, lists, retried, store=store)
    assert len(retried.calls) == 1 and [case.ok for case in sweep.cases] == [True, True]
    # The later result is the one kept.
    assert [case.ok for case in run_sweep(base, lists, Counting(), store=store).cases] == [True, True]


def test_a_lengthened_list_runs_only_the_new_cases(base, tmp_path) -> None:
    store = tmp_path / "sweep.jsonl"
    run_sweep(base, {WIRE: [0.3, 0.35], GRADE: ["N42SH"]}, Counting(), store=store)
    evaluate = Counting()
    # The order of the paths may change, and 0.3 may be written as 0.30.
    sweep = run_sweep(base, {GRADE: ["N42SH", "N52"], WIRE: [0.30, 0.35]}, evaluate, store=store)
    assert sorted((d.magnet.material, d.winding.wire_diameter_mm) for d in evaluate.calls) == [
        ("N52", 0.3), ("N52", 0.35)
    ]
    assert [case.values for case in sweep.cases][0] == {GRADE: "N42SH", WIRE: 0.3}


def test_whole_and_decimal_numbers_are_the_same_case(base, tmp_path) -> None:
    store = tmp_path / "sweep.jsonl"
    run_sweep(base, {TURNS: [20, 22]}, Counting(), store=store)
    evaluate = Counting()
    run_sweep(base, {TURNS: [20.0, 22]}, evaluate, store=store)
    assert evaluate.calls == []


def test_a_line_cut_short_by_a_stop_is_dropped(base, tmp_path) -> None:
    store = tmp_path / "sweep.jsonl"
    lists = {WIRE: [0.3, 0.35, 0.4]}
    run_sweep(base, {WIRE: [0.3]}, Counting(), store=store)
    with open(store, "a", encoding="utf-8") as file:
        file.write('{"values": {"winding.wire_diam')
    evaluate = Counting()
    sweep = run_sweep(base, lists, evaluate, store=store)
    assert len(evaluate.calls) == 2 and len(sweep.ok) == 3
    lines = store.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and all(json.loads(line) for line in lines)


def test_a_store_of_another_sweep_or_another_design_is_refused(base, tmp_path) -> None:
    store = tmp_path / "sweep.jsonl"
    run_sweep(base, {WIRE: [0.3]}, Counting(), store=store)
    with pytest.raises(ValueError, match="holds a sweep over winding.wire_diameter_mm"):
        run_sweep(base, {GRADE: ["N52"]}, Counting(), store=store)
    other = design_for(base, {"axial.stack_length_mm": 12.0})
    with pytest.raises(ValueError, match="another base design"):
        run_sweep(other, {WIRE: [0.3]}, Counting(), store=store)


@pytest.mark.parametrize("content", ["slots,poles\n12,10\n", "just one line with no end", '{"a": 1}\n'])
def test_a_file_that_is_not_a_store_is_refused_and_left_alone(base, tmp_path, content) -> None:
    store = tmp_path / "other.txt"
    store.write_text(content, encoding="utf-8")
    evaluate = Counting()
    with pytest.raises(ValueError, match="not a sweep store"):
        run_sweep(base, {WIRE: [0.3]}, evaluate, store=store)
    assert store.read_text(encoding="utf-8") == content and evaluate.calls == []


def test_a_damaged_case_line_is_reported(base, tmp_path) -> None:
    store = tmp_path / "sweep.jsonl"
    run_sweep(base, {WIRE: [0.3]}, Counting(), store=store)
    with open(store, "a", encoding="utf-8") as file:
        file.write('{"values": 3}\n')
    with pytest.raises(ValueError, match="line 3 is not a stored case"):
        run_sweep(base, {WIRE: [0.3]}, Counting(), store=store)


# With a field solve


def test_a_sweep_of_magnet_grade_against_wire_size_with_a_field_solve(base, tmp_path) -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.materials import Magnet, load_material
    from polecraft.core.mesh import build_mesh
    from polecraft.core.post import torque
    from polecraft.core.problem import build_problem
    from polecraft.core.solver import solve
    from polecraft.core.winding import phase_winding
    from polecraft.machines import machine_for

    # Placeholder magnets, not datasheet grades: only their order matters here.
    grades = {"N42SH": Magnet(remanence=1.2, relative_permeability=1.05),
              "N52": Magnet(remanence=1.4, relative_permeability=1.05)}
    steel = load_material("generic_35A_placeholder")

    def evaluate(design):
        machine = machine_for(design)
        mesh = build_mesh(machine.build_geometry(), size=2.0)
        materials = {design.steel.stator: steel, design.magnet.material: grades[design.magnet.material]}
        problem = build_problem(machine, mesh, materials, current=5.0)
        return {"torque": torque(problem, solve(problem)),
                "resistance": phase_winding(machine).resistance()}

    lists = {GRADE: ["N42SH", "N52"], WIRE: [0.3, 0.4]}
    sweep = run_sweep(base, lists, evaluate, store=tmp_path / "sweep.jsonl")
    assert len(sweep.ok) == 4
    by_case = {(case.values[GRADE], case.values[WIRE]): case.results for case in sweep.cases}
    for wire in (0.3, 0.4):
        assert by_case[("N52", wire)]["torque"] > by_case[("N42SH", wire)]["torque"] > 0.0
    for grade in grades:
        # Resistance goes as one over the wire's area; the field does not see the wire.
        thin, thick = by_case[(grade, 0.3)], by_case[(grade, 0.4)]
        assert thin["resistance"] / thick["resistance"] == pytest.approx((0.4 / 0.3) ** 2, rel=1e-9)
        assert thin["torque"] == pytest.approx(thick["torque"], rel=1e-9)
