"""Timing report."""

import time

import pytest

from polecraft.timing import Timings


def test_stages_are_recorded_in_order() -> None:
    timings = Timings()
    with timings.stage("geometry"):
        pass
    with timings.stage("mesh"):
        time.sleep(0.02)
    assert [name for name, _ in timings.stages] == ["geometry", "mesh"]
    assert timings.stages[1][1] >= 0.02
    assert timings.stages[0][1] < 0.02


def test_stages_sum_to_the_total() -> None:
    timings = Timings()
    timings.stages = [("a", 0.25), ("b", 1.5), ("c", 0.125)]
    assert timings.total == pytest.approx(1.875)


def test_a_stage_that_fails_is_still_timed() -> None:
    timings = Timings()
    with pytest.raises(RuntimeError):
        with timings.stage("mesh"):
            raise RuntimeError("no mesh")
    assert [name for name, _ in timings.stages] == ["mesh"]


def test_report_lists_every_stage_and_the_total() -> None:
    timings = Timings()
    timings.stages = [("load design", 0.004), ("mesh", 1.25)]
    assert timings.report() == (
        "Timing\n"
        "  load design     0.004 s\n"
        "  mesh            1.250 s\n"
        "  total           1.254 s"
    )


def test_report_with_no_stages() -> None:
    assert Timings().report() == "Timing\n  total     0.000 s"
