"""Timing report: how long each stage of a run took."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Timings:
    """Collects the time spent in named stages and prints them as a table.

    Stages follow one another. Do not put one inside another: both would be
    listed and the total would count the inner one twice. A stage that is
    timed again under the same name adds to the time it already has.
    """

    def __init__(self) -> None:
        self.stages: list[tuple[str, float]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the block inside ``with timings.stage("mesh"):``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def add(self, name: str, seconds: float) -> None:
        """Add time to a stage, which is listed where it was first named."""
        for index, (known, so_far) in enumerate(self.stages):
            if known == name:
                self.stages[index] = (name, so_far + seconds)
                return
        self.stages.append((name, seconds))

    @property
    def total(self) -> float:
        """Sum of all stages, in seconds."""
        return sum(seconds for _, seconds in self.stages)

    def report(self) -> str:
        """The stages, one per line, with their time in seconds and the total."""
        width = max([len(name) for name, _ in self.stages] + [len("total")])
        lines = [f"  {name:<{width}}  {seconds:8.3f} s" for name, seconds in self.stages]
        lines.append(f"  {'total':<{width}}  {self.total:8.3f} s")
        return "\n".join(["Timing"] + lines)
