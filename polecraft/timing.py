"""Timing report: how long each stage of a run took."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Timings:
    """Collects the time spent in named stages and prints them as a table.

    Stages follow one another. Do not put one inside another: both would be
    listed and the total would count the inner one twice.
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
            self.stages.append((name, time.perf_counter() - start))

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
