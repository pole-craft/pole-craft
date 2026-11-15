"""The contract every machine plug-in follows."""

from __future__ import annotations

from typing import Protocol

from polecraft.validation import Problem, Section


class MachineParameters(Protocol):
    """What a machine's parameter class must offer to the design loader."""

    @classmethod
    def from_section(cls, section: Section) -> MachineParameters | None:
        """Read the ``geometry`` block; None if any field is unusable."""
        ...

    def check(self, slots: int, poles: int) -> list[Problem]:
        """Problems that involve several fields, such as parts that do not fit."""
        ...
