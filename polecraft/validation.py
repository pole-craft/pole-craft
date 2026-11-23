"""Reading values from a design file and reporting problems in plain words.

Nothing here knows about motors. A ``Section`` wraps one block of the file
(a mapping of field names to values) and hands out checked values. Problems
are collected in a list instead of being raised one at a time, so a user sees
everything that is wrong with a file in a single run.

Warnings have the same form as problems (a field path and a message) but do
not stop a design from loading: they are engineering advice, such as a slot
that will be hard to wind.
"""

from __future__ import annotations

import difflib
import math
import numbers
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Marks a field that has no default and so must be present in the file.
REQUIRED: Any = object()


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a design: where it is and what is wrong."""

    path: str  # dotted field path such as "geometry.air_gap_mm"; "" for the whole file
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}" if self.path else self.message


def warning_text(warning: Problem, source: str | None = None) -> str:
    """One warning as a line of text, in the form problems are reported in."""
    return f"{source or 'design'}: warning: {warning}"


class DesignError(ValueError):
    """A design could not be loaded.

    ``problems`` lists every fault found, and ``warnings`` the advice found
    alongside them, so both are reported in one run.
    """

    def __init__(
        self,
        problems: Iterable[Problem],
        source: str | None = None,
        warnings: Iterable[Problem] = (),
    ) -> None:
        self.problems: list[Problem] = list(problems)
        self.source = source
        self.warnings: list[Problem] = list(warnings)
        super().__init__(self._format())

    def __reduce__(self) -> tuple:
        # Lets the error cross process boundaries intact (parallel sweeps).
        return (type(self), (self.problems, self.source, self.warnings))

    def _format(self) -> str:
        where = self.source or "design"
        if len(self.problems) == 1 and not self.warnings:
            return f"{where}: {self.problems[0]}"
        count = f"{len(self.problems)} problem{'s' if len(self.problems) != 1 else ''}"
        if self.warnings:
            count += f", {len(self.warnings)} warning{'s' if len(self.warnings) != 1 else ''}"
        lines = [f"{where}: {count}"]
        lines += [f"  {problem}" for problem in self.problems]
        lines += [f"  warning: {warning}" for warning in self.warnings]
        return "\n".join(lines)


def describe(value: object) -> str:
    """Name a value the way a user would, for use in an error message."""
    if value is None:
        return "nothing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Real):
        return f"the number {_as_float(value):g}"
    if isinstance(value, str):
        return f"the text {value!r}"
    if isinstance(value, dict):
        return "a section"
    if isinstance(value, (list, tuple)):
        return "a list"
    return f"a {type(value).__name__}"


def _is_number(value: object) -> bool:
    # bool is a subclass of int in Python, and YAML reads yes/no/on/off as
    # booleans, so it has to be excluded by hand.
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _as_float(value: Any) -> float:
    """float(value), with integers too large for a float returned as infinity."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _as_number_hint(value: object) -> str:
    """Hint for a number that arrived as text, e.g. "20" in quotes or 036."""
    if not isinstance(value, str):
        return ""
    try:
        parsed = float(value)
    except ValueError:
        return ""
    if not math.isfinite(parsed):
        return ""
    return f" (write it as a plain number: {parsed:g})"


def _limits(gt: float | None, ge: float | None, lt: float | None, le: float | None) -> str:
    parts = []
    if gt is not None:
        parts.append(f"greater than {gt:g}")
    if ge is not None:
        parts.append(f"at least {ge:g}")
    if lt is not None:
        parts.append(f"less than {lt:g}")
    if le is not None:
        parts.append(f"at most {le:g}")
    return " and ".join(parts)


class Section:
    """One block of a design file, read field by field.

    Each reader returns the checked value, or None after recording a problem.
    Call ``finish()`` on the top-level section once every field has been read;
    it reports fields in the file that nothing asked for (usually typos).
    ``warn`` records advice in ``warnings``, a list shared with nested blocks.
    """

    def __init__(
        self,
        data: object,
        path: str,
        problems: list[Problem],
        *,
        readable: bool = True,
        warnings: list[Problem] | None = None,
    ) -> None:
        self.path = path
        self.problems = problems
        self.warnings: list[Problem] = [] if warnings is None else warnings
        self._asked: list[str] = []
        self._children: list[Section] = []
        # An unreadable section stays silent: its one problem has already been
        # recorded, and a "missing" line for every field in it would be noise.
        self._readable = readable and isinstance(data, dict)
        self._data: dict[Any, Any] = data if self._readable else {}
        if readable and not isinstance(data, dict):
            self._report("", f"must be a section of named fields, got {describe(data)}")

    def _path_of(self, key: object) -> str:
        if key == "":
            return self.path
        return f"{self.path}.{key}" if self.path else str(key)

    def _report(self, key: object, message: str) -> None:
        self.problems.append(Problem(self._path_of(key), message))

    def warn(self, key: str, message: str) -> None:
        """Record advice about a field without stopping the design from loading."""
        self.warnings.append(Problem(self._path_of(key), message))

    def _raw(self, key: str, default: Any) -> tuple[bool, Any]:
        """Fetch a field. Returns (present, value).

        When the field is not present, value is its default, or None if a
        problem was recorded.
        """
        self._asked.append(key)
        if not self._readable:
            return False, None
        if key not in self._data:
            if default is REQUIRED:
                self._report(key, "required field is missing")
                return False, None
            return False, default
        value = self._data[key]
        if value is None:
            if default is None:
                return False, None
            self._report(key, "is empty (write a value after the colon)")
            return False, None
        return True, value

    def number(
        self,
        key: str,
        *,
        default: float | None = REQUIRED,
        gt: float | None = None,
        ge: float | None = None,
        lt: float | None = None,
        le: float | None = None,
    ) -> float | None:
        """Read a real number, optionally limited to a range."""
        found, value = self._raw(key, default)
        if not found:
            return value
        if not _is_number(value):
            hint = _as_number_hint(value)
            self._report(key, f"must be a number, got {describe(value)}{hint}")
            return None
        number = _as_float(value)
        if not math.isfinite(number):
            self._report(key, f"must be a finite number, got {number:g}")
            return None
        too_small = (gt is not None and number <= gt) or (ge is not None and number < ge)
        too_large = (lt is not None and number >= lt) or (le is not None and number > le)
        if too_small or too_large:
            self._report(key, f"must be {_limits(gt, ge, lt, le)}, got {number:g}")
            return None
        return number + 0.0  # turns -0.0 into 0.0

    def integer(
        self, key: str, *, default: int | None = REQUIRED, ge: int | None = None
    ) -> int | None:
        """Read a whole number. A float with no fractional part (42.0) is accepted."""
        found, value = self._raw(key, default)
        if not found:
            return value
        as_float = _as_float(value) if _is_number(value) else math.nan
        if not (math.isfinite(as_float) and as_float.is_integer()):
            if _is_number(value):
                got = f"{as_float:g}"
            else:
                got = describe(value) + _as_number_hint(value)
            self._report(key, f"must be a whole number, got {got}")
            return None
        number = int(value)
        if ge is not None and number < ge:
            self._report(key, f"must be at least {ge}, got {number}")
            return None
        return number

    def text(
        self,
        key: str,
        *,
        default: str | None = REQUIRED,
        choices: Sequence[str] | None = None,
    ) -> str | None:
        """Read a non-empty piece of text, optionally from a fixed set of choices."""
        found, value = self._raw(key, default)
        if not found:
            return value
        if not isinstance(value, str):
            hint = " (put it in quotes)" if _is_number(value) else ""
            self._report(key, f"must be text, got {describe(value)}{hint}")
            return None
        text = value.strip()
        if not text:
            self._report(key, "must not be blank")
            return None
        if choices is not None and text not in choices:
            self._report(key, f"must be one of {', '.join(choices)}; got {text!r}")
            return None
        return text

    def section(self, key: str) -> Section:
        """Open a nested block. A missing or malformed block is reported once."""
        self._asked.append(key)
        readable = self._readable
        data: object = None
        if readable:
            if key not in self._data:
                self._report(key, "required section is missing")
                readable = False
            elif self._data[key] is None:
                self._report(key, "is empty (expected a section of named fields)")
                readable = False
            else:
                data = self._data[key]
        child = Section(
            data, self._path_of(key), self.problems, readable=readable, warnings=self.warnings
        )
        self._children.append(child)
        return child

    def ignore(self, key: str) -> None:
        """Accept a field without reading it."""
        self._asked.append(key)

    def finish(self) -> None:
        """Report fields nothing asked for, here and in every nested block."""
        for key in self._data:
            if key in self._asked:
                continue
            close = difflib.get_close_matches(str(key).lower(), self._asked, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            self._report(key, f"unknown field{hint}")
        for child in self._children:
            child.finish()
