"""Sweeps: run one evaluation over every combination of design values.

A sweep names design values by their path in the design file, such as
``winding.wire_diameter_mm`` or ``magnet.material``, each with a list of
values to try. Every combination is a case: the base design with those
values put in and checked as a design file would be. A function given by the
caller evaluates each case's design and returns named results.

A case that fails (a design the checks refuse, or an error in the
evaluation) is recorded with its error and the sweep goes on. With a
``store`` file each case is written as it finishes, so a sweep that was
stopped can be run again and skips the cases already done.

The store is JSON Lines: a header naming the swept values and the base
design, then one line per finished case.

With ``workers`` above one, cases run at the same time in separate
processes, each on one core. The cases are independent, so each gives the
same result as in a serial run; only the order in which they finish, and so
the order of the store's lines and of the progress lines, differs.
"""

from __future__ import annotations

import concurrent.futures
import csv
import dataclasses
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import pickle
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from polecraft.design import Design, design_from_dict, design_to_dict

STORE_FORMAT = 1
# Columns of the result table that are not swept values or results.
FIXED_COLUMNS = ("status", "error", "warnings", "seconds")


@dataclasses.dataclass(frozen=True)
class CaseResult:
    """What one case of a sweep gave."""

    values: dict[str, Any]  # swept value by path
    status: str  # "ok" or "failed"
    results: dict[str, Any]  # what the evaluation returned; empty if it failed
    error: str | None  # why it failed
    warnings: tuple[str, ...]  # the design's warnings, if it could be built
    seconds: float  # time to build and evaluate the case
    resumed: bool = False  # read from the store, not run this time

    @property
    def ok(self) -> bool:
        """Whether the case was evaluated without an error."""
        return self.status == "ok"


@dataclasses.dataclass(frozen=True)
class Sweep:
    """The cases of a sweep, in the order of the combinations."""

    parameters: tuple[str, ...]
    cases: tuple[CaseResult, ...]
    wall: float = 0.0  # seconds from start to end of the run, s
    workers: int = 1  # processes the cases ran in

    @property
    def ok(self) -> tuple[CaseResult, ...]:
        """The cases that were evaluated."""
        return tuple(case for case in self.cases if case.ok)

    @property
    def failed(self) -> tuple[CaseResult, ...]:
        """The cases that failed."""
        return tuple(case for case in self.cases if not case.ok)

    def columns(self) -> list[str]:
        """Names of the table's columns: swept values, results, then status columns."""
        results: dict[str, None] = {}
        for case in self.cases:
            results.update(dict.fromkeys(case.results))
        return [*self.parameters, *results, *FIXED_COLUMNS]

    def table(self) -> list[dict[str, Any]]:
        """One row per case; a result a case does not have is None."""
        columns = self.columns()
        rows = []
        for case in self.cases:
            row = dict.fromkeys(columns)
            row.update(case.values)
            row.update(case.results)
            row.update(
                status=case.status,
                error=case.error,
                warnings="; ".join(case.warnings) or None,
                seconds=case.seconds,
            )
            rows.append(row)
        return rows

    def write_csv(self, path: str | Path) -> None:
        """Write the table as a CSV file; an empty cell is a missing value."""
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.columns())
            writer.writeheader()
            for row in self.table():
                writer.writerow({key: "" if value is None else value for key, value in row.items()})

    def report(self) -> str:
        """How many cases ran, failed or were taken from an earlier run, and the time taken.

        The case time is the sum of each case's own time; with several
        workers the wall time is less, and their ratio is the speed-up.
        """
        run = [case for case in self.cases if not case.resumed]
        seconds = sum(case.seconds for case in run)
        lines = [
            f"Sweep over {', '.join(self.parameters) or 'nothing (the base design)'}: "
            f"{len(self.cases)} cases, {len(self.ok)} ok, {len(self.failed)} failed",
            f"  run now        {len(run):5d}  in {seconds:9.3f} s"
            + (f"  ({seconds / len(run):.3f} s a case)" if run else ""),
            f"  from the store {len(self.cases) - len(run):5d}",
            f"  wall time            {self.wall:9.3f} s with {self.workers} "
            + ("worker" if self.workers == 1 else "workers"),
        ]
        return "\n".join(lines)


def sweep_cases(values: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Every combination of the values, the last path changing fastest.

    The number of cases is the product of the lengths of the lists. No paths
    gives one case: the base design as it is.
    """
    paths = list(values)
    return [dict(zip(paths, combination)) for combination in itertools.product(*values.values())]


def design_for(base: Design, case: Mapping[str, Any]) -> Design:
    """The base design with a case's values put in, checked as a design file is.

    Raises ValueError for a path the design does not have, and DesignError
    if the design that results is refused.
    """
    data = design_to_dict(base)
    for path, value in case.items():
        _place(data, path, value)
    label = ", ".join(f"{path}={value!r}" for path, value in case.items())
    return design_from_dict(data, source=f"sweep case ({label or 'base design'})")


def run_sweep(
    base: Design,
    values: Mapping[str, Sequence[Any]],
    evaluate: Callable[[Design], Mapping[str, Any]],
    store: str | Path | None = None,
    retry_failed: bool = True,
    progress: Callable[[str], None] | None = None,
    workers: int | None = 1,
) -> Sweep:
    """Evaluate every combination of design values.

    ``values`` maps design-file paths (``"winding.wire_diameter_mm"``) to the
    values to try. ``evaluate`` is given each case's design and returns its
    results by name, each a number, text, True/False or None.

    A case whose design is refused, or whose evaluation raises an error, is
    recorded as failed with the error, and the sweep goes on. An interrupt
    (Ctrl+C) is not caught: it stops the sweep.

    With ``store``, a file, every finished case is added to it at once. If
    the file holds an earlier run of the same sweep (the same paths and base
    design), cases it has done are not run again; with ``retry_failed`` the
    cases that failed there are. Lists may have been lengthened since: cases
    are matched by their values. Progress, one line per case, goes to
    ``progress`` if given, for example ``print``.

    ``workers`` is how many cases run at once, each in its own process; None
    uses every core this process may use, counting logical cores, though
    more workers than physical cores were found to gain nothing (see
    docs/benchmarks.md). With more than one, ``evaluate``
    must be a function defined at the top level of a module (or a
    ``functools.partial`` of one), because each process imports it, and the
    progress lines come in the order the cases finish. A case whose process
    stops without an error (a crash in a compiled library) stops the pool:
    that case and every case not finished by then are recorded as failed,
    and running the sweep again with its store retries them.

    Raises ValueError, before any case is run, for a path the design does not
    have, a list that is empty, holds the same value twice or holds a value
    that is not a single number, text, True/False or None, for a store made
    for another sweep or another base design, for fewer than one worker, and
    for an ``evaluate`` that cannot be sent to another process.
    """
    began = time.perf_counter()
    parameters = tuple(values)
    _check_values(base, values)
    workers = _check_workers(workers, evaluate)
    cases = sweep_cases(values)
    done: dict[str, CaseResult] = {}
    store_path = None
    if store is not None:
        done = _open_store(Path(store), parameters, base)
        store_path = Path(store)

    results: list[CaseResult | None] = [None] * len(cases)
    to_run = []
    finished = 0

    def record(index: int, result: CaseResult, verdict: str) -> None:
        nonlocal finished
        results[index] = result
        finished += 1
        if progress is not None:
            label = ", ".join(f"{path}={value!r}" for path, value in cases[index].items())
            progress(f"case {finished}/{len(cases)} {verdict}: {label or 'base design'}")

    for index, case in enumerate(cases):
        found = done.get(_key(case))
        if found is not None and (found.ok or not retry_failed):
            record(index, dataclasses.replace(found, values=dict(case), resumed=True), "from the store")
        else:
            to_run.append(index)

    def ran(index: int, result: CaseResult) -> None:
        if store_path is not None:
            _append(store_path, result)
        record(index, result, f"{result.status} in {result.seconds:.2f} s")

    workers = min(workers, max(len(to_run), 1))
    if workers == 1:
        for index in to_run:
            ran(index, _run_case(base, cases[index], evaluate))
    else:
        _run_parallel(base, cases, to_run, evaluate, workers, ran)
    return Sweep(
        parameters=parameters,
        cases=tuple(results),  # type: ignore[arg-type]  # every case is recorded
        wall=time.perf_counter() - began,
        workers=workers,
    )


def usable_cores() -> int:
    """How many cores this process may run on."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13 and later
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _check_workers(workers: int | None, evaluate: Callable[..., Any]) -> int:
    if workers is None:
        workers = usable_cores()
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be a whole number of at least 1, or None for every core, not {workers!r}")
    if workers > 1:
        try:
            pickle.dumps(evaluate)
        except Exception as error:
            raise ValueError(
                "with more than one worker, the evaluation must be a function defined at the top level "
                f"of a module, so that each process can import it ({type(error).__name__}: {error})"
            ) from None
    return workers


def _run_parallel(
    base: Design,
    cases: list[dict[str, Any]],
    to_run: list[int],
    evaluate: Callable[[Design], Mapping[str, Any]],
    workers: int,
    ran: Callable[[int, CaseResult], None],
) -> None:
    """Run cases in a pool of processes, handing each result to ``ran`` as it finishes."""
    # Spawned, not forked, processes: the same on every system, and safe
    # with libraries that hold threads, such as Gmsh.
    context = multiprocessing.get_context("spawn")
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context)
    try:
        futures = {pool.submit(_run_case, base, cases[index], evaluate): index for index in to_run}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
            except BrokenProcessPool as error:
                result = CaseResult(
                    values=dict(cases[index]),
                    status="failed",
                    results={},
                    error=f"the worker process stopped while this case ran or waited ({type(error).__name__})",
                    warnings=(),
                    seconds=0.0,
                )
            ran(index, result)
    except BaseException:
        # An interrupt: the cases not started are dropped; those finished are stored.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


def _run_case(base: Design, case: Mapping[str, Any], evaluate: Callable[[Design], Mapping[str, Any]]) -> CaseResult:
    began = time.perf_counter()
    warnings: tuple[str, ...] = ()
    try:
        design = design_for(base, case)
        warnings = tuple(str(warning) for warning in design.warnings)
        found = evaluate(design)
        if not isinstance(found, Mapping):
            raise TypeError(f"the evaluation must return results by name, not {type(found).__name__}")
        results = {}
        for name, value in found.items():
            if not isinstance(name, str) or name in case or name in FIXED_COLUMNS:
                raise ValueError(
                    f"result name {name!r} is not text, or is taken by a swept value or a column of the table"
                )
            results[name] = _scalar(value, f"result '{name}'")
    except Exception as error:  # a failed case is recorded; the sweep goes on
        return CaseResult(
            values=dict(case),
            status="failed",
            results={},
            error=f"{type(error).__name__}: {error}",
            warnings=warnings,
            seconds=time.perf_counter() - began,
        )
    return CaseResult(
        values=dict(case),
        status="ok",
        results=results,
        error=None,
        warnings=warnings,
        seconds=time.perf_counter() - began,
    )


def _check_values(base: Design, values: Mapping[str, Sequence[Any]]) -> None:
    data = design_to_dict(base)
    for path, choices in values.items():
        if not isinstance(path, str):
            raise ValueError(f"a swept value is named by its path in the design file, not {path!r}")
        _place(data, path, None)  # only to check the path; data is a copy
        if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
            raise ValueError(f"{path}: give a list of values to try, not {choices!r}")
        if len(choices) == 0:
            raise ValueError(f"{path}: the list of values is empty")
        keys = set()
        for value in choices:
            try:
                _scalar(value, f"{path}: value {value!r}")
            except TypeError as error:
                raise ValueError(str(error)) from None
            key = json.dumps(_normal(value))
            if key in keys:
                raise ValueError(f"{path}: {value!r} is in the list twice")
            keys.add(key)


def _place(data: dict[str, Any], path: str, value: Any) -> None:
    """Put a value at a dotted path of a design dict; the path must exist and end at a value."""
    *groups, name = path.split(".")
    here = data
    for depth, group in enumerate(groups):
        if not isinstance(here, dict) or group not in here:
            raise ValueError(f"{path}: the design has no '{'.'.join(groups[: depth + 1])}'")
        here = here[group]
    if not isinstance(here, dict) or name not in here:
        raise ValueError(f"{path}: the design has no such value")
    if isinstance(here[name], dict):
        raise ValueError(f"{path}: is a group of values; name one of them, such as {path}.{next(iter(here[name]))}")
    here[name] = value


def _scalar(value: Any, what: str) -> Any:
    item = getattr(value, "item", None)  # a NumPy number, some of which are floats too
    if callable(item) and getattr(value, "ndim", None) == 0:
        value = item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"{what} must be a single number, text, True/False or None, not {type(value).__name__}")


def _normal(value: Any) -> Any:
    """A value as compared between runs: 2 and 2.0 are the same value."""
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _key(case: Mapping[str, Any]) -> str:
    return json.dumps({path: _normal(value) for path, value in case.items()}, sort_keys=True)


def _fingerprint(base: Design) -> str:
    text = json.dumps(design_to_dict(base), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _open_store(path: Path, parameters: tuple[str, ...], base: Design) -> dict[str, CaseResult]:
    """The cases an earlier run stored, by key; a new store gets its header."""
    header = {"polecraft_sweep": STORE_FORMAT, "parameters": list(parameters), "base": _fingerprint(base)}
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(json.dumps(header) + "\n", encoding="utf-8")
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{path}: not a sweep store (not text)") from None
    lines = text.split("\n")
    complete = lines[:-1]  # the last piece has no line end: written only in part, or empty
    try:
        stored = json.loads(complete[0]) if complete else None
    except json.JSONDecodeError:
        stored = None
    if not isinstance(stored, dict) or stored.get("polecraft_sweep") != STORE_FORMAT:
        # Not ours: leave the file as it is.
        raise ValueError(f"{path}: not a sweep store of format {STORE_FORMAT}; give another file")
    if sorted(stored.get("parameters", [])) != sorted(parameters):
        raise ValueError(
            f"{path}: holds a sweep over {', '.join(stored.get('parameters', [])) or 'nothing'}, "
            f"not over {', '.join(parameters) or 'nothing'}; give another file"
        )
    if stored.get("base") != header["base"]:
        raise ValueError(f"{path}: was made for another base design; give another file")
    if lines[-1].strip():
        # Stopped while a line was being written: drop that piece, so that
        # the next line does not join it.
        with open(path, "w", encoding="utf-8", newline="") as file:
            file.write("".join(line + "\n" for line in complete))
    done: dict[str, CaseResult] = {}
    for number, line in enumerate(complete[1:], start=2):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            result = CaseResult(
                values=dict(record["values"]),
                status=str(record["status"]),
                results=dict(record["results"]),
                error=record["error"],
                warnings=tuple(record["warnings"]),
                seconds=float(record["seconds"]),
                resumed=True,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise ValueError(f"{path}: line {number} is not a stored case") from None
        done[_key(result.values)] = result  # a later line for the same case wins
    return done


def _append(path: Path, result: CaseResult) -> None:
    record = {
        "values": result.values,
        "status": result.status,
        "results": result.results,
        "error": result.error,
        "warnings": list(result.warnings),
        "seconds": result.seconds,
    }
    with open(path, "a", encoding="utf-8", newline="") as file:
        file.write(json.dumps(record) + "\n")
        file.flush()
        os.fsync(file.fileno())
