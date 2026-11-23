"""Design files: a motor described in YAML, loaded into a typed object.

A design has parts common to every machine (slot and pole counts, lengthwise
data, magnet and steel grades, winding) and a ``geometry`` block whose fields
depend on the machine ``type``. This module reads the common parts and hands
the ``geometry`` block to the parameter class registered for that type.

The design object keeps the file's names and units (``air_gap_mm`` is in
millimetres). Conversion to SI happens where a machine builds its geometry.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from polecraft.core.winding import MAX_PACKING, combination_problem, slot_fill
from polecraft.machines import machine_for, machine_types, parameters_class
from polecraft.machines.base import MachineParameters
from polecraft.validation import DesignError, Problem, Section

FORMAT_VERSION = 1
CONNECTIONS = ("star", "delta")
ABSOLUTE_ZERO_C = -273.15

# Advice limits; past them a design still loads, with a warning.
# Round wire wound on a tooth usually reaches a copper fill of 0.35 to 0.45;
# more takes special winding methods or shaped wire.
COPPER_FILL_ADVICE = 0.45
# Laminated stacks usually have 0.92 to 0.98 of their length in steel.
STACKING_FACTOR_ADVICE = 0.9


@dataclass(frozen=True)
class Axial:
    """Lengthwise data. The 2D solve only uses it in correction formulas."""

    stack_length_mm: float
    magnet_overhang_mm: float  # magnet extends past the stack by this much at each end
    skew_deg: float  # total skew, mechanical degrees
    end_winding_overhang_mm: float | None  # coil projection per end; None = estimate


@dataclass(frozen=True)
class Magnet:
    material: str  # grade name, e.g. "N42SH"
    temperature_C: float


@dataclass(frozen=True)
class Winding:
    wire_diameter_mm: float  # bare conductor
    insulated_diameter_mm: float | None  # overall, over the enamel; None = not known
    turns_per_coil: int
    parallel_strands: int  # wires in hand per turn
    conductor: str
    connection: str  # "star" or "delta"
    temperature_C: float


@dataclass(frozen=True)
class Steel:
    stator: str  # grade name, e.g. "M270-35A"
    rotor: str
    stacking_factor: float  # fraction of the stack length that is steel


@dataclass(frozen=True)
class Design:
    """A complete, checked motor design."""

    format_version: int
    type: str
    slots: int
    poles: int
    geometry: MachineParameters  # the parameter class registered for ``type``
    axial: Axial
    magnet: Magnet
    winding: Winding
    steel: Steel
    # Advice found while loading; not part of the design itself.
    warnings: tuple[Problem, ...] = field(default=(), compare=False, repr=False)


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects a field written twice in the same block.

    Plain PyYAML silently keeps the last value, which would hide a mistake
    such as two different air gaps in one file.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen = set()
        for key_node, _ in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                continue
            key = key_node.value
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"field {key!r} appears twice", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


# PyYAML follows YAML 1.1, where 036 is octal (30) and 1:30 is base 60 (90).
# A design file must mean what it says, so only plain decimal numbers are
# read as numbers; anything else arrives as text and is rejected by the
# field checks.
_INT = re.compile(r"^[-+]?(0|[1-9][0-9]*)$")
_FLOAT = re.compile(
    r"""^(?:[-+]?(?:[0-9]+\.[0-9]*(?:[eE][-+]?[0-9]+)?
                  |\.[0-9]+(?:[eE][-+]?[0-9]+)?
                  |[0-9]+[eE][-+]?[0-9]+)
           |[-+]?\.(?:inf|Inf|INF)
           |\.(?:nan|NaN|NAN))$""",
    re.VERBOSE,
)
_NUMBER_TAGS = ("tag:yaml.org,2002:int", "tag:yaml.org,2002:float")

_UniqueKeyLoader.yaml_implicit_resolvers = {
    first: [(tag, pattern) for tag, pattern in resolvers if tag not in _NUMBER_TAGS]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_UniqueKeyLoader.add_implicit_resolver(_NUMBER_TAGS[0], _INT, list("-+0123456789"))
_UniqueKeyLoader.add_implicit_resolver(_NUMBER_TAGS[1], _FLOAT, list("-+0123456789."))


def load_design(path: str | Path) -> Design:
    """Load and check a YAML design file.

    Raises DesignError, listing every problem found, if the file cannot be
    read or does not describe a buildable machine.
    """
    source = str(path)

    def fail(message: str) -> DesignError:
        return DesignError([Problem("", message)], source)

    try:
        is_folder = Path(path).is_dir()
        # utf-8-sig also accepts the byte-order mark that Windows Notepad adds.
        text = "" if is_folder else Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise fail("design file not found") from None
    except UnicodeDecodeError:
        raise fail("file is not UTF-8 text") from None
    except (OSError, ValueError) as error:
        raise fail(f"cannot read file ({getattr(error, 'strerror', None) or error})") from None
    if is_folder:
        raise fail("is a folder, not a design file")

    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        detail = (
            getattr(error, "problem", None)
            or getattr(error, "reason", None)
            or "could not be parsed"
        )
        raise fail(f"not valid YAML{where}: {detail}") from None
    except RecursionError:
        raise fail("not valid YAML: nested too deeply") from None
    except (ValueError, OverflowError) as error:
        # PyYAML lets these through for values such as an impossible date.
        raise fail(f"not valid YAML: {error}") from None

    return design_from_dict(data, source=source)


def design_from_dict(data: Any, source: str | None = None) -> Design:
    """Build a checked design from the nested dict a design file contains.

    ``source`` names where the data came from (a file path, a sweep point) and
    is used only in error messages. The input is not modified.
    """
    if data is None:
        raise DesignError([Problem("", "file is empty")], source)
    if not isinstance(data, dict):
        kind = "a list" if isinstance(data, list) else "a single value"
        raise DesignError(
            [Problem("", f"top level must be named fields such as 'slots: 36', got {kind}")],
            source,
        )

    problems: list[Problem] = []
    warnings: list[Problem] = []
    root = Section(data, "", problems, warnings=warnings)

    version = root.integer("format_version", default=FORMAT_VERSION)
    if version is not None and version != FORMAT_VERSION:
        problems.append(
            Problem(
                "format_version",
                f"this version of polecraft reads format {FORMAT_VERSION}, got {version}",
            )
        )

    machine_type = root.text("type")
    slots = root.integer("slots", ge=3)
    poles = root.integer("poles", ge=2)
    if poles is not None and poles % 2:
        problems.append(
            Problem("poles", f"must be even, since magnets come in north-south pairs; got {poles}")
        )
    elif slots is not None and poles is not None:
        unbalanced = combination_problem(slots, poles)
        if unbalanced:
            problems.append(Problem("slots", unbalanced))

    geometry = None
    if machine_type is None:
        root.ignore("geometry")
    else:
        try:
            parameters = parameters_class(machine_type)
        except KeyError:
            known = machine_types()
            close = difflib.get_close_matches(machine_type, known, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            problems.append(
                Problem(
                    "type",
                    f"unknown machine type {machine_type!r}{hint}; "
                    f"known types: {', '.join(known)}",
                )
            )
            # Without a known type there is no telling which fields belong here.
            root.ignore("geometry")
        else:
            geometry = parameters.from_section(root.section("geometry"))

    block = root.section("axial")
    axial = dict(
        stack_length_mm=block.number("stack_length_mm", gt=0),
        magnet_overhang_mm=block.number("magnet_overhang_mm", default=0.0, ge=0),
        skew_deg=block.number("skew_deg", default=0.0, ge=0),
        end_winding_overhang_mm=block.number("end_winding_overhang_mm", default=None, ge=0),
    )

    block = root.section("magnet")
    magnet = dict(
        material=block.text("material"),
        temperature_C=block.number("temperature_C", gt=ABSOLUTE_ZERO_C),
    )

    block = root.section("winding")
    winding = dict(
        wire_diameter_mm=block.number("wire_diameter_mm", gt=0),
        insulated_diameter_mm=block.number("insulated_diameter_mm", default=None, gt=0),
        turns_per_coil=block.integer("turns_per_coil", ge=1),
        parallel_strands=block.integer("parallel_strands", default=1, ge=1),
        conductor=block.text("conductor"),
        connection=block.text("connection", choices=CONNECTIONS),
        temperature_C=block.number("temperature_C", gt=ABSOLUTE_ZERO_C),
    )
    bare, overall = winding["wire_diameter_mm"], winding["insulated_diameter_mm"]
    if bare is not None and overall is not None and overall <= bare:
        problems.append(
            Problem(
                "winding.insulated_diameter_mm",
                f"must be more than the bare wire_diameter_mm of {bare:g}, got {overall:g}",
            )
        )

    block = root.section("steel")
    steel = dict(
        stator=block.text("stator"),
        rotor=block.text("rotor"),
        stacking_factor=block.number("stacking_factor", default=0.95, gt=0, le=1),
    )
    factor = steel["stacking_factor"]
    if factor is not None and factor < STACKING_FACTOR_ADVICE:
        block.warn(
            "stacking_factor",
            f"{factor:g} is low: laminated stacks are usually 0.92 to 0.98 steel; "
            "check the value",
        )

    root.finish()
    if problems:
        raise DesignError(problems, source, warnings)

    # Fit checks run only on values that passed the single-field checks, so a
    # bad number is never reported a second time as a part that does not fit.
    fit_problems = [
        Problem(f"geometry.{p.path}" if p.path else "geometry", p.message)
        for p in geometry.check(slots, poles)
    ]
    if fit_problems:
        raise DesignError(fit_problems, source, warnings)

    design = Design(
        format_version=version,
        type=machine_type,
        slots=slots,
        poles=poles,
        geometry=geometry,
        axial=Axial(**axial),
        magnet=Magnet(**magnet),
        winding=Winding(**winding),
        steel=Steel(**steel),
    )
    winding_problems = _check_slot_fill(design, warnings)
    if winding_problems:
        raise DesignError(winding_problems, source, warnings)
    return dataclasses.replace(design, warnings=tuple(warnings))


def _check_slot_fill(design: Design, warnings: list[Problem]) -> list[Problem]:
    """Reject a winding that cannot fit its slots; warn about one hard to wind."""
    try:
        shape = machine_for(design).coil_shape()
    except NotImplementedError:
        return []  # this machine does not describe its coils
    winding = design.winding
    mm = 1e-3
    fill = slot_fill(
        shape,
        winding.turns_per_coil,
        winding.parallel_strands,
        winding.wire_diameter_mm * mm,
        None if winding.insulated_diameter_mm is None else winding.insulated_diameter_mm * mm,
    )
    wires = (
        f"{winding.turns_per_coil} turns of {winding.parallel_strands} x "
        f"{winding.wire_diameter_mm:g} mm wire"
    )
    area = f"{fill.side_area / mm**2:.3g} mm2"
    if not fill.fits:
        measure = (
            "insulated"
            if fill.insulated
            else "bare (give insulated_diameter_mm to include the enamel)"
        )
        return [
            Problem(
                "winding.turns_per_coil",
                f"the slot is over-filled: {wires} take {fill.packing:.0%} of a coil side's "
                f"{area}, {measure}; round wires cannot fill more than {MAX_PACKING:.1%} even "
                "packed perfectly. Use fewer turns, fewer strands or thinner wire",
            )
        ]
    if fill.copper_fill > COPPER_FILL_ADVICE:
        warnings.append(
            Problem(
                "winding.turns_per_coil",
                f"copper fill factor {fill.copper_fill:.2f} ({wires} in a coil side of {area}) "
                f"is above {COPPER_FILL_ADVICE:g}, more than round wire wound on a tooth "
                "usually reaches; check that it can be wound",
            )
        )
    return []


def design_to_dict(design: Design) -> dict[str, Any]:
    """The design as a nested dict with the same layout as a design file.

    Defaults are written out, so ``design_from_dict(design_to_dict(d)) == d``.
    Warnings are not part of the design and are left out.
    """
    data = dataclasses.asdict(design)
    del data["warnings"]
    return data
