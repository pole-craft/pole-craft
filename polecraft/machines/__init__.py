"""Machine plug-ins. Each one turns a design into regions, excitation and symmetry.

The table below is the only place that maps a design file's ``type`` to code.
Everything outside this package looks machines up here and never names one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polecraft.machines.base import Machine, MachineParameters
from polecraft.machines.qdd_outer import QddOuter

if TYPE_CHECKING:
    from polecraft.design import Design

_MACHINES: dict[str, type[Machine]] = {
    "qdd_outer": QddOuter,
}


def machine_types() -> list[str]:
    """Names accepted as ``type`` in a design file."""
    return sorted(_MACHINES)


def parameters_class(machine_type: str) -> type[MachineParameters]:
    """The parameter class for a machine type. Raises KeyError if unknown."""
    return _MACHINES[machine_type].Parameters


def machine_for(design: Design) -> Machine:
    """The machine a design describes, ready to build its geometry."""
    return _MACHINES[design.type](design)
