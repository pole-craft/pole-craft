"""Machine plug-ins. Each one turns a design into regions, excitation and symmetry.

The table below is the only place that maps a design file's ``type`` to code.
Everything outside this package looks machines up here and never names one.
"""

from __future__ import annotations

from polecraft.machines.base import MachineParameters
from polecraft.machines.qdd_outer import QddOuterParameters

_PARAMETERS: dict[str, type[MachineParameters]] = {
    "qdd_outer": QddOuterParameters,
}


def machine_types() -> list[str]:
    """Names accepted as ``type`` in a design file."""
    return sorted(_PARAMETERS)


def parameters_class(machine_type: str) -> type[MachineParameters]:
    """The parameter class for a machine type. Raises KeyError if unknown."""
    return _PARAMETERS[machine_type]
