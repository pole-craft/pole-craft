"""The contract every machine plug-in follows."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Protocol

from polecraft.core.geometry import Geometry
from polecraft.core.winding import CoilShape, Symmetry, phase_axis_angle, symmetry
from polecraft.validation import Problem, Section

if TYPE_CHECKING:
    from polecraft.design import Design


class MachineParameters(Protocol):
    """What a machine's parameter class must offer to the design loader."""

    @classmethod
    def from_section(cls, section: Section) -> MachineParameters | None:
        """Read the ``geometry`` block; None if any field is unusable."""
        ...

    def check(self, slots: int, poles: int) -> list[Problem]:
        """Problems that involve several fields, such as parts that do not fit."""
        ...


class Machine(ABC):
    """One machine type, built around a checked design.

    A plug-in sets ``Parameters`` to the class that reads its ``geometry``
    block and implements ``build_geometry``. The rest of the program works
    with the returned regions and never asks which machine they came from.
    """

    Parameters: ClassVar[type[MachineParameters]]

    def __init__(self, design: Design) -> None:
        self.design = design

    def parameters(self) -> MachineParameters:
        """The dimensions that define this machine."""
        return self.design.geometry

    def symmetry(self) -> Symmetry:
        """The smallest repeating slice."""
        return symmetry(self.design.slots, self.design.poles)

    @abstractmethod
    def build_geometry(self, full: bool = False) -> Geometry:
        """Regions of the smallest repeating slice, or of the whole machine.

        Tooth 0 is centred on the x axis. The rotor is built at rotor angle
        0, with magnet 0 centred on the x axis and magnetised away from the
        machine axis.
        """

    def coil_shape(self) -> CoilShape:
        """Size of the coil on one tooth, for slot fill, turn length and resistance.

        A machine that does not provide it has its winding left unchecked.
        """
        raise NotImplementedError(f"{type(self).__name__} does not describe its coils")

    def excitation(self, angle: float, current: float, advance: float = 0.0) -> dict[str, float]:
        """Phase currents in amperes for a rotor position.

        ``angle`` is the rotor's mechanical angle in radians, counter-clockwise
        from the position the geometry is built in. ``current`` is the peak
        current in one coil's conductor. ``advance`` shifts the current ahead
        in electrical radians; 0 puts it in phase with the back-EMF, the
        setting for most torque per ampere in a surface-magnet machine.

        With ``p`` pole pairs and the phase A axis at electrical angle ``a``,
        the flux linked by phase A varies as cos(p * angle - a). Torque is
        current times the rate of change of that flux with angle, so current
        varying as cos(p * angle - a + pi/2) drives the rotor
        counter-clockwise. B and C follow 120 and 240 degrees later.

        This assumes a three-phase winding with a coil on every tooth and the
        placement described in ``build_geometry``. A machine that differs
        must override it.
        """
        pole_pairs = self.design.poles // 2
        axis = phase_axis_angle(self.design.slots, self.design.poles)
        phase_a = pole_pairs * angle - axis + math.pi / 2 + advance
        return {
            phase: current * math.cos(phase_a - k * 2 * math.pi / 3)
            for k, phase in enumerate("ABC")
        }
