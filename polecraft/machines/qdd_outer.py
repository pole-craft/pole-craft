"""Surface-magnet outer-rotor machine: the dimensions that define it.

The cross-section is given as one outer diameter followed by radial
thicknesses, working inwards::

    rotor yoke | magnet | air gap | tooth tip | tip taper | slot | stator yoke | bore

Every radius is derived from that stack, so the layers cannot contradict each
other. Teeth have parallel sides (constant width), which makes the slots
narrower towards the bore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polecraft.validation import Problem, Section

MAGNETISATIONS = ("parallel", "radial")


@dataclass(frozen=True)
class Rotor:
    outer_diameter_mm: float
    yoke_thickness_mm: float  # back iron behind the magnets


@dataclass(frozen=True)
class MagnetShape:
    """Arc magnet: concentric inner and outer faces, radial sides."""

    thickness_mm: float  # radial
    arc_fraction: float  # magnet angle / pole angle (360 deg / poles)
    magnetisation: str  # "parallel" to the magnet centre line, or "radial"


@dataclass(frozen=True)
class Stator:
    tooth_width_mm: float  # parallel-sided tooth body
    slot_opening_mm: float  # gap between neighbouring tooth tips at the air gap
    tooth_tip_height_mm: float  # radial height of the tip edge at the opening
    tooth_tip_taper_mm: float  # radial height of the sloped underside; 0 for a square tip
    slot_depth_mm: float  # radial, from where the tip meets the tooth to the slot bottom
    yoke_thickness_mm: float  # back iron between slot bottom and bore


@dataclass(frozen=True)
class QddOuterParameters:
    """Cross-section of an outer-rotor surface-magnet machine, in millimetres."""

    air_gap_mm: float
    rotor: Rotor
    magnet: MagnetShape
    stator: Stator

    @classmethod
    def from_section(cls, section: Section) -> QddOuterParameters | None:
        """Read the ``geometry`` block. Returns None if any field is unusable."""
        before = len(section.problems)

        air_gap = section.number("air_gap_mm", gt=0)

        rotor = section.section("rotor")
        outer_diameter = rotor.number("outer_diameter_mm", gt=0)
        rotor_yoke = rotor.number("yoke_thickness_mm", gt=0)

        magnet = section.section("magnet")
        magnet_thickness = magnet.number("thickness_mm", gt=0)
        arc_fraction = magnet.number("arc_fraction", gt=0, le=1)
        magnetisation = magnet.text(
            "magnetisation", default="parallel", choices=MAGNETISATIONS
        )

        stator = section.section("stator")
        tooth_width = stator.number("tooth_width_mm", gt=0)
        slot_opening = stator.number("slot_opening_mm", gt=0)
        tip_height = stator.number("tooth_tip_height_mm", gt=0)
        tip_taper = stator.number("tooth_tip_taper_mm", ge=0)
        slot_depth = stator.number("slot_depth_mm", gt=0)
        stator_yoke = stator.number("yoke_thickness_mm", gt=0)

        if len(section.problems) > before:
            return None
        return cls(
            air_gap_mm=air_gap,
            rotor=Rotor(outer_diameter_mm=outer_diameter, yoke_thickness_mm=rotor_yoke),
            magnet=MagnetShape(
                thickness_mm=magnet_thickness,
                arc_fraction=arc_fraction,
                magnetisation=magnetisation,
            ),
            stator=Stator(
                tooth_width_mm=tooth_width,
                slot_opening_mm=slot_opening,
                tooth_tip_height_mm=tip_height,
                tooth_tip_taper_mm=tip_taper,
                slot_depth_mm=slot_depth,
                yoke_thickness_mm=stator_yoke,
            ),
        )

    # Radii in mm, from the outside in.

    @property
    def rotor_outer_radius_mm(self) -> float:
        return self.rotor.outer_diameter_mm / 2

    @property
    def magnet_outer_radius_mm(self) -> float:
        """Inner face of the rotor yoke, where the magnets are bonded."""
        return self.rotor_outer_radius_mm - self.rotor.yoke_thickness_mm

    @property
    def magnet_inner_radius_mm(self) -> float:
        """Magnet face on the air gap."""
        return self.magnet_outer_radius_mm - self.magnet.thickness_mm

    @property
    def stator_outer_radius_mm(self) -> float:
        """Tooth-tip face on the air gap."""
        return self.magnet_inner_radius_mm - self.air_gap_mm

    @property
    def slot_top_radius_mm(self) -> float:
        """Where the underside of the tooth tip meets the tooth flank."""
        return (
            self.stator_outer_radius_mm
            - self.stator.tooth_tip_height_mm
            - self.stator.tooth_tip_taper_mm
        )

    @property
    def slot_bottom_radius_mm(self) -> float:
        return self.slot_top_radius_mm - self.stator.slot_depth_mm

    @property
    def bore_radius_mm(self) -> float:
        """Inner radius of the stator yoke."""
        return self.slot_bottom_radius_mm - self.stator.yoke_thickness_mm

    def check(self, slots: int, poles: int) -> list[Problem]:
        """Return the reasons this cross-section cannot be built, if any.

        Paths are relative to the ``geometry`` block. Later checks need the
        earlier ones to hold, so only the first failure is returned.
        """
        rotor, magnet, stator = self.rotor, self.magnet, self.stator
        outer_radius = self.rotor_outer_radius_mm

        if rotor.yoke_thickness_mm >= outer_radius:
            return [
                Problem(
                    "rotor.yoke_thickness_mm",
                    f"{rotor.yoke_thickness_mm:g} mm is not less than the outer "
                    f"radius of {outer_radius:g} mm",
                )
            ]

        if magnet.thickness_mm >= self.magnet_outer_radius_mm:
            return [
                Problem(
                    "magnet.thickness_mm",
                    f"{magnet.thickness_mm:g} mm is thicker than the rotor allows: "
                    f"only {self.magnet_outer_radius_mm:g} mm lies inside the rotor yoke",
                )
            ]

        if self.bore_radius_mm <= 0:
            layers = [
                ("rotor yoke", rotor.yoke_thickness_mm),
                ("magnet", magnet.thickness_mm),
                ("air gap", self.air_gap_mm),
                ("tooth tip", stator.tooth_tip_height_mm + stator.tooth_tip_taper_mm),
                ("slot", stator.slot_depth_mm),
                ("stator yoke", stator.yoke_thickness_mm),
            ]
            total = sum(thickness for _, thickness in layers)
            listed = " + ".join(f"{name} {thickness:g}" for name, thickness in layers)
            return [
                Problem(
                    "",
                    f"the layers add up to {total:g} mm, which does not fit inside "
                    f"the outer radius of {outer_radius:g} mm ({listed})",
                )
            ]

        # Half the angle between neighbouring teeth.
        half_pitch = math.pi / slots

        # A tooth of width w has its flank at angle asin(w / 2r) from the tooth
        # axis at radius r. The slot is narrowest at its bottom, and closes
        # there when the flank reaches the half pitch: w = 2 r sin(pi / slots).
        widest_tooth = 2 * self.slot_bottom_radius_mm * math.sin(half_pitch)
        if stator.tooth_width_mm >= widest_tooth:
            return [
                Problem(
                    "stator.tooth_width_mm",
                    f"{stator.tooth_width_mm:g} mm teeth leave no slot: with {slots} "
                    f"slots the teeth touch at the slot bottom at {widest_tooth:.4g} mm",
                )
            ]

        # The tip edge may not lie inside the radial line through the tooth
        # corner at the slot top, or the tip would be narrower than its tooth.
        flank = math.asin(stator.tooth_width_mm / (2 * self.slot_top_radius_mm))
        widest_opening = 2 * self.stator_outer_radius_mm * math.sin(half_pitch - flank)
        if stator.slot_opening_mm > widest_opening:
            return [
                Problem(
                    "stator.slot_opening_mm",
                    f"{stator.slot_opening_mm:g} mm is wider than the slot: with "
                    f"{stator.tooth_width_mm:g} mm teeth the opening can be at most "
                    f"{widest_opening:.4g} mm",
                )
            ]

        return []
