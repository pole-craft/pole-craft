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

from polecraft.core.geometry import (
    CoilSide,
    Geometry,
    Magnetisation,
    Region,
    closed_loop,
    ring_sector,
)
from polecraft.core.winding import Coil, CoilShape, concentrated_winding
from polecraft.machines.base import Machine
from polecraft.validation import Problem, Section

MM = 1e-3  # design files are in millimetres, geometry in metres

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
    slot_opening_mm: float  # straight-line gap between tooth tips at the air-gap surface
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

        # The sloped underside of the tip is a straight line from the tip
        # corner to the tooth corner at the slot top. If the tip overhangs far
        # and the taper is shallow, that line dips below the slot-top circle
        # and into the coil. It stays clear when r_tip cos(overhang) >= r_top.
        taper = stator.tooth_tip_taper_mm
        tip_radius = self.stator_outer_radius_mm - stator.tooth_tip_height_mm
        edge = half_pitch - math.asin(
            stator.slot_opening_mm / (2 * self.stator_outer_radius_mm)
        )
        overhang = edge - flank
        if 0 < taper and tip_radius * math.cos(overhang) < self.slot_top_radius_mm:
            # Rounded up, so that the suggested value is itself accepted.
            needed = tip_radius * (1 - math.cos(overhang))
            step = 10 ** (math.floor(math.log10(needed)) - 2)
            needed = math.ceil(needed / step) * step
            return [
                Problem(
                    "stator.tooth_tip_taper_mm",
                    f"{taper:g} mm is too shallow for a tip that overhangs this far: "
                    f"its underside would cut into the slot. Use about {needed:.3g} mm "
                    f"or more, or 0 for a square tip",
                )
            ]

        return []


class QddOuter(Machine):
    """Outer-rotor surface-magnet machine with a coil on every tooth.

    The stator is built tooth by tooth and the rotor pole by pole, each in its
    own frame. Tooth 0 is centred on the x axis. At rotor angle 0, magnet 0 is
    centred on the x axis too and is magnetised away from the machine axis.

    The air gap is three layers of equal thickness: one fixed to the stator,
    one fixed to the rotor, and between them the band that is reconnected as
    the rotor turns. The slice of the stator is cut along slot centre lines
    and that of the rotor between magnets, so no tooth, coil or magnet is cut.
    """

    Parameters = QddOuterParameters

    def build_geometry(self, full: bool = False) -> Geometry:
        """Regions of the smallest repeating slice, or of the whole machine."""
        design = self.design
        dims: QddOuterParameters = design.geometry
        repeat = self.symmetry()
        coils = concentrated_winding(design.slots, design.poles)

        teeth = design.slots if full else repeat.slots
        poles = design.poles if full else repeat.poles
        stator = [region for k in range(teeth) for region in self._tooth_unit(coils[k])]
        rotor = [region for j in range(poles) for region in self._pole_unit(j)]

        layer = dims.air_gap_mm * MM / 3
        return Geometry(
            stator=tuple(stator),
            rotor=tuple(rotor),
            inner_radius=dims.bore_radius_mm * MM,
            outer_radius=dims.rotor_outer_radius_mm * MM,
            band_inner_radius=dims.stator_outer_radius_mm * MM + layer,
            band_outer_radius=dims.magnet_inner_radius_mm * MM - layer,
            stator_start_angle=-math.pi / design.slots,
            rotor_start_angle=-math.pi / design.poles,
            span_angle=2 * math.pi if full else 2 * math.pi / repeat.periods,
            antiperiodic=repeat.antiperiodic and not full,
            stack_length=design.axial.stack_length_mm * MM,
        )

    def coil_shape(self) -> CoilShape:
        """The coil on a parallel-sided tooth, filling half the slot on each side.

        A coil side runs from the slot bottom to the slot top, so its mean
        thickness across the slot is its area over that depth.
        """
        dims: QddOuterParameters = self.design.geometry
        coil = concentrated_winding(self.design.slots, self.design.poles)[0]
        side = next(r for r in self._tooth_unit(coil) if r.kind == "coil")
        depth = dims.stator.slot_depth_mm * MM
        return CoilShape(
            tooth_width=dims.stator.tooth_width_mm * MM,
            thickness=side.area() / depth,
            side_area=side.area(),
        )

    def _tooth_unit(self, coil: Coil) -> list[Region]:
        """One slot pitch of the stator: a tooth, half a slot each side, yoke and air."""
        design = self.design
        dims: QddOuterParameters = design.geometry
        steel, copper = design.steel.stator, design.winding.conductor
        k = coil.tooth

        half_pitch = math.pi / design.slots
        centre = 2 * k * half_pitch
        # Written as odd multiples of the half pitch so that neighbouring
        # units compute exactly the same edge.
        low, high = (2 * k - 1) * half_pitch, (2 * k + 1) * half_pitch

        bore = dims.bore_radius_mm * MM
        bottom = dims.slot_bottom_radius_mm * MM
        top = dims.slot_top_radius_mm * MM
        surface = dims.stator_outer_radius_mm * MM
        # With no taper the underside of the tip is an arc at the slot top, and
        # the tip corner must be at exactly that radius.
        square_tip = dims.stator.tooth_tip_taper_mm == 0
        tip = top if square_tip else surface - dims.stator.tooth_tip_height_mm * MM
        gap_layer = surface + dims.air_gap_mm * MM / 3
        half_width = dims.stator.tooth_width_mm * MM / 2

        # Angle of the tooth flank from the tooth axis at the slot bottom and top.
        flank_bottom = math.asin(half_width / bottom)
        flank_top = math.asin(half_width / top)
        # Angle of the tip edge: half a pitch less half the slot opening.
        edge = half_pitch - math.asin(dims.stator.slot_opening_mm * MM / (2 * surface))

        under = "cw" if square_tip else "line"

        tooth = closed_loop(
            [
                (bottom, centre - flank_bottom, "line"),
                (top, centre - flank_top, under),
                (tip, centre - edge, "line"),
                (surface, centre - edge, "ccw"),
                (surface, centre + edge, "line"),
                (tip, centre + edge, under),
                (top, centre + flank_top, "line"),
                (bottom, centre + flank_bottom, "cw"),
            ]
        )

        # The coil sides fill the slot from its bottom to where the tip begins.
        # A square tip puts a corner of the slot air on the coil's top arc.
        tip_corner_ccw = [(top, centre + edge, "ccw")] if square_tip else []
        tip_corner_cw = [(top, centre - edge, "ccw")] if square_tip else []
        coil_ccw = closed_loop(
            [(bottom, centre + flank_bottom, "line"), (top, centre + flank_top, "ccw")]
            + tip_corner_ccw
            + [(top, high, "line"), (bottom, high, "cw")]
        )
        coil_cw = closed_loop(
            [(bottom, low, "line"), (top, low, "ccw")]
            + tip_corner_cw
            + [(top, centre - flank_top, "line"), (bottom, centre - flank_bottom, "cw")]
        )

        # Air under the tip and in the slot opening.
        if square_tip:
            air_ccw = closed_loop(
                [
                    (top, centre + edge, "line"),
                    (surface, centre + edge, "ccw"),
                    (surface, high, "line"),
                    (top, high, "cw"),
                ]
            )
            air_cw = closed_loop(
                [
                    (top, low, "line"),
                    (surface, low, "ccw"),
                    (surface, centre - edge, "line"),
                    (top, centre - edge, "cw"),
                ]
            )
        else:
            air_ccw = closed_loop(
                [
                    (top, centre + flank_top, "line"),
                    (tip, centre + edge, "line"),
                    (surface, centre + edge, "ccw"),
                    (surface, high, "line"),
                    (top, high, "cw"),
                ]
            )
            air_cw = closed_loop(
                [
                    (top, low, "line"),
                    (surface, low, "ccw"),
                    (surface, centre - edge, "line"),
                    (tip, centre - edge, "line"),
                    (top, centre - flank_top, "cw"),
                ]
            )

        yoke = ring_sector(
            bore, bottom, low, high,
            outer_breaks=(centre - flank_bottom, centre + flank_bottom),
        )
        gap = ring_sector(
            surface, gap_layer, low, high, inner_breaks=(centre - edge, centre + edge)
        )

        # Positive current in a +1 coil comes out of the plane on the
        # counter-clockwise side of its tooth and returns on the other side.
        turns = design.winding.turns_per_coil
        side_ccw = CoilSide(phase=coil.phase, tooth=k, sign=coil.direction, turns=turns)
        side_cw = CoilSide(phase=coil.phase, tooth=k, sign=-coil.direction, turns=turns)

        def steel_region(name: str, role: str, boundary: tuple) -> Region:
            return Region(name=name, kind="steel", role=role, material=steel, boundary=boundary)

        def air_region(name: str, role: str, boundary: tuple) -> Region:
            return Region(name=name, kind="air", role=role, material="air", boundary=boundary)

        def coil_region(name: str, boundary: tuple, side: CoilSide) -> Region:
            return Region(
                name=name, kind="coil", role="coil", material=copper,
                boundary=boundary, coil_side=side,
            )

        return [
            steel_region(f"stator_yoke_{k}", "stator_yoke", yoke),
            steel_region(f"tooth_{k}", "tooth", tooth),
            coil_region(f"coil_{k}_cw", coil_cw, side_cw),
            coil_region(f"coil_{k}_ccw", coil_ccw, side_ccw),
            air_region(f"slot_air_{k}_cw", "slot_air", air_cw),
            air_region(f"slot_air_{k}_ccw", "slot_air", air_ccw),
            air_region(f"gap_stator_{k}", "air_gap", gap),
        ]

    def _pole_unit(self, j: int) -> list[Region]:
        """One pole pitch of the rotor: a magnet, the air beside it, yoke and air."""
        design = self.design
        dims: QddOuterParameters = design.geometry

        half_pitch = math.pi / design.poles
        centre = 2 * j * half_pitch
        low, high = (2 * j - 1) * half_pitch, (2 * j + 1) * half_pitch

        outer = dims.rotor_outer_radius_mm * MM
        magnet_outer = dims.magnet_outer_radius_mm * MM
        magnet_inner = dims.magnet_inner_radius_mm * MM
        gap_layer = magnet_inner - dims.air_gap_mm * MM / 3

        # Magnets alternate: even ones point away from the axis, odd ones towards it.
        magnetisation = Magnetisation(
            pattern=dims.magnet.magnetisation,
            axis_angle=centre,
            polarity=1 if j % 2 == 0 else -1,
        )

        def air_region(name: str, role: str, boundary: tuple) -> Region:
            return Region(name=name, kind="air", role=role, material="air", boundary=boundary)

        def magnet_region(boundary: tuple) -> Region:
            return Region(
                name=f"magnet_{j}", kind="magnet", role="magnet",
                material=design.magnet.material, boundary=boundary,
                magnetisation=magnetisation,
            )

        if dims.magnet.arc_fraction == 1:
            # A full ring of magnets: no air between them.
            magnet = ring_sector(magnet_inner, magnet_outer, low, high)
            regions = [magnet_region(magnet)]
            corners: tuple[float, ...] = ()
        else:
            half_magnet = dims.magnet.arc_fraction * half_pitch
            start, end = centre - half_magnet, centre + half_magnet
            magnet = ring_sector(magnet_inner, magnet_outer, start, end)
            air_cw = ring_sector(magnet_inner, magnet_outer, low, start)
            air_ccw = ring_sector(magnet_inner, magnet_outer, end, high)
            regions = [
                magnet_region(magnet),
                air_region(f"magnet_gap_{j}_cw", "magnet_gap", air_cw),
                air_region(f"magnet_gap_{j}_ccw", "magnet_gap", air_ccw),
            ]
            corners = (start, end)

        yoke = ring_sector(magnet_outer, outer, low, high, inner_breaks=corners)
        gap = ring_sector(gap_layer, magnet_inner, low, high, outer_breaks=corners)
        return regions + [
            Region(
                name=f"rotor_yoke_{j}", kind="steel", role="rotor_yoke",
                material=design.steel.rotor, boundary=yoke,
            ),
            air_region(f"gap_rotor_{j}", "air_gap", gap),
        ]
