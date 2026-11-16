"""The field problem the solver is given, and how one is built for a machine.

A ``Problem`` is everything the solver needs and nothing else: a mesh, a
material for every triangle, the current density and magnetisation of every
triangle, the nodes held at a fixed potential and the node pairs that couple
the cut edges of a slice. It can be put together by hand from arrays, which
is how the solver is tested, or by ``build_problem`` from a machine and its
mesh.

Lengths are in metres. Current density is in A/m2, positive out of the plane,
and magnetisation is the remanence vector in tesla.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from polecraft.core.materials import AIR, COPPER, Magnet, Material

if TYPE_CHECKING:
    from polecraft.core.geometry import Geometry
    from polecraft.core.mesh import Mesh
    from polecraft.machines.base import Machine

# Materials every machine has, unless the caller gives its own.
USUAL = {"air": AIR, "copper": COPPER}


@dataclass(frozen=True, eq=False)
class Problem:
    """One magnetic field problem on a triangle mesh.

    Only ``nodes``, ``triangles``, ``materials`` and the fixed nodes have to
    be given. Left out, the triangles all take the first material, there is
    no current, no magnetisation and no coupling, and the fixed nodes are
    held at zero. A single current density, remanence vector or fixed value
    stands for all triangles or fixed nodes. A magnetised triangle needs a
    linear material; the solver refuses it otherwise.
    """

    nodes: NDArray[np.float64]  # (n, 2) positions in metres
    triangles: NDArray[np.intp]  # (m, 3) node numbers
    materials: tuple[Material, ...]
    triangle_material: NDArray[np.intp] = None  # (m,) index into materials
    current_density: NDArray[np.float64] = None  # (m,) A/m2, positive out of the plane
    remanence: NDArray[np.float64] = None  # (m, 2) Br in tesla, zero outside magnets
    fixed_nodes: NDArray[np.intp] = ()  # (k,) nodes held at a potential
    fixed_values: NDArray[np.float64] = 0.0  # (k,) their potential in Wb/m
    periodic: NDArray[np.intp] = None  # (p, 2) node pairs: copy, original
    antiperiodic: bool = False  # a copy takes the opposite of its original

    def __post_init__(self) -> None:
        nodes = np.array(self.nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(f"nodes must be an (n, 2) array of positions, not {nodes.shape}")
        triangles = _whole_numbers(self.triangles, "triangles")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(
                f"triangles must be an (m, 3) array of node numbers, not {triangles.shape}"
            )
        _check_nodes(triangles, len(nodes), "a triangle")
        count = len(triangles)

        materials = tuple(self.materials)
        if not materials:
            raise ValueError("a problem needs at least one material")
        if self.triangle_material is None:
            material = np.zeros(count, dtype=np.intp)
        else:
            material = _whole_numbers(self.triangle_material, "triangle_material")
            if material.shape != (count,):
                raise ValueError(
                    f"there must be one material per triangle ({count}), not {material.shape}"
                )
            wrong = material[(material < 0) | (material >= len(materials))]
            if wrong.size:
                raise ValueError(
                    f"a triangle names material {wrong[0]}, but there are materials "
                    f"0 to {len(materials) - 1}"
                )

        density = _per_triangle(self.current_density, (count,), "current density")
        remanence = _per_triangle(self.remanence, (count, 2), "remanence")

        fixed = _whole_numbers(self.fixed_nodes, "fixed_nodes").ravel()
        _check_nodes(fixed, len(nodes), "fixed_nodes")
        values = np.array(self.fixed_values, dtype=float)
        if values.ndim == 0:
            values = np.full(fixed.size, float(values))
        elif values.shape != (fixed.size,):
            raise ValueError(
                f"fixed values must be one value, or one per fixed node ({fixed.size}), "
                f"not {values.shape}"
            )

        if self.periodic is None or np.size(self.periodic) == 0:
            pairs = np.zeros((0, 2), dtype=np.intp)
        else:
            pairs = _whole_numbers(self.periodic, "periodic")
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError(
                    f"periodic must be pairs of node numbers, shape (p, 2), not {pairs.shape}"
                )
            _check_nodes(pairs, len(nodes), "periodic")

        for name, value in (
            ("nodes", nodes),
            ("triangles", triangles),
            ("materials", materials),
            ("triangle_material", material),
            ("current_density", density),
            ("remanence", remanence),
            ("fixed_nodes", fixed),
            ("fixed_values", values),
            ("periodic", pairs),
            ("antiperiodic", bool(self.antiperiodic)),
        ):
            object.__setattr__(self, name, value)


def _whole_numbers(values: ArrayLike, name: str) -> NDArray[np.intp]:
    array = np.asarray(values)
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must hold whole numbers")
    return array.astype(np.intp)


def _check_nodes(numbers: NDArray[np.intp], count: int, name: str) -> None:
    wrong = numbers[(numbers < 0) | (numbers >= count)]
    if wrong.size:
        raise ValueError(f"{name} names node {wrong.flat[0]}, but there are nodes 0 to {count - 1}")


def _per_triangle(values: ArrayLike | None, shape: tuple[int, ...], name: str) -> NDArray:
    """One value for every triangle, from nothing, one value or a full array."""
    if values is None:
        return np.zeros(shape)
    array = np.array(values, dtype=float)
    if array.shape == shape[1:]:
        array = np.broadcast_to(array, shape).copy()
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape} or {shape[1:]}, not {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite numbers")
    return array


def build_problem(
    machine: Machine,
    mesh: Mesh,
    materials: Mapping[str, Material],
    current: float = 0.0,
    advance: float = 0.0,
) -> Problem:
    """The field problem of a machine on its mesh, at rotor angle 0.

    ``mesh`` is the mesh of the machine's repeating slice or of the whole
    machine. ``materials`` maps the grade names used in the design (steel,
    magnet) to their models; a magnet grade needs a ``Magnet``. Air and
    copper are known already. ``current`` is the peak phase current in
    amperes and ``advance`` its angle ahead of the back-EMF in electrical
    radians, both as in ``Machine.excitation``.

    Each coil side carries its turns times its phase current, spread evenly
    over its area. Each magnet is magnetised to its grade's remanence along
    its own direction. The inner and outer surfaces of the machine are field
    lines (potential 0): no flux leaves the iron there. The cut edges of a
    slice are coupled as the mesh says.

    Only what the machine contract offers is used: the regions from
    ``build_geometry`` and the phase currents from ``excitation``.
    """
    geometry = _geometry_of(machine, mesh)
    known = {**USUAL, **materials}
    models: list[Material] = []
    triangle_material = np.zeros(len(mesh.triangles), dtype=np.intp)
    density = np.zeros(len(mesh.triangles))
    remanence = np.zeros((len(mesh.triangles), 2))

    def index_of(name: str) -> int:
        if name not in known:
            raise ValueError(f"no material model for '{name}'")
        model = known[name]
        if model not in models:
            models.append(model)
        return models.index(model)

    air = index_of("air")
    triangle_material[:] = air  # the band, which is no region, is air
    phase_currents = machine.excitation(0.0, current, advance)
    area = mesh.triangle_areas()
    centre = mesh.nodes[mesh.triangles].mean(axis=1)

    for number, region in enumerate(geometry.regions):
        inside = mesh.triangle_region == number
        triangle_material[inside] = index_of(region.material)
        if region.coil_side is not None:
            side = region.coil_side
            ampere_turns = side.sign * side.turns * phase_currents[side.phase]
            density[inside] = ampere_turns / area[inside].sum()
        if region.magnetisation is not None:
            grade = known[region.material]
            if not isinstance(grade, Magnet):
                raise ValueError(
                    f"'{region.material}' is the grade of {region.name}, "
                    "so its model must be a magnet"
                )
            remanence[inside] = grade.remanence * _directions(region.magnetisation, centre[inside])

    radius = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    on_surface = np.isclose(radius, geometry.inner_radius) | np.isclose(
        radius, geometry.outer_radius
    )
    return Problem(
        nodes=mesh.nodes,
        triangles=mesh.triangles,
        materials=tuple(models),
        triangle_material=triangle_material,
        current_density=density,
        remanence=remanence,
        fixed_nodes=np.flatnonzero(on_surface),
        fixed_values=0.0,
        periodic=mesh.periodic,
        antiperiodic=mesh.antiperiodic,
    )


# Meshed and exact region areas differ by the chords that stand in for arcs,
# under 1 % on any usable mesh. More than this and the mesh is of another
# machine.
AREA_MISMATCH = 0.03


def _geometry_of(machine: Machine, mesh: Mesh) -> Geometry:
    """The machine's geometry that the mesh was made from: slice or whole.

    The regions must have the machine's names, radii and areas. That catches
    a mesh of another design; it cannot tell two designs apart that differ
    by less than a few percent in every region.
    """
    names: Sequence[str] = mesh.region_names[:-1]  # the last name is the band
    radius = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    for full in (False, True):
        geometry = machine.build_geometry(full=full)
        if tuple(region.name for region in geometry.regions) != tuple(names):
            continue
        exact = np.array([region.area() for region in geometry.regions])
        meshed = np.bincount(
            mesh.triangle_region, weights=mesh.triangle_areas(), minlength=len(exact) + 1
        )[: len(exact)]
        if (
            np.isclose(radius.min(), geometry.inner_radius)
            and np.isclose(radius.max(), geometry.outer_radius)
            and np.all(np.abs(meshed - exact) <= AREA_MISMATCH * exact)
        ):
            return geometry
    raise ValueError("the mesh does not belong to this machine: its regions are not the machine's")


def _directions(magnetisation: object, points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit vectors of a magnetisation at many points, shape (k, 2)."""
    if magnetisation.pattern == "parallel":
        angle = np.full(len(points), magnetisation.axis_angle)
    else:
        angle = np.arctan2(points[:, 1], points[:, 0])
    return magnetisation.polarity * np.column_stack([np.cos(angle), np.sin(angle)])
