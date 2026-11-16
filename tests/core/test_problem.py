"""The solver's problem: built by hand, and built from a machine and its mesh."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from polecraft.core.materials import AIR, COPPER, MU0, FrohlichSteel, LinearMaterial, Magnet
from polecraft.core.problem import Problem, build_problem

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
TWO_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


# A problem built by hand


def test_smallest_problem_fills_in_the_rest() -> None:
    problem = Problem(
        nodes=SQUARE, triangles=TWO_TRIANGLES, materials=(AIR, LinearMaterial(1000.0)),
        fixed_nodes=[0],
    )
    assert np.array_equal(problem.triangle_material, [0, 0])
    assert np.array_equal(problem.current_density, [0.0, 0.0])
    assert np.array_equal(problem.remanence, np.zeros((2, 2)))
    assert np.array_equal(problem.fixed_nodes, [0])
    assert np.array_equal(problem.fixed_values, [0.0])
    assert problem.periodic.shape == (0, 2)
    assert not problem.antiperiodic


def test_problem_keeps_what_it_is_given() -> None:
    iron = LinearMaterial(1000.0)
    problem = Problem(
        nodes=SQUARE,
        triangles=TWO_TRIANGLES,
        materials=(AIR, iron),
        triangle_material=[1, 0],
        current_density=[2e6, 0.0],
        remanence=[[0.0, 0.0], [0.0, 1.2]],
        fixed_nodes=[0, 1],
        fixed_values=[0.0, 0.5],
        periodic=[[3, 0]],
        antiperiodic=True,
    )
    assert problem.materials == (AIR, iron)
    assert np.array_equal(problem.triangle_material, [1, 0])
    assert np.array_equal(problem.current_density, [2e6, 0.0])
    assert np.array_equal(problem.remanence, [[0.0, 0.0], [0.0, 1.2]])
    assert np.array_equal(problem.fixed_values, [0.0, 0.5])
    assert np.array_equal(problem.periodic, [[3, 0]])
    assert problem.antiperiodic
    assert problem.nodes.dtype == np.float64
    assert problem.current_density.dtype == np.float64


def test_problem_copies_its_arrays() -> None:
    nodes, density = SQUARE.copy(), np.array([1.0, 2.0])
    problem = Problem(
        nodes=nodes, triangles=TWO_TRIANGLES, materials=(AIR,), current_density=density,
        fixed_nodes=[0], fixed_values=0.0,
    )
    nodes[0, 0], density[0] = 9.0, 9.0
    assert problem.nodes[0, 0] == 0.0
    assert problem.current_density[0] == 1.0


def test_one_value_can_stand_for_all_triangles() -> None:
    problem = Problem(
        nodes=SQUARE, triangles=TWO_TRIANGLES, materials=(AIR,), current_density=3.0,
        remanence=[0.0, 1.1], fixed_nodes=[0, 1], fixed_values=0.25,
    )
    assert np.array_equal(problem.current_density, [3.0, 3.0])
    assert np.array_equal(problem.remanence, [[0.0, 1.1], [0.0, 1.1]])
    assert np.array_equal(problem.fixed_values, [0.25, 0.25])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"nodes": np.zeros((4, 3))}, "nodes"),
        ({"triangles": [[0, 1, 2, 3]]}, "triangles"),
        ({"triangles": [[0, 1, 7]]}, "node 7"),
        ({"triangles": [[0.0, 1.0, 2.0]]}, "whole numbers"),
        ({"fixed_nodes": [0.5]}, "whole numbers"),
        ({"periodic": [[9, 0]]}, "node 9"),
        ({"periodic": [[3.0, 0.0]]}, "whole numbers"),
        ({"materials": ()}, "at least one material"),
        ({"triangle_material": [0, 1]}, "material 1"),
        ({"triangle_material": [0, -1]}, "material -1"),
        ({"triangle_material": [0]}, "one material per triangle"),
        ({"current_density": [1.0, 2.0, 3.0]}, "current density"),
        ({"current_density": [1.0, math.nan]}, "current density"),
        ({"remanence": [1.0, 2.0, 3.0]}, "remanence"),
        ({"fixed_values": [0.0, 1.0]}, "fixed values"),
        ({"fixed_nodes": [9]}, "node 9"),
        ({"periodic": [[3, 0, 1]]}, "periodic"),
    ],
)
def test_bad_problem_is_rejected(changes: dict, message: str) -> None:
    fields = dict(
        nodes=SQUARE, triangles=TWO_TRIANGLES, materials=(AIR,), fixed_nodes=[0], fixed_values=0.0
    )
    fields.update(changes)
    with pytest.raises(ValueError, match=message):
        Problem(**fields)


def test_problem_module_needs_neither_gmsh_nor_matplotlib() -> None:
    code = (
        "import sys, polecraft.core.problem, polecraft.core.solver;"
        "assert 'gmsh' not in sys.modules;"
        "assert 'matplotlib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# A problem built from a machine


STEEL = FrohlichSteel(saturation=2.0, half_field=300.0)
MAGNET = Magnet(remanence=1.2, relative_permeability=1.05)
MATERIALS = {"M270-35A": STEEL, "N42SH": MAGNET, "copper": COPPER}


@pytest.fixture(scope="module")
def reference() -> dict:
    """The reference design, its slice geometry and its mesh."""
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / "qdd_outer_36s42p.yaml"))
    geometry = machine.build_geometry()
    return {"machine": machine, "geometry": geometry, "mesh": build_mesh(geometry)}


@pytest.fixture(scope="module")
def loaded(reference: dict) -> Problem:
    return build_problem(reference["machine"], reference["mesh"], MATERIALS, current=8.0)


def in_region(reference: dict, name: str) -> np.ndarray:
    mesh = reference["mesh"]
    return mesh.triangle_region == mesh.region_names.index(name)


def test_problem_takes_the_mesh_as_it_is(reference: dict, loaded: Problem) -> None:
    mesh = reference["mesh"]
    assert np.array_equal(loaded.nodes, mesh.nodes)
    assert np.array_equal(loaded.triangles, mesh.triangles)
    assert np.array_equal(loaded.periodic, mesh.periodic)
    assert loaded.antiperiodic == mesh.antiperiodic
    assert loaded.antiperiodic  # 6 slots and 7 poles in the slice


def test_every_triangle_gets_the_material_of_its_region(reference: dict, loaded: Problem) -> None:
    mesh, geometry = reference["mesh"], reference["geometry"]
    expected = {"steel": STEEL, "coil": COPPER, "magnet": MAGNET, "air": AIR}
    for region in geometry.regions:
        used = {loaded.materials[k] for k in loaded.triangle_material[in_region(reference, region.name)]}
        assert used == {expected[region.kind]}, region.name
    band = loaded.triangle_material[in_region(reference, "band")]
    assert {loaded.materials[k] for k in band} == {AIR}
    assert len(loaded.triangle_material) == len(mesh.triangles)


def test_each_material_is_listed_once(loaded: Problem) -> None:
    # Steel, magnet and air; copper is the same model as air.
    assert len(loaded.materials) == len(set(loaded.materials)) == 3


def test_each_coil_side_carries_its_ampere_turns(reference: dict, loaded: Problem) -> None:
    mesh, machine = reference["mesh"], reference["machine"]
    currents = machine.excitation(0.0, 8.0)
    area = mesh.triangle_areas()
    coils = [region for region in reference["geometry"].regions if region.kind == "coil"]
    assert len(coils) == 12
    for region in coils:
        inside = in_region(reference, region.name)
        side = region.coil_side
        total = (loaded.current_density[inside] * area[inside]).sum()
        assert total == pytest.approx(side.sign * side.turns * currents[side.phase], rel=1e-12)
        # Spread evenly over the coil side.
        assert np.ptp(loaded.current_density[inside]) == 0.0


def test_current_flows_only_in_coils(reference: dict, loaded: Problem) -> None:
    in_coil = np.zeros(len(reference["mesh"].triangles), dtype=bool)
    for region in reference["geometry"].regions:
        if region.kind == "coil":
            in_coil |= in_region(reference, region.name)
    assert np.all(loaded.current_density[~in_coil] == 0.0)
    assert np.all(loaded.current_density[in_coil] != 0.0)


def test_the_two_sides_of_a_coil_carry_opposite_currents(reference: dict, loaded: Problem) -> None:
    area = reference["mesh"].triangle_areas()
    for tooth in range(6):
        totals = [
            (loaded.current_density[inside] * area[inside]).sum()
            for inside in (
                in_region(reference, f"coil_{tooth}_cw"),
                in_region(reference, f"coil_{tooth}_ccw"),
            )
        ]
        assert totals[0] == pytest.approx(-totals[1], rel=1e-12)


def test_no_current_means_no_current_density(reference: dict) -> None:
    problem = build_problem(reference["machine"], reference["mesh"], MATERIALS)
    assert np.all(problem.current_density == 0.0)


def test_current_advance_reaches_the_excitation(reference: dict) -> None:
    machine, mesh = reference["machine"], reference["mesh"]
    shifted = build_problem(machine, mesh, MATERIALS, current=8.0, advance=0.6)
    currents = machine.excitation(0.0, 8.0, 0.6)
    inside = in_region(reference, "coil_0_ccw")
    side = reference["geometry"].region("coil_0_ccw").coil_side
    total = (shifted.current_density[inside] * mesh.triangle_areas()[inside]).sum()
    assert total == pytest.approx(side.sign * side.turns * currents[side.phase], rel=1e-12)


def test_magnets_are_magnetised_along_their_own_direction(reference: dict, loaded: Problem) -> None:
    mesh = reference["mesh"]
    centre = mesh.nodes[mesh.triangles].mean(axis=1)
    magnets = [region for region in reference["geometry"].regions if region.kind == "magnet"]
    assert len(magnets) == 7
    for region in magnets:
        inside = in_region(reference, region.name)
        expected = np.array(
            [region.magnetisation.direction(x, y) for x, y in centre[inside]]
        ) * MAGNET.remanence
        assert loaded.remanence[inside] == pytest.approx(expected, abs=1e-12)
        assert np.linalg.norm(loaded.remanence[inside], axis=1) == pytest.approx(1.2)


def test_neighbouring_magnets_point_opposite_ways(reference: dict, loaded: Problem) -> None:
    mesh = reference["mesh"]
    centre = mesh.nodes[mesh.triangles].mean(axis=1)
    outward = centre / np.linalg.norm(centre, axis=1)[:, None]
    radial = (loaded.remanence * outward).sum(axis=1)
    for j in range(7):
        inside = in_region(reference, f"magnet_{j}")
        assert np.all(np.sign(radial[inside]) == (-1) ** j)


def test_only_magnets_are_magnetised(reference: dict, loaded: Problem) -> None:
    in_magnet = np.zeros(len(reference["mesh"].triangles), dtype=bool)
    for j in range(7):
        in_magnet |= in_region(reference, f"magnet_{j}")
    assert np.all(loaded.remanence[~in_magnet] == 0.0)


def test_radial_magnets_follow_the_radius(reference: dict) -> None:
    from polecraft.core.mesh import build_mesh
    from polecraft.design import design_from_dict, design_to_dict
    from polecraft.machines import machine_for

    data = design_to_dict(reference["machine"].design)
    data["geometry"]["magnet"]["magnetisation"] = "radial"
    machine = machine_for(design_from_dict(data))
    mesh = build_mesh(machine.build_geometry())
    problem = build_problem(machine, mesh, MATERIALS)
    centre = mesh.nodes[mesh.triangles].mean(axis=1)
    outward = centre / np.linalg.norm(centre, axis=1)[:, None]
    inside = mesh.triangle_region == mesh.region_names.index("magnet_1")
    assert problem.remanence[inside] == pytest.approx(-1.2 * outward[inside], abs=1e-12)


def test_the_inner_and_outer_surfaces_are_field_lines(reference: dict, loaded: Problem) -> None:
    mesh, geometry = reference["mesh"], reference["geometry"]
    radius = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    on_surface = np.isclose(radius, geometry.inner_radius) | np.isclose(
        radius, geometry.outer_radius
    )
    assert np.array_equal(np.sort(loaded.fixed_nodes), np.flatnonzero(on_surface))
    assert np.all(loaded.fixed_values == 0.0)
    assert radius.min() == pytest.approx(geometry.inner_radius)
    assert radius.max() == pytest.approx(geometry.outer_radius)
    assert 20 < len(loaded.fixed_nodes) < len(mesh.nodes) / 10


def test_air_and_copper_need_not_be_listed(reference: dict) -> None:
    problem = build_problem(
        reference["machine"], reference["mesh"], {"M270-35A": STEEL, "N42SH": MAGNET}
    )
    assert COPPER in problem.materials
    assert AIR in problem.materials


def test_the_callers_copper_replaces_the_usual_one(reference: dict) -> None:
    own = LinearMaterial(0.999991)
    problem = build_problem(
        reference["machine"], reference["mesh"], {**MATERIALS, "copper": own}
    )
    in_coil = in_region(reference, "coil_2_cw")
    assert {problem.materials[k] for k in problem.triangle_material[in_coil]} == {own}
    in_air = in_region(reference, "band")
    assert {problem.materials[k] for k in problem.triangle_material[in_air]} == {AIR}


def test_a_missing_material_is_named(reference: dict) -> None:
    with pytest.raises(ValueError, match="no material model for 'M270-35A'"):
        build_problem(reference["machine"], reference["mesh"], {"N42SH": MAGNET})
    with pytest.raises(ValueError, match="no material model for 'N42SH'"):
        build_problem(reference["machine"], reference["mesh"], {"M270-35A": STEEL})


def test_a_magnet_region_needs_a_magnet_material(reference: dict) -> None:
    with pytest.raises(ValueError, match="'N42SH' .* magnet"):
        build_problem(
            reference["machine"], reference["mesh"],
            {"M270-35A": STEEL, "N42SH": LinearMaterial(1.05)},
        )


def test_a_mesh_of_another_machine_is_rejected(reference: dict) -> None:
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    other = machine_for(load_design(root / "designs" / "qdd_outer_12s10p.yaml"))
    with pytest.raises(ValueError, match="mesh does not belong to this machine"):
        build_problem(other, reference["mesh"], MATERIALS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (("geometry", "rotor", "outer_diameter_mm"), 94.0),  # larger all over
        (("geometry", "magnet", "thickness_mm"), 3.0),  # same radii, thicker magnets
    ],
)
def test_a_mesh_of_a_machine_with_other_dimensions_is_rejected(
    reference: dict, field: tuple, value: float
) -> None:
    from polecraft.design import design_from_dict, design_to_dict
    from polecraft.machines import machine_for

    data = design_to_dict(reference["machine"].design)
    block = data
    for name in field[:-1]:
        block = block[name]
    assert block[field[-1]] != value
    block[field[-1]] = value
    other = machine_for(design_from_dict(data))
    with pytest.raises(ValueError, match="mesh does not belong to this machine"):
        build_problem(other, reference["mesh"], MATERIALS)


def test_a_mesh_of_the_whole_machine_has_no_coupling() -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    from polecraft.core.mesh import build_mesh
    from polecraft.design import load_design
    from polecraft.machines import machine_for

    root = Path(__file__).resolve().parents[2]
    machine = machine_for(load_design(root / "designs" / "qdd_outer_12s10p.yaml"))
    mesh = build_mesh(machine.build_geometry(full=True))
    materials = {"M270-35A": STEEL, "N42SH": MAGNET}
    problem = build_problem(machine, mesh, materials, current=5.0)
    assert problem.periodic.shape == (0, 2)
    assert not problem.antiperiodic
    area = mesh.triangle_areas()
    # All twelve coils are there, and the phase currents add up to nothing.
    assert (problem.current_density * area).sum() == pytest.approx(0.0, abs=1e-9)
    assert np.count_nonzero(np.linalg.norm(problem.remanence, axis=1)) > 0


def test_the_problem_holds_the_material_models_themselves(loaded: Problem) -> None:
    # Not reluctivities: nothing is evaluated until the solver asks.
    steel = [m for m in loaded.materials if m is STEEL]
    assert len(steel) == 1
    assert steel[0].reluctivity(0.0) == pytest.approx(1.0 / (MU0 + 2.0 / 300.0))
