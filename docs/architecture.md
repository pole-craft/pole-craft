# Architecture

How Pole-Craft fits together, for anyone who wants to read or change the
code. This describes the package as it stands in the repository, which can
be ahead of the latest release; it will grow with the package.

What works today: a design file is loaded and checked, turned into geometry,
meshed, and the magnetic field is solved at one rotor position, with steel
that saturates. Torque and the flux linked by each phase are read off the
solved field. What is not built yet: rotation of the rotor, the studies
(cogging, back-EMF, efficiency), losses, a full material library, and any
machine other than the outer-rotor one. Nothing on the command line runs
the solver yet; it is reached from Python.

## From design file to results

```
design file (YAML)
   |  polecraft/design.py        load and check
   v
Design
   |  polecraft/machines/        the machine for the design's type
   v
Machine
   |  build_geometry()           regions, in metres
   v
Geometry
   |  polecraft/core/mesh.py     triangles (needs Gmsh)
   v
Mesh
   |  polecraft/core/problem.py  materials, currents, magnets, boundaries
   v
Problem
   |  polecraft/core/solver.py   field solve
   v
Solution                         potential and flux density
   |  polecraft/core/post.py     torque, flux linkage
   v
results
```

1. `polecraft/design.py` reads a YAML file such as
   `designs/qdd_outer_36s42p.yaml` into a `Design`. Every mistake in the
   file is reported with the path of its field, through
   `polecraft/validation.py`. The part of the file that depends on the
   machine type, the `geometry` block, is read by that machine's own
   parameter class.
2. `machine_for` in `polecraft/machines/__init__.py` returns the machine for
   the design. Its `build_geometry` gives a `Geometry`
   (`polecraft/core/geometry.py`): closed regions of steel, air, coil and
   magnet, for the slice of the machine that repeats or for the whole
   machine. `polecraft/core/winding.py` works out that slice and which
   phase each coil belongs to.
3. `build_mesh` in `polecraft/core/mesh.py` meshes the regions with
   triangles and returns plain arrays. It also lists the node pairs on the
   two cut edges of a slice, and says whether the field repeats there
   unchanged or with its sign reversed.
4. `build_problem` in `polecraft/core/problem.py` puts a material, a current
   density and a magnetisation on every triangle, and fixes the potential
   on the inner and outer surface. The material models are in
   `polecraft/core/materials.py`.
5. `solve` in `polecraft/core/solver.py` solves for the vector potential by
   Newton iteration and returns the potential, the flux density in every
   triangle and a record of the iteration.
6. `torque` and `flux_linkage` in `polecraft/core/post.py` take the problem
   and its solution and give the torque on the rotor and the flux linked by
   each phase, for the whole machine.

The `polecraft mesh` command (`polecraft/cli.py`) runs steps 1 to 3 and
saves a picture with `polecraft/plot.py`. `polecraft/timing.py` times the
stages. `examples/mesh_a_design.py` does the same from Python.

The `polecraft solve` command runs all six steps and saves a picture of
the field, with a stand-in steel and magnet until there is material data.

## What each package does

**`polecraft/core/`** is the part that knows nothing about motor types. It
works with regions, materials, coils and magnets only.

- `polecraft/core/geometry.py`: regions and their outlines.
- `polecraft/core/winding.py`: the repeating slice, the coil layout of a
  tooth-wound three-phase winding, and its winding factor.
- `polecraft/core/mesh.py`: the mesher. It is the only module that uses
  Gmsh, and it imports Gmsh only when a mesh is asked for. The thin band of
  air between stator and rotor is built here, not by Gmsh, so that the
  rotor can later be turned without meshing again.
- `polecraft/core/materials.py`: reluctivity against flux density, from a
  BH table or a formula, and magnets. Material files are in
  `polecraft/data/materials/`. The one there now is a placeholder made from
  a formula, not data for a real steel.
- `polecraft/core/problem.py`: the `Problem` the solver is given. It can be
  built for a machine, or by hand from arrays, which is how the solver is
  tested. It also says where the windings and the air gap are, for the
  results.
- `polecraft/core/solver.py`: the field solve. It takes a `Problem` and
  nothing else.
- `polecraft/core/compute.py`: the heavy arithmetic. The solver hands its
  sparse linear systems to it by the name of a solver, so that a faster one
  can be added in one place, and it builds the matrix without working out
  again which entries it has. The solution says how its time was spent.
  The solver that comes with SciPy is the default. One solver, PARDISO, is
  in an optional package and is used only when asked for by name; where
  the package is missing the default is used instead.
- `polecraft/core/post.py`: results from a solved field. Torque comes from
  the Maxwell stress in the band of air between stator and rotor, and flux
  linkage from the potential over the coil sides. The flux density in the
  middle of the air gap can be read at any angle.

**`polecraft/femm.py`** writes the same model as a script for FEMM, a
separate field program, from the same regions, materials and currents. It
is how the solver is checked: the package never runs FEMM. The stored FEMM
results and the script that makes them are in `tests/data/femm/`.

**`polecraft/machines/`** holds one module per machine type, and the
contract they follow in `polecraft/machines/base.py`. The only machine so
far is `qdd_outer`, in `polecraft/machines/qdd_outer.py`. The table that
maps a design file's `type` to a machine is in
`polecraft/machines/__init__.py`, and it is the only place where a machine
type is named.

**`polecraft/studies/`** is for calculations over many solves, such as
cogging torque or back-EMF. It is empty for now. A study will call the
solver and never name a machine type.

Three rules keep these apart:

- `polecraft.core` never names a machine type. Where it needs a machine,
  it uses it only through the contract below.
- `polecraft.studies` never names a machine type.
- Designs and materials are data, not code: a new design is a file in
  `designs/`, and a new material is a file in `polecraft/data/materials/`.

## The machine contract

Every machine is a subclass of `Machine` in `polecraft/machines/base.py` and
offers four things:

- `parameters()`: the dimensions that define the machine, as read from the
  design file.
- `build_geometry()`: the regions of the repeating slice or of the whole
  machine, each with its material, and for coils and magnets their phase,
  direction and magnetisation.
- `excitation()`: the phase currents for a rotor angle, a peak current and
  a current advance.
- `symmetry()`: the smallest slice of the machine that repeats, and whether
  the next slice has the field reversed.

The conventions that go with them (which tooth and which magnet sit on the
x axis, which way positive current flows, which way a magnet with positive
polarity points) are stated in the docstrings of that module and of
`polecraft/core/geometry.py`.

## Units

A design file is in the units an engineer types: millimetres, degrees and
degrees Celsius, with the unit in the field name, as in `air_gap_mm`. The
`Design` object keeps those names and units.

Geometry and field quantities in `polecraft/core/` are in SI: metres,
radians, tesla, amperes. The few values kept in degrees for reporting say
so in their name, as in `angle_deg`. The change of units happens in one
place for each machine, where it builds its geometry.

Fields are per metre of depth in the solver. The stack length goes into the
problem as its depth, and the results (flux linkage, torque) are scaled by
it and by the number of slices in the machine.

## Adding a machine

1. Write a parameter class for the machine's `geometry` block, as a frozen
   dataclass, so that designs can be compared and written back out. It
   needs a `from_section` class method that reads the fields, and a `check`
   method that reports dimensions that do not fit together. See
   `QddOuterParameters` in `polecraft/machines/qdd_outer.py`.
2. Write a subclass of `Machine`, set its `Parameters` to that class, and
   implement `build_geometry`. Convert to SI there. `excitation` and
   `symmetry` come from the base class and suit a three-phase winding with
   a coil on every tooth; override them for anything else.
3. Add the machine to the table in `polecraft/machines/__init__.py`.
4. Add a design file to `designs/` and tests under `tests/machines/`. The
   tests for the outer-rotor machine show what to check: region areas
   against hand formulas, regions that tile the slice without overlap, and
   the sign conventions.
5. Name the new module and type in this document, and update the two tests
   that list the machine types and the example designs:
   `tests/machines/test_machine_contract.py` and `tests/test_cli.py`.

Nothing in `polecraft/core/` should need to change. If it does, the contract
is missing something, and that is worth raising as an issue first.

## Out of scope

- Thermal simulation.
- A graphical interface.
- Three-dimensional field solving inside this package. A check against an
  external 3D solver is planned for later.
- Gear tooth strength and gear life. A later gear module will treat the
  gearbox as ratios, losses and inertias only.
- Replacing a general-purpose finite-element package. The aim is fast,
  scriptable comparison of motor designs.
