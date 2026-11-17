# Changelog

All notable changes to Pole-Craft are listed here, newest first. Version
numbers follow [semantic versioning](https://semver.org); while the version
is 0.0.x the design-file format and the Python interface may change between
releases.

## [0.0.2] - unreleased

### Added

- Material model for the field solver (`polecraft.core.materials`):
  reluctivity against flux density, and its derivative, from a BH table for
  steel; constant permeability for air and copper; magnets by remanence and
  recoil permeability; and a saturation law given by a formula, for tests.
- A placeholder steel curve, `generic_35A_placeholder`, in the material
  folder. It is worked out from a formula and is not data for any real
  steel; measured curves come later.
- Field solver (`polecraft.core.solver`): the vector potential on
  first-order triangles, with current density in conductors and the
  remanence of permanent magnets as sources, fixed-value boundaries, and
  periodic or anti-periodic coupling of the cut edges of a slice. Steel
  that saturates is solved by Newton iteration with relaxation, with a
  report of the residual at every iteration.
- `polecraft.core.problem`: builds the solver's problem for a machine on
  its mesh (material, current density and magnetisation of every triangle,
  boundary nodes and slice coupling) at rotor angle 0. The same problem
  object can be put together by hand from arrays.
- Results from a solved field (`polecraft.core.post`): torque on the rotor
  from the Maxwell stress in the air gap, and the flux linked by each
  phase, both for the whole machine at rotor angle 0.
- `docs/architecture.md`: how the package fits together, the contract a
  machine type follows, the unit rule and how to add a machine type.

### Changed

- When Gmsh is not installed, a command that needs it now stops at once
  with one line giving the install command, `pip install "polecraft[mesh]"`
  (it said `pip install gmsh` before, and only after reading the design).
  The example script prints the same line in place of a traceback when it
  reaches the meshing step, and `polecraft mesh --help` says that the
  command needs Gmsh.
- The README states in one sentence that a plain `pip install polecraft`
  has no mesher and that Gmsh is GPL-licensed.
- When Gmsh is installed but cannot load, as on Linux without its system
  libraries, the message now says so and points to the README.

## [0.0.1] - 2026-11-16

First release. It loads a design file, builds the geometry of an outer-rotor
motor, meshes it and saves a picture. There is no field solver yet, so it
does not report torque, back-EMF or losses.

### Added

- Design files in YAML (`format_version: 1`) for outer-rotor surface-magnet
  motors (`type: qdd_outer`), with strict checking: unknown fields, wrong
  types and values out of range are all reported in one message, and
  dimensions that do not fit together are then reported one at a time.
- Symmetry slice and winding layout for any balanced three-phase slot and
  pole combination with a double-layer tooth-wound winding, and its winding
  factor.
- Cross-section geometry as labelled regions (stator and rotor steel, coil
  sides, magnets, air), for the repeating slice or the whole machine.
- Triangle meshing through Gmsh, with a sliding band in the air gap prepared
  for rotation without remeshing.
- `polecraft mesh <design.yaml>` command: saves a coloured picture of the
  mesh (PNG, SVG or PDF) and prints the mesh size and a timing report.
- The same steps from Python (`load_design`, `machine_for`, `build_mesh`,
  `plot_mesh`), with an example script in `examples/`.
- Three example designs: 36 slots with 42 poles, 24 slots with 28 poles and
  12 slots with 10 poles. Their dimensions are placeholders, not measured or
  optimised designs.

### Known limits

- No field solution: nothing is simulated yet.
- Outer-rotor surface-magnet motors only, with parallel-sided teeth.
- Meshing needs the optional Gmsh package (`pip install "polecraft[mesh]"`).
  On Linux, Gmsh also needs a few system libraries; the README lists them.
- The material names in a design file are stored but not yet checked against
  a material library.
- Winding, lengthwise and temperature values are read but not used yet: for
  example, nothing checks that the turns fit in the slot.
- There is no air region outside the rotor yoke or inside the stator bore.
- The air-gap band is one layer of triangles. Very fine meshes (`--size`
  below about 0.5) and slot and pole counts with little in common give thin
  triangles in the air gap.
- Pictures of very fine meshes look dark, because the triangle edge lines do
  not thin out.
- Meshing cannot be interrupted part-way; Ctrl-C takes effect when the mesher
  returns.

[0.0.1]: https://github.com/pole-craft/pole-craft/releases/tag/v0.0.1
