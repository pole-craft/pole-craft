# Changelog

All notable changes to Pole-Craft are listed here, newest first. Version
numbers follow [semantic versioning](https://semver.org); while the version
is 0.0.x the design-file format and the Python interface may change between
releases.

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
