# Changelog

All notable changes to Pole-Craft are listed here, newest first. Version
numbers follow [semantic versioning](https://semver.org); while the version
is 0.0.x the design-file format and the Python interface may change between
releases.

## [Unreleased]

### Added

- Rotor rotation (`polecraft.core.motion`): `rotate(mesh, angle)` gives the
  mesh with the rotor turned to any angle. Nothing is meshed again: the
  rotor's nodes are moved and only the band of air between stator and rotor
  is joined again, in a fraction of a millisecond. `build_problem` then
  sets up the field problem at that angle, with the magnets turned and the
  phase currents following the rotor.
- `step_angle(mesh)`: the angle between neighbouring nodes of the band.
  At whole multiples of it no triangle changes shape. Between them the
  rotor's air next to the band leans by up to half a spacing, which adds
  mesh noise to the torque that falls with a finer mesh. On the reference
  motor with the default mesh it is about 0.004 N m: 0.2 % of the torque at
  10 A, but about twice that motor's cogging torque, which the default mesh
  does not resolve at any angle (it needs `size` 0.5 or finer).
- `solve_angles(mesh, angles, problem_at)` solves the field at a series of
  rotor angles, each solve starting from the field at the angle before.
  The results are those of separate solves; with saturating steel it takes
  fewer Newton iterations (about 30 % fewer on the reference motor at 10 A).
  `RotorRun.timing_report()` adds up the time over all angles.
- Cogging torque (`polecraft.studies.cogging`): `cogging_torque(machine,
  mesh, materials)` gives the torque with no current over one or more
  cogging periods of 360 / LCM(slots, poles) mechanical degrees, and
  `cogging_period(slots, poles)` the period itself. A mesh with fewer than
  10 node spacings around the air gap in a period is warned about: the
  waveform is then mostly mesh noise (the default mesh of the 36-slot
  42-pole reference motor has 4; `size` 0.5 gives 10). Known limit: where
  cogging torque is about a thousandth of the rated torque or less, its
  period and mean come out right but its size does not settle with a finer
  mesh. On the reference motor (about 2 mN m against 3.19 N m at 10 A) it
  moved between 1.8 and 2.4 mN m over `size` 0.5 to 0.2, and on the
  24-slot 28-pole example by 27 % from `size` 0.35 to 0.25; take such values
  as an order of magnitude. The 12-slot 10-pole example, with larger
  cogging, changes by about 3 % from `size` 0.5 to 0.35.
- Back-EMF (`polecraft.studies.back_emf`): `back_emf(machine, mesh,
  materials, speed)` gives the flux linked by each phase over whole
  electrical cycles with no current, and the back-EMF of each phase at that
  speed (rad/s), with its electrical frequency, fundamental peak and the
  back-EMF constant in V s/rad. On the three example designs the fundamental
  is within 2.4 % of the formula from the air-gap field and changes by
  0.1 % from mesh `size` 1 to 0.5. The phase EMFs carry a third harmonic
  (6 to 9 % on the examples), which is the same in all three phases, so the
  three phases sum to three times it rather than to zero; line-to-line EMFs
  do not carry it. 2D values, with no correction for the ends of the machine.

### Changed

- A `Mesh` now records the rotor angle it has been turned to and how its
  band is joined (`rotor_angle`, `rotor_turn`, `band`). Meshes at rotor
  angle 0 are the same as before, node for node.
- The band of a slice always gets an even number of node spacings. With an
  odd number, which some slot and pole counts gave (a slice of 9 slots,
  for one), the torque of the slice was off by up to about 1 %. The
  example designs were not affected.
- Field pictures: region outlines are drawn as built, at rotor angle 0,
  and do not follow a turned rotor yet.

## [0.0.2] - 2026-11-18

The first release with a field solver. It solves the magnetic field of an
outer-rotor surface-magnet motor at one rotor position, with saturating
steel and permanent magnets, reports the torque and the flux linked by each
phase, and draws the field. On the reference motor the solver agrees with
FEMM within 0.3 % on torque, with the same placeholder materials in both.
There is no real material data yet, so the numbers show how the tool
behaves, not how a motor performs.

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
- Compute layer (`polecraft.core.compute`): the field solve takes the name
  of a sparse solver, and every solution carries a timing report of where
  its time went. `superlu`, `superlu_symmetric` and `banded` come with
  SciPy. `pardiso` is optional and faster
  (`pip install "polecraft[pardiso]"`; it brings Intel's MKL, which is not
  open source and exists for Windows and Linux on Intel and AMD processors
  only). SuperLU is the default; PARDISO is used when asked for by name
  (`solve(problem, solver="pardiso")`). Its first solve in a run takes
  about a third of a second longer while MKL loads, which the timing report
  lists on its own line. Where it is not installed, the timing report of a
  solution (`Solution.timing_report()`) says how to install it, and asking
  for it by name gives a warning and the solve carries on with SuperLU.
- Results from a solved field (`polecraft.core.post`): torque on the rotor
  from the Maxwell stress in the air gap, and the flux linked by each
  phase, both for the whole machine at rotor angle 0.
- Flux density in the middle of the air gap at any angle
  (`polecraft.core.post.gap_flux_density`).
- A check of the solver against FEMM, an independent field program.
  `polecraft.femm` writes the same model as a FEMM script, from the same
  regions, steel table, magnet and currents; the package never runs FEMM.
  FEMM's results for the reference design are stored in `tests/data/femm/`
  with the script that makes them, and the tests compare the solver with
  them. With the placeholder steel and magnet, at rotor angle 0 and at 0,
  10 and 40 A: torque within 0.3 %, the flux linked by each phase within
  0.7 % of the largest, and the main wave of the gap field within 0.3 %.
  Making the solver's mesh twice as fine changes its torque by under 0.2 %.
  The README has the figures side by side, with the time each program took.
  This checks the 2D solver, not the motor: both programs solve the same
  2D model, with no end effects.
- `polecraft solve <design.yaml>`: meshes a design, solves its field at
  rotor angle 0 for a given current (`--current`, `--advance`), prints
  torque, flux linkage, the convergence of the solve and a timing report,
  and saves a picture of the field. The steel and the magnet are stand-ins
  until the package has material data, and the command says so.
- Pictures of a solved field (`polecraft.plot.plot_field`, `draw_field`):
  flux density as colour with a scale in tesla, field lines, and
  optionally the outline of every region.
- Nine more example scripts in `examples/`: winding layouts, material
  models, solving a motor, torque against current, the slice against the
  whole machine, a picture of the field, choosing a solver, and two
  hand-built problems checked against formulas (a saturating core with an
  air gap, and the inductance of a coil).
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

### Known limits

- One rotor position only (rotor angle 0). There is no rotation yet, so no
  cogging torque, back-EMF over a cycle or torque ripple.
- No real material data. The steel curve is a placeholder worked out from a
  formula, `polecraft solve` uses a stand-in magnet, and the material names
  in a design file are still not checked against a material library. Do not
  use the results to judge a motor.
- The comparison with FEMM is for one motor, at three currents, with the
  same placeholder materials in both programs. It checks the 2D solver, not
  the motor.
- Two-dimensional only: no end effects, which matter in short, wide motors,
  and no end windings, skew, losses or efficiency.
- Torque and flux linkage for the whole machine are worked out from the
  repeating slice, which assumes the slices are identical.
- Outer-rotor surface-magnet motors only, with parallel-sided teeth. The
  FEMM model is written for parallel magnetisation only, and the check
  against FEMM is for that.
- There is no air region outside the rotor yoke or inside the stator bore.
  In FEMM, on the reference motor at 10 A and 40 A, adding air inside the
  bore and out to 1.5 times the outer radius changed the torque by under
  0.02 %. This was not checked with Pole-Craft's own solver.
- Winding, lengthwise and temperature values are read but not used yet: for
  example, nothing checks that the turns fit in the slot.
- The mesh-refinement check and the FEMM comparison cover the default mesh
  of the reference motor only. Coarser meshes are accepted without a
  warning and give a different torque: on the reference motor at 10 A,
  `--size 5` changed it by about 1 % and `--size 6` to `--size 8` by about
  4 % to 5 %. `polecraft solve` refuses only a mesh so coarse that the
  regions no longer have the areas of the design, and that point depends on
  the design (`--size 10` on the reference motor). Use the default size, or
  check with a finer mesh. The air-gap band is one layer of triangles, so
  very fine meshes (`--size` below about 0.5) and slot and pole counts with
  little in common give thin triangles there.
- Solving needs the optional Gmsh package (`pip install "polecraft[mesh]"`).
  The faster PARDISO solver is optional too and exists for Windows and
  Linux on Intel and AMD processors only.
- Pictures of the field are saved files; there is no window to click in.
- Pictures of very fine meshes look dark, because the triangle edge lines do
  not thin out. Meshing cannot be interrupted part-way.

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

[0.0.2]: https://github.com/pole-craft/pole-craft/releases/tag/v0.0.2
[0.0.1]: https://github.com/pole-craft/pole-craft/releases/tag/v0.0.1
