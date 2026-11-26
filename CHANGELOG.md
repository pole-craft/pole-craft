# Changelog

All notable changes to Pole-Craft are listed here, newest first. Version
numbers follow [semantic versioning](https://semver.org); while the version
is 0.0.x the design-file format and the Python interface may change between
releases.

## [0.0.5] - 2026-11-26

The first release with motor-level answers. From the 2D field it now gives
the copper and core loss, flux and torque maps, the torque-speed curve and
the efficiency map under a drive's voltage and current limits, torque against
current with saturation, a demagnetisation check, mass, rotor inertia and
the motor constant, an end-effect correction for a short stack, and a stall
heating estimate. A U8-class motor is compared with its published
measurements, and the gaps are reported in `docs/u8_comparison.md`. The
commands and studies still take a stand-in steel and magnet rather than the
materials a design names, and there is no real magnet grade yet, so the
numbers show how the tool behaves, not how a motor performs.

### Added

- Copper loss study (`polecraft.studies.copper_loss`): the loss of all three
  phases for a peak phase current and a winding temperature, from the DC
  resistance (no skin or proximity effect).
- Core loss study (`polecraft.studies.core_loss`): hysteresis, eddy-current
  and excess loss in the steel from the flux density waveform of every
  triangle over one electrical cycle, with the rotor's steel followed in its
  own frame. One run gives the loss at any speed (`CoreLoss.at_speed`). The
  loss is that of the sheet as measured in a sine-wave field: no allowance
  for stamping or stress, none for a solid rotor yoke, and no eddy currents
  in the magnets.
- Flux and torque maps (`polecraft.studies.flux_map`): d- and q-axis flux
  linkage and mean torque solved on a grid of d and q currents, then
  interpolated at any operating point inside it (`FluxMap.at`) instead of
  solved again. `operating_point` solves a single point.
- Torque-speed curve (`polecraft.studies.torque_speed`): the most torque at
  each speed within a peak phase voltage and current, from a flux map, with
  field weakening above the base speed and the winding's resistance and
  end-turn inductance included. `phase_limits` turns a drive's DC bus
  voltage and current limit into phase limits for star or delta.
- Efficiency map (`polecraft.studies.efficiency`): copper and core loss
  and efficiency over speed and shaft torque, at the d and q current that
  loses least within the drive's voltage and current limits. Flux maps can
  now carry the core loss (`flux_map(..., losses=...)`, `FluxMap.core_loss`)
  from the same solves. Not included: friction and windage, magnet and
  winding eddy currents, and the drive's own loss.
- End-effect correction (`polecraft.studies.end_effect`): a factor for
  the flux of a finite stack against the 2D model, from a separate 2D model
  of the stack end (magnet end leakage, fringing into the stator's end
  face, magnet overhang). An approximation: leakage between poles at the
  ends and slotting are left out.
- Stall heating (`polecraft.studies.stall_heating`): how long the worst
  phase can hold a current at standstill before reaching a temperature
  limit, with the copper heating alone; `stall_current` gives the current
  for a hold time. Copper now carries its specific heat.
- Comparison with a measured motor (`docs/u8_comparison.md`): a U8-class
  motor, as published for the MIT Mini Cheetah actuator, with its turns
  fitted to the measured torque constant. The model understates the fall of
  the torque constant at peak current (2 % against 13 %) and gives about
  twice the measured inductance; both gaps are reported, not tuned away.
- Mechanical metrics (`polecraft.studies.mechanical`): mass of the stator
  steel, rotor steel, magnets and copper, rotor inertia, torque density and
  the motor constant. Active parts only; densities come from the material
  library where it has them and are otherwise given by the caller.
- Torque against current up to peak (`polecraft.studies.saturation`): mean
  torque, torque constant and its fall with saturation at each current.
- Demagnetisation check (`polecraft.studies.demagnetisation`): the most
  demagnetising field in every magnet triangle over a cycle, at a current
  and magnet temperature, against the knee of the magnet's curve; flags any
  part of the magnet past it. `magnet_field` in `polecraft.core.post` gives
  the field along the magnetisation.
- Magnet grades can give the knee as `knee_to_intrinsic_coercivity_ratio`
  (Hk / Hcj); `Magnet.intrinsic_coercivity_at` and `Magnet.knee_field` give
  Hcj and Hk at a temperature.
- `core_loss_of_run` in `polecraft.studies.core_loss`: the core loss from
  solves already made.
- `core_loss_density` and `material_flux_density` in `polecraft.core.post`:
  loss per kilogram from any flux density waveform, and the field a turning
  triangle sees in its own frame.

- Example `examples/torque_speed_and_efficiency.py`: a flux map of the
  12-slot, 10-pole motor, then its torque-speed curve and efficiency map on a
  24 V drive.

### Changed

- Copper (`copper.yaml`) carries a specific heat, 384.5 J/(kg K), worked
  out from its molar heat capacity; `Conductor.specific_heat` is optional,
  and an entry without one cannot be used for stall heating.
- Flux maps can carry the core loss (`flux_map(..., losses=...)`), from the
  same solves as the flux linkages.
- `copper_mass` in `polecraft.studies.mechanical` is public; it was private.

### Known limits

- No real magnet grade, and the commands and studies still use a stand-in
  steel and magnet, not the materials a design names. Do not use the
  results to judge a motor.
- Compared with a published U8-class motor (`docs/u8_comparison.md`), with
  its turns fitted to the measured torque constant, the model understates
  how far the torque constant falls at peak current (2 % against 13 %) and
  gives about twice the measured inductance. Neither gap is explained, and
  neither was tuned away. The motor's dimensions, steel and magnet are
  estimates and placeholders.
- The end-effect factor leaves out leakage between poles at the ends and
  slotting. It finds a loss of only 0.4 % of the 2D flux for the reference
  motor, where short wide motors are expected to lose more; the 3D check is
  still to come.
- Stall heating takes the copper alone, with no heat leaving it, so it
  answers how long a hold of a few seconds can last, not a continuous
  rating. There is no thermal simulation.
- The efficiency map counts copper and core loss only: no friction,
  windage, magnet or winding eddy currents, and none for the drive. Core loss
  is that of the sheet as measured in a sine-wave field, with no allowance
  for stamping or stress. The loss coefficients are fitted over 50 to
  1000 Hz and warn outside it.
- Copper loss uses the DC resistance, with no skin or proximity effect.
- The cogging torque of the 36-slot, 42-pole motors still does not settle
  with the mesh, so it is an order of magnitude only.
- The limits of 0.0.4 still hold where this list does not replace them: the
  studies are run from Python, the model is 2D, outer-rotor surface-magnet
  motors only, and solving needs the optional Gmsh package.

## [0.0.4] - 2026-11-24

The first release with material data and a winding model. The library has
M270-35A steel from the maker's brochure, copper and a placeholder magnet;
a phase winding has its resistance, slot fill and design warnings; and the
d- and q-axis inductance, with an end-turn estimate, is within 1 % of FEMM on
the reference motor. The commands and studies still take a stand-in steel
and magnet rather than the materials a design names, and there is no real
magnet grade yet, so the numbers show how the tool behaves, not how a motor
performs.

### Added

- Material library (`polecraft.core.materials`): `load_material(name)` gives
  a `BHCurve` for a steel, a `Magnet` for a magnet grade and a `Conductor`
  for a winding conductor. Every entry is a file that states its kind and
  where its numbers come from. The library has M270-35A steel, copper, and
  two placeholders that are not datasheet values: the steel made from a
  formula that 0.0.3 had, and a sintered NdFeB magnet. There is no real
  magnet grade yet, and the commands and studies do not take a design's
  materials from the library yet.
- Magnets against temperature: `Magnet.at_temperature(t)` gives the magnet
  at `t` degrees Celsius from its 20 C entry. The remanence follows its
  coefficient (`remanence_temp_coefficient_per_K` in a material file) in a
  straight line; the recoil permeability stays constant, so the coercivity
  on the recoil line, Hcb = Br / (mu_r mu0), follows the remanence and has
  no coefficient of its own. A temperature below absolute zero, or one at
  which the line gives no remanence, is refused. The intrinsic coercivity
  Hcj and its coefficient, which are what a datasheet quotes, can be stored
  as `intrinsic_coercivity_A_per_m` and
  `intrinsic_coercivity_temp_coefficient_per_K`; they are kept for a later
  demagnetisation check and are not used in any result yet. A field named
  `coercivity_temp_coefficient_per_K` is refused, with a message saying
  what to write: the name does not say which coercivity it means, and a
  datasheet's coefficient is for Hcj, not Hcb.
- Copper (`copper` in the library) with the values of the International
  Annealed Copper Standard at 20 C: resistivity 1.7241e-8 ohm m,
  temperature coefficient 0.00393 per kelvin, density 8890 kg/m3.
  `Conductor.resistivity_at(t)` follows the straight line and refuses a
  temperature at which it gives zero or less (about -234 C for copper).
- M270-35A electrical steel, typical values from the maker's brochure
  (Cogent Power, 2002-11), with the page of each table cited in the file:
  polarisation against field strength at 50 Hz from 0.1 to 1.8 T, and
  specific loss against polarisation at 50 to 2500 Hz. The rows are 0.1 T
  apart; at 5000 A/m the curve gives 1.638 T, 0.012 T below the brochure's
  separate table of typical properties, whose values are rounded to 0.01 T.
- `BHCurve` beyond its last row: the polarisation J = B - mu0 H levels off
  and the slope tends to that of air. `saturation_polarisation_T` in a
  material file (`BHCurve(b, h, saturation)`) gives the level; without it a
  stated rule sets one from the last table step. For M270-35A, whose
  brochure has nothing above 1.8 T, the rule gives 1.830 T at 18 kA/m,
  1.849 T at 30 kA/m, 1.865 T at 50 kA/m and 1.884 T at 100 kA/m, tending
  to 1.899 T. That is likely below the real saturation, so flux in steel
  above 1.8 T is understated until a measured saturation value is given.
  The slope at the last row is held to that of air at most, so the curve
  beyond the table never asks for more field than air would.
  `BHCurve.stacked(factor)` gives the curve of a lamination stack.
- Core loss: `load_core_loss(name)` gives a steel's loss tables with its
  thickness, density and resistivity, and `CoreLossData.fit()` fits the
  hysteresis and excess coefficients of p = kh f B^2 + kc f^2 B^2 +
  ke f^1.5 B^1.5, with the classical eddy-current kc from the sheet. For
  M270-35A the fit over 50 to 1000 Hz from 0.3 T up reproduces the tables
  within 10 % RMS and 25 % at worst. The coefficients carry the range they
  were fitted over (`frequency_range`, `min_polarisation`,
  `max_polarisation`, and `polarisation_limits`, the highest fitted
  polarisation at each frequency), and `CoreLossCoefficients.loss` gives a
  `LossRangeWarning` outside it, still returning the value. For M270-35A
  the tables reach 1.8 T at 50 Hz, 1.5 T at 100 to 400 Hz and 1.1 T at
  1000 Hz.
- Winding as a circuit (`polecraft.core.winding`): `phase_winding(machine)`
  gives a `PhaseWinding` with the phase resistance at the winding's
  temperature, the resistance between terminals, line and phase currents
  and voltages for star and delta, and the copper loss for an RMS phase
  current. The mean turn length comes from the coil's shape
  (`Machine.coil_shape()`, `mean_turn_length`) and the stack length, with
  `axial.end_winding_overhang_mm` if the design gives it.
- Slot fill and design warnings: `slot_fill` gives the copper fill factor
  of a coil side. Design files take an optional
  `winding.insulated_diameter_mm`, the wire's diameter over its enamel. A
  design is loaded with warnings (`Design.warnings`), which
  `polecraft mesh` and `polecraft solve` print, when its copper fill is
  above 0.45 or its stacking factor below 0.9.
- d- and q-axis inductance (`polecraft.studies.inductance`):
  `inductance(machine, mesh, materials, current)` gives the apparent Ld and
  Lq of a phase at a test current, over the rotor angles of one electrical
  cycle, with the magnet's flux linkage with and without load. On the
  reference motor the 2D values are within 1 % of FEMM, from stored results
  made by `tests/data/femm/make_inductance.py`.
- End-turn inductance: `end_turn_inductance` and `end_winding_inductance`
  estimate what the end turns add to the 2D value, from the coil in air by
  Neumann's formula and half of what an ideal iron end face would add; the
  study adds it to Ld and Lq. It is an estimate, good to perhaps 40 %:
  6.9 uH on 0.61 mH for the reference motor and 5.7 uH on 0.157 mH for the
  12-slot 10-pole example.
- An example script, `examples/magnet_temperature.py`: torque and back-EMF
  of the 12-slot 10-pole example with M270-35A steel and the placeholder
  magnet at 20, 60 and 100 C.

### Changed

- Design files: `axial.end_winding_overhang_mm` must now be above 0. A coil
  has to cross the end of its tooth, so it cannot stand out by nothing;
  leave the field empty (`null`) to have the overhang estimated.
- Design files: `magnet.temperature_C` and `winding.temperature_C` must be
  between -100 and 300 degrees Celsius; anything above absolute zero used
  to load. The range is a sanity bound, set by where the straight line for
  copper resistance is still reasonable (it reaches zero at about -234 C).
  It does not say that a magnet's straight line is good that far: that one
  holds only up to the grade's working temperature, which the material
  library does not store yet.
- A design whose wires cannot fit its slots, even packed as tightly as
  circles can be, is refused.
- A design with `connection: delta` loads with a warning: the studies take
  the phase currents as sinusoidal and leave out the third-harmonic current
  that the phase back-EMFs drive round a closed delta.
- `polecraft solve --help` says that the command does not use the material
  library yet, not that the package has no material data.
- Material files are read the way design files are: a field written twice
  is refused, and forms such as `010` or `1:30` are not read as numbers. Each file must also give its `name`,
  `kind`, `description` and `source`.
- A BH table whose polarisation J = B - mu0 H falls from one row to the
  next is refused, with the row named: such a table asks for more field
  than air would need. A `jh_curve` in a material file is checked for a
  falling J column as written.
- A `BHCurve` used beyond its last row tends to the slope of air (see
  above); it used to go on as a straight line with the slope of the last
  row.
- `solve_angles` refuses a problem whose nodes are not those of the turned
  mesh. A problem built on the unturned mesh of a whole machine has the
  right number of nodes and used to pass.

### Known limits

- No real magnet grade: the library's magnet is a placeholder, and the
  commands and studies (`polecraft solve`, the example scripts) still use a
  stand-in steel and magnet, not the materials a design names. Do not use
  the results to judge a motor.
- The steel curve beyond its table (above 1.8 T for M270-35A) is an
  extension by a stated rule, probably below the real saturation, so flux in
  steel above 1.8 T is understated. Beyond 1000 Hz the core-loss fit
  over-predicts (16 to 36 % at 2500 Hz); nothing yet uses it in a result.
- The magnet's straight line is only good up to its grade's working
  temperature, which the library does not store, and nothing checks for
  demagnetisation. The intrinsic coercivity is stored at 20 C and does not
  change with temperature.
- A winding in delta is loaded with a warning: the third-harmonic current
  that the phase back-EMFs drive round a closed delta is not modelled.
- The end-turn inductance is an estimate, good to perhaps 40 %, and is not
  meant for an overhang below about half the coil thickness.
- The inductance is compared with FEMM on the reference motor only, with
  the placeholder materials in both programs. The M270-35A curve and loss
  data are typical brochure values, checked against the brochure but not
  against measurements.
- The limits of 0.0.3 still hold: studies are run from Python, cogging
  torque of the 36-slot 42-pole reference motor does not settle with the
  mesh, and the model is 2D, without end effects, skew, losses in a result
  or an efficiency map, for outer-rotor surface-magnet motors only. Solving
  needs the optional Gmsh package.

## [0.0.3] - 2026-11-20

The first release with a moving rotor. It turns the rotor of an outer-rotor
surface-magnet motor without meshing again, and works out over a cycle of
rotor angle the cogging torque, the back-EMF, and the torque with the phase
currents following the rotor, with its ripple and torque constant. Checked
against FEMM on the example motors: back-EMF within 0.14 %, and the cogging
torque of the 12-slot 10-pole example within 3 %. There is still no real
material data, so the numbers show how the tool behaves, not how a motor
performs.

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
  motor with the default mesh it is about 0.004 N m: 0.1 % of the torque at
  10 A, but about twice that motor's cogging torque.
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
  42-pole reference motor has 4; `size` 0.5 gives 10).
- Back-EMF (`polecraft.studies.back_emf`): `back_emf(machine, mesh,
  materials, speed)` gives the flux linked by each phase over whole
  electrical cycles with no current, and the back-EMF of each phase at that
  speed (rad/s), with its electrical frequency, fundamental peak and the
  back-EMF constant in V s/rad. On the three example designs the fundamental
  is within 2.4 % of the formula from the air-gap field and changes by
  0.1 % from mesh `size` 1 to 0.5. The phase EMFs carry a third harmonic
  (6 to 9 % on the examples), which is the same in all three phases, so the
  three phases sum to three times it rather than to zero; line-to-line EMFs
  do not carry it.
- Loaded torque (`polecraft.studies.loaded_torque`): `loaded_torque(machine,
  mesh, materials, current, advance)` gives the torque over one electrical
  cycle with the phase currents locked to the rotor, with its mean,
  peak-to-peak ripple and torque constant (N m per ampere of peak phase
  current). The mean is also found from the phase flux linkages
  (`mean_from_linkage`), independently of the air-gap stress; on the
  reference motor the two agree within 0.2 % at 1 and 10 A and 0.31 % at
  40 A, and within 0.05 % on a mesh of `size` 0.5. The torque constant at
  low current is 1.5 times the back-EMF constant to 0.2 %, as power balance
  requires. On the reference motor (placeholder steel) the torque constant
  is 0.12 % lower at 5 A than at 0.5 A, 0.7 % lower at 10 A and 26 % lower
  at 40 A, where the steel saturates.
- `femm_script(..., rotor_angles=...)` turns the rotor of the FEMM model for
  each case. Cogging and back-EMF waveforms are checked against FEMM from
  stored results, made by `tests/data/femm/make_waveforms.py`, so the tests
  need no FEMM. On the 36-slot 42-pole reference motor the back-EMF
  fundamental agrees within 0.14 %, its third harmonic within 0.2 % and the
  waveform point by point within 0.2 % of its peak. On the 12-slot 10-pole
  example the cogging torque's fundamental agrees within 3 % and its timing
  within 0.2 degrees (mesh `size` 0.35). The reference motor's much smaller
  cogging torque, about 2 mN m, agrees with FEMM's fine mesh in timing
  within 1 degree, but in size it is 28 % above at `size` 0.5, 5.6 % above at
  0.35, 16 % above at 0.25 and 1.9 % below at 0.2: it does not settle; see
  the known limits.
- An example script, `examples/motor_over_a_cycle.py`: cogging torque,
  back-EMF and loaded torque of the 12-slot 10-pole example motor over a
  cycle, with the torque constant checked against the back-EMF constant.

### Changed

- A `Mesh` now records the rotor angle it has been turned to and how its
  band is joined (`rotor_angle`, `rotor_turn`, `band`). Meshes at rotor
  angle 0 are the same as before, node for node.
- The band of a slice always gets an even number of node spacings. With an
  odd number, which some slot and pole counts gave (a slice of 9 slots,
  for one), the torque of the slice was off by up to about 1 %. The
  example designs were not affected.

### Known limits

- The studies are run from Python (see the example script). `polecraft
  solve` still solves at rotor angle 0 only, and the outlines in field
  pictures are drawn at rotor angle 0, so they do not follow a turned rotor.
- No real material data. The steel curve is a placeholder worked out from a
  formula, `polecraft solve` and the example scripts use a stand-in magnet,
  and the material names in a design file are still not checked against a
  material library. Do not use the results to judge a motor.
- Cogging torque is hard to resolve. It needs a mesh with at least 10 node
  spacings around the air gap in a period, which the default mesh gives
  only on some designs (the reference motor's `size` 0.5 does; its default
  does not, and a warning says so). Where cogging torque is about a
  thousandth of the rated torque, as on the reference motor (about 2 mN m
  against 3.19 N m at 10 A), its period and mean come out right but its size
  does not settle with a finer mesh: 1.8 to 2.4 mN m over `size` 0.5 to
  0.2, and 27 % between `size` 0.35 and 0.25 on the 24-slot 28-pole
  example. Take such values as an order of magnitude. The 12-slot 10-pole
  example, with larger cogging, changes by about 3 % from `size` 0.5 to
  0.35. FEMM has the same difficulty: its normal mesh puts the reference
  motor's cogging 41 % above its fine one.
- The waveform comparison with FEMM covers the 36-slot 42-pole and 12-slot
  10-pole examples with the same placeholder materials in both programs; the
  24-slot 28-pole example is not compared. The loaded torque over a cycle,
  its ripple and torque constant are not compared with FEMM either; they
  are checked against the back-EMF constant and a second way of working
  out the mean torque. It checks the 2D solver and the studies, not the
  motor.
- Two-dimensional only: no end effects, which matter in short, wide motors,
  and no end windings, skew, losses or efficiency. Back-EMF, torque and
  ripple are 2D values times the stack length.
- Torque and flux linkage for the whole machine are worked out from the
  repeating slice, which assumes the slices are identical.
- Outer-rotor surface-magnet motors only, with parallel-sided teeth. The
  FEMM model is written for parallel magnetisation only, and the checks
  against FEMM are for that.
- There is no air region outside the rotor yoke or inside the stator bore.
  In FEMM, on the reference motor at 10 A and 40 A, adding air inside the
  bore and out to 1.5 times the outer radius changed the torque by under
  0.02 %. This was not checked with Pole-Craft's own solver.
- Winding, lengthwise and temperature values are read but not used yet: for
  example, nothing checks that the turns fit in the slot.
- The mesh-refinement check and the static FEMM comparison cover the default
  mesh of the reference motor only. Coarser meshes are accepted without a
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

[0.0.5]: https://github.com/pole-craft/pole-craft/releases/tag/v0.0.5
[0.0.4]: https://github.com/pole-craft/pole-craft/releases/tag/v0.0.4
[0.0.3]: https://github.com/pole-craft/pole-craft/releases/tag/v0.0.3
[0.0.2]: https://github.com/pole-craft/pole-craft/releases/tag/v0.0.2
[0.0.1]: https://github.com/pole-craft/pole-craft/releases/tag/v0.0.1
