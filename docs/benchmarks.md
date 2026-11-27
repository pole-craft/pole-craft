# Benchmarks: Pole-Craft against FEMM 4.2

This is the baseline for the speed work of phase F: how long Pole-Craft 0.0.5
takes on one problem, section by section, and how that compares with FEMM 4.2
on the same problem. The raw numbers are in
[benchmarks/baseline.json](../benchmarks/baseline.json), made by
[benchmarks/femm_comparison.py](../benchmarks/femm_comparison.py). The script
is a manual tool: it needs FEMM, which runs on Windows, and it is not part of
the test suite. `polecraft.benchmark.time_machine` is the part that times a
Pole-Craft run, and it is tested.

Later speed-ups should be measured against these figures, on the same machine.
Times from another machine are not comparable.

## The problem

- Design: `designs/qdd_outer_36s42p.yaml` (36 slots, 42 poles, 90 mm outer diameter).
- Placeholder steel (`generic_35A_placeholder`) and a magnet with remanence 1.2 T and
  relative permeability 1.05: the ones of the FEMM checks, not real grades.
- 10 A peak, phase currents following the rotor.
- **Static:** one solve at rotor angle 0.
- **Sweep:** 12 rotor angles evenly across one cogging period, which for this
  design is 1/252 of a turn.

## What it ran on

| | |
|---|---|
| Processor | AMD Ryzen 9 5900HX, 16 logical cores |
| Memory | 31.4 GB |
| System | Windows 11 (10.0.26300) |
| Software | Python 3.11.5, NumPy 2.4.6, SciPy 1.17.1, Gmsh 4.15.2, Pole-Craft 0.0.5 |
| FEMM | 4.2 (executable dated 2019-04-21), run as `femm.exe -lua-script=... -windowhide` |
| Date | 2026-11-26 |

Pole-Craft used its default solver (SuperLU). How many threads FEMM's solver uses was not checked.

## Results

Seconds, best of the repeats (static: 3, sweep: 2). Node counts are of the mesh as made. FEMM's coarse
mesh was chosen to have about the node count of Pole-Craft's default whole-machine mesh, and
Pole-Craft's fine mesh to have about the node count of FEMM's normal mesh. FEMM's normal mesh is
the one of the FEMM checks (`tests/data/femm/make_reference.py`).

| Program and mesh | Problem | Nodes | geometry | mesh | setup | solve | post | other | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pole-Craft, slice, default mesh | static | 2,480 | 0.00 | 0.16 | 0.00 | 0.08 | 0.00 | 0.00 | 0.24 |
| Pole-Craft, slice, default mesh | sweep | 2,480 | 0.00 | 0.16 | 0.04 | 0.43 | 0.02 | 0.00 | 0.65 |
| Pole-Craft, whole machine, default mesh | static | 14,706 | 0.01 | 0.96 | 0.03 | 0.46 | 0.01 | 0.00 | 1.47 |
| Pole-Craft, whole machine, default mesh | sweep | 14,706 | 0.01 | 0.99 | 0.34 | 2.42 | 0.12 | 0.01 | 3.89 |
| FEMM, coarse mesh | static | 15,031 | 3.48 | 0.16 | 0.00 | 2.16 | 0.13 | 0.18 | 6.11 |
| FEMM, coarse mesh | sweep | 15,031 | 3.48 | 0.00 | 2.73 | 28.85 | 1.54 | 0.30 | 36.90 |
| Pole-Craft, whole machine, fine mesh | static | 68,108 | 0.01 | 2.45 | 0.07 | 2.72 | 0.06 | 0.00 | 5.31 |
| Pole-Craft, whole machine, fine mesh | sweep | 68,108 | 0.01 | 2.43 | 0.94 | 15.84 | 0.69 | 0.04 | 19.95 |
| FEMM, normal mesh | static | 70,246 | 3.62 | 0.61 | 0.00 | 36.99 | 0.53 | 0.18 | 41.92 |
| FEMM, normal mesh | sweep | 70,246 | 3.31 | 0.00 | 2.84 | 716.28 | 7.13 | 0.34 | 729.91 |

The columns are the stages of both programs mapped onto the same names:

- **geometry**: Pole-Craft builds the regions; FEMM's script draws every point, line and
  arc through Lua commands.
- **mesh**: meshing the model (Gmsh for Pole-Craft, FEMM's mesher for FEMM). In a FEMM
  sweep the mesh is made inside the solve call, so its time is part of the solve column and
  this column reads 0. (A separate mesh call on a model that has been turned stopped FEMM 4.2
  with no error, so it is timed apart only for the first angle.)
- **setup**: turning the rotor and building the problem (Pole-Craft); turning the rotor and
  setting the currents (FEMM).
- **solve**: assembling and solving the nonlinear field.
- **post**: torque and flux linkage. For FEMM this includes reading its solution file back,
  because Pole-Craft's solution is already in memory.
- **other**: what the program's own clock does not see. For Pole-Craft it is the stepping code
  between sections; for FEMM it is starting and quitting the program and its files. The
  sections add up to the total by construction, so a large *other* would show an untimed part.

## How much faster

| Comparison (same node count) | Static | Sweep |
|---|---:|---:|
| About 15,000 nodes: FEMM coarse against Pole-Craft whole machine, default mesh | 4.2 times | 9.5 times |
| About 70,000 nodes: FEMM normal against Pole-Craft whole machine, fine mesh | 7.9 times | 36.6 times |
| About 15,000 nodes: FEMM coarse against Pole-Craft slice (2,480 nodes) | 25 times | 56 times |

Read these with the caveats below. The figures are ratios of total time, which for FEMM includes
about 3.5 s of drawing and start-up that a person using FEMM's window would pay once.

## How steady the times are

Pole-Craft's repeats agree to within about 3 %. FEMM's do not. Three repeats of its normal-mesh
static solve took 70, 74 and 42 s, and two repeats of its normal-mesh sweep 730 and 732 s. An earlier
run of the same script on the same machine, which was discarded because its node counts were not
matched, gave 322 s for that sweep, 2.3 times less. The cause was not found (a power setting, background
load or file scanning are possible). So the FEMM times, and the ratios made from them, are good
to a factor of about two at the largest mesh, and the best of the repeats favours FEMM. At the coarse mesh the repeats
agreed to 2 %.

## Same answer, or not

Both programs solve the same machine, but a faster answer only counts if it is as good. The
largest difference in torque, over the angles solved and in percent of the mean torque (about
3.19 N m), against FEMM's normal mesh:

| | Static | Sweep |
|---|---:|---:|
| Pole-Craft, slice, default mesh | 0.04 % | 0.17 % |
| Pole-Craft, whole machine, default mesh | 0.04 % | 0.17 % |
| Pole-Craft, whole machine, fine mesh | 0.05 % | 0.12 % |
| FEMM, coarse mesh | 0.37 % | 0.46 % |

This is the difference from FEMM's normal mesh, not the error against the true answer, which
neither program knows. It shows that at about 15,000 nodes Pole-Craft is not faster by being less
accurate: its default mesh is closer to FEMM's finest than FEMM's own mesh of the same size.

## Which side each choice favours

Favours Pole-Craft:

- **The slice.** The repeating slice is 1/6 of the machine (36 and 42 have 6 as their greatest common divisor); FEMM was given the whole machine.
  The whole-machine rows are the like-for-like ones; the slice rows show what the symmetry buys.
- **No remeshing between angles.** Pole-Craft turns the rotor by moving nodes in the air-gap
  layer; FEMM meshes again at every angle. This is the main reason the sweep ratios are larger
  than the static ones, and it is by design (see `docs/architecture.md`).
- **Warm start.** Each angle starts from the field at the one before.
- **In-memory results.** FEMM writes a solution file and reads it back.
- **Scripting overhead.** FEMM is driven through Lua here, and drawing the model through script
  commands costs about 3.5 s. Its mesh and solve times are not affected by this.

Favours FEMM:

- **Mesh time.** Gmsh takes 0.96 s of Pole-Craft's 1.47 s for one static solve on the whole
  machine. FEMM's mesher takes 0.16 s at the same node count. With one solve, meshing is
  Pole-Craft's largest cost; the speed comes from solving many angles on one mesh.
- **One job at a time.** Both programs ran one job at a time. FEMM can be run in parallel as separate processes, as the waveform
  reference does with 12; Pole-Craft's parallel runs are not built yet (phase F3).
- **Node count is not accuracy.** Equal node counts were used because the plan asks for them, but
  the two meshes place their nodes differently. The torque table above is the check.

Neither side was tuned for the comparison. FEMM's mesh sizes are those of the FEMM checks
(`polecraft.femm`); Pole-Craft's are its defaults, and its fine mesh is a smaller `size`
(0.397) found only to match the node count.

## Repeating it

```
pip install -e ".[dev,mesh]"
python benchmarks/femm_comparison.py
```

It takes about 40 minutes, nearly all of it FEMM. Do not use the computer for anything else
meanwhile. It overwrites `benchmarks/baseline.json`. To time Pole-Craft alone, from Python:

```python
from polecraft.benchmark import time_machine
run = time_machine(machine, materials, angles, current=10.0, full=True)
print(run.report())
```
