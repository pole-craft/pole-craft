# Pole-Craft

Open-source design and simulation tool for quasi-direct-drive (QDD) motors.

> **Status: early development (version 0.0.1).** Pole-Craft loads a motor design file and checks it, works out the winding layout and the slice of the motor that repeats, builds the cross-section of an outer-rotor motor as labelled regions (steel, coils, magnets, air), meshes it with triangles and saves a picture with the `polecraft mesh` command. **It does not simulate anything yet:** there is no field solver, so it reports no torque, back-EMF or losses. The roadmap below shows what comes next.

![Mesh of the repeating slice of a 36-slot, 42-pole outer-rotor motor](https://raw.githubusercontent.com/pole-craft/pole-craft/main/docs/images/mesh_36s42p.png)

## What it will do

Pole-Craft takes a motor described in a simple design file and tells you how it performs:

- Builds the motor geometry from parameters (slots, poles, diameters, magnet shape)
- Runs 2D electromagnetic finite-element analysis
- Reports torque, cogging, back-EMF, torque constant, losses and efficiency
- Lets you change magnet grade, steel grade, wire size and turns, and compare the results
- Sweeps many design variants to find the best one for a target

The first focus is surface-magnet outer-rotor and inner-rotor motors of the kind used in QDD robot actuators.

## Why

Commercial motor design tools are expensive, and their free student versions cannot be used for commercial work. Existing open-source tools are general-purpose. Pole-Craft aims to make exploring QDD motor designs fast, scriptable and free for anyone to use.

## Install

Needs Python 3.11 or newer.

```
pip install "polecraft[mesh]"
```

The `mesh` extra installs [Gmsh](https://gmsh.info), which does the meshing. Gmsh is licensed under the GPL, so it is kept optional: Pole-Craft does not include any Gmsh code and only calls it if you install it. Plain `pip install polecraft` gives everything except meshing.

On Linux, Gmsh needs a few system libraries that servers, containers and WSL often lack. If `polecraft mesh` reports `libGLU.so.1: cannot open shared object file` (or a similar library name), install them. On Debian or Ubuntu:

```
sudo apt-get install libglu1-mesa libxcursor1 libxft2 libxinerama1 libxrender1 libgomp1
```

## Try it

Save an example design, for instance [qdd_outer_36s42p.yaml](https://raw.githubusercontent.com/pole-craft/pole-craft/main/designs/qdd_outer_36s42p.yaml), into the folder you are working in. The others are in the [designs folder](https://github.com/pole-craft/pole-craft/tree/main/designs), and a clone of the repository contains them all. Then mesh it and save a picture:

```
polecraft mesh qdd_outer_36s42p.yaml
```

This writes `qdd_outer_36s42p_mesh.png` in the current folder and prints the mesh size and how long each stage took. Only the slice of the motor that repeats is meshed; add `--full` for the whole machine, `-o` to choose the output file (`.png`, `.svg` or `.pdf`), and `--size 0.5` for a finer mesh.

The design files are plain YAML, with lengths in millimetres. The dimensions in the examples are placeholders typical of each motor size, not measured or optimised designs. The design-file format may change between 0.0.x versions.

## Use from Python

The same steps can be run from a script:

```python
from polecraft import load_design, machine_for
from polecraft.core.mesh import build_mesh
from polecraft.plot import plot_mesh

design = load_design("qdd_outer_36s42p.yaml")
machine = machine_for(design)
print(machine.symmetry())                # the slice that repeats

geometry = machine.build_geometry()      # full=True for the whole machine
mesh = build_mesh(geometry, size=1.0)    # needs Gmsh; smaller size, finer mesh
print(len(mesh.nodes), len(mesh.triangles))

plot_mesh(mesh, geometry, "motor.png")
```

A longer example that also prints the winding layout is in [examples/mesh_a_design.py](https://github.com/pole-craft/pole-craft/blob/main/examples/mesh_a_design.py). The Python interface may change between 0.0.x versions.

## Roadmap

- [x] Parametric geometry and meshing for QDD motors (outer rotor; inner rotor to follow)
- [ ] 2D field solver with saturated steel and permanent magnets
- [ ] Rotation: cogging torque, back-EMF, torque ripple
- [ ] Material and winding library
- [ ] Losses, torque-speed curve and efficiency map
- [ ] Parallel and GPU-accelerated design sweeps
- [ ] Validation against FEMM and measured motors
- [ ] Gear module for actuator-level results
- [ ] 3D verification

Changes in each release are listed in the [changelog](https://github.com/pole-craft/pole-craft/blob/main/CHANGELOG.md).

## Built with

Python, [Gmsh](https://gmsh.info) for meshing, and [FEMM](https://www.femm.info) as an independent reference for validation.

## Development setup and running the tests

Clone the repository, then install it in a virtual environment together with the test tools:

```
git clone https://github.com/pole-craft/pole-craft.git
cd pole-craft
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,mesh]"
pytest
```

Without the `mesh` extra everything except meshing works, and the meshing tests are skipped.

## Contributing

The project is just starting. Ideas, questions, bug reports and motor data for validation are welcome: please [open an issue](https://github.com/pole-craft/pole-craft/issues).

## License

Apache License 2.0. See [LICENSE](https://github.com/pole-craft/pole-craft/blob/main/LICENSE).
