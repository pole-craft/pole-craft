# Pole-Craft

Open-source design and simulation tool for quasi-direct-drive (QDD) motors.

> **Status: early development.** Design files can be loaded and checked. There is no geometry, meshing or simulation yet; the plan below describes what is being built.

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

## Roadmap

- [ ] Parametric geometry and meshing for QDD motors
- [ ] 2D field solver with saturated steel and permanent magnets
- [ ] Rotation: cogging torque, back-EMF, torque ripple
- [ ] Material and winding library
- [ ] Losses, torque-speed curve and efficiency map
- [ ] Parallel and GPU-accelerated design sweeps
- [ ] Validation against FEMM and measured motors
- [ ] Gear module for actuator-level results
- [ ] 3D verification

## Built with

Python, [Gmsh](https://gmsh.info) for meshing, and [FEMM](https://www.femm.info) as an independent reference for validation.

## Development setup

Needs Python 3.11 or newer. The package installs and its tests run, but it does not simulate anything yet.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Contributing

The project is just starting. Ideas, questions and motor data for validation are welcome: please open an issue.

## License

Apache License 2.0. See [LICENSE](LICENSE).
