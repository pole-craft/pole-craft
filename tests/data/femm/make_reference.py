"""Make the FEMM reference results that the tests compare the solver with.

Needs FEMM 4.2 (https://www.femm.info), which runs on Windows. Run from the
repository:

    python tests/data/femm/make_reference.py
    python tests/data/femm/make_reference.py --femm "C:\\femm42\\bin\\femm.exe"

It writes a FEMM script for the reference design from the same regions,
steel table, magnet and currents that the solver is given, runs FEMM on it
and stores what FEMM finds in qdd_outer_36s42p.json beside this file. FEMM
is run three times: on a normal mesh, on a finer one to show how far its own
answer still moves, and with air round the machine to show what the solver's
closed outer boundary leaves out. Expect it to take a while.

The steel curve and the magnet are placeholders, not the grades the design
names: the comparison checks the solver, not the motor.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from polecraft import design_to_dict, load_design, machine_for
from polecraft.core.materials import Magnet, load_material
from polecraft.femm import femm_script, read_results

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DESIGN = "designs/qdd_outer_36s42p.yaml"
STEEL = "generic_35A_placeholder"
REMANENCE, PERMEABILITY = 1.2, 1.05
CURRENTS = (0.0, 10.0, 40.0)  # peak phase current in amperes
GAP_POINTS = 1440
# name: fineness of FEMM's mesh, and how far air reaches round the machine.
RUNS = {
    "normal": (1.0, None),
    "fine": (2.0, None),
    "air_outside": (1.0, 1.5),
}
USUAL_PLACES = (r"C:\femm42\bin\femm.exe", r"C:\Program Files\femm42\bin\femm.exe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--femm", help="path of femm.exe")
    parser.add_argument("--keep", help="folder to keep FEMM's scripts and files in")
    parser.add_argument("--output", default=str(HERE / "qdd_outer_36s42p.json"))
    arguments = parser.parse_args()
    femm = arguments.femm or next((p for p in USUAL_PLACES if Path(p).exists()), None)
    if femm is None or not Path(femm).exists():
        sys.exit("FEMM was not found; give its place with --femm <path of femm.exe>")

    design = load_design(ROOT / DESIGN)
    machine = machine_for(design)
    steel = load_material(STEEL)
    materials = {
        design.steel.stator: steel,
        design.steel.rotor: steel,
        design.magnet.material: Magnet(remanence=REMANENCE, relative_permeability=PERMEABILITY),
    }
    geometry = machine.build_geometry(full=True)
    cases = [machine.excitation(0.0, current) for current in CURRENTS]
    pole_pairs = design.poles // 2

    reference = {
        "what": "FEMM results for the reference design, to check the solver against. "
        "Placeholder steel curve and magnet, not real grades.",
        "made_by": "tests/data/femm/make_reference.py",
        "made_on": datetime.date.today().isoformat(),
        "femm": "FEMM 4.2, file of " + _file_date(femm),
        "design": DESIGN,
        "design_values": design_to_dict(design),
        "steel": {"name": STEEL, "bh_curve": [[float(b), float(h)] for b, h in zip(steel.b, steel.h)]},
        "magnet": {"remanence": REMANENCE, "relative_permeability": PERMEABILITY},
        "rotor_angle": 0.0,
        "advance": 0.0,
        "runs": {},
    }
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(arguments.keep or temporary)
        folder.mkdir(parents=True, exist_ok=True)
        for name, (fineness, surround) in RUNS.items():
            results = folder / f"{name}.txt"
            results.unlink(missing_ok=True)
            script = folder / f"{name}.lua"
            script.write_text(
                femm_script(
                    geometry,
                    materials,
                    cases,
                    results,
                    fineness=fineness,
                    gap_points=GAP_POINTS,
                    surround=surround,
                ),
                encoding="utf-8",
            )
            print(f"{name}: running FEMM ...", flush=True)
            began = time.perf_counter()
            try:
                subprocess.run(
                    [femm, f"-lua-script={script}", "-windowhide"], check=False, timeout=3600
                )
            except subprocess.TimeoutExpired:
                sys.exit(f"{name}: FEMM did not finish in an hour; it may be showing a message")
            if not results.exists() or not results.read_text(encoding="utf-8").strip():
                sys.exit(f"{name}: FEMM wrote no results; run {script} from FEMM to see why")
            found = read_results(results)
            if len(found) != len(CURRENTS):
                sys.exit(f"{name}: FEMM gave {len(found)} cases, not {len(CURRENTS)}")
            print(f"{name}: {found[0]['nodes']} nodes, {time.perf_counter() - began:.0f} s")
            run = {
                "fineness": fineness,
                "air_round_the_machine": surround,
                "nodes": found[0]["nodes"],
                "elements": found[0]["elements"],
                "gap_radius": found[0]["radius"],
                "cases": [],
            }
            for current, case in zip(CURRENTS, found):
                wave = 2.0 * np.fft.fft(case["radial"])[pole_pairs] / GAP_POINTS
                stored = {
                    "current": current,
                    "phase_currents": case["current"],
                    "torque": case["torque"],
                    "flux_linkage": case["flux_linkage"],
                    # Size of the part of the radial gap field with one wave per pole pair.
                    "gap_fundamental": float(abs(wave)),
                }
                if name == "fine":
                    stored["gap_radial"] = [round(float(b), 5) for b in case["radial"]]
                    stored["gap_tangential"] = [round(float(b), 5) for b in case["tangential"]]
                run["cases"].append(stored)
                print(f"  {current:5.1f} A: torque {case['torque']:.4f} N m")
            reference["runs"][name] = run
    # The table FEMM was given is the one in the material file today.
    library = ROOT / "polecraft" / "data" / "materials" / f"{STEEL}.yaml"
    assert yaml.safe_load(library.read_text(encoding="utf-8"))["bh_curve"] == [
        [b, h] for b, h in reference["steel"]["bh_curve"]
    ]
    Path(arguments.output).write_text(json.dumps(reference, indent=1) + "\n", encoding="utf-8")
    print(f"written: {arguments.output}")


def _file_date(path: str) -> str:
    return datetime.date.fromtimestamp(Path(path).stat().st_mtime).isoformat()


if __name__ == "__main__":
    main()
