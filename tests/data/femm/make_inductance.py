"""Make the FEMM d- and q-axis flux linkages that the inductance tests compare with.

Needs FEMM 4.2 (https://www.femm.info), which runs on Windows. Run from the
repository:

    python tests/data/femm/make_inductance.py
    python tests/data/femm/make_inductance.py --femm "C:\\femm42\\bin\\femm.exe"

FEMM solves the whole reference machine at rotor angle 0 with no current,
with current along the d axis (opposing the magnet at 10 and 40 A, aiding it
at 10 A) and with 10 A along the q axis, each on its normal mesh and on one
twice as fine. The phase currents come from the machine's ``excitation``,
as in the study. The phase flux linkages and torque go to inductance.json
beside this file. Each case is a FEMM process of its own, all at once.

The steel curve and the magnet are placeholders, as for make_reference.py.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from polecraft import design_to_dict, load_design, machine_for
from polecraft.core.materials import Magnet, load_material
from polecraft.femm import femm_script, read_results

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DESIGN = "qdd_outer_36s42p"
STEEL = "generic_35A_placeholder"
REMANENCE, PERMEABILITY = 1.2, 1.05
# name: peak phase current in A, advance in electrical radians (+pi/2 is -d, -pi/2 is +d)
CASES = {
    "none": (0.0, 0.0),
    "d_10": (10.0, 0.5 * math.pi),
    "d_40": (40.0, 0.5 * math.pi),
    "d_aiding_10": (10.0, -0.5 * math.pi),
    "q_10": (10.0, 0.0),
}
FINENESS = {"normal": 1.0, "fine": 2.0}
USUAL_PLACES = (r"C:\femm42\bin\femm.exe", r"C:\Program Files\femm42\bin\femm.exe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--femm", help="path of femm.exe")
    parser.add_argument("--keep", help="folder to keep FEMM's scripts and files in")
    parser.add_argument("--processes", type=int, default=10, help="FEMM processes at once")
    parser.add_argument("--output", default=str(HERE / "inductance.json"))
    arguments = parser.parse_args()
    femm = arguments.femm or next((p for p in USUAL_PLACES if Path(p).exists()), None)
    if femm is None or not Path(femm).exists():
        sys.exit("FEMM was not found; give its place with --femm <path of femm.exe>")

    steel = load_material(STEEL)
    magnet = Magnet(remanence=REMANENCE, relative_permeability=PERMEABILITY)
    design = load_design(ROOT / "designs" / f"{DESIGN}.yaml")
    machine = machine_for(design)
    materials = {design.steel.stator: steel, design.steel.rotor: steel, design.magnet.material: magnet}
    geometry = machine.build_geometry(full=True)
    reference = {
        "what": "FEMM phase flux linkages at rotor angle 0 with d- and q-axis currents, "
        "to check the inductance study against. Placeholder steel curve and magnet, not real grades.",
        "made_by": "tests/data/femm/make_inductance.py",
        "made_on": datetime.date.today().isoformat(),
        "femm": "FEMM 4.2, file of " + _file_date(femm),
        "design": f"designs/{DESIGN}.yaml",
        "design_values": design_to_dict(design),
        "steel": {"name": STEEL, "bh_curve": [[float(b), float(h)] for b, h in zip(steel.b, steel.h)]},
        "magnet": {"remanence": REMANENCE, "relative_permeability": PERMEABILITY},
        "rotor_angle": 0.0,
        "runs": {run: {"fineness": fineness, "cases": {}} for run, fineness in FINENESS.items()},
    }

    jobs = []  # (run, case, script, results)
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(arguments.keep or temporary)
        folder.mkdir(parents=True, exist_ok=True)
        for run, fineness in FINENESS.items():
            for case, (current, advance) in CASES.items():
                results = folder / f"{run}_{case}.txt"
                results.unlink(missing_ok=True)
                script = folder / f"{run}_{case}.lua"
                currents = machine.excitation(0.0, current, advance)
                script.write_text(
                    femm_script(geometry, materials, [currents], results, fineness=fineness, gap_points=8),
                    encoding="utf-8",
                )
                reference["runs"][run]["cases"][case] = {
                    "current": current,
                    "advance": advance,
                    "phase_currents": currents,
                }
                jobs.append((run, case, script, results))

        print(f"{len(jobs)} FEMM runs, {arguments.processes} at a time ...", flush=True)
        began = time.perf_counter()
        waiting, running = list(jobs), []
        while waiting or running:
            while waiting and len(running) < arguments.processes:
                job = waiting.pop(0)
                process = subprocess.Popen([femm, f"-lua-script={job[2]}", "-windowhide"])
                running.append((job, process, time.perf_counter()))
            for item in list(running):
                job, process, started = item
                if process.poll() is not None:
                    running.remove(item)
                    print(f"  {job[0]} {job[1]}: {time.perf_counter() - started:.0f} s", flush=True)
                elif time.perf_counter() - started > 3600:
                    process.kill()
                    sys.exit(f"{job[0]} {job[1]}: FEMM did not finish in an hour; it may be showing a message")
            time.sleep(1.0)
        print(f"all done in {time.perf_counter() - began:.0f} s")

        for run, case, script, results in jobs:
            if not results.exists() or not results.read_text(encoding="utf-8").strip():
                sys.exit(f"{run} {case}: FEMM wrote no results; run {script} from FEMM to see why")
            [found] = read_results(results)
            stored = reference["runs"][run]["cases"][case]
            stored["torque"] = found["torque"]
            stored["flux_linkage"] = found["flux_linkage"]
            reference["runs"][run].setdefault("nodes", found["nodes"])
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
