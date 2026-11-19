"""Make the FEMM cogging and back-EMF waveforms that the tests compare the studies with.

Needs FEMM 4.2 (https://www.femm.info), which runs on Windows. Run from the
repository:

    python tests/data/femm/make_waveforms.py
    python tests/data/femm/make_waveforms.py --femm "C:\\femm42\\bin\\femm.exe"

FEMM solves the whole machine with no current at a series of rotor angles,
turning the rotor and meshing again at each, and the torque and phase flux
linkages are stored in waveforms.json beside this file:

- the reference design's flux linkage over one electrical cycle (back-EMF);
- its cogging torque over one cogging period, on FEMM's normal mesh and on
  one twice as fine, since that torque is about a thousandth of the loaded
  torque and FEMM's own mesh noise is of the same size on the normal mesh;
- the same for the 12-slot 10-pole example, whose cogging is larger.

The angles are cut into short runs that go to several FEMM processes at
once. Expect it to take a while all the same. The steel curve and the magnet
are placeholders, as for make_reference.py.
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
STEEL = "generic_35A_placeholder"
REMANENCE, PERMEABILITY = 1.2, 1.05
COGGING_POINTS = 12  # a cogging period
EMF_POINTS = 24  # an electrical cycle
CHUNK = 4  # angles a FEMM process solves
# name: design, study, fineness of FEMM's mesh
RUNS = {
    "reference_back_emf": ("qdd_outer_36s42p", "back_emf", 1.0),
    "reference_cogging": ("qdd_outer_36s42p", "cogging", 1.0),
    "reference_cogging_fine": ("qdd_outer_36s42p", "cogging", 2.0),
    "small_cogging": ("qdd_outer_12s10p", "cogging", 1.0),
    "small_cogging_fine": ("qdd_outer_12s10p", "cogging", 2.0),
}
USUAL_PLACES = (r"C:\femm42\bin\femm.exe", r"C:\Program Files\femm42\bin\femm.exe")


def angles_for(design, study: str) -> list[float]:
    """Rotor angles in radians, evenly over a cogging period or an electrical cycle."""
    if study == "cogging":
        period = 2.0 * math.pi / math.lcm(design.slots, design.poles)
        return [period * k / COGGING_POINTS for k in range(COGGING_POINTS)]
    cycle = 2.0 * math.pi / (design.poles // 2)
    return [cycle * k / EMF_POINTS for k in range(EMF_POINTS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--femm", help="path of femm.exe")
    parser.add_argument("--keep", help="folder to keep FEMM's scripts and files in")
    parser.add_argument("--processes", type=int, default=12, help="FEMM processes at once")
    parser.add_argument("--output", default=str(HERE / "waveforms.json"))
    arguments = parser.parse_args()
    femm = arguments.femm or next((p for p in USUAL_PLACES if Path(p).exists()), None)
    if femm is None or not Path(femm).exists():
        sys.exit("FEMM was not found; give its place with --femm <path of femm.exe>")

    steel = load_material(STEEL)
    magnet = Magnet(remanence=REMANENCE, relative_permeability=PERMEABILITY)
    reference = {
        "what": "FEMM cogging torque and no-current flux linkage against rotor angle, "
        "to check the studies against. Placeholder steel curve and magnet, not real grades.",
        "made_by": "tests/data/femm/make_waveforms.py",
        "made_on": datetime.date.today().isoformat(),
        "femm": "FEMM 4.2, file of " + _file_date(femm),
        "steel": {"name": STEEL, "bh_curve": [[float(b), float(h)] for b, h in zip(steel.b, steel.h)]},
        "magnet": {"remanence": REMANENCE, "relative_permeability": PERMEABILITY},
        "designs": {},
        "runs": {},
    }

    jobs = []  # (run name, chunk number, script, results, angles)
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(arguments.keep or temporary)
        folder.mkdir(parents=True, exist_ok=True)
        for name, (design_name, study, fineness) in RUNS.items():
            design = load_design(ROOT / "designs" / f"{design_name}.yaml")
            machine = machine_for(design)
            materials = {design.steel.stator: steel, design.steel.rotor: steel, design.magnet.material: magnet}
            reference["designs"][design_name] = design_to_dict(design)
            geometry = machine.build_geometry(full=True)
            angles = angles_for(design, study)
            reference["runs"][name] = {
                "design": design_name,
                "study": study,
                "fineness": fineness,
                "angles": angles,
            }
            for number, start in enumerate(range(0, len(angles), CHUNK)):
                part = angles[start : start + CHUNK]
                results = folder / f"{name}_{number}.txt"
                results.unlink(missing_ok=True)
                script = folder / f"{name}_{number}.lua"
                script.write_text(
                    femm_script(
                        geometry,
                        materials,
                        [machine.excitation(angle, 0.0) for angle in part],
                        results,
                        fineness=fineness,
                        gap_points=8,
                        rotor_angles=part,
                    ),
                    encoding="utf-8",
                )
                jobs.append((name, number, script, results, part))

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
                    print(f"  {job[0]} part {job[1]}: {time.perf_counter() - started:.0f} s", flush=True)
                elif time.perf_counter() - started > 3600:
                    process.kill()
                    sys.exit(f"{job[0]} part {job[1]}: FEMM did not finish in an hour; it may be showing a message")
            time.sleep(1.0)
        print(f"all done in {time.perf_counter() - began:.0f} s")

        for name, number, script, results, part in jobs:
            if not results.exists() or not results.read_text(encoding="utf-8").strip():
                sys.exit(f"{name} part {number}: FEMM wrote no results; run {script} from FEMM to see why")
            found = read_results(results)
            if [case["rotor_angle"] for case in found] != part:
                sys.exit(f"{name} part {number}: FEMM's angles are not the ones asked for")
            run = reference["runs"][name]
            run.setdefault("nodes", found[0]["nodes"])
            run.setdefault("torque", []).extend(case["torque"] for case in found)
            linkage = run.setdefault("flux_linkage", {})
            for case in found:
                for phase, value in case["flux_linkage"].items():
                    linkage.setdefault(phase, []).append(value)
    for name, run in reference["runs"].items():
        print(f"{name}: {run['nodes']} nodes, torque {min(run['torque']):.5f} to {max(run['torque']):.5f} N m")
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
