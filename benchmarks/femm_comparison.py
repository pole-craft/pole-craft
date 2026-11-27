"""Time Pole-Craft and FEMM 4.2 on the same problem, section by section.

A manual tool, not part of the tests. Needs FEMM 4.2 (https://www.femm.info,
Windows) and Gmsh. Run from the repository:

    python benchmarks/femm_comparison.py
    python benchmarks/femm_comparison.py --femm "C:\\femm42\\bin\\femm.exe" --repeats 3

The problem is the reference design (36 slots, 42 poles) with the placeholder
steel and magnet of the FEMM checks, at 10 A peak: one static solve, and a
torque sweep over 12 rotor angles across one cogging period. Each is run on:

- Pole-Craft's repeating slice at its default mesh;
- Pole-Craft's whole machine at its default mesh, and at a finer mesh with
  about the node count of FEMM's normal mesh;
- FEMM's whole machine at its normal mesh (the one of make_reference.py), and
  at a coarser mesh with about the node count of Pole-Craft's whole machine.

Times, node counts, torques and the machine they ran on are written to
benchmarks/baseline.json, and the table that docs/benchmarks.md quotes is
printed. Nothing else should be running while this does: it takes about
30 minutes, most of it FEMM.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from polecraft import design_to_dict, load_design, machine_for
from polecraft.benchmark import Benchmark, machine_info, time_machine
from polecraft.core.materials import Magnet, load_material
from polecraft.core.mesh import build_mesh
from polecraft.femm import femm_script, read_results

ROOT = Path(__file__).resolve().parents[1]
DESIGN = "designs/qdd_outer_36s42p.yaml"
STEEL = "generic_35A_placeholder"
REMANENCE, PERMEABILITY = 1.2, 1.05
CURRENT = 10.0  # A peak
SWEEP_POINTS = 12  # rotor angles over one cogging period
USUAL_PLACES = (r"C:\femm42\bin\femm.exe", r"C:\Program Files\femm42\bin\femm.exe")
# Columns of the common table; both programs' own stages are mapped onto them.
COLUMNS = ("geometry", "mesh", "setup", "solve", "post", "other")


def pole_craft_columns(run: Benchmark) -> dict[str, float]:
    """A Pole-Craft run in the common columns. Setup is turning the rotor and building the problem."""
    columns = {
        "geometry": run.seconds("geometry"),
        "mesh": run.seconds("mesh"),
        "setup": run.seconds("turn rotor") + run.seconds("build problem"),
        "solve": run.seconds("solve"),
        "post": run.seconds("post-processing"),
        "other": run.seconds("other"),
    }
    columns["total"] = sum(columns.values())
    return columns


def femm_columns(cases: list[dict], process_seconds: float) -> dict[str, float]:
    """FEMM's own stage times, added over the cases, in the common columns.

    Loading FEMM's solution file counts as post-processing: Pole-Craft's is
    in memory already. ``other`` is what the program's own clock does not
    see: starting and quitting FEMM, and writing and reading its files.
    In a sweep only the first case's mesh is timed apart (a separate mesh
    call on a turned model stops FEMM), so there the mesh column is folded
    into solve and reads 0: solve then means mesh and solve.
    """
    def add(name: str) -> float:
        return sum(case["times"].get(name, 0.0) for case in cases)

    folded = len(cases) > 1
    columns = {
        "geometry": add("draw"),
        "mesh": 0.0 if folded else add("mesh"),
        "setup": add("setup"),
        "solve": add("solve") + (add("mesh") if folded else 0.0),
        "post": add("load") + add("post"),
    }
    columns["other"] = max(0.0, process_seconds - sum(columns.values()))
    columns["total"] = sum(columns.values())
    return columns


def run_pole_craft(machine, materials, angles, size: float, full: bool) -> dict:
    run = time_machine(machine, materials, angles, current=CURRENT, size=size, full=full)
    return {"columns": pole_craft_columns(run), "nodes": run.nodes, "elements": run.elements,
            "torque": list(run.torque)}


def run_femm(femm: str, machine, geometry, materials, angles, fineness: float, folder: Path, name: str) -> dict:
    results = folder / f"{name}.txt"
    results.unlink(missing_ok=True)
    script = folder / f"{name}.lua"
    cases = [machine.excitation(angle, CURRENT) for angle in angles]
    script.write_text(
        femm_script(geometry, materials, cases, results, fineness=fineness, gap_points=8,
                    rotor_angles=list(angles), timing=True),
        encoding="utf-8",
    )
    began = time.perf_counter()
    try:
        subprocess.run([femm, f"-lua-script={script}", "-windowhide"], check=False, timeout=7200)
    except subprocess.TimeoutExpired:
        sys.exit(f"{name}: FEMM did not finish in two hours; it may be showing a message")
    seconds = time.perf_counter() - began
    if not results.exists() or not results.read_text(encoding="utf-8").strip():
        sys.exit(f"{name}: FEMM wrote no results; run {script} from FEMM to see why")
    found = read_results(results)
    if len(found) != len(angles) or any("times" not in case for case in found):
        sys.exit(f"{name}: FEMM did not report a time for every case")
    return {"columns": femm_columns(found, seconds), "nodes": found[0]["nodes"],
            "elements": found[0]["elements"], "torque": [case["torque"] for case in found],
            "process_seconds": seconds}


def match_nodes(target: int, measure, value: float, nodes: int, slope: float) -> tuple[float, int]:
    """The setting that gives about ``target`` nodes, and what it gave.

    ``measure(setting)`` returns the node count. ``nodes`` is what ``value``
    gives, and ``slope`` is a first guess of how the log of the count changes
    with the log of the setting. Stops within 5 % of the target.
    """
    for _ in range(8):
        if abs(nodes - target) <= 0.05 * target:
            break
        new = round(value * (target / nodes) ** (1.0 / slope), 3)
        if new == value:
            break
        new_nodes = measure(new)
        slope = math.log(new_nodes / nodes) / math.log(new / value)
        if abs(slope) < 0.1:  # a flat step says little: keep to a gentle slope
            slope = math.copysign(0.1, slope)
        value, nodes = new, new_nodes
    return value, nodes


def best(repeats: list[dict]) -> dict:
    """The repeat with the least total time."""
    return min(repeats, key=lambda run: run["columns"]["total"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--femm", help="path of femm.exe")
    parser.add_argument("--repeats", type=int, default=3, help="repeats of each static solve")
    parser.add_argument("--sweep-repeats", type=int, default=2, help="repeats of each sweep")
    parser.add_argument("--keep", help="folder to keep FEMM's scripts and files in")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "baseline.json"))
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
    period = 2.0 * math.pi / math.lcm(design.slots, design.poles)
    sweep = [period * k / SWEEP_POINTS for k in range(SWEEP_POINTS)]
    problems = {"static": [0.0], "sweep": sweep}
    repeats = {"static": arguments.repeats, "sweep": arguments.sweep_repeats}

    print("warm-up (not recorded) ...", flush=True)
    run_pole_craft(machine, materials, [0.0], 2.0, False)

    pole_craft: dict[str, dict] = {}
    femm_runs: dict[str, dict] = {}

    def pole_craft_case(label: str, size: float, full: bool) -> None:
        for problem, angles in problems.items():
            runs = [run_pole_craft(machine, materials, angles, size, full) for _ in range(repeats[problem])]
            pole_craft[f"{label} {problem}"] = {
                "size": size, "whole_machine": full, "problem": problem, "repeats": runs,
            }
            top = best(runs)
            print(f"Pole-Craft {label} {problem}: {top['nodes']} nodes, {top['columns']['total']:.2f} s", flush=True)

    pole_craft_case("slice, default mesh", 1.0, False)
    pole_craft_case("whole, default mesh", 1.0, True)
    default_nodes = best(pole_craft["whole, default mesh static"]["repeats"])["nodes"]

    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(arguments.keep or temporary)
        folder.mkdir(parents=True, exist_ok=True)

        def femm_case(label: str, fineness: float) -> None:
            for problem, angles in problems.items():
                runs = []
                for number in range(repeats[problem]):
                    print(f"FEMM {label} {problem} ({number + 1}/{repeats[problem]}) ...", flush=True)
                    runs.append(run_femm(femm, machine, geometry, materials, angles, fineness,
                                         folder, f"{label.split()[0]}_{problem}_{number}"))
                femm_runs[f"{label} {problem}"] = {"fineness": fineness, "problem": problem, "repeats": runs}
                top = best(runs)
                print(f"FEMM {label} {problem}: {top['nodes']} nodes, {top['columns']['total']:.2f} s", flush=True)

        femm_case("normal mesh", 1.0)
        normal_nodes = best(femm_runs["normal mesh static"]["repeats"])["nodes"]

        # Node counts do not follow a simple power of either setting, so look
        # for the setting that gives the count wanted, a probe at a time.
        def femm_nodes(fineness: float) -> int:
            print(f"  probing FEMM fineness {fineness} ...", flush=True)
            probe = run_femm(femm, machine, geometry, materials, [0.0], fineness, folder, "probe")
            return probe["nodes"]

        def pole_craft_nodes(size: float) -> int:
            return len(build_mesh(geometry, size=size).nodes)

        fineness, _ = match_nodes(default_nodes, femm_nodes, 0.2, femm_nodes(0.2), 0.7)
        femm_case("coarse mesh", fineness)
        size, _ = match_nodes(normal_nodes, pole_craft_nodes, 0.5, pole_craft_nodes(0.5), -1.6)
        pole_craft_case("whole, fine mesh", size, True)

    # Same problem both ways. FEMM's normal mesh is the finest, so it is the reference.
    check = {}
    for problem in problems:
        reference = best(femm_runs[f"normal mesh {problem}"]["repeats"])["torque"]
        runs = {f"Pole-Craft, {label}": pole_craft[f"{label} {problem}"] for label in
                ("slice, default mesh", "whole, default mesh", "whole, fine mesh")}
        runs["FEMM, coarse mesh"] = femm_runs[f"coarse mesh {problem}"]
        for label, entry in runs.items():
            values = best(entry["repeats"])["torque"]
            check[f"{label} {problem}"] = max(abs(a - b) for a, b in zip(values, reference)) / statistics.mean(
                abs(b) for b in reference
            )
    data = {
        "what": "Pole-Craft and FEMM 4.2 timed on the same problem; see docs/benchmarks.md.",
        "made_by": "benchmarks/femm_comparison.py",
        "made_on": datetime.date.today().isoformat(),
        "machine": machine_info(),
        "femm": "FEMM 4.2, file of " + datetime.date.fromtimestamp(Path(femm).stat().st_mtime).isoformat(),
        "design": DESIGN,
        "design_values": design_to_dict(design),
        "steel": STEEL,
        "magnet": {"remanence": REMANENCE, "relative_permeability": PERMEABILITY},
        "current": CURRENT,
        "sweep_angles": sweep,
        "femm_fineness_coarse": fineness,
        "pole_craft_size_fine": size,
        "torque_difference_from_femm_normal_mesh": check,
        "pole_craft": pole_craft,
        "femm_runs": femm_runs,
    }
    Path(arguments.output).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"written: {arguments.output}\n")
    print(table(data))
    print("\nFEMM sweeps: mesh time is inside the solve column (see femm_columns).")


def table(data: dict) -> str:
    """Markdown table of the best repeat of every run, in seconds."""
    lines = ["| Program and mesh | Problem | Nodes | " + " | ".join(COLUMNS) + " | total |",
             "|---|---|---:|" + "---:|" * (len(COLUMNS) + 1)]
    for program, runs in (("Pole-Craft", data["pole_craft"]), ("FEMM", data["femm_runs"])):
        for label, entry in runs.items():
            top = best(entry["repeats"])
            cells = " | ".join(f"{top['columns'][name]:.2f}" for name in (*COLUMNS, "total"))
            lines.append(f"| {program}, {label.rsplit(' ', 1)[0]} | {entry['problem']} | {top['nodes']} | {cells} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
