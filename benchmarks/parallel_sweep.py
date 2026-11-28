"""Time one sweep run with 1, 2, 4, ... worker processes.

A manual tool, not part of the tests. Needs Gmsh. Run from the repository,
with nothing else running:

    python benchmarks/parallel_sweep.py
    python benchmarks/parallel_sweep.py --workers 1 2 4 8 --repeats 2

The sweep is magnet grade against wire size on the reference design (36
slots, 42 poles): 2 placeholder magnets (remanence 1.2 and 1.4 T, not
datasheet grades) by 8 wire sizes, 16 cases. Each case meshes its repeating
slice at the default size and solves the torque over one electrical cycle
(48 rotor angles) at 10 A peak. The wall time of each worker count, the
speed-up over one worker and a check that every count gives the serial
results are written to benchmarks/parallel_sweep.json and printed.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from polecraft import load_design, machine_for
from polecraft.benchmark import machine_info
from polecraft.core.materials import Magnet, load_material
from polecraft.core.mesh import build_mesh
from polecraft.core.winding import phase_winding
from polecraft.studies.loaded_torque import loaded_torque
from polecraft.studies.sweep import run_sweep, usable_cores

ROOT = Path(__file__).resolve().parents[1]
DESIGN = "designs/qdd_outer_36s42p.yaml"
STEEL = "generic_35A_placeholder"
GRADES = {"N42SH": 1.2, "N52": 1.4}  # placeholder remanences, T
PERMEABILITY = 1.05
WIRES = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54]  # mm
CURRENT = 10.0  # A peak


def evaluate(design) -> dict[str, float]:
    """Mean torque and ripple over one electrical cycle, and the phase resistance."""
    machine = machine_for(design)
    mesh = build_mesh(machine.build_geometry())
    magnet = Magnet(remanence=GRADES[design.magnet.material], relative_permeability=PERMEABILITY)
    materials = {design.steel.stator: load_material(STEEL), design.magnet.material: magnet}
    run = loaded_torque(machine, mesh, materials, CURRENT)
    return {"torque": run.mean, "ripple": run.ripple, "resistance": phase_winding(machine).resistance()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, nargs="+", default=None, help="worker counts to time")
    parser.add_argument("--repeats", type=int, default=2, help="runs of each count; the best is kept")
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "parallel_sweep.json"))
    arguments = parser.parse_args()
    cores = usable_cores()
    counts = arguments.workers or [n for n in (1, 2, 4, 8, 12, 16) if n <= cores]
    base = load_design(ROOT / DESIGN)
    lists = {"magnet.material": list(GRADES), "winding.wire_diameter_mm": WIRES}

    serial = None
    rows = []
    for workers in counts:
        walls = []
        for _ in range(arguments.repeats):
            sweep = run_sweep(base, lists, evaluate, workers=workers)
            walls.append(sweep.wall)
            if len(sweep.ok) != len(sweep.cases):
                raise SystemExit(f"{len(sweep.failed)} cases failed: {sweep.failed[0].error}")
            results = [case.results for case in sweep.cases]
            if serial is None:
                serial = results
            same = results == serial
        rows.append({
            "workers": workers,
            "wall": min(walls),
            "walls": walls,
            "case_seconds": sum(case.seconds for case in sweep.cases),
            "same_as_serial": same,
        })
        print(sweep.report())
    one = next((row["wall"] for row in rows if row["workers"] == 1), None)
    for row in rows:
        row["speed_up"] = one / row["wall"] if one else None
        row["efficiency"] = row["speed_up"] / row["workers"] if one else None

    data = {
        "date": datetime.date.today().isoformat(),
        "machine": machine_info(),
        "usable_cores": cores,
        "design": DESIGN,
        "cases": len(sweep.cases),
        "angles_per_case": 48,
        "repeats": arguments.repeats,
        "runs": rows,
    }
    Path(arguments.output).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(table(data))


def table(data: dict) -> str:
    """The table that docs/benchmarks.md quotes."""
    lines = ["| Workers | Wall time (s) | Speed-up | Per worker | Same results as serial |",
             "|---:|---:|---:|---:|---|"]
    for row in data["runs"]:
        speed = f"{row['speed_up']:.2f}" if row["speed_up"] else "-"
        share = f"{row['efficiency']:.0%}" if row["efficiency"] else "-"
        lines.append(f"| {row['workers']} | {row['wall']:.1f} | {speed} | {share} | "
                     f"{'yes' if row['same_as_serial'] else 'NO'} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
