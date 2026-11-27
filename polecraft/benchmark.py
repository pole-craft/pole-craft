"""Time a field solve section by section, and describe the machine it ran on.

``time_machine`` does what a study does at its start (geometry, mesh, rotor
angles, solve, torque and flux linkage) and times each part on its own, so
that two runs, or Pole-Craft and another program, can be compared part by
part. The recorded results are in ``benchmarks/`` and ``docs/benchmarks.md``.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import sys
import time
from typing import Mapping, Sequence

from polecraft.core.materials import Material
from polecraft.core.mesh import build_mesh
from polecraft.core.motion import solve_angles
from polecraft.core.post import flux_linkage, torque
from polecraft.core.problem import build_problem
from polecraft.machines.base import Machine
from polecraft.timing import Timings

# Sections in the order a run goes through them.
SECTIONS = ("geometry", "mesh", "turn rotor", "build problem", "solve", "post-processing", "other")


@dataclasses.dataclass(frozen=True, eq=False)
class Benchmark:
    """Time of one run by section, and the size of the problem it solved.

    ``wall`` is the whole run timed from outside. ``sections`` adds up to it
    exactly: ``other`` is what no section accounts for, so a large ``other``
    means something is not being timed.
    """

    sections: Timings
    wall: float  # s
    nodes: int  # of the mesh as meshed (the band's copies come on top)
    elements: int
    angles: int  # rotor angles solved
    torque: tuple[float, ...]  # N m at each angle, from the same solutions

    @property
    def total(self) -> float:
        """Sum of the sections in seconds, equal to ``wall``."""
        return self.sections.total

    def seconds(self, name: str) -> float:
        """Time of one section in seconds; 0 if it did not occur."""
        return dict(self.sections.stages).get(name, 0.0)

    def report(self) -> str:
        """The sections as a table, with the size of the problem."""
        heading = f"Benchmark ({self.nodes} nodes, {self.elements} elements, {self.angles} rotor angles)"
        return "\n".join([heading] + self.sections.report().splitlines()[1:])


def time_machine(
    machine: Machine,
    materials: Mapping[str, Material],
    angles: Sequence[float] = (0.0,),
    current: float = 0.0,
    size: float = 1.0,
    full: bool = False,
    solver: str | None = None,
) -> Benchmark:
    """Solve a machine at each rotor angle and time every section.

    ``angles`` are mechanical radians; one angle is a single static solve.
    ``current`` is the peak phase current in amperes, with the phases
    following the rotor as in the studies. ``full`` solves the whole machine
    instead of its repeating slice. Needs the optional Gmsh package.
    """
    sections = Timings()
    began = time.perf_counter()
    with sections.stage("geometry"):
        geometry = machine.build_geometry(full=full)
    with sections.stage("mesh"):
        mesh = build_mesh(geometry, size=size)
    run = solve_angles(
        mesh,
        angles,
        lambda turned: build_problem(machine, turned, materials, current),
        solver=solver,
    )
    spent = dict(run.timings.stages)
    # Turning the rotor and building the problem are timed by the stepping
    # driver; the rest of its stages belong to the solver itself.
    sections.add("turn rotor", spent["turn rotor"])
    sections.add("build problem", spent["build problem"])
    sections.add("solve", sum(step.solution.timings.total for step in run.steps))
    values = []
    with sections.stage("post-processing"):
        for step in run.steps:
            values.append(torque(step.problem, step.solution))
            flux_linkage(step.problem, step.solution)
    wall = time.perf_counter() - began
    sections.add("other", max(0.0, wall - sections.total))
    return Benchmark(
        sections=sections,
        wall=max(wall, sections.total),
        nodes=len(mesh.nodes),
        elements=len(mesh.triangles),
        angles=len(run.steps),
        torque=tuple(values),
    )


def machine_info() -> dict[str, str | int | float | None]:
    """What the benchmark ran on: processor, memory and software versions.

    Only the standard library is used. A value that cannot be found is None.
    """
    import numpy
    import scipy

    import polecraft

    try:
        import gmsh

        gmsh_version: str | None = str(gmsh.__version__)
    except Exception:  # missing, or its library cannot be loaded
        gmsh_version = None
    return {
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "gmsh": gmsh_version,
        "polecraft": polecraft.__version__,
    }


def _cpu_name() -> str | None:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            with key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as file:
                for line in file:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("page_total", ctypes.c_ulonglong),
                    ("page_free", ctypes.c_ulonglong),
                    ("virtual_total", ctypes.c_ulonglong),
                    ("virtual_free", ctypes.c_ulonglong),
                    ("extended_free", ctypes.c_ulonglong),
                ]

            status = Status()
            status.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.total / 2**30, 1)
        elif sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as file:
                for line in file:
                    if line.startswith("MemTotal"):
                        return round(int(line.split()[1]) / 2**20, 1)
        else:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (OSError, ValueError, AttributeError):
        pass
    return None
