"""The example scripts run as written."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Every example, and whether it needs Gmsh.
EXAMPLES = {
    "choose_a_solver.py": True,
    "core_with_an_air_gap.py": False,
    "inductance_of_a_coil.py": False,
    "material_models.py": False,
    "mesh_a_design.py": True,
    "motor_over_a_cycle.py": True,
    "picture_of_the_field.py": True,
    "slice_against_whole_machine.py": True,
    "solve_a_motor.py": True,
    "torque_against_current.py": True,
    "winding_layouts.py": False,
}
NEEDS_GMSH = sorted(name for name, needed in EXAMPLES.items() if needed)


def run_example(repo_root: Path, name: str, folder: Path) -> subprocess.CompletedProcess:
    """Run an example in an empty folder, as a user would from anywhere."""
    return subprocess.run(
        [sys.executable, str(repo_root / "examples" / name)],
        cwd=folder, capture_output=True, text=True,
    )


def numbers(line: str) -> list[float]:
    return [float(text) for text in re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?", line)]


def test_mesh_a_design_example_runs(repo_root: Path, tmp_path: Path) -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")
    script = repo_root / "examples" / "mesh_a_design.py"
    done = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    lines = done.stdout.splitlines()
    assert lines[0] == "36 slots, 42 poles"
    assert lines[1] == "repeats 6 times, every 60 degrees"
    assert lines[2] == "winding layout: " + "ACcbBAacCBba" * 3
    assert lines[3] == "winding factor: 0.9330"
    assert lines[4].startswith("77 regions, magnet area ")
    assert lines[5] == "2480 nodes, 4838 triangles"
    assert (tmp_path / "qdd_outer_36s42p_mesh.png").read_bytes().startswith(PNG_SIGNATURE)


def test_readme_shows_the_calls_the_example_uses(repo_root: Path) -> None:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    for call in ("load_design(", "machine_for(", "build_geometry(", "build_mesh(", "plot_mesh("):
        assert call in readme, call
        assert call in (repo_root / "examples" / "mesh_a_design.py").read_text(encoding="utf-8")


def test_mesh_a_design_example_explains_a_missing_gmsh(repo_root: Path, tmp_path: Path) -> None:
    script = repo_root / "examples" / "mesh_a_design.py"
    code = (
        "import runpy, sys; sys.modules['gmsh'] = None; "
        "runpy.run_path(sys.argv[1], run_name='__main__')"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert done.stderr.strip().splitlines() == [
        "meshing needs the optional Gmsh package, which is not installed; "
        'install it with: pip install "polecraft[mesh]"'
    ]
    # What does not need a mesh is still printed.
    assert done.stdout.splitlines()[0] == "36 slots, 42 poles"
    assert list(tmp_path.iterdir()) == []


# All the examples


def test_every_example_is_listed(repo_root: Path) -> None:
    # A new example must be added to the table above, and so to these tests.
    found = sorted(path.name for path in (repo_root / "examples").glob("*.py"))
    assert found == sorted(EXAMPLES)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_says_how_to_run_it_and_what_it_needs(repo_root: Path, name: str) -> None:
    text = (repo_root / "examples" / name).read_text(encoding="utf-8")
    heading = text.split('"""')[1]
    assert f"Run from any folder: python examples/{name}" in heading
    if EXAMPLES[name]:
        assert 'Needs the optional Gmsh package (pip install "polecraft[mesh]").' in heading
    else:
        assert "Needs no mesher" in heading
        assert "polecraft.core.mesh" not in text
    assert 'if __name__ == "__main__":' in text


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_that_uses_stand_in_materials_says_so(repo_root: Path, name: str) -> None:
    text = (repo_root / "examples" / name).read_text(encoding="utf-8")
    if "generic_35A_placeholder" in text:
        heading = text.split('"""')[1]
        assert "placeholder" in heading
        # No grade name is typed in: the design's own names are used as keys.
        assert "M270" not in text and "N42" not in text


@pytest.mark.parametrize("name", NEEDS_GMSH)
def test_example_explains_a_missing_gmsh(repo_root: Path, tmp_path: Path, name: str) -> None:
    code = (
        "import runpy, sys; sys.modules['gmsh'] = None; "
        "runpy.run_path(sys.argv[1], run_name='__main__')"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(repo_root / "examples" / name)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert done.stderr.strip().splitlines() == [
        "meshing needs the optional Gmsh package, which is not installed; "
        'install it with: pip install "polecraft[mesh]"'
    ]
    assert list(tmp_path.iterdir()) == []


# The examples that need no mesher


def test_winding_layouts_example(repo_root: Path, tmp_path: Path) -> None:
    done = run_example(repo_root, "winding_layouts.py", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    lines = done.stdout.splitlines()
    assert len(lines) == 8
    by_counts = {tuple(line.split()[:2]): line for line in lines[1:]}
    reference = by_counts[("36", "42")]
    assert "6 x" in reference and "reversed" in reference and "0.9330" in reference
    # One letter per tooth, and only the slice that repeats: 6 of 36 teeth,
    # 3 of 12, 6 of 12, and all 9 where nothing repeats.
    assert reference.split()[-1] == "ACcbBA"
    assert "the same" in by_counts[("12", "8")] and by_counts[("12", "8")].split()[-1] == "ABC"
    assert by_counts[("12", "10")].split()[-1] == "AabBCc"
    assert by_counts[("9", "8")].split()[-1] == "AabBbcCca"
    assert "cannot carry a three-phase winding" in by_counts[("10", "8")]
    assert list(tmp_path.iterdir()) == []


def test_core_with_an_air_gap_example(repo_root: Path, tmp_path: Path) -> None:
    done = run_example(repo_root, "core_with_an_air_gap.py", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    lines = done.stdout.splitlines()
    rows = [numbers(line) for line in lines[1:6]]
    assert [row[0] for row in rows] == [50.0, 400.0, 700.0, 1000.0, 1800.0]
    for ampere_turns, by_formula, solved, iterations in rows:
        assert solved == pytest.approx(by_formula, abs=2e-6)  # as printed, six decimals
        assert 1 <= iterations <= 15
    assert rows[0][1] < 0.2 and rows[-1][1] > 2.0  # from nearly linear to deep saturation
    # The formula column, worked out here by other means: at 400 ampere-turns
    # the gap takes B g / mu0 and the iron H l, with B = mu0 H + 2 H / (300 + H).
    b = rows[1][1]
    h = (400.0 - b * 0.0005 / (4e-7 * 3.141592653589793)) / 0.020
    assert b == pytest.approx(4e-7 * 3.141592653589793 * h + 2.0 * h / (300.0 + h), abs=1e-5)
    assert lines[-1] == "Without saturation the last case would give 4.5 T."
    assert list(tmp_path.iterdir()) == []


def test_inductance_of_a_coil_example(repo_root: Path, tmp_path: Path) -> None:
    done = run_example(repo_root, "inductance_of_a_coil.py", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    lines = done.stdout.splitlines()
    assert numbers(lines[0]) == [147.4454]  # mu0 40^2 0.05 (0.012 + 0.008 / 3) / 0.01
    rows = [numbers(line) for line in lines[2:6]]
    assert [row[0] for row in rows] == [4.0, 8.0, 16.0, 32.0]
    differences = [abs(row[2]) for row in rows]
    assert differences[0] < 0.3  # percent
    for coarse, fine in zip(differences, differences[1:]):
        assert 3.0 < coarse / fine < 5.0  # a quarter each time
    assert lines[-1] == "Halving the triangles cuts the difference to a quarter."
    assert list(tmp_path.iterdir()) == []


def test_material_models_example(repo_root: Path, tmp_path: Path) -> None:
    done = run_example(repo_root, "material_models.py", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    lines = done.stdout.splitlines()
    assert lines[0] == "materials in the library: copper, generic_35A_placeholder, generic_NdFeB_placeholder"
    assert "relative permeability 1.00" in lines[2]
    assert "remanence 1.2 T" in lines[3] and "1.05" in lines[3] and "stand-in" in lines[3]
    assert "placeholder steel (not real steel data)" in lines
    rows = [numbers(line) for line in lines[7:13]]
    assert [row[0] for row in rows] == [0.5, 1.0, 1.5, 1.8, 2.0, 2.2]
    for b, from_table, from_formula, permeability in rows:
        # The table is the formula, interpolated between its rows.
        assert from_table == pytest.approx(from_formula, rel=0.02)
        h = from_formula
        assert b == pytest.approx(4e-7 * 3.141592653589793 * h + 2.0 * h / (300.0 + h), abs=2e-3)
    permeabilities = [row[3] for row in rows]
    assert permeabilities == sorted(permeabilities, reverse=True)  # the steel saturates
    assert "H rises with B all the way: True" in lines
    assert list(tmp_path.iterdir()) == []


# The examples that solve a motor


@pytest.fixture
def needs_gmsh() -> None:
    pytest.importorskip("gmsh", reason="meshing needs the optional Gmsh package")


def test_solve_a_motor_example(needs_gmsh: None, repo_root: Path, tmp_path: Path) -> None:
    done = run_example(repo_root, "solve_a_motor.py", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    out = done.stdout
    lines = out.splitlines()
    assert lines[0] == "36 slots, 42 poles, placeholder steel and magnet"
    assert lines[1] == "mesh: 2480 nodes, 4838 triangles"
    assert "current: 10 A peak" in lines
    assert "Newton iteration" in lines
    converged = [line for line in lines if line.startswith("  converged in ")]
    assert len(converged) == 1 and 4 <= numbers(converged[0])[0] <= 20
    (torque_line,) = [line for line in lines if line.startswith("torque: ")]
    assert 2.5 < numbers(torque_line)[0] < 4.0  # with the stand-in materials
    linked = [numbers(line)[0] for line in lines if line.startswith("flux linkage of phase ")]
    assert len(linked) == 3 and 5.0 < max(abs(value) for value in linked) < 20.0  # mWb
    (steel_line,) = [line for line in lines if line.startswith("flux density in the steel: ")]
    assert 1.0 < numbers(steel_line)[0] < 2.5
    timing = lines[lines.index(next(line for line in lines if line.startswith("Timing (solver: "))) :]
    assert [line.split("  ")[1].strip() for line in timing[1:8]] == [
        "solver start-up", "assemble", "boundary conditions", "linear solve",
        "materials and residual", "other", "total",
    ]
    assert list(tmp_path.iterdir()) == []


def test_torque_against_current_example(needs_gmsh: None, repo_root: Path, tmp_path: Path) -> None:
    done = run_example(repo_root, "torque_against_current.py", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    lines = done.stdout.splitlines()
    assert "placeholder steel and magnet" in lines[0]
    rows = [numbers(line) for line in lines[3:8]]
    assert [row[0] for row in rows] == [0.0, 5.0, 10.0, 20.0, 40.0]
    torque = [row[1] for row in rows]
    assert abs(torque[0]) < 0.01  # no current, no torque at this position
    assert torque[1:] == sorted(torque[1:]) and torque[1] > 1.0
    per_ampere = [row[2] for row in rows[1:]]
    assert per_ampere == sorted(per_ampere, reverse=True)  # falls as the steel saturates
    assert per_ampere[-1] < 0.85 * per_ampere[0]
    # A solve that starts from the field before it takes fewer iterations
    # than the first, which starts from nothing.
    iterations = [row[-1] for row in rows]
    assert iterations[1] < iterations[0]
    assert list(tmp_path.iterdir()) == []


def test_motor_over_a_cycle_example(needs_gmsh: None, repo_root: Path, tmp_path: Path) -> None:
    done = run_example(repo_root, "motor_over_a_cycle.py", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""  # the default mesh of this design is fine enough for its cogging
    lines = done.stdout.splitlines()
    assert lines[0] == "12 slots, 10 poles, placeholder steel and magnet"
    assert "One period is 6.0 degrees" in lines[2] and "360 / LCM(12, 10)" in lines[2]

    # Cogging: twelve angles over the period, a clean wave that averages to zero.
    rows = [numbers(line) for line in lines[5:17]]
    assert [row[0] for row in rows] == [0.5 * k for k in range(12)]
    cogging = [row[1] for row in rows]
    spread = max(cogging) - min(cogging)
    assert 5.0 < spread < 15.0  # mN m
    assert abs(sum(cogging) / 12) < 0.05 * spread
    # One wave a period: half a period on, the torque is the opposite (0.5 mN m off at most here).
    for k in range(6):
        assert cogging[k + 6] == pytest.approx(-cogging[k], abs=0.15 * spread)
    assert cogging[3] < 0.0 < cogging[9]  # the sign of the torque at 1.5 and at 4.5 degrees
    assert numbers(lines[17])[0] == pytest.approx(spread, abs=0.02)

    # Back-EMF: twelve angles over the electrical cycle, 30 electrical degrees
    # apart, so each phase is the one before shifted by four rows.
    at = lines.index(next(line for line in lines if line.startswith("Back-EMF at 1000 rpm")))
    assert "72 degrees of rotor angle" in lines[at]
    table = [numbers(line) for line in lines[at + 3 : at + 15]]
    assert [row[0] for row in table] == [6.0 * k for k in range(12)]
    a, b, c = ([row[phase] for row in table] for phase in (1, 2, 3))
    assert a[0] < 0.0 < b[0] and c[0] < 0.0  # phase B is at its positive peak at angle 0
    for k in range(12):
        assert b[k] == pytest.approx(a[(k + 8) % 12], abs=0.01)
        assert c[k] == pytest.approx(a[(k + 4) % 12], abs=0.01)
    voltage, frequency, constant = numbers(lines[at + 15])
    assert frequency == pytest.approx(5 * 1000 / 60, abs=0.05)  # pole pairs times turns per second
    assert 3.0 < voltage < 3.7
    assert voltage == pytest.approx(constant * 1000 * 3.141592653589793 / 30, abs=0.005)
    assert 5.0 < numbers(lines[at + 16])[0] < 15.0  # percent

    # Loaded torque: the torque constant is 1.5 times the back-EMF constant.
    at = lines.index(next(line for line in lines if line.startswith("Torque with 5 A peak")))
    mean, ripple = numbers(lines[at + 1])
    assert 0.2 < mean < 0.28 and 0.3 < ripple < 2.5
    per_ampere, _, by_power_balance = numbers(lines[at + 2])
    assert per_ampere == pytest.approx(by_power_balance, rel=0.01)
    assert per_ampere == pytest.approx(constant * 1.5, rel=0.01)
    assert numbers(lines[at + 3])[0] == pytest.approx(mean, rel=0.01)  # the other method
    assert list(tmp_path.iterdir()) == []


def test_slice_against_whole_machine_example(
    needs_gmsh: None, repo_root: Path, tmp_path: Path
) -> None:
    done = run_example(repo_root, "slice_against_whole_machine.py", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    lines = done.stdout.splitlines()
    assert lines[0] == "36 slots, 42 poles, placeholder steel and magnet"
    assert lines[1] == "repeats 6 times; the field in the next slice is reversed"
    part = numbers(next(line for line in lines if line.startswith("slice ")))
    whole = numbers(next(line for line in lines if line.startswith("whole machine ")))
    # Nodes, seconds, torque, flux linkage. The whole machine has about six
    # times the nodes and gives the same results.
    assert 5.0 < whole[0] / part[0] < 7.0
    assert part[2] == pytest.approx(whole[2], rel=5e-3) and part[2] > 1.0
    assert part[3] == pytest.approx(whole[3], rel=5e-3)
    for line in lines[-3:-1]:
        assert numbers(line)[0] < 0.5  # percent
    assert list(tmp_path.iterdir()) == []


def test_choose_a_solver_example(needs_gmsh: None, repo_root: Path, tmp_path: Path) -> None:
    from polecraft.core.compute import available_solvers, known_solvers

    done = run_example(repo_root, "choose_a_solver.py", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""  # a solver that is not installed is listed, not warned about
    lines = done.stdout.splitlines()
    assert any("placeholder steel and magnet" in line for line in lines)
    listed = [line.split(":")[0].strip() for line in lines[1 : 1 + len(known_solvers())]]
    assert listed[: len(available_solvers())] == available_solvers()
    assert sorted(listed) == sorted(known_solvers())
    timed = [line for line in lines if " ms, of which " in line]
    assert [line.split()[0] for line in timed] == available_solvers()
    iterations = {numbers(line.split(";")[1])[0] for line in timed}
    assert len(iterations) == 1  # every solver takes the same number of iterations
    for line in timed:
        assert numbers(line.split("differs from the first by")[1])[0] < 1e-9
    assert list(tmp_path.iterdir()) == []


def test_picture_of_the_field_example(needs_gmsh: None, repo_root: Path, tmp_path: Path) -> None:
    done = run_example(repo_root, "picture_of_the_field.py", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    lines = done.stdout.splitlines()
    assert 1.5 < numbers(lines[0])[0] < 3.0  # largest flux density, tesla
    assert lines[0].endswith("with placeholder steel and magnet")
    assert lines[1] == "picture: qdd_outer_36s42p_field.png"
    picture = tmp_path / "qdd_outer_36s42p_field.png"
    data = picture.read_bytes()
    assert data.startswith(PNG_SIGNATURE) and len(data) > 50_000
    assert [path.name for path in tmp_path.iterdir()] == [picture.name]


def test_readme_lists_every_example(repo_root: Path) -> None:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    base = "https://github.com/pole-craft/pole-craft/blob/main/examples/"
    for name, needs_gmsh in EXAMPLES.items():
        rows = [line for line in readme.splitlines() if line.startswith(f"| [{name}]({base}{name})")]
        assert len(rows) == 1, name
        # The last column says whether the example needs Gmsh.
        assert rows[0].endswith("| yes |" if needs_gmsh else "| no |"), name


def test_examples_use_only_public_names(repo_root: Path) -> None:
    for name in EXAMPLES:
        text = (repo_root / "examples" / name).read_text(encoding="utf-8")
        imported = re.findall(r"^\s*(?:from polecraft[\w.]* import|import polecraft)(.*)$", text, re.M)
        assert imported or name == "nothing", name
        assert not re.search(r"\bimport _|\b_[a-z]\w*\(|\._[a-z]", text), name
