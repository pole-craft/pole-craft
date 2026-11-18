"""Release files agree with the package: version, changelog, metadata, manifest."""

import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

import polecraft

REPOSITORY = "https://github.com/pole-craft/pole-craft"


@pytest.fixture
def project(repo_root: Path) -> dict:
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_changelog_starts_with_the_current_version(repo_root: Path) -> None:
    text = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    entries = re.findall(r"^## \[(.+?)\] - (\d{4}-\d{2}-\d{2})$", text, flags=re.MULTILINE)
    assert entries, "no release entries found"
    assert entries[0][0] == polecraft.__version__
    # Newest first, and no version listed twice.
    versions = [version for version, _ in entries]
    assert len(set(versions)) == len(versions)
    assert [date for _, date in entries] == sorted((date for _, date in entries), reverse=True)


def test_changelog_entry_states_the_limits(repo_root: Path) -> None:
    text = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    newest = text.split(f"\n## [{polecraft.__version__}]")[1].split("\n## [")[0]
    assert re.search(r"^### (Added|Changed|Fixed|Removed)$", newest, flags=re.MULTILINE)
    assert "### Known limits" in newest


def test_package_metadata(project: dict) -> None:
    assert project["name"] == "polecraft"
    assert project["license"] == "Apache-2.0"
    assert project["requires-python"] == ">=3.11"
    assert project["authors"] and all(author.get("name") for author in project["authors"])
    assert "Development Status :: 2 - Pre-Alpha" in project["classifiers"]
    urls = project["urls"]
    assert urls["Repository"] == REPOSITORY
    assert urls["Issues"] == f"{REPOSITORY}/issues"
    assert urls["Changelog"] == f"{REPOSITORY}/blob/main/CHANGELOG.md"


def package_names(requirements: list[str]) -> list[str]:
    return sorted(re.split(r"[<>=!~\[ ;]", requirement)[0].lower() for requirement in requirements)


def test_dependency_lists_are_as_intended(project: dict) -> None:
    assert package_names(project["dependencies"]) == ["matplotlib", "numpy", "pyyaml", "scipy"]
    extras = project["optional-dependencies"]
    assert sorted(extras) == ["dev", "mesh", "pardiso"]
    # Gmsh is GPL-licensed: it is in the mesh extra and nowhere else.
    assert package_names(extras["mesh"]) == ["gmsh"]
    assert package_names(extras["dev"]) == ["pytest"]
    # PARDISO comes with Intel's MKL, which is not open source: its own extra.
    assert package_names(extras["pardiso"]) == ["pypardiso"]
    # MKL does not exist for Macs or ARM processors. There the extra must
    # install nothing, or the whole install command would fail.
    (requirement,) = extras["pardiso"]
    marker = requirement.split(";")[1]
    assert "sys_platform != 'darwin'" in marker
    assert "platform_machine == 'x86_64'" in marker and "platform_machine == 'AMD64'" in marker


def test_the_pardiso_marker_leaves_macs_and_arm_out(project: dict) -> None:
    packaging = pytest.importorskip("packaging.markers")
    (requirement,) = project["optional-dependencies"]["pardiso"]
    marker = packaging.Marker(requirement.split(";")[1])
    wanted = {
        ("win32", "AMD64"): True,
        ("linux", "x86_64"): True,
        ("linux", "aarch64"): False,
        ("darwin", "arm64"): False,
        ("darwin", "x86_64"): False,
    }
    for (system, machine), expected in wanted.items():
        environment = {"sys_platform": system, "platform_machine": machine}
        assert marker.evaluate(environment) is expected, (system, machine)


def test_no_test_job_installs_pardiso_on_a_mac(repo_root: Path) -> None:
    workflow = repo_root / ".github" / "workflows" / "tests.yml"
    if not workflow.exists():
        pytest.skip("the workflow files are in the repository, not in a source distribution")
    text = workflow.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    text = "\n".join(lines)
    jobs = re.split(r"\n  (?=[\w-]+:\n    )", text.split("\njobs:\n")[1])
    with_pardiso = [job for job in jobs if "pardiso" in job.lower()]
    assert len(with_pardiso) == 1
    job = with_pardiso[0]
    assert "macos" not in job
    # It runs where the solver exists, and names those systems itself.
    assert "os: [ubuntu-latest, windows-latest]" in job
    assert "runs-on: ${{ matrix.os }}" in job
    # The jobs that do run on a Mac install nothing of it.
    for other in jobs:
        if "macos" in other:
            assert "pardiso" not in other.lower()


def test_readme_says_how_to_install_the_faster_solver(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    section = text.split("\n## Install\n")[1].split("\n## ")[0]
    commands = re.findall(r"```\n(.*?)\n```", section, flags=re.DOTALL)
    assert 'pip install "polecraft[mesh,pardiso]"' in commands
    flat = " ".join(section.split())
    assert "not open source" in flat and "MKL" in flat
    assert "on a Mac" in flat  # where it cannot be had
    assert "From version 0.0.2" in flat  # version 0.0.1 had no solver
    assert "a third of a second" in flat  # the cost of loading it
    assert "Without it" in flat  # what happens when it is not installed
    assert 'SuperLU stays the default' in flat and 'solver="pardiso"' in flat  # used by name only
    # The command in the message a user gets is one the README explains.
    from polecraft.core import compute
    from polecraft.core.compute import install_hint

    # As on a machine the solver exists for, whatever this one is.
    monkeypatch.setattr(compute, "_machine", lambda: ("linux", "x86_64"))
    assert install_hint("pardiso") == 'pip install "polecraft[pardiso]"'
    assert install_hint("pardiso") in flat


def test_gmsh_is_only_an_optional_dependency(project: dict) -> None:
    assert not any("gmsh" in requirement.lower() for requirement in project["dependencies"])
    assert any(r.startswith("gmsh") for r in project["optional-dependencies"]["mesh"])
    required = metadata.requires("polecraft") or []
    for requirement in required:
        if requirement.lower().startswith("gmsh"):
            assert "extra ==" in requirement


def test_source_distribution_includes_tests_and_designs(repo_root: Path) -> None:
    lines = (repo_root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    for wanted in ("graft tests", "graft designs", "include CHANGELOG.md"):
        assert wanted in lines
    # Compiled files carry the path of the machine that made them.
    assert "global-exclude *.py[cod]" in lines


def test_readme_says_how_to_install_and_test(repo_root: Path) -> None:
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert 'pip install "polecraft[mesh]"' in text
    assert 'pip install -e ".[dev,mesh]"' in text
    assert "\npytest\n" in text
    assert "may change between 0.0.x versions" in text


def test_readme_install_section_leads_with_the_mesh_extra(
    repo_root: Path, package_dir: Path
) -> None:
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    section = text.split("\n## Install\n")[1].split("\n## ")[0]
    commands = re.findall(r"```\n(.*?)\n```", section, flags=re.DOTALL)
    assert commands[0] == 'pip install "polecraft[mesh]"'
    # One sentence says the plain install has no mesher and that Gmsh is GPL-licensed.
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(section.split()))
    telling = [s for s in sentences if "GPL-licensed" in s and "no mesher" in s]
    assert len(telling) == 1
    assert "`pip install polecraft`" in telling[0]
    assert "Gmsh" in telling[0]
    # The command is the one the missing-Gmsh message gives.
    source = (package_dir / "core" / "mesh.py").read_text(encoding="utf-8")
    assert "install it with: pip install \"polecraft[mesh]\"" in source


def test_changelog_says_how_a_missing_gmsh_is_reported(repo_root: Path) -> None:
    text = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = text.split("\n## [0.0.2]")[1].split("\n## [")[0]
    assert 'pip install "polecraft[mesh]"' in " ".join(entry.split())
    assert "GPL" in entry


def test_readme_status_names_the_current_version(repo_root: Path) -> None:
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert f"(version {polecraft.__version__})" in text


def test_readme_links_work_on_the_package_index(repo_root: Path) -> None:
    """The package index shows the README away from the repository, so links must be absolute."""
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)]+)\)", text)
    assert targets
    for target in targets:
        assert target.startswith("https://"), target
