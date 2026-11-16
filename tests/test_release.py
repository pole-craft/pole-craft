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
