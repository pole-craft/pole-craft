"""The architecture document names things that exist."""

import importlib.util
import re
from pathlib import Path

import pytest

from polecraft.machines import machine_types
from polecraft.machines.base import Machine

REPOSITORY = "https://github.com/pole-craft/pole-craft"


@pytest.fixture(scope="module")
def document() -> str:
    path = Path(__file__).resolve().parents[1] / "docs" / "architecture.md"
    return path.read_text(encoding="utf-8")


def code_spans(text: str) -> list[str]:
    """Everything written between backticks, outside fenced blocks."""
    without_blocks = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return re.findall(r"`([^`\n]+)`", without_blocks)


def named_paths(text: str) -> list[str]:
    """Code spans that are files or folders of the repository."""
    return sorted(
        {
            span
            for span in code_spans(text)
            if re.fullmatch(r"[\w./-]+", span)
            and ("/" in span or re.search(r"\.(py|md|yaml|toml)$", span))
        }
    )


def paths_in_blocks(text: str) -> list[str]:
    """Repository paths written inside fenced blocks, such as the flow diagram."""
    blocks = "\n".join(re.findall(r"```.*?```", text, flags=re.DOTALL))
    return sorted(set(re.findall(r"\b(?:polecraft|designs|examples|tests|docs)/[\w./-]*", blocks)))


def named_modules(text: str) -> list[str]:
    """Code spans that are dotted module names of the package."""
    return sorted({s for s in code_spans(text) if re.fullmatch(r"polecraft(\.\w+)+", s)})


def test_document_names_files_and_modules(document: str) -> None:
    # Guards the two tests below against passing on an empty list.
    assert len(named_paths(document)) >= 15
    assert len(named_modules(document)) >= 2
    assert "polecraft/core/solver.py" in named_paths(document)
    assert "polecraft/machines/" in named_paths(document)


def found_at(path: str, repo_root: Path, package_dir: Path) -> Path:
    """Where a path of the document is: the package in the package, the rest in the repository."""
    return package_dir.parent / path if path.startswith("polecraft/") else repo_root / path


def test_every_path_in_the_document_exists(
    document: str, repo_root: Path, package_dir: Path
) -> None:
    paths = named_paths(document) + paths_in_blocks(document)
    missing = [path for path in paths if not found_at(path, repo_root, package_dir).exists()]
    assert not missing, f"docs/architecture.md names paths that do not exist: {missing}"


def test_the_flow_diagram_is_checked_too(document: str) -> None:
    assert "polecraft/core/solver.py" in paths_in_blocks(document)
    assert len(paths_in_blocks(document)) >= 5


def test_folders_are_written_with_a_closing_slash(
    document: str, repo_root: Path, package_dir: Path
) -> None:
    for path in named_paths(document):
        assert found_at(path, repo_root, package_dir).is_dir() == path.endswith("/"), path


def test_every_module_in_the_document_can_be_found(document: str) -> None:
    missing = [name for name in named_modules(document) if importlib.util.find_spec(name) is None]
    assert not missing, f"docs/architecture.md names modules that do not exist: {missing}"


def test_every_core_and_machine_module_is_described(document: str, package_dir: Path) -> None:
    # A new module must be added to the document in the same step.
    modules = [m for folder in ("core", "machines") for m in (package_dir / folder).glob("*.py")]
    assert len(modules) >= 10  # the check below is not run over an empty list
    for folder in ("core", "machines"):
        for module in sorted((package_dir / folder).glob("*.py")):
            if module.name != "__init__.py":
                assert f"polecraft/{folder}/{module.name}" in named_paths(document), module.name


def test_the_machine_contract_in_the_document_is_the_real_one(document: str) -> None:
    section = document.split("## The machine contract")[1].split("\n## ")[0]
    methods = set(re.findall(r"`(\w+)\(", section))
    assert methods == {"parameters", "build_geometry", "excitation", "symmetry", "coil_shape"}
    for method in methods:
        assert callable(getattr(Machine, method))


def test_machine_types_in_the_document_are_registered(document: str) -> None:
    for name in machine_types():
        assert f"`{name}`" in document


def test_document_covers_what_it_should(document: str) -> None:
    headings = re.findall(r"^## (.+)$", document, flags=re.MULTILINE)
    assert headings == [
        "From design file to results",
        "What each package does",
        "The machine contract",
        "Units",
        "Adding a machine",
        "Out of scope",
    ]


def test_document_says_what_is_not_built_yet(document: str) -> None:
    assert "not built yet" in document


def test_document_stays_short(document: str) -> None:
    assert len(document.split()) < 1500


def test_document_links_stay_inside_the_repository_or_are_absolute(
    document: str, repo_root: Path
) -> None:
    for target in re.findall(r"\]\(([^)]+)\)", document):
        if target.startswith("https://"):
            continue
        assert ((repo_root / "docs") / target).resolve().exists(), target


def test_readme_links_to_the_document(repo_root: Path) -> None:
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert f"]({REPOSITORY}/blob/main/docs/architecture.md)" in text


def test_document_is_in_the_source_distribution(repo_root: Path) -> None:
    # The tests travel with the source distribution, and this one reads the document.
    lines = (repo_root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    assert "graft docs" in lines


def test_core_and_studies_do_not_name_a_machine_type(package_dir: Path) -> None:
    # The rule the document states: only the machines package knows the types.
    assert len(list((package_dir / "core").glob("*.py"))) >= 8  # not run over an empty list
    for folder in ("core", "studies"):
        for module in sorted((package_dir / folder).glob("*.py")):
            text = module.read_text(encoding="utf-8")
            for name in machine_types():
                assert name not in text, f"{module.name} names the machine type {name}"
            # The machine contract may be named for type hints only.
            for line in text.splitlines():
                if "polecraft.machines" in line and "import" in line:
                    assert line.startswith("    from polecraft.machines.base import"), line
