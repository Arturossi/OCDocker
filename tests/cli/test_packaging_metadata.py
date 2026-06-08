#!/usr/bin/env python3

# Description
###############################################################################
'''
Packaging metadata consistency tests.

These tests keep pyproject optional extras and requirements.txt aligned.
'''

# Imports
###############################################################################
import importlib
import tomllib

from pathlib import Path

import pytest

# Functions
###############################################################################
## Private ##


def _load_pyproject() -> dict:
    with Path("pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _load_requirements_lines() -> list[str]:
    lines: list[str] = []
    for raw in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _dep_name(dep: str) -> str:
    return dep.split(";", maxsplit=1)[0].split("[", maxsplit=1)[0].split(">=", maxsplit=1)[0].split("==", maxsplit=1)[0].strip()


## Public ##


@pytest.mark.order(468)
def test_pyproject_toml_parses():
    data = _load_pyproject()
    assert data["project"]["name"] == "OCDocker"
    assert "dependencies" in data["project"]


@pytest.mark.order(469)
def test_pyproject_uses_ml_extra_name():
    data = _load_pyproject()
    extras = data["project"]["optional-dependencies"]

    assert "ml" in extras
    assert "optuna" not in extras


@pytest.mark.order(470)
def test_requirements_match_core_pyproject_dependencies():
    data = _load_pyproject()
    pyproject_core = data["project"]["dependencies"]
    requirements_core = _load_requirements_lines()

    assert requirements_core == pyproject_core


@pytest.mark.order(471)
def test_core_dependencies_are_minimal():
    data = _load_pyproject()
    core = {_dep_name(dep) for dep in data["project"]["dependencies"]}

    expected = {
        "configargparse",
        "joblib",
        "packaging",
        "pydantic",
        "pydantic-settings",
        "pyyaml",
        "requests",
        "rich",
        "tqdm",
    }
    assert core == expected


@pytest.mark.order(472)
def test_ml_extra_contains_ml_runtime_stack():
    data = _load_pyproject()
    ml_extra = data["project"]["optional-dependencies"]["ml"]

    expected_prefixes = (
        "torch",
        "torchaudio",
        "torchvision",
        "xgboost",
        "visualtorch",
        "optuna",
        "optuna-dashboard",
        "optuna-integration",
        "scikit-learn",
        "numpy",
        "pandas",
        "scipy",
    )
    for prefix in expected_prefixes:
        assert any(dep.startswith(prefix) for dep in ml_extra)

    assert "torchsummary" in ml_extra
    assert "torchviz" in ml_extra


@pytest.mark.order(473)
def test_core_dependencies_exclude_ml_runtime_stack():
    data = _load_pyproject()
    pyproject_core = data["project"]["dependencies"]
    excluded_prefixes = (
        "torch",
        "torchaudio",
        "torchvision",
        "xgboost",
        "visualtorch",
        "optuna",
        "optuna-dashboard",
        "optuna-integration",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "rdkit",
    )

    for prefix in excluded_prefixes:
        assert not any(dep.startswith(prefix) for dep in pyproject_core)

    assert "torchsummary" not in pyproject_core
    assert "torchviz" not in pyproject_core


@pytest.mark.order(474)
def test_optional_extras_include_expected_groups():
    extras = _load_pyproject()["project"]["optional-dependencies"]

    for name in ("analysis", "docking", "db", "workflow", "docs", "dev", "build", "all", "full"):
        assert name in extras, f"missing extra: {name}"


@pytest.mark.order(475)
def test_docking_extra_includes_chemistry_and_clustering_stack():
    docking = _load_pyproject()["project"]["optional-dependencies"]["docking"]
    names = {_dep_name(dep) for dep in docking}

    assert {"rdkit", "openbabel-wheel", "biopython", "spyrmsd", "numpy", "pandas", "scipy", "scikit-learn"} <= names


@pytest.mark.order(476)
def test_analysis_extra_includes_plotting_stack():
    analysis = _load_pyproject()["project"]["optional-dependencies"]["analysis"]
    names = {_dep_name(dep) for dep in analysis}

    assert {"matplotlib", "seaborn", "statsmodels", "pingouin", "networkx"} <= names


@pytest.mark.order(477)
def test_base_package_imports_without_optional_stacks():
    import OCDocker

    assert OCDocker.__version__


@pytest.mark.order(478)
def test_rmsd_clustering_matplotlib_error_message():
    from OCDocker.Processing.Preprocessing import RMSDClustering as mod

    original = mod._require_matplotlib

    def _raise_import_error():
        try:
            raise ImportError("No module named 'matplotlib'")
        except ImportError as exc:
            raise ImportError(
                "Matplotlib is required for RMSD clustering plots. "
                "Install with `pip install 'ocdocker[analysis]'`."
            ) from exc

    mod._require_matplotlib = _raise_import_error
    try:
        with pytest.raises(ImportError, match=r"ocdocker\[analysis\]"):
            mod._require_matplotlib()
    finally:
        mod._require_matplotlib = original
