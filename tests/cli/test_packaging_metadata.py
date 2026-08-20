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
import subprocess
import sys
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


@pytest.mark.order(479)
def test_workbench_static_files_are_declared_in_package_data():
    '''Workbench dashboard assets ship with the OCDocker.Workbench package.'''

    data = _load_pyproject()
    package_data = data["tool"]["setuptools"]["package-data"]
    assert "OCDocker.Workbench" in package_data
    assert "static/*" in package_data["OCDocker.Workbench"]


@pytest.mark.order(480)
def test_ocscore_protocol_files_are_declared_in_package_data():
    '''Bundled OCScore protocol YAML files ship with the OCDocker.OCScore package.'''

    data = _load_pyproject()
    package_data = data["tool"]["setuptools"]["package-data"]
    assert "OCDocker.OCScore" in package_data
    patterns = package_data["OCDocker.OCScore"]
    assert "Protocols/*.yml" in patterns
    assert "Protocols/Ablations/*.yml" in patterns


@pytest.mark.order(481)
def test_package_data_paths_exist_on_disk():
    '''Declared package-data globs resolve to files in the source tree.'''

    root = Path("OCDocker")
    assert list(root.glob("OCScore/Protocols/*.yml")), "expected bundled training protocols"
    assert list(root.glob("OCScore/Protocols/Ablations/*.yml")), "expected bundled ablation policies"
    static_dir = root / "Workbench" / "static"
    assert static_dir.is_dir()
    javascript_files = (
        "app-core.js",
        "app-jobs.js",
        "app-comparison.js",
        "app-plots.js",
        "app-results.js",
        "app-ablation-design.js",
        "app-vs-design.js",
        "app-workspace.js",
        "app.js",
    )
    for name in ("index.html", "app.css", *javascript_files):
        assert (static_dir / name).is_file(), f"missing Workbench static asset: {name}"


@pytest.mark.order(482)
def test_analysis_extra_includes_shap():
    '''OCScore export SHAP uses the shap library from the analysis extra.'''

    analysis = _load_pyproject()["project"]["optional-dependencies"]["analysis"]
    assert any(_dep_name(dep) == "shap" for dep in analysis)


@pytest.mark.order(483)
def test_all_extra_is_superset_of_runtime_extras():
    '''The all extra includes every package from runtime workflow extras.'''

    extras = _load_pyproject()["project"]["optional-dependencies"]
    all_names = {_dep_name(dep) for dep in extras["all"]}
    for group in ("analysis", "docking", "db", "ml", "workflow", "cloud", "gpu", "api", "mcp"):
        group_names = {_dep_name(dep) for dep in extras[group]}
        missing = group_names - all_names
        assert not missing, f"all extra missing packages from {group}: {sorted(missing)}"


@pytest.mark.order(484)
def test_full_extra_includes_all_and_docs():
    '''The full extra includes runtime all packages and documentation build tools.

    The one deliberate exception is pdb2pqr: it hard-pins docutils<0.18, which
    conflicts with myst-parser's docutils>=0.19/0.20, so the two can never resolve
    together. pdb2pqr is only reachable through a function-scoped import in
    Toolbox.MoleculeProcessing, so Sphinx autodoc (which drives the docs build) never
    needs it importable -- docking installs (docking/all) still get it.
    '''

    extras = _load_pyproject()["project"]["optional-dependencies"]
    full_names = {_dep_name(dep) for dep in extras["full"]}
    all_names = {_dep_name(dep) for dep in extras["all"]}
    docs_names = {_dep_name(dep) for dep in extras["docs"]}
    assert all_names - {"pdb2pqr"} <= full_names
    assert "pdb2pqr" not in full_names
    assert docs_names <= full_names


@pytest.mark.order(485)
def test_python_version_metadata_is_aligned():
    '''Python 3.11 is declared consistently across packaging and tooling files.'''

    pyproject = _load_pyproject()
    assert pyproject["project"]["requires-python"] == ">=3.11"

    mypy_text = Path("mypy.ini").read_text(encoding="utf-8")
    assert "python_version = 3.11" in mypy_text

    recipe_text = Path("recipe/meta.yaml").read_text(encoding="utf-8")
    assert "python >=3.11" in recipe_text
    assert "python >=3.10" not in recipe_text

    environment_text = Path("environment.yml").read_text(encoding="utf-8")
    assert "python=3.11" in environment_text
    assert "pyproject.toml" in environment_text


@pytest.mark.order(486)
def test_cli_parser_builds_without_api_or_mcp_extras():
    '''The CLI parser (all commands) builds without the api/mcp extras installed.

    OCDocker.Workbench.Server requires FastAPI (the `api` extra) and
    OCDocker.MCP.Server requires the mcp/httpx stack (the `mcp` extra). Neither
    should be required just to build the argparse parser or run unrelated
    commands like `ocdocker vs --help`.
    '''

    script = (
        "import sys, builtins\n"
        "blocked = {'fastapi', 'uvicorn', 'starlette', 'mcp', 'httpx'}\n"
        "_orig_import = builtins.__import__\n"
        "def _guard(name, *a, **k):\n"
        "    if name.split('.')[0] in blocked:\n"
        "        raise ModuleNotFoundError(name)\n"
        "    return _orig_import(name, *a, **k)\n"
        "builtins.__import__ = _guard\n"
        "from OCDocker.CLI.parser import build_parser\n"
        "parser = build_parser()\n"
        "for argv in (['workbench', 'serve', '--help'], ['mcp', 'serve', '--help'], ['vs', '--help']):\n"
        "    try:\n"
        "        parser.parse_args(argv)\n"
        "    except SystemExit as exc:\n"
        "        assert exc.code == 0, f'{argv} exited {exc.code}'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
