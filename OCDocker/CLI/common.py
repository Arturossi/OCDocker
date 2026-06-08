#!/usr/bin/env python3
"""Shared CLI bootstrap, parsing helpers, and optional-dependency hints."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
from pathlib import Path
from typing import Optional, Tuple

def _bootstrap_ocdocker_env(ns: argparse.Namespace) -> None:
    '''Bootstrap OCDocker.Initialise explicitly (no import-time side effects).

    - Set `OCDOCKER_CONFIG` env var if provided
    - Call `OCDocker.Initialise.bootstrap(ns)`

    Parameters
    ----------
    ns : argparse.Namespace
        Parsed command-line arguments.
    '''

    if ns.config_file:
        os.environ["OCDOCKER_CONFIG"] = ns.config_file
    init_db = bool(getattr(ns, "_ocdocker_init_db", True))
    create_db_if_missing = bool(getattr(ns, "_ocdocker_create_db_if_missing", False))
    init_mod = importlib.import_module("OCDocker.Initialise")
    if hasattr(init_mod, "bootstrap"):
        bootstrap = init_mod.bootstrap
        parameters = inspect.signature(bootstrap).parameters
        if "create_db_if_missing" in parameters:
            bootstrap(ns, init_db=init_db, create_db_if_missing=create_db_if_missing)
        else:
            bootstrap(ns, init_db=init_db)
    else:
        raise RuntimeError("OCDocker.Initialise.bootstrap not found")


def _print_optional_dependency_hint(
    *,
    feature: str,
    extra: str,
    exc: ModuleNotFoundError,
) -> int:
    '''Print a concise optional dependency hint and return CLI error code.

    Parameters
    ----------
    feature : str
        User-facing feature name that failed.
    extra : str
        Extra name to suggest in pip install hint.
    exc : ModuleNotFoundError
        Original missing dependency exception.

    Returns
    -------
    int
        Exit code for CLI command failures caused by missing dependencies.
    '''

    missing = getattr(exc, "name", "") or "unknown"
    print(f"Error: missing optional dependency '{missing}' required for {feature}.")
    print(f"Install with: pip install \"ocdocker[{extra}]\"")
    return 2


def _suggest_extra_for_missing_module(module_name: str) -> str:
    '''Map missing module names to a recommended pip extra.'''

    mod = (module_name or "").strip()
    if (
        mod.startswith("optuna")
        or mod.startswith("torch")
        or mod.startswith("torchaudio")
        or mod.startswith("torchvision")
        or mod.startswith("xgboost")
        or mod.startswith("torchsummary")
        or mod.startswith("torchviz")
        or mod.startswith("visualtorch")
    ):
        return "ml"
    if mod.startswith("sqlalchemy") or mod.startswith("psycopg") or mod.startswith("pymysql"):
        return "db"
    if mod.startswith("rdkit") or mod.startswith("Bio") or mod.startswith("openbabel") or mod.startswith("spyrmsd"):
        return "docking"
    if mod.startswith("snakemake"):
        return "workflow"
    if (
        mod.startswith("matplotlib")
        or mod.startswith("seaborn")
        or mod.startswith("statsmodels")
        or mod.startswith("pingouin")
        or mod.startswith("lime")
        or mod.startswith("networkx")
        or mod.startswith("rustworkx")
        or mod.startswith("skimage")
    ):
        return "analysis"
    if mod in {"numpy", "pandas", "scipy", "sklearn"} or mod.startswith(("numpy.", "pandas.", "scipy.", "sklearn.")):
        return "docking"
    return "all"


def _db_dependencies_available() -> Tuple[bool, Optional[ModuleNotFoundError]]:
    '''Return whether required DB runtime dependencies are importable.'''

    required_modules = ("sqlalchemy", "sqlalchemy_utils")
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            return False, exc
    return True, None



def _preparse_global_args(argv: list[str]) -> argparse.Namespace:
    '''Extract global flags from anywhere in argv.

    Works around argparse limitation when global options appear after the subcommand.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed global arguments.
    '''

    ns = argparse.Namespace(
        version=False,
        multiprocess=True,
        update=False,
        config_file=None,
        output_level=1,
        overwrite=False,
        log_file=None,
        no_stdout_log=False,
        no_splash=False,
    )

    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--version":
            ns.version = True
            i += 1
            continue
        if tok == "--multiprocess":
            ns.multiprocess = True
            i += 1
            continue
        if tok == "--no-multiprocess":
            ns.multiprocess = False
            i += 1
            continue
        if tok in ("-u", "--update-databases"):
            ns.update = True
            i += 1
            continue
        if tok == "--conf" and i + 1 < len(argv):
            ns.config_file = argv[i + 1]
            i += 2
            continue
        if tok == "--output-level" and i + 1 < len(argv):
            try:
                ns.output_level = int(argv[i + 1])
            except (ValueError, TypeError):
                # Ignore invalid output level values
                pass
            i += 2
            continue
        if tok == "--overwrite":
            ns.overwrite = True
            i += 1
            continue
        if tok == "--log-file" and i + 1 < len(argv):
            ns.log_file = argv[i + 1]
            i += 2
            continue
        if tok == "--no-stdout-log":
            ns.no_stdout_log = True
            i += 1
            continue
        if tok == "--no-splash":
            ns.no_splash = True
            i += 1
            continue
        # skip token
        i += 1
    return ns


def _require_file(p: str, label: str) -> Path:
    '''Ensure a file path exists. Print a helpful message and exit if not.

    Also warns if the path seems to contain a Unicode ellipsis (…)
    which is often a placeholder, not a real path.

    Parameters
    ----------
    p : str
        The file path to check.
    label : str
        A label for the file path (used in error messages).

    Returns
    -------
    Path
        The resolved file path.

    Raises
    ------
    SystemExit
        If the file path is invalid or not found.
    '''

    if "…" in p:
        print(f"Error: {label} contains an ellipsis character (…). Replace it with a real path.")
        raise SystemExit(2)
    path = Path(p).resolve()
    if not path.is_file():
        print(f"Error: {label} file not found: {p}")
        raise SystemExit(2)
    return path
