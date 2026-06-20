#!/usr/bin/env python3

# Description
###############################################################################
'''
Run-bundle builders for prepared Workbench runs.
'''

# Imports
###############################################################################
from __future__ import annotations

import json
import shlex

from pathlib import Path

import yaml

from OCDocker.Workbench.IO import model_to_data
from OCDocker.Workbench.IO import write_model
from OCDocker.Workbench.Models import RunBundle
from OCDocker.Workbench.Models import WorkbenchSpec
from OCDocker.Workbench.Planner import build_run_manifest
from OCDocker.Workbench.Planner import plan_command

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################

BUNDLE_SPEC_FILENAME = "spec.yml"
BUNDLE_PLAN_FILENAME = "plan.json"
BUNDLE_RUN_MANIFEST_FILENAME = "run_manifest.yml"
BUNDLE_MANIFEST_FILENAME = "bundle.json"


# Functions
###############################################################################
## Private ##


def _shell_command(command: tuple[str, ...]) -> str:
    '''Format a command tuple as a shell-safe string.

    Parameters
    ----------
    command : tuple[str, ...]
        Command arguments.

    Returns
    -------
    str
        Shell-safe command string.
    '''

    return " ".join(shlex.quote(part) for part in command)


def _write_json(path: Path, payload: dict) -> Path:
    '''Write a JSON payload to disk.

    Parameters
    ----------
    path : pathlib.Path
        Output path.
    payload : dict
        JSON-compatible payload.

    Returns
    -------
    pathlib.Path
        Written path.
    '''

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_yaml(path: Path, payload: dict) -> Path:
    '''Write a YAML payload to disk.

    Parameters
    ----------
    path : pathlib.Path
        Output path.
    payload : dict
        YAML-compatible payload.

    Returns
    -------
    pathlib.Path
        Written path.
    '''

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _existing_bundle_files(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    '''Return bundle files that already exist.

    Parameters
    ----------
    paths : tuple[pathlib.Path, ...]
        Candidate file paths.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Existing paths.
    '''

    return tuple(path for path in paths if path.exists())


## Public ##


def build_run_bundle(
    spec: WorkbenchSpec,
    bundle_dir: str | Path,
    *,
    run_id: str,
    overwrite: bool = False,
    ocdocker_executable: str = "ocdocker",
    snakemake_executable: str = "snakemake",
) -> RunBundle:
    '''Build a prepared run bundle without executing the planned command.

    Parameters
    ----------
    spec : WorkbenchSpec
        Validated Workbench spec.
    bundle_dir : str or pathlib.Path
        Directory that will receive the bundle files.
    run_id : str
        Stable run identifier for the run manifest.
    overwrite : bool
        If True, overwrite existing bundle files.
    ocdocker_executable : str
        OCDocker executable to use in planned OCScore commands.
    snakemake_executable : str
        Snakemake executable to use in planned VS campaign commands.

    Returns
    -------
    RunBundle
        Prepared bundle summary.
    '''

    root = Path(bundle_dir)
    spec_path = root / BUNDLE_SPEC_FILENAME
    plan_path = root / BUNDLE_PLAN_FILENAME
    run_manifest_path = root / BUNDLE_RUN_MANIFEST_FILENAME
    bundle_manifest_path = root / BUNDLE_MANIFEST_FILENAME
    bundle_files = (spec_path, plan_path, run_manifest_path, bundle_manifest_path)

    existing = _existing_bundle_files(bundle_files)
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing Workbench bundle file(s): {joined}"
        )

    root.mkdir(parents=True, exist_ok=True)
    plan = plan_command(
        spec,
        ocdocker_executable=ocdocker_executable,
        snakemake_executable=snakemake_executable,
    )
    run_manifest = build_run_manifest(spec, plan, run_id=run_id, workspace=root)
    bundle = RunBundle(
        root=root,
        spec_path=spec_path,
        plan_path=plan_path,
        run_manifest_path=run_manifest_path,
        bundle_manifest_path=bundle_manifest_path,
        run_id=run_id,
        spec_type=spec.type,
        name=spec.name,
        command=plan.command,
    )

    write_model(spec_path, spec)
    _write_json(
        plan_path,
        {
            "plan": plan.model_dump(mode="json", exclude_none=True),
            "shell_command": _shell_command(plan.command),
        },
    )
    _write_yaml(run_manifest_path, model_to_data(run_manifest))
    _write_json(bundle_manifest_path, model_to_data(bundle))
    return bundle


__all__ = [
    "BUNDLE_MANIFEST_FILENAME",
    "BUNDLE_PLAN_FILENAME",
    "BUNDLE_RUN_MANIFEST_FILENAME",
    "BUNDLE_SPEC_FILENAME",
    "build_run_bundle",
]
