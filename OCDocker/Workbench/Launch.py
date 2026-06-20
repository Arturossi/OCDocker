#!/usr/bin/env python3

# Description
###############################################################################
'''
Non-executing launch-plan helpers for prepared Workbench run bundles.
'''

# Imports
###############################################################################
from __future__ import annotations

import shlex

from pathlib import Path

from OCDocker.Workbench.IO import read_run_manifest
from OCDocker.Workbench.Models import RunLaunchPlan
from OCDocker.Workbench.Registry import RUN_MANIFEST_FILENAMES

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################

DEFAULT_LAUNCH_LOG_DIR = "logs"
DEFAULT_LAUNCH_SCRIPT_NAME = "run.sh"

# Functions
###############################################################################
## Private ##


def _direct_manifest_paths(directory: Path) -> tuple[Path, ...]:
    '''Return direct child run manifest paths in deterministic order.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to inspect.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Matching direct child run manifest paths.
    '''

    return tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.name.lower() in RUN_MANIFEST_FILENAMES
            ),
            key=lambda path: path.name,
        )
    )


def _resolve_run_manifest_path(target: str | Path) -> Path:
    '''Resolve a run manifest path from a manifest file or bundle directory.

    Parameters
    ----------
    target : str or pathlib.Path
        Run manifest file or prepared bundle directory.

    Returns
    -------
    pathlib.Path
        Resolved run manifest path.
    '''

    target_path = Path(target)
    if target_path.is_file():
        return target_path
    if not target_path.exists():
        raise FileNotFoundError(
            f"Workbench launch-plan target does not exist: {target_path}"
        )
    if not target_path.is_dir():
        raise ValueError(
            f"Workbench launch-plan target is not a file or directory: {target_path}"
        )

    candidates = _direct_manifest_paths(target_path)
    if not candidates:
        raise FileNotFoundError(
            f"No Workbench run manifest found directly in directory: {target_path}"
        )
    return candidates[0]


def _resolve_path(base_path: Path, path: str | Path) -> Path:
    '''Resolve a relative path from a base directory without requiring it to exist.

    Parameters
    ----------
    base_path : pathlib.Path
        Base directory.
    path : str or pathlib.Path
        Path to resolve.

    Returns
    -------
    pathlib.Path
        Absolute path or path relative to ``base_path``.
    '''

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base_path / candidate


def _shell_command(command: tuple[str, ...]) -> str:
    '''Format command arguments as a shell-safe command string.

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


def _safe_filename(value: str) -> str:
    '''Return a conservative filename fragment for generated launch files.

    Parameters
    ----------
    value : str
        Input value.

    Returns
    -------
    str
        Filename-safe value.
    '''

    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ).strip("._")
    return cleaned or "run"


def _foreground_command(plan: RunLaunchPlan) -> str:
    '''Build a shell command that runs in the foreground with log redirection.

    Parameters
    ----------
    plan : RunLaunchPlan
        Launch-plan payload.

    Returns
    -------
    str
        Shell command.
    '''

    return (
        f"mkdir -p {shlex.quote(str(plan.log_dir))} && "
        f"cd {shlex.quote(str(plan.cwd))} && "
        f"{plan.shell_command} > {shlex.quote(str(plan.stdout_log))} "
        f"2> {shlex.quote(str(plan.stderr_log))}"
    )


def _background_command(plan: RunLaunchPlan) -> str:
    '''Build a shell command that backgrounds the run and records a PID file.

    Parameters
    ----------
    plan : RunLaunchPlan
        Launch-plan payload.

    Returns
    -------
    str
        Shell command.
    '''

    return (
        f"mkdir -p {shlex.quote(str(plan.log_dir))} && "
        f"cd {shlex.quote(str(plan.cwd))} && "
        f"nohup {plan.shell_command} > {shlex.quote(str(plan.stdout_log))} "
        f"2> {shlex.quote(str(plan.stderr_log))} & "
        f"echo $! > {shlex.quote(str(plan.pid_file))}"
    )


## Public ##


def build_run_launch_plan(
    target: str | Path,
    *,
    log_dir: str | Path = DEFAULT_LAUNCH_LOG_DIR,
    script_path: str | Path | None = None,
) -> RunLaunchPlan:
    '''Build a non-executing launch plan from a prepared Workbench bundle.

    Parameters
    ----------
    target : str or pathlib.Path
        Run manifest file or prepared bundle directory.
    log_dir : str or pathlib.Path
        Log directory. Relative values are resolved from the run workspace.
    script_path : str or pathlib.Path or None
        Optional shell-script path. Relative values are resolved from the run workspace.

    Returns
    -------
    RunLaunchPlan
        Non-executing launch-plan payload for GUI or CLI consumers.
    '''

    manifest_path = _resolve_run_manifest_path(target)
    manifest = read_run_manifest(manifest_path)
    if not manifest.command:
        raise ValueError(
            f"Workbench run manifest has no command to launch: {manifest_path}"
        )

    manifest_dir = manifest_path.parent
    workspace = _resolve_path(manifest_dir, manifest.workspace)
    resolved_log_dir = _resolve_path(workspace, log_dir)
    run_fragment = _safe_filename(manifest.run_id)
    resolved_script_path = (
        _resolve_path(workspace, script_path) if script_path is not None else None
    )
    shell_command = _shell_command(manifest.command)
    plan = RunLaunchPlan(
        manifest_path=manifest_path,
        run_id=manifest.run_id,
        spec_type=manifest.spec_type,
        name=manifest.name,
        status=manifest.status,
        workspace=workspace,
        cwd=workspace,
        command=manifest.command,
        shell_command=shell_command,
        foreground_command="",
        background_command="",
        log_dir=resolved_log_dir,
        stdout_log=resolved_log_dir / f"{run_fragment}.stdout.log",
        stderr_log=resolved_log_dir / f"{run_fragment}.stderr.log",
        pid_file=resolved_log_dir / f"{run_fragment}.pid",
        script_path=resolved_script_path,
    )
    return plan.model_copy(
        update={
            "foreground_command": _foreground_command(plan),
            "background_command": _background_command(plan),
        }
    )


def build_launch_script(plan: RunLaunchPlan) -> str:
    '''Build shell script content for a Workbench launch plan.

    Parameters
    ----------
    plan : RunLaunchPlan
        Launch-plan payload.

    Returns
    -------
    str
        Shell script content.
    '''

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"mkdir -p {shlex.quote(str(plan.log_dir))}",
        f"cd {shlex.quote(str(plan.cwd))}",
        f"{plan.shell_command} > {shlex.quote(str(plan.stdout_log))} "
        f"2> {shlex.quote(str(plan.stderr_log))}",
        "",
    ]
    return "\n".join(lines)


def write_launch_script(
    plan: RunLaunchPlan, *, overwrite: bool = False
) -> RunLaunchPlan:
    '''Write a launch script for a plan without executing it.

    Parameters
    ----------
    plan : RunLaunchPlan
        Launch-plan payload with ``script_path`` set.
    overwrite : bool
        If True, overwrite an existing script file.

    Returns
    -------
    RunLaunchPlan
        Plan updated with ``script_written`` set to True.
    '''

    if plan.script_path is None:
        raise ValueError("script_path is required to write a launch script.")
    if plan.script_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing launch script: {plan.script_path}"
        )

    plan.script_path.parent.mkdir(parents=True, exist_ok=True)
    plan.script_path.write_text(build_launch_script(plan), encoding="utf-8")
    plan.script_path.chmod(0o755)
    return plan.model_copy(update={"script_written": True})


__all__ = [
    "DEFAULT_LAUNCH_LOG_DIR",
    "DEFAULT_LAUNCH_SCRIPT_NAME",
    "build_launch_script",
    "build_run_launch_plan",
    "write_launch_script",
]
