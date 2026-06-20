#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only preflight checks for Workbench specs before execution.
'''

# Imports
###############################################################################
from __future__ import annotations

import os
import shutil

from pathlib import Path

from OCDocker.Workbench.IO import read_spec
from OCDocker.Workbench.Models import OCScoreAblationSpec
from OCDocker.Workbench.Models import OCScoreStudySpec
from OCDocker.Workbench.Models import PreflightCheck
from OCDocker.Workbench.Models import PreflightReport
from OCDocker.Workbench.Models import PreflightSeverity
from OCDocker.Workbench.Models import VSCampaignSpec
from OCDocker.Workbench.Models import WorkbenchSpec
from OCDocker.Workbench.Planner import plan_command

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Private ##


def _resolve_path(base_path: Path, path: str | Path) -> Path:
    '''Resolve a spec-relative path without touching the filesystem.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory used for relative paths.
    path : str or pathlib.Path
        Path to resolve.

    Returns
    -------
    pathlib.Path
        Absolute path or path relative to the spec directory.
    '''

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base_path / candidate


def _check(
    *,
    code: str,
    severity: PreflightSeverity,
    passed: bool,
    message: str,
    path: Path | None = None,
    subject: str = "",
) -> PreflightCheck:
    '''Create one preflight check payload.

    Parameters
    ----------
    code : str
        Stable check code.
    severity : PreflightSeverity
        Check severity.
    passed : bool
        Whether the check passed.
    message : str
        Human-readable message.
    path : pathlib.Path or None
        Optional related path.
    subject : str
        Optional related subject.

    Returns
    -------
    PreflightCheck
        Check payload.
    '''

    return PreflightCheck(
        code=code,
        severity=severity,
        passed=passed,
        message=message,
        path=path,
        subject=subject,
    )


def _check_executable(command: tuple[str, ...]) -> PreflightCheck:
    '''Check whether the planned command executable is discoverable.

    Parameters
    ----------
    command : tuple[str, ...]
        Planned command tuple.

    Returns
    -------
    PreflightCheck
        Executable availability check.
    '''

    if not command:
        return _check(
            code="command.empty",
            severity="error",
            passed=False,
            message="Planned command is empty.",
        )
    executable = command[0]
    discovered = shutil.which(executable)
    if discovered is not None:
        return _check(
            code="command.executable",
            severity="error",
            passed=True,
            message=f"Executable found: {executable}",
            path=Path(discovered),
            subject=executable,
        )
    executable_path = Path(executable)
    if executable_path.exists():
        executable_is_runnable = executable_path.is_file() and os.access(
            executable_path, os.X_OK
        )
        return _check(
            code="command.executable",
            severity="error",
            passed=executable_is_runnable,
            message=(
                f"Executable path is runnable: {executable}"
                if executable_is_runnable
                else f"Executable path is not runnable: {executable}"
            ),
            path=executable_path,
            subject=executable,
        )
    return _check(
        code="command.executable",
        severity="error",
        passed=False,
        message=f"Executable was not found on PATH: {executable}",
        subject=executable,
    )


def _path_check(
    base_path: Path,
    path: str | Path,
    *,
    code: str,
    subject: str,
    expected: str,
    severity: PreflightSeverity = "error",
) -> PreflightCheck:
    '''Check whether a referenced path exists and has the expected kind.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory used for relative paths.
    path : str or pathlib.Path
        Path to check.
    code : str
        Stable check code.
    subject : str
        Related path subject.
    expected : str
        Expected path kind: ``any``, ``file``, or ``directory``.
    severity : PreflightSeverity
        Severity used when the check fails.

    Returns
    -------
    PreflightCheck
        Path existence/kind check.
    '''

    resolved = _resolve_path(base_path, path)
    if not resolved.exists():
        return _check(
            code=code,
            severity=severity,
            passed=False,
            message=f"Required {expected} path does not exist: {resolved}",
            path=resolved,
            subject=subject,
        )
    if expected == "file" and not resolved.is_file():
        return _check(
            code=code,
            severity=severity,
            passed=False,
            message=f"Expected a file but found a different path kind: {resolved}",
            path=resolved,
            subject=subject,
        )
    if expected == "directory" and not resolved.is_dir():
        return _check(
            code=code,
            severity=severity,
            passed=False,
            message=f"Expected a directory but found a different path kind: {resolved}",
            path=resolved,
            subject=subject,
        )
    return _check(
        code=code,
        severity=severity,
        passed=True,
        message=f"Found {expected} path: {resolved}",
        path=resolved,
        subject=subject,
    )


def _write_path_checks(
    base_path: Path, writes: tuple[Path, ...]
) -> tuple[PreflightCheck, ...]:
    '''Build non-blocking checks for planned output paths.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory used for relative paths.
    writes : tuple[pathlib.Path, ...]
        Planned write paths.

    Returns
    -------
    tuple[PreflightCheck, ...]
        Output path checks.
    '''

    checks: list[PreflightCheck] = []
    for path in writes:
        resolved = _resolve_path(base_path, path)
        if resolved.exists():
            checks.append(
                _check(
                    code="output.exists",
                    severity="warning",
                    passed=False,
                    message=f"Planned output path already exists: {resolved}",
                    path=resolved,
                    subject="output",
                )
            )
            continue
        parent = resolved.parent
        checks.append(
            _check(
                code="output.parent",
                severity="warning",
                passed=parent.exists(),
                message=(
                    f"Output parent exists: {parent}"
                    if parent.exists()
                    else f"Output parent does not exist yet: {parent}"
                ),
                path=parent,
                subject="output_parent",
            )
        )
    return tuple(checks)


def _ocscore_input_checks(
    base_path: Path, spec: OCScoreStudySpec | OCScoreAblationSpec
) -> tuple[PreflightCheck, ...]:
    '''Build checks for OCScore input and feature-policy paths.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory used for relative paths.
    spec : OCScoreStudySpec or OCScoreAblationSpec
        OCScore Workbench spec.

    Returns
    -------
    tuple[PreflightCheck, ...]
        OCScore path checks.
    '''

    checks: list[PreflightCheck] = []
    inputs = spec.inputs
    if inputs.raw_input_dir is not None:
        checks.append(
            _path_check(
                base_path,
                inputs.raw_input_dir,
                code="input.raw_input_dir",
                subject="raw_input_dir",
                expected="directory",
            )
        )
    if inputs.merged_input is not None:
        checks.append(
            _path_check(
                base_path,
                inputs.merged_input,
                code="input.merged_input",
                subject="merged_input",
                expected="file",
            )
        )
    for subject, path in (
        ("pdbbind_input", inputs.pdbbind_input),
        ("dudez_input", inputs.dudez_input),
    ):
        if path is not None:
            checks.append(
                _path_check(
                    base_path,
                    path,
                    code=f"input.{subject}",
                    subject=subject,
                    expected="file",
                )
            )

    for policy_dir in spec.feature_policies.policy_dirs:
        checks.append(
            _path_check(
                base_path,
                policy_dir,
                code="feature_policy.dir",
                subject="feature_policy_dir",
                expected="directory",
            )
        )
    for policy_yml in spec.feature_policies.policy_ymls:
        checks.append(
            _path_check(
                base_path,
                policy_yml,
                code="feature_policy.yml",
                subject="feature_policy_yml",
                expected="file",
            )
        )
    return tuple(checks)


def _vs_input_checks(
    base_path: Path, spec: VSCampaignSpec
) -> tuple[PreflightCheck, ...]:
    '''Build checks for VS campaign workflow and input paths.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory used for relative paths.
    spec : VSCampaignSpec
        Virtual-screening campaign spec.

    Returns
    -------
    tuple[PreflightCheck, ...]
        VS campaign path checks.
    '''

    checks: list[PreflightCheck] = [
        _path_check(
            base_path,
            spec.workflow.snakefile,
            code="workflow.snakefile",
            subject="snakefile",
            expected="file",
        )
    ]
    if spec.workflow.workdir is not None:
        checks.append(
            _path_check(
                base_path,
                spec.workflow.workdir,
                code="workflow.workdir",
                subject="workdir",
                expected="directory",
                severity="warning",
            )
        )
    for index, input_spec in enumerate(spec.inputs, start=1):
        prefix = f"input.{index}"
        checks.extend(
            (
                _path_check(
                    base_path,
                    input_spec.receptor,
                    code=f"{prefix}.receptor",
                    subject=f"{input_spec.sample}:receptor",
                    expected="file",
                ),
                _path_check(
                    base_path,
                    input_spec.ligand,
                    code=f"{prefix}.ligand",
                    subject=f"{input_spec.sample}:ligand",
                    expected="file",
                ),
                _path_check(
                    base_path,
                    input_spec.box,
                    code=f"{prefix}.box",
                    subject=f"{input_spec.sample}:box",
                    expected="file",
                ),
            )
        )
    return tuple(checks)


def _spec_path_checks(
    base_path: Path, spec: WorkbenchSpec
) -> tuple[PreflightCheck, ...]:
    '''Dispatch read-only path checks for a Workbench spec.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory used for relative paths.
    spec : WorkbenchSpec
        Validated Workbench spec.

    Returns
    -------
    tuple[PreflightCheck, ...]
        Spec-specific path checks.
    '''

    if isinstance(spec, VSCampaignSpec):
        return _vs_input_checks(base_path, spec)
    if isinstance(spec, (OCScoreStudySpec, OCScoreAblationSpec)):
        return _ocscore_input_checks(base_path, spec)
    return ()


## Public ##


def preflight_spec(
    spec: WorkbenchSpec,
    *,
    spec_path: str | Path | None = None,
    ocdocker_executable: str = "ocdocker",
    snakemake_executable: str = "snakemake",
) -> PreflightReport:
    '''Build a read-only preflight report for a validated Workbench spec.

    Parameters
    ----------
    spec : WorkbenchSpec
        Validated Workbench spec.
    spec_path : str or pathlib.Path or None
        Optional source spec path used to resolve relative paths.
    ocdocker_executable : str
        OCDocker executable used in planned OCScore commands.
    snakemake_executable : str
        Snakemake executable used in planned VS campaign commands.

    Returns
    -------
    PreflightReport
        Read-only preflight report.
    '''

    base_path = Path(spec_path).parent if spec_path is not None else Path.cwd()
    source_spec_path = Path(spec_path) if spec_path is not None else None
    plan = plan_command(
        spec,
        ocdocker_executable=ocdocker_executable,
        snakemake_executable=snakemake_executable,
    )
    checks = (
        _check_executable(plan.command),
        *_spec_path_checks(base_path, spec),
        *_write_path_checks(base_path, plan.writes),
    )
    error_count = sum(
        1 for check in checks if check.severity == "error" and not check.passed
    )
    warning_count = sum(
        1 for check in checks if check.severity == "warning" and not check.passed
    )
    info_count = sum(1 for check in checks if check.severity == "info")
    return PreflightReport(
        spec_path=source_spec_path,
        spec_type=spec.type,
        name=spec.name,
        ready=error_count == 0,
        planned_command=plan.command,
        checks=checks,
        error_count=error_count,
        warning_count=warning_count,
        info_count=info_count,
    )


def preflight_spec_file(
    path: str | Path,
    *,
    ocdocker_executable: str = "ocdocker",
    snakemake_executable: str = "snakemake",
) -> PreflightReport:
    '''Build a read-only preflight report from a Workbench spec file.

    Parameters
    ----------
    path : str or pathlib.Path
        Workbench spec path.
    ocdocker_executable : str
        OCDocker executable used in planned OCScore commands.
    snakemake_executable : str
        Snakemake executable used in planned VS campaign commands.

    Returns
    -------
    PreflightReport
        Read-only preflight report.
    '''

    spec_path = Path(path)
    spec = read_spec(spec_path)
    return preflight_spec(
        spec,
        spec_path=spec_path,
        ocdocker_executable=ocdocker_executable,
        snakemake_executable=snakemake_executable,
    )


__all__ = ["preflight_spec", "preflight_spec_file"]
