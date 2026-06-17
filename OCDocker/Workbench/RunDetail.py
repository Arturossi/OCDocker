#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only aggregate run drill-down helpers for Workbench GUI integrations.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path

from OCDocker.Workbench.Logs import DEFAULT_LOG_BYTE_LIMIT
from OCDocker.Workbench.Logs import DEFAULT_LOG_LINE_LIMIT
from OCDocker.Workbench.Logs import preview_run_logs
from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import ResultSummary
from OCDocker.Workbench.Models import RunDetail
from OCDocker.Workbench.Models import RunLogPreview
from OCDocker.Workbench.Models import RunStatusReport
from OCDocker.Workbench.Results import summarize_results
from OCDocker.Workbench.Status import inspect_run_status

# License
###############################################################################
"""
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
"""

# Functions
###############################################################################
## Private ##


def _issue(path: Path, message: str) -> InventoryIssue:
    '''Build a non-fatal run-detail issue.

    Parameters
    ----------
    path : pathlib.Path
        Path related to the issue.
    message : str
        Issue message.

    Returns
    -------
    InventoryIssue
        Non-fatal issue payload.
    '''

    return InventoryIssue(path=path, message=message)


def _result_summary_target(status_report: RunStatusReport) -> Path:
    '''Return the preferred manifest path for result summarization.

    Parameters
    ----------
    status_report : RunStatusReport
        Read-only run status report.

    Returns
    -------
    pathlib.Path
        Result manifest path when present, otherwise the run manifest path.
    '''

    if status_report.result_manifest_path is not None:
        return status_report.result_manifest_path
    return status_report.manifest_path


def _safe_log_preview(
    target: Path,
    *,
    lines: int,
    max_bytes: int,
    encoding: str,
    issues: list[InventoryIssue],
) -> RunLogPreview | None:
    '''Build a log preview while preserving partial run detail payloads.

    Parameters
    ----------
    target : pathlib.Path
        Run manifest path or run directory.
    lines : int
        Maximum returned lines per log file.
    max_bytes : int
        Maximum bytes read per log file.
    encoding : str
        Text encoding used to decode logs.
    issues : list[InventoryIssue]
        Mutable issue list updated on non-fatal failures.

    Returns
    -------
    RunLogPreview or None
        Log preview when it can be built.
    '''

    try:
        return preview_run_logs(
            target,
            lines=lines,
            max_bytes=max_bytes,
            encoding=encoding,
        )
    except Exception as exc:
        issues.append(_issue(target, f"Could not preview logs: {exc}"))
        return None


def _safe_result_summary(
    status_report: RunStatusReport,
    *,
    issues: list[InventoryIssue],
) -> ResultSummary | None:
    '''Build a result summary while preserving partial run detail payloads.

    Parameters
    ----------
    status_report : RunStatusReport
        Read-only run status report.
    issues : list[InventoryIssue]
        Mutable issue list updated on non-fatal failures.

    Returns
    -------
    ResultSummary or None
        Result summary when it can be built.
    '''

    target = _result_summary_target(status_report)
    try:
        return summarize_results(target)
    except Exception as exc:
        issues.append(_issue(target, f"Could not summarize results: {exc}"))

    if target == status_report.manifest_path:
        return None
    try:
        return summarize_results(status_report.manifest_path)
    except Exception as exc:
        issues.append(
            _issue(
                status_report.manifest_path,
                f"Could not summarize run manifest artifacts: {exc}",
            )
        )
        return None


## Public ##


def build_run_detail(
    target: str | Path,
    *,
    lines: int = DEFAULT_LOG_LINE_LIMIT,
    max_bytes: int = DEFAULT_LOG_BYTE_LIMIT,
    encoding: str = "utf-8",
) -> RunDetail:
    '''Build a read-only aggregate detail payload for one Workbench run.

    Parameters
    ----------
    target : str or pathlib.Path
        Run manifest file or prepared bundle directory.
    lines : int
        Maximum returned lines per declared log file.
    max_bytes : int
        Maximum bytes read from the end of each log file.
    encoding : str
        Text encoding used to decode log bytes.

    Returns
    -------
    RunDetail
        Aggregate run detail payload for GUI drill-downs.
    '''

    target_path = Path(target)
    issues: list[InventoryIssue] = []
    status_report = inspect_run_status(target_path)
    log_preview = _safe_log_preview(
        target_path,
        lines=lines,
        max_bytes=max_bytes,
        encoding=encoding,
        issues=issues,
    )
    result_summary = _safe_result_summary(status_report, issues=issues)

    return RunDetail(
        target=target_path,
        manifest_path=status_report.manifest_path,
        run_id=status_report.run_id,
        spec_type=status_report.spec_type,
        name=status_report.name,
        status=status_report.status,
        status_report=status_report,
        log_preview=log_preview,
        result_summary=result_summary,
        issue_count=len(issues),
        issues=tuple(issues),
    )


__all__ = ["build_run_detail"]
