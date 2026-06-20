#!/usr/bin/env python3

# Description
###############################################################################
'''
Composed Workbench analysis reports for GUI and publication workflows.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import Any

from OCDocker.Workbench.Decision import build_metrics_catalog
from OCDocker.Workbench.Decision import build_pareto_front
from OCDocker.Workbench.Decision import parse_pareto_objective
from OCDocker.Workbench.Leaderboard import build_metric_leaderboard
from OCDocker.Workbench.MetricsMatrix import build_metric_matrix
from OCDocker.Workbench.Models import MetricCatalog
from OCDocker.Workbench.Models import MetricCatalogEntry
from OCDocker.Workbench.Models import MetricLeaderboard
from OCDocker.Workbench.Models import MetricSortMode
from OCDocker.Workbench.Models import ParetoFront
from OCDocker.Workbench.Models import ParetoObjective
from OCDocker.Workbench.Models import PreflightSeverity
from OCDocker.Workbench.Models import WorkbenchAnalysisReport
from OCDocker.Workbench.Models import WorkbenchReportFinding
from OCDocker.Workbench.Models import WorkspaceOverview
from OCDocker.Workbench.Overview import build_workspace_overview

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################

DEFAULT_MINIMIZE_HINTS = frozenset(
    {
        "cost",
        "duration",
        "error",
        "latency",
        "loss",
        "mae",
        "mse",
        "rmse",
        "time",
    }
)
MARKDOWN_METRIC_LIMIT = 20

# Functions
###############################################################################
## Private ##


def _clean_metric_name(value: str) -> str:
    '''Normalize a metric name and reject empty values.

    Parameters
    ----------
    value : str
        Metric name.

    Returns
    -------
    str
        Cleaned metric name.
    '''

    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("metric name must not be empty.")
    return cleaned


def _infer_metric_mode(metric_name: str) -> MetricSortMode:
    '''Infer a conservative metric sort mode from a metric name.

    Parameters
    ----------
    metric_name : str
        Metric name.

    Returns
    -------
    MetricSortMode
        ``min`` for loss/error-like metrics and ``max`` otherwise.
    '''

    lowered = metric_name.lower()
    if any(hint in lowered for hint in DEFAULT_MINIMIZE_HINTS):
        return "min"
    return "max"


def _normalize_metric_specs(
    metric_specs: tuple[ParetoObjective, ...],
) -> tuple[ParetoObjective, ...]:
    '''Validate report metric selections.

    Parameters
    ----------
    metric_specs : tuple[ParetoObjective, ...]
        Requested metric selections.

    Returns
    -------
    tuple[ParetoObjective, ...]
        Validated metric selections.
    '''

    names = tuple(_clean_metric_name(spec.metric_name) for spec in metric_specs)
    if len(set(names)) != len(names):
        raise ValueError("report metric names must be unique.")
    return tuple(
        ParetoObjective(metric_name=name, mode=spec.mode)
        for name, spec in zip(names, metric_specs)
    )


def _default_metric_specs(
    catalog: MetricCatalog, *, top_n: int
) -> tuple[ParetoObjective, ...]:
    '''Build default leaderboard metric selections from metric coverage.

    Parameters
    ----------
    catalog : MetricCatalog
        Metric coverage catalog.
    top_n : int
        Maximum number of metric selections.

    Returns
    -------
    tuple[ParetoObjective, ...]
        Default metric selections.
    '''

    numeric_metrics = tuple(
        entry for entry in catalog.metrics if entry.numeric_count > 0
    )
    ranked = sorted(
        numeric_metrics,
        key=lambda entry: (
            -entry.numeric_count,
            entry.missing_count,
            entry.metric_name,
        ),
    )
    return tuple(
        ParetoObjective(
            metric_name=entry.metric_name,
            mode=_infer_metric_mode(entry.metric_name),
        )
        for entry in ranked[:top_n]
    )


def _issue_count(
    overview: WorkspaceOverview,
    catalog: MetricCatalog,
    leaderboards: tuple[MetricLeaderboard, ...],
    pareto_front: ParetoFront | None,
) -> int:
    '''Count non-fatal scan issues across report components.

    Parameters
    ----------
    overview : WorkspaceOverview
        Workspace overview.
    catalog : MetricCatalog
        Metric catalog.
    leaderboards : tuple[MetricLeaderboard, ...]
        Metric leaderboards.
    pareto_front : ParetoFront or None
        Optional Pareto front.

    Returns
    -------
    int
        Combined issue count.
    '''

    count = overview.issue_count + catalog.issue_count
    count += sum(leaderboard.issue_count for leaderboard in leaderboards)
    if pareto_front is not None:
        count += pareto_front.issue_count
    return count


def _finding(
    *,
    kind: str,
    severity: PreflightSeverity,
    title: str,
    message: str,
    run_id: str = "",
    metric_name: str = "",
    metric_value: float | None = None,
    manifest_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> WorkbenchReportFinding:
    '''Build one normalized report finding.

    Parameters
    ----------
    kind : str
        Finding kind.
    severity : PreflightSeverity
        Finding severity.
    title : str
        Short finding title.
    message : str
        Finding message.
    run_id : str
        Optional run id.
    metric_name : str
        Optional metric name.
    metric_value : float or None
        Optional metric value.
    manifest_path : pathlib.Path or None
        Optional source manifest path.
    metadata : dict[str, Any] or None
        Optional extra metadata.

    Returns
    -------
    WorkbenchReportFinding
        Report finding.
    '''

    return WorkbenchReportFinding(
        kind=kind,
        severity=severity,
        title=title,
        message=message,
        run_id=run_id,
        metric_name=metric_name,
        metric_value=metric_value,
        manifest_path=manifest_path,
        metadata={} if metadata is None else metadata,
    )


def _incomplete_metric_entries(
    catalog: MetricCatalog, *, top_n: int
) -> tuple[MetricCatalogEntry, ...]:
    '''Return incomplete metric coverage entries worth surfacing.

    Parameters
    ----------
    catalog : MetricCatalog
        Metric coverage catalog.
    top_n : int
        Maximum number of entries.

    Returns
    -------
    tuple[MetricCatalogEntry, ...]
        Incomplete metric entries.
    '''

    incomplete = tuple(
        entry
        for entry in catalog.metrics
        if entry.missing_count > 0 or entry.non_numeric_count > 0
    )
    return tuple(
        sorted(
            incomplete,
            key=lambda entry: (
                -(entry.missing_count + entry.non_numeric_count),
                entry.metric_name,
            ),
        )[:top_n]
    )


def _build_findings(
    *,
    overview: WorkspaceOverview,
    catalog: MetricCatalog,
    leaderboards: tuple[MetricLeaderboard, ...],
    pareto_front: ParetoFront | None,
    top_n: int,
) -> tuple[WorkbenchReportFinding, ...]:
    '''Build decision-support findings from report components.

    Parameters
    ----------
    overview : WorkspaceOverview
        Workspace overview.
    catalog : MetricCatalog
        Metric catalog.
    leaderboards : tuple[MetricLeaderboard, ...]
        Metric leaderboards.
    pareto_front : ParetoFront or None
        Optional Pareto front.
    top_n : int
        Maximum entries for repeated finding families.

    Returns
    -------
    tuple[WorkbenchReportFinding, ...]
        Report findings.
    '''

    findings: list[WorkbenchReportFinding] = []
    if overview.result_manifest_count == 0:
        findings.append(
            _finding(
                kind="no_results",
                severity="warning",
                title="No result manifests found",
                message="The report root does not contain result manifests within the scan depth.",
            )
        )
    for issue in overview.issues[:top_n]:
        findings.append(
            _finding(
                kind="workspace_issue",
                severity="warning",
                title="Workspace scan issue",
                message=issue.message,
                manifest_path=issue.path,
            )
        )
    if overview.missing_artifact_count:
        findings.append(
            _finding(
                kind="missing_artifact",
                severity="warning",
                title="Missing declared artifacts",
                message=(
                    f"{overview.missing_artifact_count} declared artifact(s) were "
                    "missing from run manifests."
                ),
                metadata={"missing_artifact_count": overview.missing_artifact_count},
            )
        )
    for entry in _incomplete_metric_entries(catalog, top_n=top_n):
        findings.append(
            _finding(
                kind="incomplete_metric",
                severity="warning",
                title=f"Incomplete metric coverage: {entry.metric_name}",
                message=(
                    f"Metric {entry.metric_name!r} was missing in {entry.missing_count} "
                    f"manifest(s) and non-numeric in {entry.non_numeric_count} manifest(s)."
                ),
                metric_name=entry.metric_name,
                metadata={
                    "missing_count": entry.missing_count,
                    "non_numeric_count": entry.non_numeric_count,
                    "numeric_count": entry.numeric_count,
                },
            )
        )
    for leaderboard in leaderboards:
        if leaderboard.best_entry is None:
            findings.append(
                _finding(
                    kind="incomplete_metric",
                    severity="warning",
                    title=f"No rankable entries: {leaderboard.metric_name}",
                    message=(
                        f"No result manifest had a numeric {leaderboard.metric_name!r} "
                        "value for ranking."
                    ),
                    metric_name=leaderboard.metric_name,
                )
            )
            continue
        best = leaderboard.best_entry
        findings.append(
            _finding(
                kind="best_metric",
                severity="info",
                title=f"Best {leaderboard.metric_name}",
                message=(
                    f"Run {best.run_id!r} ranks first for {leaderboard.metric_name!r} "
                    f"using {leaderboard.mode} mode."
                ),
                run_id=best.run_id,
                metric_name=leaderboard.metric_name,
                metric_value=best.metric_value,
                manifest_path=best.manifest_path,
                metadata={"mode": leaderboard.mode, "rank": best.rank},
            )
        )
    if pareto_front is not None:
        for entry in pareto_front.front_entries[:top_n]:
            findings.append(
                _finding(
                    kind="pareto_candidate",
                    severity="info",
                    title=f"Pareto candidate: {entry.run_id}",
                    message=(
                        f"Run {entry.run_id!r} is non-dominated for the requested "
                        "objectives."
                    ),
                    run_id=entry.run_id,
                    manifest_path=entry.manifest_path,
                    metadata={"metric_values": entry.metric_values},
                )
            )
        if pareto_front.skipped_entries:
            findings.append(
                _finding(
                    kind="pareto_skipped",
                    severity="warning",
                    title="Pareto skipped entries",
                    message=(
                        f"{len(pareto_front.skipped_entries)} result manifest(s) were "
                        "skipped because objective metrics were missing or non-numeric."
                    ),
                    metadata={"skipped_count": len(pareto_front.skipped_entries)},
                )
            )
    return tuple(findings)


def _markdown_cell(value: Any) -> str:
    '''Return a safe Markdown table cell.

    Parameters
    ----------
    value : Any
        Input value.

    Returns
    -------
    str
        Markdown-safe text.
    '''

    if value is None:
        text = ""
    elif isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _metric_value_text(value: float | None) -> str:
    '''Format a numeric metric for Markdown.

    Parameters
    ----------
    value : float or None
        Metric value.

    Returns
    -------
    str
        Formatted metric value.
    '''

    return "" if value is None else f"{value:.6g}"


def _status_count_lines(report: WorkbenchAnalysisReport) -> list[str]:
    '''Build Markdown table lines for status counts.

    Parameters
    ----------
    report : WorkbenchAnalysisReport
        Analysis report.

    Returns
    -------
    list[str]
        Markdown lines.
    '''

    lines = ["| Status | Count |", "| --- | ---: |"]
    for status, count in report.overview.status_counts.items():
        lines.append(f"| {_markdown_cell(status)} | {count} |")
    return lines


def _metric_catalog_lines(report: WorkbenchAnalysisReport) -> list[str]:
    '''Build Markdown table lines for metric coverage.

    Parameters
    ----------
    report : WorkbenchAnalysisReport
        Analysis report.

    Returns
    -------
    list[str]
        Markdown lines.
    '''

    if not report.metrics_catalog.metrics:
        return ["No metrics were discovered."]
    entries = report.metrics_catalog.metrics[:MARKDOWN_METRIC_LIMIT]
    lines = [
        "| Metric | Numeric | Missing | Non-numeric | Min | Max | Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in entries:
        lines.append(
            "| "
            f"{_markdown_cell(entry.metric_name)} | {entry.numeric_count} | "
            f"{entry.missing_count} | {entry.non_numeric_count} | "
            f"{_metric_value_text(entry.min_value)} | "
            f"{_metric_value_text(entry.max_value)} | "
            f"{_metric_value_text(entry.mean_value)} |"
        )
    if len(report.metrics_catalog.metrics) > MARKDOWN_METRIC_LIMIT:
        lines.append(
            f"\nShowing {MARKDOWN_METRIC_LIMIT} of "
            f"{len(report.metrics_catalog.metrics)} metrics."
        )
    return lines


def _leaderboard_lines(report: WorkbenchAnalysisReport) -> list[str]:
    '''Build Markdown sections for metric leaderboards.

    Parameters
    ----------
    report : WorkbenchAnalysisReport
        Analysis report.

    Returns
    -------
    list[str]
        Markdown lines.
    '''

    if not report.leaderboards:
        return ["No leaderboards were requested or inferred."]
    lines: list[str] = []
    for leaderboard in report.leaderboards:
        lines.extend(
            [
                f"### {leaderboard.metric_name} ({leaderboard.mode})",
                "",
            ]
        )
        if not leaderboard.ranked_entries:
            lines.extend(["No rankable entries were found.", ""])
            continue
        lines.extend(
            [
                "| Rank | Run ID | Value | Status | Manifest |",
                "| ---: | --- | ---: | --- | --- |",
            ]
        )
        for entry in leaderboard.ranked_entries[: report.top_n]:
            lines.append(
                "| "
                f"{entry.rank} | {_markdown_cell(entry.run_id)} | "
                f"{_metric_value_text(entry.metric_value)} | "
                f"{_markdown_cell(entry.status)} | `{_markdown_cell(entry.manifest_path)}` |"
            )
        lines.append("")
    return lines


def _pareto_lines(report: WorkbenchAnalysisReport) -> list[str]:
    '''Build Markdown lines for an optional Pareto front.

    Parameters
    ----------
    report : WorkbenchAnalysisReport
        Analysis report.

    Returns
    -------
    list[str]
        Markdown lines.
    '''

    if report.pareto_front is None:
        return ["No Pareto objectives were requested."]
    if not report.pareto_front.front_entries:
        return ["No non-dominated entries were found."]
    objective_names = tuple(
        objective.metric_name for objective in report.pareto_front.objectives
    )
    header = (
        "| Run ID | "
        + " | ".join(_markdown_cell(name) for name in objective_names)
        + " |"
    )
    separator = "| --- | " + " | ".join("---:" for _ in objective_names) + " |"
    lines = [header, separator]
    for entry in report.pareto_front.front_entries[: report.top_n]:
        values = " | ".join(
            _metric_value_text(entry.metric_values.get(name))
            for name in objective_names
        )
        lines.append(f"| {_markdown_cell(entry.run_id)} | {values} |")
    return lines


def _finding_lines(report: WorkbenchAnalysisReport) -> list[str]:
    '''Build Markdown lines for report findings.

    Parameters
    ----------
    report : WorkbenchAnalysisReport
        Analysis report.

    Returns
    -------
    list[str]
        Markdown lines.
    '''

    if not report.findings:
        return ["No findings were generated."]
    lines = [
        "| Severity | Kind | Title | Message |",
        "| --- | --- | --- | --- |",
    ]
    for finding in report.findings:
        lines.append(
            "| "
            f"{_markdown_cell(finding.severity)} | {_markdown_cell(finding.kind)} | "
            f"{_markdown_cell(finding.title)} | {_markdown_cell(finding.message)} |"
        )
    return lines


## Public ##


def parse_report_metric(value: str) -> ParetoObjective:
    '''Parse a report metric selection.

    Parameters
    ----------
    value : str
        Metric selection in ``metric`` or ``metric:min|max`` form.

    Returns
    -------
    ParetoObjective
        Parsed metric selection.
    '''

    return parse_pareto_objective(value)


def render_analysis_report_markdown(report: WorkbenchAnalysisReport) -> str:
    '''Render a Workbench analysis report as Markdown.

    Parameters
    ----------
    report : WorkbenchAnalysisReport
        Analysis report.

    Returns
    -------
    str
        Markdown report text.
    '''

    lines = [
        "# OCDocker Workbench Analysis Report",
        "",
        f"- Root: `{report.root}`",
        f"- Scanned at: `{report.scanned_at.isoformat()}`",
        f"- Runs: `{report.overview.run_count}`",
        f"- Result manifests: `{report.overview.result_manifest_count}`",
        f"- Missing artifacts: `{report.overview.missing_artifact_count}`",
        f"- Scan issues: `{report.issue_count}`",
        "",
        "## Status Counts",
        "",
        *_status_count_lines(report),
        "",
        "## Metric Coverage",
        "",
        *_metric_catalog_lines(report),
        "",
        "## Leaderboards",
        "",
        *_leaderboard_lines(report),
        "## Pareto Front",
        "",
        *_pareto_lines(report),
        "",
        "## Findings",
        "",
        *_finding_lines(report),
        "",
    ]
    return "\n".join(lines)


def build_analysis_report(
    root: str | Path,
    *,
    leaderboards: tuple[ParetoObjective, ...] = (),
    pareto_objectives: tuple[ParetoObjective, ...] = (),
    max_depth: int = 6,
    recent_limit: int = 20,
    top_n: int = 5,
) -> WorkbenchAnalysisReport:
    '''Build a composed read-only analysis report from Workbench manifests.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or manifest file to inspect.
    leaderboards : tuple[ParetoObjective, ...]
        Metric selections used to build leaderboards. When empty, numeric metrics are inferred.
    pareto_objectives : tuple[ParetoObjective, ...]
        Optional objectives used to build a Pareto front.
    max_depth : int
        Maximum directory depth below root to scan.
    recent_limit : int
        Maximum recent runs included in the overview.
    top_n : int
        Maximum entries surfaced in repeated report sections.

    Returns
    -------
    WorkbenchAnalysisReport
        Composed analysis report.
    '''

    if top_n < 1:
        raise ValueError("top_n must be greater than or equal to one.")
    if recent_limit < 1:
        raise ValueError("recent_limit must be greater than or equal to one.")

    root_path = Path(root)
    overview = build_workspace_overview(
        root_path,
        max_depth=max_depth,
        recent_limit=recent_limit,
    )
    catalog = build_metrics_catalog(root_path, max_depth=max_depth)
    metric_specs = _normalize_metric_specs(tuple(leaderboards))
    if not metric_specs:
        metric_specs = _default_metric_specs(catalog, top_n=top_n)
    leaderboard_payloads = tuple(
        build_metric_leaderboard(
            root_path,
            metric_name=spec.metric_name,
            mode=spec.mode,
            max_depth=max_depth,
        )
        for spec in metric_specs
    )
    matrix_metric_names = tuple(spec.metric_name for spec in metric_specs)
    metric_matrix = build_metric_matrix(
        root_path,
        metric_names=matrix_metric_names or None,
        max_depth=max_depth,
    )
    normalized_pareto = _normalize_metric_specs(tuple(pareto_objectives))
    pareto_front = (
        build_pareto_front(
            root_path,
            objectives=normalized_pareto,
            max_depth=max_depth,
        )
        if normalized_pareto
        else None
    )
    issue_count = _issue_count(
        overview,
        catalog,
        leaderboard_payloads,
        pareto_front,
    )
    report = WorkbenchAnalysisReport(
        root=root_path,
        max_depth=max_depth,
        recent_limit=recent_limit,
        top_n=top_n,
        overview=overview,
        metrics_catalog=catalog,
        metric_matrix=metric_matrix,
        leaderboards=leaderboard_payloads,
        pareto_front=pareto_front,
        issue_count=issue_count,
    )
    findings = _build_findings(
        overview=overview,
        catalog=catalog,
        leaderboards=leaderboard_payloads,
        pareto_front=pareto_front,
        top_n=top_n,
    )
    report = report.model_copy(update={"findings": findings})
    return report.model_copy(
        update={"markdown": render_analysis_report_markdown(report)}
    )


__all__ = [
    "DEFAULT_MINIMIZE_HINTS",
    "MARKDOWN_METRIC_LIMIT",
    "build_analysis_report",
    "parse_report_metric",
    "render_analysis_report_markdown",
]
