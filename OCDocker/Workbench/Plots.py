#!/usr/bin/env python3

# Description
###############################################################################
'''
Plot-ready Workbench payloads for GUI and notebook integrations.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import Any

from OCDocker.Workbench.Decision import build_pareto_front
from OCDocker.Workbench.Leaderboard import build_metric_leaderboard
from OCDocker.Workbench.MetricsMatrix import build_metric_matrix
from OCDocker.Workbench.Models import MetricMatrixRow
from OCDocker.Workbench.Models import MetricSortMode
from OCDocker.Workbench.Models import ParetoEntry
from OCDocker.Workbench.Models import ParetoObjective
from OCDocker.Workbench.Models import WorkbenchPlot

# License
###############################################################################
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Constants
###############################################################################

PLOTLY_CONFIG = {"responsive": True, "displaylogo": False}
PLOTLY_TEMPLATE = "plotly_white"

# Functions
###############################################################################
## Private ##


def _clean_metric_name(value: str, field_name: str = "metric") -> str:
    '''Return a stripped non-empty metric name.

    Parameters
    ----------
    value : str
        Metric name.
    field_name : str
        Field name used in validation errors.

    Returns
    -------
    str
        Cleaned metric name.
    '''

    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty.")
    return cleaned


def _unique_metric_names(metric_names: tuple[str, ...]) -> tuple[str, ...]:
    '''Normalize metric names and require uniqueness.

    Parameters
    ----------
    metric_names : tuple[str, ...]
        Metric names.

    Returns
    -------
    tuple[str, ...]
        Unique cleaned metric names.
    '''

    cleaned = tuple(_clean_metric_name(name) for name in metric_names)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("metric names must be unique.")
    return cleaned


def _layout(title: str, **extra: Any) -> dict[str, Any]:
    '''Build a shared Plotly layout dictionary.

    Parameters
    ----------
    title : str
        Plot title.
    **extra : Any
        Additional layout fields.

    Returns
    -------
    dict[str, Any]
        Plotly layout payload.
    '''

    payload: dict[str, Any] = {"title": title, "template": PLOTLY_TEMPLATE}
    payload.update(extra)
    return payload


def _row_has_metrics(row: MetricMatrixRow, metric_names: tuple[str, ...]) -> bool:
    '''Return whether a matrix row has all requested numeric metrics.

    Parameters
    ----------
    row : MetricMatrixRow
        Metric matrix row.
    metric_names : tuple[str, ...]
        Required metric names.

    Returns
    -------
    bool
        True when all requested metrics are numeric in the row.
    '''

    return all(name in row.metric_values for name in metric_names)


def _row_hover(row: MetricMatrixRow) -> str:
    '''Build hover text for a metric matrix row.

    Parameters
    ----------
    row : MetricMatrixRow
        Metric matrix row.

    Returns
    -------
    str
        Hover text.
    '''

    return f"Run: {row.run_id}<br>Status: {row.status}<br>Manifest: {row.manifest_path}"


def _pareto_hover(entry: ParetoEntry) -> str:
    '''Build hover text for a Pareto entry.

    Parameters
    ----------
    entry : ParetoEntry
        Pareto entry.

    Returns
    -------
    str
        Hover text.
    '''

    return f"Run: {entry.run_id}<br>Status: {entry.status}<br>Manifest: {entry.manifest_path}"


def _parallel_dimension(name: str, rows: tuple[MetricMatrixRow, ...]) -> dict[str, Any]:
    '''Build one Plotly parallel-coordinate dimension.

    Parameters
    ----------
    name : str
        Metric name.
    rows : tuple[MetricMatrixRow, ...]
        Included metric rows.

    Returns
    -------
    dict[str, Any]
        Plotly parallel-coordinate dimension.
    '''

    values = [row.metric_values[name] for row in rows]
    dimension: dict[str, Any] = {"label": name, "values": values}
    if values:
        dimension["range"] = [min(values), max(values)]
    return dimension


## Public ##


def build_leaderboard_plot(
    root: str | Path,
    *,
    metric_name: str,
    mode: MetricSortMode = "max",
    max_depth: int = 6,
    top_n: int = 20,
) -> WorkbenchPlot:
    '''Build a Plotly-compatible bar plot from a metric leaderboard.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or result manifest file to scan.
    metric_name : str
        Metric name or dotted metric path to rank.
    mode : MetricSortMode
        Ranking mode, either ``max`` or ``min``.
    max_depth : int
        Maximum directory depth below root to scan.
    top_n : int
        Maximum ranked entries to include in the plotted trace.

    Returns
    -------
    WorkbenchPlot
        Plot-ready leaderboard bar payload.
    '''

    if top_n < 1:
        raise ValueError("top_n must be greater than or equal to one.")
    metric = _clean_metric_name(metric_name)
    leaderboard = build_metric_leaderboard(
        root,
        metric_name=metric,
        mode=mode,
        max_depth=max_depth,
    )
    entries = leaderboard.ranked_entries[:top_n]
    trace = {
        "type": "bar",
        "name": metric,
        "x": [entry.run_id for entry in entries],
        "y": [entry.metric_value for entry in entries],
        "text": [f"rank {entry.rank}" for entry in entries],
        "hovertext": [
            f"Run: {entry.run_id}<br>Rank: {entry.rank}<br>"
            f"{metric}: {entry.metric_value}<br>Manifest: {entry.manifest_path}"
            for entry in entries
        ],
        "hoverinfo": "text",
        "marker": {"color": "#4477AA"},
    }
    return WorkbenchPlot(
        root=leaderboard.root,
        plot_kind="leaderboard_bar",
        title=f"{metric} leaderboard ({mode})",
        metric_names=(metric,),
        data=(trace,),
        layout=_layout(
            f"{metric} leaderboard ({mode})",
            xaxis={"title": "Run ID"},
            yaxis={"title": metric},
            margin={"l": 60, "r": 30, "t": 60, "b": 110},
        ),
        config=PLOTLY_CONFIG,
        included_count=len(leaderboard.ranked_entries),
        skipped_count=len(leaderboard.skipped_entries),
        issue_count=leaderboard.issue_count,
        issues=leaderboard.issues,
        metadata={"mode": mode, "source": "metric_leaderboard", "top_n": top_n},
    )


def build_metric_scatter_plot(
    root: str | Path,
    *,
    x_metric: str,
    y_metric: str,
    color_metric: str | None = None,
    max_depth: int = 6,
) -> WorkbenchPlot:
    '''Build a Plotly-compatible metric scatter plot from result manifests.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or result manifest file to scan.
    x_metric : str
        Metric plotted on the x axis.
    y_metric : str
        Metric plotted on the y axis.
    color_metric : str or None
        Optional numeric metric used for marker color.
    max_depth : int
        Maximum directory depth below root to scan.

    Returns
    -------
    WorkbenchPlot
        Plot-ready scatter payload.
    '''

    metrics = [
        _clean_metric_name(x_metric, "x_metric"),
        _clean_metric_name(y_metric, "y_metric"),
    ]
    if color_metric is not None:
        metrics.append(_clean_metric_name(color_metric, "color_metric"))
    metric_names = _unique_metric_names(tuple(metrics))
    matrix = build_metric_matrix(root, metric_names=metric_names, max_depth=max_depth)
    included = tuple(row for row in matrix.rows if _row_has_metrics(row, metric_names))
    marker: dict[str, Any] = {"size": 10, "opacity": 0.82, "color": "#4477AA"}
    if color_metric is not None:
        marker = {
            "size": 10,
            "opacity": 0.82,
            "color": [row.metric_values[color_metric] for row in included],
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": color_metric},
        }
    trace = {
        "type": "scatter",
        "mode": "markers",
        "name": f"{metric_names[1]} vs {metric_names[0]}",
        "x": [row.metric_values[metric_names[0]] for row in included],
        "y": [row.metric_values[metric_names[1]] for row in included],
        "text": [row.run_id for row in included],
        "hovertext": [_row_hover(row) for row in included],
        "hoverinfo": "text+x+y",
        "marker": marker,
        "customdata": [
            {
                "run_id": row.run_id,
                "status": row.status,
                "manifest_path": str(row.manifest_path),
            }
            for row in included
        ],
    }
    return WorkbenchPlot(
        root=matrix.root,
        plot_kind="metric_scatter",
        title=f"{metric_names[1]} vs {metric_names[0]}",
        metric_names=metric_names,
        data=(trace,),
        layout=_layout(
            f"{metric_names[1]} vs {metric_names[0]}",
            xaxis={"title": metric_names[0]},
            yaxis={"title": metric_names[1]},
            margin={"l": 70, "r": 30, "t": 60, "b": 70},
        ),
        config=PLOTLY_CONFIG,
        included_count=len(included),
        skipped_count=len(matrix.rows) - len(included),
        issue_count=matrix.issue_count,
        issues=matrix.issues,
        metadata={"source": "metric_matrix"},
    )


def build_parallel_coordinates_plot(
    root: str | Path,
    *,
    metric_names: tuple[str, ...] | list[str],
    max_depth: int = 6,
) -> WorkbenchPlot:
    '''Build a Plotly-compatible parallel-coordinate metric plot.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or result manifest file to scan.
    metric_names : tuple[str, ...] or list[str]
        Metrics used as parallel-coordinate dimensions.
    max_depth : int
        Maximum directory depth below root to scan.

    Returns
    -------
    WorkbenchPlot
        Plot-ready parallel-coordinate payload.
    '''

    metrics = _unique_metric_names(tuple(metric_names))
    if len(metrics) < 2:
        raise ValueError("parallel-coordinate plots require at least two metrics.")
    matrix = build_metric_matrix(root, metric_names=metrics, max_depth=max_depth)
    included = tuple(row for row in matrix.rows if _row_has_metrics(row, metrics))
    trace: dict[str, Any] = {
        "type": "parcoords",
        "dimensions": [_parallel_dimension(name, included) for name in metrics],
        "labelangle": 30,
    }
    if included:
        trace["line"] = {
            "color": [row.metric_values[metrics[0]] for row in included],
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": metrics[0]},
        }
    return WorkbenchPlot(
        root=matrix.root,
        plot_kind="parallel_coordinates",
        title="Parallel metric comparison",
        metric_names=metrics,
        data=(trace,),
        layout=_layout(
            "Parallel metric comparison",
            margin={"l": 80, "r": 80, "t": 80, "b": 60},
        ),
        config=PLOTLY_CONFIG,
        included_count=len(included),
        skipped_count=len(matrix.rows) - len(included),
        issue_count=matrix.issue_count,
        issues=matrix.issues,
        metadata={
            "source": "metric_matrix",
            "run_ids": [row.run_id for row in included],
        },
    )


def build_pareto_scatter_plot(
    root: str | Path,
    *,
    objectives: tuple[ParetoObjective, ...],
    max_depth: int = 6,
) -> WorkbenchPlot:
    '''Build a Plotly-compatible two-objective Pareto scatter plot.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or result manifest file to scan.
    objectives : tuple[ParetoObjective, ...]
        Exactly two Pareto objectives.
    max_depth : int
        Maximum directory depth below root to scan.

    Returns
    -------
    WorkbenchPlot
        Plot-ready Pareto scatter payload.
    '''

    if len(objectives) != 2:
        raise ValueError("Pareto scatter plots require exactly two objectives.")
    front = build_pareto_front(root, objectives=objectives, max_depth=max_depth)
    x_objective, y_objective = front.objectives
    x_name = x_objective.metric_name
    y_name = y_objective.metric_name

    def trace(
        name: str, entries: tuple[ParetoEntry, ...], color: str
    ) -> dict[str, Any]:
        return {
            "type": "scatter",
            "mode": "markers",
            "name": name,
            "x": [entry.metric_values[x_name] for entry in entries],
            "y": [entry.metric_values[y_name] for entry in entries],
            "text": [entry.run_id for entry in entries],
            "hovertext": [_pareto_hover(entry) for entry in entries],
            "hoverinfo": "text+x+y",
            "marker": {"size": 11, "opacity": 0.86, "color": color},
            "customdata": [
                {
                    "run_id": entry.run_id,
                    "status": entry.status,
                    "manifest_path": str(entry.manifest_path),
                    "dominated_by": entry.dominated_by,
                }
                for entry in entries
            ],
        }

    data = (
        trace("Pareto front", front.front_entries, "#228833"),
        trace("Dominated", front.dominated_entries, "#CC6677"),
    )
    return WorkbenchPlot(
        root=front.root,
        plot_kind="pareto_scatter",
        title=f"Pareto front: {x_name} vs {y_name}",
        metric_names=(x_name, y_name),
        data=data,
        layout=_layout(
            f"Pareto front: {x_name} vs {y_name}",
            xaxis={"title": f"{x_name} ({x_objective.mode})"},
            yaxis={"title": f"{y_name} ({y_objective.mode})"},
            margin={"l": 70, "r": 30, "t": 60, "b": 70},
        ),
        config=PLOTLY_CONFIG,
        included_count=len(front.front_entries) + len(front.dominated_entries),
        skipped_count=len(front.skipped_entries),
        issue_count=front.issue_count,
        issues=front.issues,
        metadata={
            "source": "pareto_front",
            "objectives": [
                objective.model_dump(mode="json") for objective in front.objectives
            ],
        },
    )


__all__ = [
    "PLOTLY_CONFIG",
    "PLOTLY_TEMPLATE",
    "build_leaderboard_plot",
    "build_metric_scatter_plot",
    "build_parallel_coordinates_plot",
    "build_pareto_scatter_plot",
]
