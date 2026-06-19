#!/usr/bin/env python3

# Description
###############################################################################
'''
Plot cross-validation artifacts written by :func:`save_cross_validation_result`.

Usage::

    from OCDocker.OCScore.Analysis.Plotting import CrossValidationPlots as occvplot

    occvplot.save_cross_validation_figures("/path/to/best_model/cross_validation")
'''

# Imports
###############################################################################
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from OCDocker.OCScore.Analysis.Metrics.Ranking import (
    DEFAULT_SCREENING_RANKING_METRICS,
    SCREENING_CONFUSION_METRICS,
)
from OCDocker.OCScore.Optimization.ModelCrossValidation import OCSCORE_MODEL_SCORER_NAME
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import (
    DESCRIPTOR_AGGREGATE_NAME_PREFIX,
    DESCRIPTOR_AGGREGATE_SCORER_TYPE,
)

from .Core import apply_basic_style, new_fig

# License
###############################################################################
'''
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
'''

# Constants
###############################################################################
RESULTS_JSON_NAME = "cross_validation_results.json"
MEAN_STD_CSV_NAME = "cross_validation_scorer_mean_std.csv"
FOLD_COMPARISON_CSV_NAME = "cross_validation_fold_comparison.csv"
FOLD_COMPARISON_PLOT_PREFIX = "cv_fold_comparison_"
LEGACY_FOLD_LINES_PLOT_PREFIX = "cv_fold_lines_"
OCSCORE_WINS_CSV_NAME = "cross_validation_ocscore_wins.csv"
PER_TARGET_CSV_NAME = "cross_validation_per_target_metrics.csv"

OCSCORE_COLOR = "#c0392b"
SF_COLOR = "#7f8c8d"

PINNED_FOLD_COMPARISON_SCORERS = ("sf_max",)

DEFAULT_CV_PLOT_METRICS = DEFAULT_SCREENING_RANKING_METRICS
NON_VISUAL_COMPARISON_METRICS = frozenset(SCREENING_CONFUSION_METRICS)

# Functions
###############################################################################
## Private ##

def _validation_column(metric: str) -> str:
    return f"validation_{metric}"


def _safe_filename(metric: str) -> str:
    return metric.replace("%", "pct").replace("/", "_").replace(" ", "_")


def _is_descriptor_aggregate_scorer(scorer: str, scorer_type: Optional[str] = None) -> bool:
    if scorer_type == DESCRIPTOR_AGGREGATE_SCORER_TYPE:
        return True
    return str(scorer).startswith(DESCRIPTOR_AGGREGATE_NAME_PREFIX)


def _fold_comparison_plot_pool(fold_comparison: pd.DataFrame) -> pd.DataFrame:
    '''Drop descriptor-row aggregates (``desc_*``) from fold-comparison plots.'''

    if fold_comparison.empty:
        return fold_comparison
    if "scorer_type" in fold_comparison.columns:
        mask = (
            fold_comparison["scorer_type"].astype(str) != DESCRIPTOR_AGGREGATE_SCORER_TYPE
        )
        return fold_comparison.loc[mask]
    return fold_comparison.loc[
        ~fold_comparison["scorer"].astype(str).str.startswith(DESCRIPTOR_AGGREGATE_NAME_PREFIX)
    ]


def _pin_fold_comparison_scorers(
        scorers: Sequence[str],
        available_scorers: Iterable[str],
        *,
        pinned: Sequence[str] = PINNED_FOLD_COMPARISON_SCORERS,
    ) -> list[str]:
    available = {str(scorer) for scorer in available_scorers}
    ordered = list(scorers)
    for scorer in pinned:
        if scorer in available and scorer not in ordered:
            ordered.append(scorer)
    return ordered


def _select_scorers_for_plot(
        mean_std: pd.DataFrame,
        metric: str,
        *,
        top_n: Optional[int],
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
    ) -> list[str]:
    subset = mean_std[mean_std["metric"] == metric].copy()
    if subset.empty:
        return []
    subset = subset[np.isfinite(subset["mean"].astype(float))]
    subset = subset.sort_values("mean", ascending=False)
    scorers = [
        scorer
        for scorer in subset["scorer"].astype(str).tolist()
        if not _is_descriptor_aggregate_scorer(scorer)
    ]
    if reference_scorer in scorers:
        scorers.remove(reference_scorer)
    if top_n is None or top_n <= 0 or len(scorers) <= top_n:
        ordered = scorers
    else:
        ordered = scorers[: max(0, top_n - 1)]
    ordered = _pin_fold_comparison_scorers(ordered, scorers)
    if reference_scorer in mean_std["scorer"].astype(str).values:
        return [reference_scorer, *ordered]
    return ordered


def _save_figure(fig: plt.Figure, path: Path, *, dpi: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _resolve_plot_metrics(
        metrics: Optional[Sequence[str]],
        results: dict[str, Any],
        mean_std: pd.DataFrame,
    ) -> list[str]:
    '''Return metrics to plot, excluding raw confusion-matrix counts by default.'''

    if metrics is not None:
        return [str(item) for item in metrics]

    summary_metrics = (results.get("scorer_comparison_summary") or {}).get("comparison_metrics")
    if summary_metrics:
        candidates = [
            str(item)
            for item in summary_metrics
            if str(item) not in NON_VISUAL_COMPARISON_METRICS
        ]
    elif not mean_std.empty:
        candidates = [
            str(item)
            for item in sorted(mean_std["metric"].astype(str).unique())
            if str(item) not in NON_VISUAL_COMPARISON_METRICS
        ]
    else:
        candidates = list(DEFAULT_CV_PLOT_METRICS)

    if not mean_std.empty:
        available = set(mean_std["metric"].astype(str))
        candidates = [metric for metric in candidates if metric in available]

    if candidates:
        return candidates

    objective = str(results.get("objective_metric") or "BEDROC")
    return [objective]


def _remove_legacy_fold_line_plots(output_path: Path) -> None:
    '''Drop obsolete ``cv_fold_lines_*`` PNGs after the bar-chart rename.'''

    for path in output_path.glob(f"{LEGACY_FOLD_LINES_PLOT_PREFIX}*.png"):
        path.unlink(missing_ok=True)


def _prune_obsolete_cv_figures(output_path: Path, active_metrics: Sequence[str]) -> None:
    '''Remove PNG artifacts for metrics that are no longer plotted.'''

    active_safe = {_safe_filename(metric) for metric in active_metrics}
    patterns = (
        "cv_mean_std_*.png",
        "cv_heatmap_*.png",
        f"{FOLD_COMPARISON_PLOT_PREFIX}*.png",
        "per_target_validation_*_boxplot.png",
        "per_target_validation_*_heatmap.png",
        "per_target_validation_*_heatmap_part*.png",
        "per_target_test_*_boxplot.png",
        "per_target_test_*_heatmap.png",
        "per_target_test_*_heatmap_part*.png",
    )
    prefix_lengths = {
        "cv_mean_std_": len("cv_mean_std_"),
        "cv_heatmap_": len("cv_heatmap_"),
        FOLD_COMPARISON_PLOT_PREFIX: len(FOLD_COMPARISON_PLOT_PREFIX),
        "per_target_validation_": len("per_target_validation_"),
        "per_target_test_": len("per_target_test_"),
    }

    for pattern in patterns:
        for path in output_path.glob(pattern):
            name = path.name
            if "_heatmap_part" in name:
                stem = name.split("_heatmap_part", maxsplit=1)[0]
                for prefix, length in prefix_lengths.items():
                    if stem.startswith(prefix):
                        metric = stem[length:]
                        break
                else:
                    continue
            elif name.startswith("cv_mean_std_"):
                metric = name[len("cv_mean_std_") : -len(".png")]
            elif name.startswith("cv_heatmap_"):
                metric = name[len("cv_heatmap_") : -len(".png")]
            elif name.startswith(FOLD_COMPARISON_PLOT_PREFIX):
                metric = name[len(FOLD_COMPARISON_PLOT_PREFIX) : -len(".png")]
            elif name.startswith("per_target_validation_") and name.endswith("_boxplot.png"):
                metric = name[len("per_target_validation_") : -len("_boxplot.png")]
            elif name.startswith("per_target_validation_") and name.endswith("_heatmap.png"):
                metric = name[len("per_target_validation_") : -len("_heatmap.png")]
            elif name.startswith("per_target_test_") and name.endswith("_boxplot.png"):
                metric = name[len("per_target_test_") : -len("_boxplot.png")]
            elif name.startswith("per_target_test_") and name.endswith("_heatmap.png"):
                metric = name[len("per_target_test_") : -len("_heatmap.png")]
            else:
                continue
            if metric not in active_safe:
                path.unlink(missing_ok=True)


## Public ##

def resolve_cross_validation_dir(path: str | Path) -> Path:
    '''Resolve a cross-validation directory from an export or CV path.

    Parameters
    ----------
    path : str | Path
        Either ``<export>/cross_validation`` or ``<export>/best_model`` (or any
        directory containing ``cross_validation_results.json``).

    Returns
    -------
    Path
        Directory with cross-validation artifacts.
    '''

    candidate = Path(path)
    if (candidate / RESULTS_JSON_NAME).exists():
        return candidate.resolve()
    nested = candidate / "cross_validation"
    if (nested / RESULTS_JSON_NAME).exists():
        return nested.resolve()
    raise FileNotFoundError(
        f"Cross-validation results not found under {candidate!r} "
        f"(expected {RESULTS_JSON_NAME} or {nested / RESULTS_JSON_NAME})."
    )


def load_cross_validation_artifacts(cv_dir: str | Path) -> dict[str, Any]:
    '''Load JSON/CSV artifacts from a cross-validation output directory.

    Parameters
    ----------
    cv_dir : str | Path
        Cross-validation directory (see :func:`resolve_cross_validation_dir`).

    Returns
    -------
    dict[str, Any]
        Keys: ``cv_dir``, ``results``, ``mean_std``, ``fold_comparison``,
        ``ocscore_wins``, ``per_target`` (DataFrames may be empty if files are missing).
    '''

    root = resolve_cross_validation_dir(cv_dir)
    artifacts: dict[str, Any] = {"cv_dir": root}

    results_path = root / RESULTS_JSON_NAME
    artifacts["results"] = json.loads(results_path.read_text(encoding="utf-8"))

    mean_std_path = root / MEAN_STD_CSV_NAME
    artifacts["mean_std"] = (
        pd.read_csv(mean_std_path) if mean_std_path.exists() else pd.DataFrame()
    )

    fold_comparison_path = root / FOLD_COMPARISON_CSV_NAME
    artifacts["fold_comparison"] = (
        pd.read_csv(fold_comparison_path) if fold_comparison_path.exists() else pd.DataFrame()
    )

    wins_path = root / OCSCORE_WINS_CSV_NAME
    artifacts["ocscore_wins"] = pd.read_csv(wins_path) if wins_path.exists() else pd.DataFrame()

    per_target_path = root / PER_TARGET_CSV_NAME
    artifacts["per_target"] = (
        pd.read_csv(per_target_path) if per_target_path.exists() else pd.DataFrame()
    )

    return artifacts


def _select_scorers_from_per_target(
        per_target: pd.DataFrame,
        metric: str,
        *,
        top_n: Optional[int],
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
        split: Optional[str] = None,
    ) -> list[str]:
    subset = per_target.copy()
    if split is not None and "split" in subset.columns:
        subset = subset[subset["split"].astype(str) == split]
    if metric not in subset.columns:
        return []
    ranked = (
        subset.groupby("scorer", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
    )
    scorers = ranked["scorer"].astype(str).tolist()
    if reference_scorer in scorers:
        scorers.remove(reference_scorer)
    if top_n is not None and top_n > 0 and len(scorers) > max(0, top_n - 1):
        scorers = scorers[: max(0, top_n - 1)]
    if reference_scorer in ranked["scorer"].astype(str).values:
        return [reference_scorer, *scorers]
    return scorers


def _ordered_groups_for_per_target(
        per_target: pd.DataFrame,
        metric: str,
        *,
        split: Optional[str],
        reference_scorer: str,
        max_groups: Optional[int],
    ) -> list[str]:
    subset = per_target.copy()
    if split is not None and "split" in subset.columns:
        subset = subset[subset["split"].astype(str) == split]
    if metric not in subset.columns:
        return []
    subset = subset.dropna(subset=[metric])
    if subset.empty or "group" not in subset.columns:
        return []

    subset = subset.assign(
        _group=subset["group"].astype(str),
        _scorer=subset["scorer"].astype(str),
    )
    reference_rows = subset[subset["_scorer"] == reference_scorer]
    if reference_rows.empty:
        ordered = sorted(subset["_group"].unique().tolist())
    else:
        ordered = (
            reference_rows.groupby("_group", as_index=False)[metric]
            .mean(numeric_only=True)
            .sort_values(metric, ascending=False)["_group"]
            .astype(str)
            .tolist()
        )

    if max_groups is not None and max_groups > 0:
        return ordered[:max_groups]
    return ordered


def _remove_stale_heatmap_parts(output_path: Path, split_label: str, metric: str) -> None:
    '''Delete legacy receptor-chunk heatmap PNGs for one metric.'''

    pattern = f"per_target_{split_label}_{_safe_filename(metric)}_heatmap_part*.png"
    for path in output_path.glob(pattern):
        path.unlink(missing_ok=True)


def _heatmap_layout(n_rows: int, n_cols: int) -> dict[str, float]:
    '''Square-cell canvas sizing and typography for per-target heatmaps.

    Figure dimensions are derived from a single ``cell_in`` value so each grid
    cell renders square; fonts scale up on larger grids.
    '''

    span = max(n_rows, n_cols)
    if span >= 35:
        cell_in = 0.44
        row_font = 17.0
        col_font = 16.0
        axis_font = 17.0
        title_font = 20.0
        cbar_label = 16.0
        cbar_tick = 15.0
        margin_left = 3.0
        margin_bottom = 2.6
        margin_top = 1.1
        margin_right = 0.55
    elif span >= 22:
        cell_in = 0.40
        row_font = 15.0
        col_font = 14.0
        axis_font = 15.0
        title_font = 18.0
        cbar_label = 14.0
        cbar_tick = 13.0
        margin_left = 2.6
        margin_bottom = 2.2
        margin_top = 1.0
        margin_right = 0.50
    else:
        cell_in = 0.36
        row_font = 13.0
        col_font = 12.0
        axis_font = 14.0
        title_font = 16.0
        cbar_label = 12.0
        cbar_tick = 11.0
        margin_left = 2.0
        margin_bottom = 1.8
        margin_top = 0.9
        margin_right = 0.45

    plot_width = cell_in * max(1, n_cols)
    plot_height = cell_in * max(1, n_rows)
    width = margin_left + plot_width + margin_right
    height = margin_top + plot_height + margin_bottom

    return {
        "cell_in": cell_in,
        "row_font": row_font,
        "col_font": col_font,
        "axis_font": axis_font,
        "title_font": title_font,
        "cbar_label": cbar_label,
        "cbar_tick": cbar_tick,
        "width": width,
        "height": height,
        "left": margin_left / width,
        "bottom": margin_bottom / height,
        "right": 1.0 - (margin_right / width),
        "top": 1.0 - (margin_top / height),
    }


def _position_heatmap_colorbar(
        fig: plt.Figure,
        ax: plt.Axes,
        metric: str,
        *,
        tick_fontsize: float,
        label_fontsize: float,
    ) -> None:
    '''Place a compact colorbar beside the rendered heatmap axes.'''

    if len(fig.axes) < 2:
        return
    fig.canvas.draw()
    pos = ax.get_position()
    cbar_ax = fig.axes[-1]
    cbar_height = pos.height * 0.68
    cbar_y = pos.y0 + (pos.height - cbar_height) / 2.0
    cbar_width = 0.012
    cbar_x = pos.x1 + 0.012
    cbar_ax.set_position([cbar_x, cbar_y, cbar_width, cbar_height])
    cbar_ax.tick_params(labelsize=tick_fontsize, length=2.5, width=0.8)
    cbar_ax.set_ylabel(metric, fontsize=label_fontsize, rotation=270, labelpad=14)
    cbar_ax.yaxis.set_label_coords(3.8, 0.5)


def aggregate_cv_per_target_metrics(per_target: pd.DataFrame) -> pd.DataFrame:
    '''Average per-receptor metrics across CV folds for plotting.

    CV exports one row per ``(fold_index, group, scorer)``; this collapses folds
    into a single row per ``(group, scorer, scorer_type)`` with mean metrics.
    '''

    if per_target.empty:
        return per_target
    if "fold_index" not in per_target.columns:
        return per_target

    group_cols = ["group", "scorer", "scorer_type"]
    metric_cols = [
        column
        for column in per_target.columns
        if column not in {*group_cols, "fold_index", "split"}
        and pd.api.types.is_numeric_dtype(per_target[column])
    ]
    if not metric_cols:
        return pd.DataFrame(columns=[*group_cols, "split", *metric_cols])

    aggregated = (
        per_target.groupby(group_cols, as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )
    aggregated["split"] = "validation"
    return aggregated


def plot_per_target_heatmap(
        per_target: pd.DataFrame,
        metric: str,
        *,
        split: Optional[str] = "test",
        top_n: Optional[int] = 15,
        max_groups: Optional[int] = None,
        groups: Optional[Sequence[str]] = None,
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
        size: tuple[float, float] = (12, 8),
        annotate: Optional[bool] = None,
        annotation_cell_limit: int = 80,
        title_suffix: str = "",
        transpose: Optional[bool] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
    '''Heatmap of a metric with scorers on rows and receptors on columns.'''

    if metric not in per_target.columns:
        raise ValueError(f"Metric {metric!r} missing from per-target table.")

    subset = per_target.copy()
    if split is not None and "split" in subset.columns:
        subset = subset[subset["split"].astype(str) == split]
    subset = subset.dropna(subset=[metric])
    if subset.empty:
        raise ValueError(f"No per-target values for metric {metric!r}.")

    scorers = _select_scorers_from_per_target(
        subset,
        metric,
        top_n=top_n,
        reference_scorer=reference_scorer,
        split=split,
    )
    subset = subset.assign(
        _group=subset["group"].astype(str),
        _scorer=subset["scorer"].astype(str),
    )
    subset = subset[subset["_scorer"].isin(scorers)]

    if groups is not None:
        selected_groups = [str(group) for group in groups]
    else:
        selected_groups = _ordered_groups_for_per_target(
            subset,
            metric,
            split=None,
            reference_scorer=reference_scorer,
            max_groups=max_groups,
        )
    if selected_groups:
        subset = subset[subset["_group"].isin(selected_groups)]

    table = subset.pivot_table(index="_scorer", columns="_group", values=metric, aggfunc="mean")
    table = table.loc[[scorer for scorer in scorers if scorer in table.index]]
    if selected_groups:
        table = table.reindex(columns=[group for group in selected_groups if group in table.columns])

    n_scorers, n_groups = table.shape
    use_transpose = (n_groups <= n_scorers) if transpose is None else bool(transpose)
    if use_transpose:
        table = table.T
        n_rows, n_cols = n_groups, n_scorers
        y_label, x_label = "Receptor", "Scorer"
    else:
        n_rows, n_cols = n_scorers, n_groups
        y_label, x_label = "Scorer", "Receptor"

    apply_basic_style()
    n_cells = int(n_rows * n_cols)
    show_annotations = n_cells <= annotation_cell_limit if annotate is None else bool(annotate)
    layout = _heatmap_layout(n_rows, n_cols)
    row_fontsize = layout["row_font"]
    col_fontsize = layout["col_font"]
    axis_label_fontsize = layout["axis_font"]
    title_fontsize = layout["title_font"]
    cbar_label_fontsize = layout["cbar_label"]
    cbar_tick_fontsize = layout["cbar_tick"]
    width = max(size[0], layout["width"])
    height = max(size[1], layout["height"])
    fig, ax = new_fig((width, height))
    sns.heatmap(
        table.astype(float),
        annot=show_annotations,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={
            "label": metric,
            "shrink": 0.68,
            "aspect": 32,
            "pad": 0.015,
            "fraction": 0.035,
        },
        annot_kws={"fontsize": max(row_fontsize, col_fontsize) - 1.0},
    )
    title_split = f" ({split})" if split else ""
    title_extra = "" if show_annotations else " - color scale only"
    suffix = f" {title_suffix}" if title_suffix else ""
    ax.set_title(
        f"Per-receptor {metric}{title_split}{title_extra}{suffix}",
        fontsize=title_fontsize,
        pad=12,
    )
    ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
    xlabels = ax.get_xticklabels()
    ylabels = ax.get_yticklabels()
    plt.setp(xlabels, rotation=45, ha="right", fontsize=col_fontsize)
    plt.setp(ylabels, fontsize=row_fontsize)
    fig.subplots_adjust(
        bottom=layout["bottom"],
        left=layout["left"],
        right=layout["right"],
        top=layout["top"],
    )
    ax.set_aspect("equal", adjustable="box")
    _position_heatmap_colorbar(
        fig,
        ax,
        metric,
        tick_fontsize=cbar_tick_fontsize,
        label_fontsize=cbar_label_fontsize,
    )
    return fig, ax


def plot_per_target_boxplot(
        per_target: pd.DataFrame,
        metric: str,
        *,
        split: Optional[str] = "test",
        top_n: Optional[int] = 15,
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
        size: tuple[float, float] = (10, 5),
    ) -> tuple[plt.Figure, plt.Axes]:
    '''Boxplot of per-receptor metric values for selected scorers.'''

    if metric not in per_target.columns:
        raise ValueError(f"Metric {metric!r} missing from per-target table.")

    subset = per_target.copy()
    if split is not None and "split" in subset.columns:
        subset = subset[subset["split"].astype(str) == split]
    subset = subset.dropna(subset=[metric])
    scorers = _select_scorers_from_per_target(
        subset,
        metric,
        top_n=top_n,
        reference_scorer=reference_scorer,
        split=split,
    )
    plot_df = subset[subset["scorer"].astype(str).isin(scorers)].copy()
    plot_df["scorer"] = pd.Categorical(
        plot_df["scorer"].astype(str),
        categories=scorers,
        ordered=True,
    )

    apply_basic_style()
    height = max(size[1], 0.38 * max(1, len(scorers)) + 1.8)
    width = max(size[0], 8.0)
    palette = {s: OCSCORE_COLOR if s == reference_scorer else SF_COLOR for s in scorers}
    fig, ax = new_fig((width, height))
    box_data = [
        plot_df.loc[plot_df["scorer"] == scorer, metric].dropna().astype(float).values
        for scorer in scorers
    ]
    boxplot = ax.boxplot(
        box_data,
        orientation="horizontal",
        tick_labels=scorers,
        patch_artist=True,
        showfliers=True,
        widths=0.62,
        medianprops={"color": "#1a1a1a", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.0, "color": "#555555"},
        capprops={"linewidth": 1.0, "color": "#555555"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#666666",
            "markersize": 3.5,
            "alpha": 0.8,
        },
    )
    for patch, scorer in zip(boxplot["boxes"], scorers):
        patch.set_facecolor(palette[scorer])
        patch.set_edgecolor("#444444")
        patch.set_alpha(0.82)
        patch.set_linewidth(1.0)
    positions = np.arange(1, len(scorers) + 1)
    point_rng = np.random.default_rng(0)
    for position, scorer in zip(positions, scorers):
        values = plot_df.loc[plot_df["scorer"] == scorer, metric].dropna().astype(float).values
        if values.size == 0:
            continue
        jitter = point_rng.uniform(-0.14, 0.14, size=values.size)
        ax.scatter(
            values,
            np.full(values.size, position) + jitter,
            color="#222222",
            alpha=0.35,
            s=12,
            linewidths=0,
            zorder=3,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(scorers)
    title_split = f" ({split})" if split else ""
    ax.set_title(f"Per-receptor {metric} distribution{title_split}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Scorer")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(0.4, len(scorers) + 0.6)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.subplots_adjust(left=0.28 if len(scorers) > 12 else 0.22)
    return fig, ax


def plot_per_target_ocscore_wins(
        per_target: pd.DataFrame,
        metric: str = "BEDROC",
        *,
        split: Optional[str] = "test",
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
        size: tuple[float, float] = (8, 5),
    ) -> tuple[plt.Figure, plt.Axes]:
    '''Bar chart: receptors where OCScore beats each other scorer on ``metric``.'''

    if metric not in per_target.columns:
        raise ValueError(f"Metric {metric!r} missing from per-target table.")

    subset = per_target.copy()
    if split is not None and "split" in subset.columns:
        subset = subset[subset["split"].astype(str) == split]
    subset = subset.dropna(subset=[metric])
    if reference_scorer not in set(subset["scorer"].astype(str)):
        model_rows = subset[subset["scorer_type"].astype(str) == "model"]
        if model_rows.empty:
            raise ValueError(f"Reference scorer {reference_scorer!r} not found.")
        reference_scorer = str(model_rows["scorer"].iloc[0])

    pivot = subset.pivot_table(
        index="group",
        columns="scorer",
        values=metric,
        aggfunc="mean",
    )
    if reference_scorer not in pivot.columns:
        raise ValueError(f"Reference scorer {reference_scorer!r} not in pivot.")

    ref_values = pivot[reference_scorer]
    win_counts: dict[str, int] = {}
    for column in pivot.columns:
        if column == reference_scorer:
            continue
        wins = int((ref_values > pivot[column]).sum())
        win_counts[str(column)] = wins

    if not win_counts:
        raise ValueError("No comparators for OCScore win counts.")

    ranked = sorted(win_counts.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in ranked]
    values = [item[1] for item in ranked]

    apply_basic_style()
    fig, ax = new_fig(size)
    ax.barh(labels[::-1], values[::-1], color=SF_COLOR, alpha=0.85)
    ax.set_xlabel(f"Receptors where {reference_scorer} wins on {metric}")
    title_split = f" ({split})" if split else ""
    ax.set_title(f"OCScore per-receptor wins{title_split}")
    ax.set_xlim(left=0.0)
    return fig, ax


def save_per_target_figures(
        per_target_source: str | Path | pd.DataFrame,
        figures_dir: str | Path,
        *,
        split: Optional[str] = "test",
        metrics: Optional[Sequence[str]] = None,
        top_n: Optional[int] = 15,
        heatmap_top_n: Optional[int] = None,
        max_groups: Optional[int] = None,
        dpi: int = 150,
    ) -> dict[str, str]:
    '''Generate per-receptor heatmap, boxplot, and OCScore-win charts.

    Parameters
    ----------
    per_target_source : str | Path | pd.DataFrame
        Path to a per-target CSV or an in-memory table.
    '''

    if isinstance(per_target_source, pd.DataFrame):
        per_target = per_target_source.copy()
    else:
        csv_path = Path(per_target_source)
        per_target = pd.read_csv(csv_path)
        if per_target.empty:
            raise ValueError(f"Per-target table is empty: {csv_path}")

    metric_list = (
        [str(item) for item in metrics]
        if metrics
        else [column for column in ("BEDROC", "ROC-AUC", "PR-AUC", "EF1%") if column in per_target.columns]
    )
    if not metric_list:
        raise ValueError(f"No plottable metrics in {csv_path}")

    output_path = Path(figures_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    reference = OCSCORE_MODEL_SCORER_NAME
    if reference not in set(per_target["scorer"].astype(str)):
        model_rows = per_target[per_target.get("scorer_type", pd.Series(dtype=str)).astype(str) == "model"]
        if not model_rows.empty:
            reference = str(model_rows["scorer"].iloc[0])

    split_label = split or "all"
    heatmap_scorers = heatmap_top_n if heatmap_top_n is not None else top_n
    for metric in metric_list:
        if metric not in per_target.columns:
            continue
        _remove_stale_heatmap_parts(output_path, split_label, metric)
        try:
            fig, _ = plot_per_target_heatmap(
                per_target,
                metric,
                split=split,
                top_n=heatmap_scorers,
                max_groups=max_groups,
                reference_scorer=reference,
            )
            key = f"per_target_heatmap_{split_label}_{metric}"
            written[key] = _save_figure(
                fig,
                output_path / f"per_target_{split_label}_{_safe_filename(metric)}_heatmap.png",
                dpi=dpi,
            )
        except ValueError:
            pass

        try:
            fig, _ = plot_per_target_boxplot(
                per_target,
                metric,
                split=split,
                top_n=top_n,
                reference_scorer=reference,
            )
            key = f"per_target_box_{split_label}_{metric}"
            written[key] = _save_figure(
                fig,
                output_path / f"per_target_{split_label}_{_safe_filename(metric)}_boxplot.png",
                dpi=dpi,
            )
        except ValueError:
            pass

    try:
        fig, _ = plot_per_target_ocscore_wins(
            per_target,
            metric="BEDROC" if "BEDROC" in per_target.columns else metric_list[0],
            split=split,
            reference_scorer=reference,
        )
        key = f"per_target_ocscore_wins_{split_label}"
        written[key] = _save_figure(
            fig,
            output_path / f"per_target_{split_label}_ocscore_wins.png",
            dpi=dpi,
        )
    except ValueError:
        pass

    return written


def plot_mean_std_bars(
        mean_std: pd.DataFrame,
        metric: str,
        *,
        top_n: Optional[int] = 25,
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
        width: float = 8.0,
    ) -> tuple[plt.Figure, plt.Axes]:
    '''Bar chart of mean ± std per scorer for one metric.

    Parameters
    ----------
    mean_std : pd.DataFrame
        ``cross_validation_scorer_mean_std.csv`` contents.
    metric : str
        Metric name (e.g. ``BEDROC``).
    top_n : int | None, optional
        Maximum scorers to show (OCScore is always included). Default: 25.
    reference_scorer : str, optional
        Highlighted scorer name. Default: ``OCScore``.
    width : float, optional
        Figure width in inches; height scales with the number of scorers.

    Returns
    -------
    (Figure, Axes)
    '''

    apply_basic_style()
    scorers = _select_scorers_for_plot(
        mean_std,
        metric,
        top_n=top_n,
        reference_scorer=reference_scorer,
    )
    subset = mean_std[
        (mean_std["metric"] == metric) & (mean_std["scorer"].astype(str).isin(scorers))
    ].copy()
    if subset.empty:
        raise ValueError(f"No mean/std rows for metric {metric!r}.")

    subset["scorer"] = pd.Categorical(
        subset["scorer"].astype(str),
        categories=list(reversed(scorers)),
        ordered=True,
    )
    subset = subset.sort_values("scorer", ascending=True)
    height = max(4.0, 0.32 * len(subset))
    fig, ax = new_fig((width, height))

    colors = [
        OCSCORE_COLOR if str(row.scorer) == reference_scorer else SF_COLOR
        for row in subset.itertuples(index=False)
    ]
    y_pos = np.arange(len(subset))
    ax.barh(
        y_pos,
        subset["mean"].astype(float),
        xerr=subset["std"].astype(float),
        color=colors,
        capsize=3,
        alpha=0.9,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subset["scorer"].astype(str))
    ax.set_xlabel(metric)
    ax.set_title(f"Cross-validation: {metric} (mean ± std)")
    ax.invert_yaxis()
    means = subset["mean"].astype(float)
    stds = subset["std"].astype(float)
    if means.notna().any():
        xmax = float((means + stds).max())
        if np.isfinite(xmax):
            upper = xmax * 1.08 if xmax > 0 else 1.0
            ax.set_xlim(0.0, upper)
        else:
            ax.set_xlim(left=0.0)
    else:
        ax.set_xlim(left=0.0)
    return fig, ax


def plot_fold_metric_heatmap(
        fold_comparison: pd.DataFrame,
        metric: str,
        *,
        top_n: Optional[int] = 25,
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
        size: tuple[float, float] = (10, 8),
    ) -> tuple[plt.Figure, plt.Axes]:
    '''Heatmap of validation metric values (scorers × folds).

    Parameters
    ----------
    fold_comparison : pd.DataFrame
        ``cross_validation_fold_comparison.csv`` contents.
    metric : str
        Metric name (e.g. ``BEDROC``).
    top_n : int | None, optional
        Limit scorers by mean across folds. Default: 25.
    reference_scorer : str, optional
        Always-included scorer. Default: ``OCScore``.
    size : tuple[float, float], optional
        Figure size in inches.

    Returns
    -------
    (Figure, Axes)
    '''

    value_col = _validation_column(metric)
    if value_col not in fold_comparison.columns:
        raise ValueError(f"Column {value_col!r} missing from fold comparison table.")

    apply_basic_style()
    pool = _fold_comparison_plot_pool(fold_comparison)
    pivot_source = pool[["scorer", "fold_index", value_col]].dropna()
    if pivot_source.empty:
        raise ValueError(f"No fold values for metric {metric!r}.")

    scorers = _select_fold_comparison_scorers(
        pool,
        value_col,
        top_n=top_n,
        reference_scorer=reference_scorer,
    )

    filtered = pivot_source[pivot_source["scorer"].astype(str).isin(scorers)]
    table = filtered.pivot(index="scorer", columns="fold_index", values=value_col)
    table = table.loc[[s for s in scorers if s in table.index]]

    fig, ax = new_fig(size)
    sns.heatmap(
        table.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": metric},
    )
    ax.set_title(f"Per-fold {metric}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Scorer")
    return fig, ax


_CROSS_FOLD_MEAN_LABEL = "Mean"


def _select_fold_comparison_scorers(
        fold_comparison: pd.DataFrame,
        value_col: str,
        *,
        scorers: Optional[Sequence[str]] = None,
        top_n: Optional[int] = 15,
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
    ) -> list[str]:
    pool = _fold_comparison_plot_pool(fold_comparison)
    available_scorers = pool["scorer"].astype(str).unique()
    if scorers is not None:
        return [
            str(scorer)
            for scorer in scorers
            if str(scorer) in available_scorers
            and not _is_descriptor_aggregate_scorer(str(scorer))
        ]

    ranked = (
        pool.groupby("scorer", as_index=False)[value_col]
        .mean()
        .sort_values(value_col, ascending=False)
    )
    scorer_list = ranked["scorer"].astype(str).tolist()
    if reference_scorer in scorer_list:
        scorer_list.remove(reference_scorer)
    if top_n is not None and top_n > 0 and len(scorer_list) > max(0, top_n - 1):
        scorer_list = scorer_list[: max(0, top_n - 1)]
    scorer_list = _pin_fold_comparison_scorers(scorer_list, available_scorers)
    if reference_scorer in available_scorers:
        return [reference_scorer, *scorer_list]
    return scorer_list


def plot_fold_metric_bars(
        fold_comparison: pd.DataFrame,
        metric: str,
        *,
        scorers: Optional[Sequence[str]] = None,
        top_n: Optional[int] = 15,
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
        size: tuple[float, float] = (8, 5),
    ) -> tuple[plt.Figure, plt.Axes]:
    '''Grouped bar chart of per-fold metric values plus cross-fold mean ± std.

    Each x-axis group is one CV fold, except the final group which shows the
    mean across folds per scorer with standard-deviation error bars.

    Parameters
    ----------
    fold_comparison : pd.DataFrame
        ``cross_validation_fold_comparison.csv`` contents.
    metric : str
        Metric name.
    scorers : Sequence[str] | None, optional
        Explicit scorer list. When ``None``, uses ``top_n`` best by mean.
    top_n : int | None, optional
        Used when ``scorers`` is ``None``. Default: 15.
    reference_scorer : str, optional
        Highlighted scorer. Default: ``OCScore``.
    size : tuple[float, float], optional
        Figure size in inches.

    Returns
    -------
    (Figure, Axes)
    '''

    value_col = _validation_column(metric)
    if value_col not in fold_comparison.columns:
        raise ValueError(f"Column {value_col!r} missing from fold comparison table.")

    apply_basic_style()
    pool = _fold_comparison_plot_pool(fold_comparison)
    scorers = _select_fold_comparison_scorers(
        pool,
        value_col,
        scorers=scorers,
        top_n=top_n,
        reference_scorer=reference_scorer,
    )
    if not scorers:
        raise ValueError(f"No scorers available for metric {metric!r}.")

    fold_indices = sorted(fold_comparison["fold_index"].astype(int).unique())
    group_labels = [str(fold) for fold in fold_indices] + [_CROSS_FOLD_MEAN_LABEL]
    n_folds = len(fold_indices)
    n_groups = len(group_labels)
    n_scorers = len(scorers)
    mean_group_x = float(n_folds)

    fold_values_by_scorer: dict[str, list[float]] = {}
    mean_by_scorer: dict[str, float] = {}
    std_by_scorer: dict[str, float] = {}
    for scorer in scorers:
        rows = pool[pool["scorer"].astype(str) == scorer].sort_values(
            "fold_index"
        )
        fold_values = rows.set_index(rows["fold_index"].astype(int))[value_col].astype(float)
        per_fold = [float(fold_values.get(fold, np.nan)) for fold in fold_indices]
        fold_values_by_scorer[scorer] = per_fold
        finite = np.asarray([value for value in per_fold if np.isfinite(value)], dtype=float)
        if finite.size:
            mean_by_scorer[scorer] = float(np.mean(finite))
            std_by_scorer[scorer] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        else:
            mean_by_scorer[scorer] = np.nan
            std_by_scorer[scorer] = 0.0

    fig_width = max(size[0], 9.0 + 0.55 * n_groups + 0.04 * n_scorers)
    fig, ax = new_fig((fig_width, size[1]))
    x = np.arange(n_groups, dtype=float)
    group_width = min(0.88, max(0.45, 0.94 - 0.012 * n_scorers))
    bar_width = group_width / n_scorers
    cmap = plt.get_cmap("tab20")

    for index, scorer in enumerate(scorers):
        offset = (index - (n_scorers - 1) / 2) * bar_width
        is_reference = scorer == reference_scorer
        color = OCSCORE_COLOR if is_reference else cmap(index % 20)
        alpha = 0.95 if is_reference else 0.78
        edgecolor = "#922b21" if is_reference else "white"
        linewidth = 0.8 if is_reference else 0.25
        ax.bar(
            x[:n_folds] + offset,
            fold_values_by_scorer[scorer],
            bar_width,
            label=scorer,
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.bar(
            mean_group_x + offset,
            mean_by_scorer[scorer],
            bar_width,
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.errorbar(
            mean_group_x + offset,
            mean_by_scorer[scorer],
            yerr=std_by_scorer[scorer],
            fmt="none",
            ecolor=edgecolor if is_reference else "#566573",
            elinewidth=1.4 if is_reference else 1.0,
            capsize=3.0,
            capthick=1.2 if is_reference else 0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    if fold_indices:
        ax.axvline(n_folds - 0.5, color="#bdc3c7", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Fold / mean ± std")
    ax.set_ylabel(metric)
    ax.set_title(f"Fold comparison: {metric}")
    ax.set_ylim(bottom=0.0)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
        borderaxespad=0.0,
    )
    max_label_len = max((len(str(scorer)) for scorer in scorers), default=10)
    fig.subplots_adjust(right=min(0.78, max(0.52, 0.96 - 0.013 * max_label_len)))
    return fig, ax


def plot_fold_metric_lines(
        fold_comparison: pd.DataFrame,
        metric: str,
        *,
        scorers: Optional[Sequence[str]] = None,
        top_n: Optional[int] = 15,
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
        size: tuple[float, float] = (8, 5),
    ) -> tuple[plt.Figure, plt.Axes]:
    '''Backward-compatible alias for :func:`plot_fold_metric_bars`.'''

    return plot_fold_metric_bars(
        fold_comparison,
        metric,
        scorers=scorers,
        top_n=top_n,
        reference_scorer=reference_scorer,
        size=size,
    )


def plot_ocscore_wins(
        ocscore_wins: pd.DataFrame,
        *,
        size: tuple[float, float] = (7, 4),
    ) -> tuple[plt.Figure, plt.Axes]:
    '''Bar chart of how often OCScore ranked first per metric.

    Parameters
    ----------
    ocscore_wins : pd.DataFrame
        ``cross_validation_ocscore_wins.csv`` contents.

    Returns
    -------
    (Figure, Axes)
    '''

    if ocscore_wins.empty:
        raise ValueError("OCScore wins table is empty.")

    apply_basic_style()
    fig, ax = new_fig(size)
    metrics = ocscore_wins["metric"].astype(str)
    wins = ocscore_wins["n_folds_won"].astype(float)
    compared = ocscore_wins["n_folds_compared"].astype(float).replace(0, np.nan)
    ax.bar(metrics, wins, color=OCSCORE_COLOR, alpha=0.85, label="Folds won")
    ax.plot(metrics, compared, color=SF_COLOR, marker="o", linestyle="--", label="Folds compared")
    ax.set_ylabel("Fold count")
    ax.set_title("OCScore top rank per fold")
    ax.set_ylim(bottom=0.0)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        borderaxespad=0.0,
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.subplots_adjust(right=0.82)
    return fig, ax


def save_cross_validation_figures(
        cv_dir: str | Path,
        figures_dir: str | Path | None = None,
        *,
        metrics: Optional[Sequence[str]] = None,
        top_n: Optional[int] = 25,
        dpi: int = 150,
    ) -> dict[str, str]:
    '''Generate standard PNG plots from a cross-validation output directory.

    Parameters
    ----------
    cv_dir : str | Path
        Cross-validation or export directory.
    figures_dir : str | Path | None, optional
        Destination for PNG files. Default: ``<cv_dir>/figures``.
    metrics : Sequence[str] | None, optional
        Metrics to plot. Default: ranking metrics from results JSON (excludes
        raw TP/TN/FP/FN counts).
    top_n : int | None, optional
        Maximum scoring functions per chart (OCScore always shown). Default: 25.
    dpi : int, optional
        PNG resolution. Default: 150.

    Returns
    -------
    dict[str, str]
        Map of plot label to written file path.
    '''

    artifacts = load_cross_validation_artifacts(cv_dir)
    root = Path(artifacts["cv_dir"])
    output_path = Path(figures_dir) if figures_dir is not None else root / "figures"

    results = artifacts["results"]
    mean_std: pd.DataFrame = artifacts["mean_std"]
    fold_comparison: pd.DataFrame = artifacts["fold_comparison"]
    ocscore_wins: pd.DataFrame = artifacts["ocscore_wins"]

    metric_list = _resolve_plot_metrics(metrics, results, mean_std)

    output_path.mkdir(parents=True, exist_ok=True)
    _prune_obsolete_cv_figures(output_path, metric_list)
    _remove_legacy_fold_line_plots(output_path)

    written: dict[str, str] = {}

    if not mean_std.empty:
        for metric in metric_list:
            if metric not in set(mean_std["metric"].astype(str)):
                continue
            fig, _ = plot_mean_std_bars(mean_std, metric, top_n=top_n)
            key = f"mean_std_{metric}"
            written[key] = _save_figure(
                fig,
                output_path / f"cv_mean_std_{_safe_filename(metric)}.png",
                dpi=dpi,
            )

    if not fold_comparison.empty:
        for metric in metric_list:
            value_col = _validation_column(metric)
            if value_col not in fold_comparison.columns:
                continue
            if fold_comparison[value_col].dropna().empty:
                continue
            fig, _ = plot_fold_metric_heatmap(fold_comparison, metric, top_n=top_n)
            written[f"heatmap_{metric}"] = _save_figure(
                fig,
                output_path / f"cv_heatmap_{_safe_filename(metric)}.png",
                dpi=dpi,
            )
            fig, _ = plot_fold_metric_bars(fold_comparison, metric, top_n=min(15, top_n or 15))
            written[f"fold_comparison_{metric}"] = _save_figure(
                fig,
                output_path / f"{FOLD_COMPARISON_PLOT_PREFIX}{_safe_filename(metric)}.png",
                dpi=dpi,
            )

    if not ocscore_wins.empty:
        fig, _ = plot_ocscore_wins(ocscore_wins)
        written["ocscore_wins"] = _save_figure(
            fig,
            output_path / "cv_ocscore_fold_wins.png",
            dpi=dpi,
        )

    per_target: pd.DataFrame = artifacts.get("per_target", pd.DataFrame())
    if not per_target.empty:
        aggregated = aggregate_cv_per_target_metrics(per_target)
        if not aggregated.empty:
            per_target_path = output_path / "cv_per_target_aggregated.csv"
            aggregated.to_csv(per_target_path, index=False)
            written["per_target_aggregated_csv"] = str(per_target_path.resolve())
            try:
                pt_plots = save_per_target_figures(
                    aggregated,
                    output_path,
                    split="validation",
                    metrics=metric_list,
                    top_n=top_n,
                    dpi=dpi,
                )
                written.update(pt_plots)
            except (ValueError, TypeError):
                pass

    return written


def save_baseline_comparison_figures(
        comparison_csv: str | Path,
        figures_dir: str | Path | None = None,
        *,
        split: str = "test",
        metrics: Optional[Sequence[str]] = None,
        top_n: Optional[int] = 25,
        dpi: int = 150,
    ) -> dict[str, str]:
    '''Plot DUDEz baseline comparison CSV from example 19.

    Parameters
    ----------
    comparison_csv : str | Path
        Path to ``dudez_sf_baseline_comparison.csv``.
    figures_dir : str | Path | None, optional
        Output directory. Default: ``<csv-parent>/figures``.
    split : str, optional
        Split to plot, by default ``test``.
    metrics : Sequence[str] | None, optional
        Metrics to plot. Default: BEDROC and ROC-AUC when present.
    top_n : int | None, optional
        Max scorers per chart. Default: 25.
    dpi : int, optional
        PNG resolution. Default: 150.

    Returns
    -------
    dict[str, str]
        Map of plot label to written file path.
    '''

    csv_path = Path(comparison_csv)
    table = pd.read_csv(csv_path)
    if table.empty:
        raise ValueError(f"Baseline comparison table is empty: {csv_path}")

    subset = table[table["split"].astype(str) == split].copy()
    if subset.empty:
        raise ValueError(f"No rows for split {split!r} in {csv_path}")

    metric_list = (
        [str(item) for item in metrics]
        if metrics
        else [column for column in ("BEDROC", "ROC-AUC", "EF1%", "NDCG@1%") if column in subset.columns]
    )
    if not metric_list:
        raise ValueError(f"No plottable metric columns in {csv_path}")

    output_path = Path(figures_dir) if figures_dir is not None else csv_path.parent / "figures"
    written: dict[str, str] = {}

    for metric in metric_list:
        rows = subset[["scorer", "scorer_type", metric]].dropna(subset=[metric])
        if rows.empty:
            continue
        rows = rows.sort_values(metric, ascending=False)
        scorers = rows["scorer"].astype(str).tolist()
        if top_n is not None and top_n > 0 and len(scorers) > top_n:
            ocscore_name = next(
                (name for name, stype in zip(scorers, rows["scorer_type"].astype(str)) if stype == "model"),
                None,
            )
            trimmed = scorers[:top_n]
            if ocscore_name and ocscore_name not in trimmed:
                trimmed = [ocscore_name, *[s for s in scorers if s != ocscore_name][: max(0, top_n - 1)]]
            scorers = trimmed
        plot_rows = rows[rows["scorer"].astype(str).isin(scorers)].sort_values(metric, ascending=True)

        apply_basic_style()
        height = max(4.0, 0.32 * len(plot_rows))
        fig, ax = new_fig((8.0, height))
        colors = [
            OCSCORE_COLOR if str(st) == "model" else SF_COLOR
            for st in plot_rows["scorer_type"].astype(str)
        ]
        y_pos = np.arange(len(plot_rows))
        ax.barh(y_pos, plot_rows[metric].astype(float), color=colors, alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_rows["scorer"].astype(str))
        ax.set_xlabel(metric)
        ax.set_title(f"DUDEz baseline comparison ({split})")
        ax.invert_yaxis()
        ax.set_xlim(left=0.0)
        key = f"baseline_{split}_{metric}"
        written[key] = _save_figure(
            fig,
            output_path / f"baseline_{split}_{_safe_filename(metric)}.png",
            dpi=dpi,
        )

    return written


def save_calibration_reliability_figures(
        y_true: np.ndarray,
        logits: np.ndarray,
        figures_dir: str | Path,
        *,
        split: str = "test",
        calibrator: Optional[Any] = None,
        dpi: int = 150,
    ) -> dict[str, str]:
    '''Write reliability diagrams for OCScore sigmoid and optional calibrated probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels for the split.
    logits : np.ndarray
        Classifier logits.
    figures_dir : str | Path
        Output directory for PNG files.
    split : str, optional
        Split label used in filenames, by default ``test``.
    calibrator : Any | None, optional
        Fitted :class:`~OCDocker.OCScore.Analysis.Metrics.Calibration.ProbabilityCalibrator`.
    dpi : int, optional
        PNG resolution.

    Returns
    -------
    dict[str, str]
        Map of plot label to written path.
    '''

    from OCDocker.OCScore.Analysis.Metrics.Calibration import logits_to_probabilities
    from OCDocker.OCScore.Analysis.Plotting.MetricsPlots import reliability_plot

    output_path = Path(figures_dir)
    written: dict[str, str] = {}
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    logits = np.asarray(logits, dtype=float).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return written

    uncalibrated = logits_to_probabilities(logits)
    fig, _ = reliability_plot(y_true, uncalibrated, label="Sigmoid (uncalibrated)")
    written["reliability_uncalibrated"] = _save_figure(
        fig,
        output_path / f"reliability_{split}_uncalibrated.png",
        dpi=dpi,
    )

    if calibrator is not None:
        calibrated = calibrator.predict(logits)
        fig, _ = reliability_plot(y_true, calibrated, label=f"Calibrated ({calibrator.method})")
        written["reliability_calibrated"] = _save_figure(
            fig,
            output_path / f"reliability_{split}_calibrated.png",
            dpi=dpi,
        )
    return written


__all__ = [
    "aggregate_cv_per_target_metrics",
    "load_cross_validation_artifacts",
    "plot_fold_metric_heatmap",
    "plot_fold_metric_bars",
    "plot_fold_metric_lines",
    "plot_mean_std_bars",
    "plot_ocscore_wins",
    "plot_per_target_boxplot",
    "plot_per_target_heatmap",
    "plot_per_target_ocscore_wins",
    "resolve_cross_validation_dir",
    "save_baseline_comparison_figures",
    "save_calibration_reliability_figures",
    "save_cross_validation_figures",
    "save_per_target_figures",
]
