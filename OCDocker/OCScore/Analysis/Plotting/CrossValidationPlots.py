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

from OCDocker.OCScore.Optimization.ModelCrossValidation import OCSCORE_MODEL_SCORER_NAME

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
OCSCORE_WINS_CSV_NAME = "cross_validation_ocscore_wins.csv"
PER_TARGET_CSV_NAME = "cross_validation_per_target_metrics.csv"

OCSCORE_COLOR = "#c0392b"
SF_COLOR = "#7f8c8d"

# Functions
###############################################################################
## Private ##

def _validation_column(metric: str) -> str:
    return f"validation_{metric}"


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
    scorers = subset["scorer"].astype(str).tolist()
    if reference_scorer in scorers:
        scorers.remove(reference_scorer)
    if top_n is None or top_n <= 0 or len(scorers) <= top_n:
        ordered = scorers
    else:
        ordered = scorers[: max(0, top_n - 1)]
    if reference_scorer in mean_std["scorer"].astype(str).values:
        return [reference_scorer, *ordered]
    return ordered


def _save_figure(fig: plt.Figure, path: Path, *, dpi: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


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


def _chunks(values: Sequence[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        return []
    return [list(values[index:index + chunk_size]) for index in range(0, len(values), chunk_size)]


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
        max_groups: Optional[int] = 40,
        groups: Optional[Sequence[str]] = None,
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
        size: tuple[float, float] = (12, 8),
        annotate: Optional[bool] = None,
        annotation_cell_limit: int = 300,
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

    apply_basic_style()
    n_scorers, n_groups = table.shape
    n_cells = int(n_scorers * n_groups)
    show_annotations = n_cells <= annotation_cell_limit if annotate is None else bool(annotate)
    width = max(size[0], 3.5 + 0.36 * max(1, n_groups))
    height = max(size[1], 1.8 + 0.36 * max(1, n_scorers))
    tick_fontsize = 6 if n_groups > 35 else 7
    scorer_fontsize = 7 if n_scorers > 18 else 8
    fig, ax = new_fig((width, height))
    sns.heatmap(
        table.astype(float),
        annot=show_annotations,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.25 if n_cells > annotation_cell_limit else 0.5,
        ax=ax,
        cbar_kws={"label": metric, "shrink": 0.85},
        annot_kws={"fontsize": 6},
    )
    title_split = f" ({split})" if split else ""
    title_extra = "" if show_annotations else " - values in CSV"
    ax.set_title(f"Per-receptor {metric}{title_split}{title_extra}")
    ax.set_xlabel("Receptor")
    ax.set_ylabel("Scorer")
    xlabels = ax.get_xticklabels()
    if n_groups > 24:
        label_step = max(1, (n_groups + 23) // 24)
        for index, label in enumerate(xlabels):
            label.set_visible(index % label_step == 0)
    plt.setp(xlabels, rotation=90, ha="center", fontsize=tick_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=scorer_fontsize)
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
    height = max(size[1], 0.34 * max(1, len(scorers)) + 1.6)
    width = max(size[0], 8.0)
    palette = {s: OCSCORE_COLOR if s == reference_scorer else SF_COLOR for s in scorers}
    fig, ax = new_fig((width, height))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="vert: bool will be deprecated.*",
            category=PendingDeprecationWarning,
        )
        sns.boxplot(
            data=plot_df,
            x=metric,
            y="scorer",
            hue="scorer",
            order=scorers,
            hue_order=scorers,
            ax=ax,
            palette=palette,
            legend=False,
            dodge=False,
            width=0.58,
            showfliers=False,
        )
    sns.stripplot(
        data=plot_df,
        x=metric,
        y="scorer",
        order=scorers,
        ax=ax,
        color="#222222",
        alpha=0.35,
        size=2.5,
        jitter=0.22,
    )
    title_split = f" ({split})" if split else ""
    ax.set_title(f"Per-receptor {metric} distribution{title_split}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Scorer")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    plt.setp(ax.get_yticklabels(), fontsize=8)
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
    return fig, ax


def save_per_target_figures(
        per_target_source: str | Path | pd.DataFrame,
        figures_dir: str | Path,
        *,
        split: Optional[str] = "test",
        metrics: Optional[Sequence[str]] = None,
        top_n: Optional[int] = 15,
        max_groups: Optional[int] = 40,
        heatmap_chunk_size: int = 16,
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
    for metric in metric_list:
        if metric not in per_target.columns:
            continue
        try:
            fig, _ = plot_per_target_heatmap(
                per_target,
                metric,
                split=split,
                top_n=top_n,
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

        selected_groups = _ordered_groups_for_per_target(
            per_target,
            metric,
            split=split,
            reference_scorer=reference,
            max_groups=max_groups,
        )
        if heatmap_chunk_size > 0 and len(selected_groups) > heatmap_chunk_size:
            for chunk_index, group_chunk in enumerate(_chunks(selected_groups, heatmap_chunk_size), start=1):
                try:
                    fig, _ = plot_per_target_heatmap(
                        per_target,
                        metric,
                        split=split,
                        top_n=top_n,
                        max_groups=None,
                        groups=group_chunk,
                        reference_scorer=reference,
                    )
                    key = f"per_target_heatmap_{split_label}_{metric}_part{chunk_index:02d}"
                    written[key] = _save_figure(
                        fig,
                        output_path / (
                            f"per_target_{split_label}_{_safe_filename(metric)}_heatmap_part{chunk_index:02d}.png"
                        ),
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
    pivot_source = fold_comparison[["scorer", "fold_index", value_col]].dropna()
    if pivot_source.empty:
        raise ValueError(f"No fold values for metric {metric!r}.")

    mean_by_scorer = (
        pivot_source.groupby("scorer", as_index=False)[value_col]
        .mean()
        .sort_values(value_col, ascending=False)
    )
    scorers = mean_by_scorer["scorer"].astype(str).tolist()
    if reference_scorer in scorers:
        scorers.remove(reference_scorer)
    if top_n is not None and top_n > 0 and len(scorers) > max(0, top_n - 1):
        scorers = scorers[: max(0, top_n - 1)]
    if reference_scorer in pivot_source["scorer"].astype(str).unique():
        scorers = [reference_scorer, *scorers]

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


def plot_fold_metric_lines(
        fold_comparison: pd.DataFrame,
        metric: str,
        *,
        scorers: Optional[Sequence[str]] = None,
        top_n: Optional[int] = 15,
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
        size: tuple[float, float] = (8, 5),
    ) -> tuple[plt.Figure, plt.Axes]:
    '''Line plot of a metric across folds for selected scorers.

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
    if scorers is None:
        ranked = (
            fold_comparison.groupby("scorer", as_index=False)[value_col]
            .mean()
            .sort_values(value_col, ascending=False)
        )
        scorer_list = ranked["scorer"].astype(str).tolist()
        if reference_scorer in scorer_list:
            scorer_list.remove(reference_scorer)
        if top_n is not None and top_n > 0 and len(scorer_list) > max(0, top_n - 1):
            scorer_list = scorer_list[: max(0, top_n - 1)]
        if reference_scorer in fold_comparison["scorer"].astype(str).unique():
            scorers = [reference_scorer, *scorer_list]
        else:
            scorers = scorer_list

    fig, ax = new_fig(size)
    for scorer in scorers:
        rows = fold_comparison[fold_comparison["scorer"].astype(str) == scorer].sort_values(
            "fold_index"
        )
        if rows.empty:
            continue
        color = OCSCORE_COLOR if scorer == reference_scorer else None
        linewidth = 2.5 if scorer == reference_scorer else 1.2
        ax.plot(
            rows["fold_index"],
            rows[value_col],
            marker="o",
            label=scorer,
            color=color,
            linewidth=linewidth,
        )
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across folds")
    ax.legend(loc="best", fontsize=8)
    return fig, ax


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
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
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
        Metrics to plot. Default: from ``comparison_metrics`` in results JSON,
        or unique metrics in the mean/std table.
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

    if metrics is None:
        summary_metrics = (results.get("scorer_comparison_summary") or {}).get(
            "comparison_metrics"
        )
        if summary_metrics:
            metric_list = [str(item) for item in summary_metrics]
        elif not mean_std.empty:
            metric_list = sorted(mean_std["metric"].astype(str).unique())
        else:
            objective = str(results.get("objective_metric") or "BEDROC")
            metric_list = [objective]
    else:
        metric_list = [str(item) for item in metrics]

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
            fig, _ = plot_fold_metric_lines(fold_comparison, metric, top_n=min(15, top_n or 15))
            written[f"fold_lines_{metric}"] = _save_figure(
                fig,
                output_path / f"cv_fold_lines_{_safe_filename(metric)}.png",
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


def _safe_filename(metric: str) -> str:
    return metric.replace("%", "pct").replace("/", "_").replace(" ", "_")


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
