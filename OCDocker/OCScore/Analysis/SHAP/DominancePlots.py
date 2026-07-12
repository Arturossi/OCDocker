#!/usr/bin/env python3

# Description
###############################################################################
'''
Stacked-bar plotting for cross-replica SHAP family composition.

Usage:

from OCDocker.OCScore.Analysis.SHAP.DominancePlots import save_family_composition_stacked_plot
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
DEFAULT_MIN_LABEL_PCT = 6.0

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

def _ensure_dir(path: Union[str, Path]) -> None:
    '''Ensure that a directory exists.

    Parameters
    ----------
    path : str | Path
        Directory path to create.
    '''

    Path(path).mkdir(parents=True, exist_ok=True)


## Public ##

def save_family_composition_stacked_plot(
        composition_df: pd.DataFrame,
        policy_order: Sequence[str],
        output_dir: Union[str, Path],
        *,
        family_order: Optional[Sequence[str]] = None,
        policy_labels: Optional[Mapping[str, str]] = None,
        colors: Optional[Mapping[str, str]] = None,
        dpi: int = 300,
        figsize: Optional[Tuple[float, float]] = None,
        min_label_pct: float = DEFAULT_MIN_LABEL_PCT,
        file_stem: str = "shap_family_composition",
        title: str = "SHAP importance composition by descriptor family",
        xlabel: str = "Relative SHAP importance (%, mean across replicas)",
        family_labels: Optional[Mapping[str, str]] = None,
        legend_ncol: Optional[int] = None,
    ) -> dict[str, str]:
    '''Save a stacked horizontal bar plot of per-policy SHAP family composition.

    Parameters
    ----------
    composition_df : pd.DataFrame
        Output of ``SHAP.Dominance.aggregate_family_composition``, with columns
        ``policy``, ``family``, ``relative_importance_pct_mean``.
    policy_order : sequence[str]
        Policies to plot, top-to-bottom.
    output_dir : str | Path
        Output directory.
    family_order : sequence[str] | None, optional
        Stacking order of families, left-to-right. Defaults to the families'
        total-importance order (largest first).
    policy_labels : mapping[str, str] | None, optional
        Optional display label per policy (e.g. a rank number), used for the
        y-axis tick labels instead of the raw policy name.
    colors : mapping[str, str] | None, optional
        Optional color per family. Defaults to a ``Set2`` palette.
    dpi : int, optional
        Figure DPI, by default 300.
    figsize : tuple[float, float] | None, optional
        Figure size. Defaults to a size scaled to the number of policies.
    min_label_pct : float, optional
        Minimum segment share (%) to annotate with a direct value label, by default 6.0.
    file_stem : str, optional
        Output file stem (without extension), by default ``"shap_family_composition"``.
    title : str, optional
        Plot title. Override to render the figure in another language.
    xlabel : str, optional
        X-axis label. Override to render the figure in another language.
    family_labels : mapping[str, str] | None, optional
        Optional display label per family, used in the legend instead of the raw
        family name. Override to render the figure in another language.
    legend_ncol : int | None, optional
        Legend columns. Defaults to one per family, which is fine for short names,
        but long labels make the legend wider than the figure; the saved image is
        then padded out to fit it, and the plot shrinks when that image is later
        embedded at a fixed width. Lower this when the labels are long.

    Returns
    -------
    dict[str, str]
        Output artifact paths (``family_composition_png``, ``family_composition_csv``).
    '''

    output_path = Path(output_dir)
    _ensure_dir(output_path)

    pivot = (
        composition_df[composition_df["policy"].isin(policy_order)]
        .pivot(index="policy", columns="family", values="relative_importance_pct_mean")
        .fillna(0.0)
        .reindex(policy_order)
        .iloc[::-1]  # barh draws the first row at the bottom; reverse to match the documented top-to-bottom order
    )
    if family_order is None:
        family_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot[family_order]

    if colors is None:
        palette = sns.color_palette("Set2", n_colors=len(family_order))
        colors = dict(zip(family_order, palette))

    fig_height = figsize[1] if figsize is not None else max(3.0, 0.5 * len(policy_order) + 1.5)
    fig_width = figsize[0] if figsize is not None else 9.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y_pos = list(range(len(pivot)))
    left = pd.Series(0.0, index=pivot.index)
    for family in family_order:
        values = pivot[family]
        legend_label = (family_labels or {}).get(family, family)
        ax.barh(y_pos, values, left=left.to_numpy(), height=0.62, color=colors[family], label=legend_label)
        for i, (policy, value) in enumerate(values.items()):
            if value >= min_label_pct:
                ax.text(
                    left.loc[policy] + value / 2, i, f"{value:.0f}",
                    va="center", ha="center", fontsize=8.5, color="white", fontweight="bold",
                )
        left = left + values

    labels = [(policy_labels or {}).get(policy, policy) for policy in pivot.index]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 100)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ncol = legend_ncol if legend_ncol is not None else min(4, len(family_order))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=False)
    fig.tight_layout()

    png_path = output_path / f"{file_stem}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_path / f"{file_stem}.csv"
    pivot.reset_index().to_csv(csv_path, index=False)

    return {"family_composition_png": str(png_path), "family_composition_csv": str(csv_path)}


__all__ = [
    "save_family_composition_stacked_plot",
]
