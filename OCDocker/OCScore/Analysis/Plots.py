from __future__ import annotations
"""
Test2 plotting helpers (library style, no I/O).

Each function produces a single chart and returns (fig, ax).
It will only save if you pass output_path.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "plot_roc_auc_heatmap",
    "plot_pooled_roc_forest",
    "plot_efroc_by_target",
    "plot_efroc_pooled",
]

def plot_roc_auc_heatmap(
    roc_per_target: pd.DataFrame,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of per-target ROC AUC (targets x models).
    Expects columns: ["target","model","roc_auc"].
    """
    pivot = roc_per_target.pivot_table(index="target", columns="model", values="roc_auc", aggfunc="mean")
    data = pivot.to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]), labels=list(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), labels=list(pivot.index))
    ax.set_xlabel("Model")
    ax.set_ylabel("Target")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("ROC AUC")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig, ax

def plot_pooled_roc_forest(
    pooled_roc: pd.DataFrame,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Forest plot for pooled ROC AUC with 95% CI.
    Expects columns: ["model","roc_auc","ci_low","ci_high"].
    """
    d = pooled_roc.copy().sort_values("roc_auc", ascending=True).reset_index(drop=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = np.arange(d.shape[0])
    x = d["roc_auc"].to_numpy()
    xerr = np.vstack([x - d["ci_low"].to_numpy(), d["ci_high"].to_numpy() - x])
    ax.errorbar(x=x, y=y, xerr=xerr, fmt="o", capsize=3)
    ax.set_yticks(y, labels=d["model"].tolist())
    ax.set_xlabel("ROC AUC")
    if title:
        ax.set_title(title)
    ax.set_ylim(-0.5, d.shape[0] - 0.5)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig, ax

def plot_efroc_by_target(
    efroc_per_tgt: pd.DataFrame,
    target: str,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    EF-ROC curves for a single target, multiple models.
    Expects columns: ["target","model","epsilon","ef_roc"].
    """
    d = efroc_per_tgt.loc[efroc_per_tgt["target"] == target]
    models = d["model"].unique().tolist()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for m in models:
        dm = d.loc[d["model"] == m].sort_values("epsilon")
        ax.plot(dm["epsilon"].to_numpy(), dm["ef_roc"].to_numpy(), marker="o", label=str(m))
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("FPR (epsilon)")
    ax.set_ylabel("EF_ROC (TPR/epsilon)")
    ax.legend(loc="best")
    if title:
        ax.set_title(title if title else f"EF-ROC — {target}")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig, ax

def plot_efroc_pooled(
    efroc_pool: pd.DataFrame,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    EF-ROC pooled curves per model.
    Expects columns: ["model","epsilon","ef_roc"].
    """
    d = efroc_pool.copy()
    models = d["model"].unique().tolist()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for m in models:
        dm = d.loc[d["model"] == m].sort_values("epsilon")
        ax.plot(dm["epsilon"].to_numpy(), dm["ef_roc"].to_numpy(), marker="o", label=str(m))
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("FPR (epsilon)")
    ax.set_ylabel("EF_ROC (TPR/epsilon)")
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig, ax
