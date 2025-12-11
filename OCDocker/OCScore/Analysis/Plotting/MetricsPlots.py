
#!/usr/bin/env python3

# Description
###############################################################################
'''
Core metric plots (ROC/PR/enrichment) for Analysis.

They are imported as:

from OCDocker.OCScore.Analysis.Plotting import MetricsPlots as PlottingMetrics
'''

# Imports
###############################################################################

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from .Core import apply_basic_style, new_fig

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Methods
###############################################################################


def roc_plot(y_true: Union[Sequence[float], np.ndarray],
             y_score: Union[Sequence[float], np.ndarray],
             *,
             size: Tuple[float, float] = (6, 4)) -> Tuple[Figure, Axes]:
    '''Plot a ROC curve with AUC in legend.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels (0/1 or boolean).
    y_score : array-like of shape (n_samples,)
        Continuous prediction scores (higher = more positive).
    size : tuple(float, float), optional
        Figure size in inches (width, height). Default: (6, 4).

    Returns
    -------
    tuple(Figure, Axes)
        Matplotlib figure and axes objects.
    '''

    apply_basic_style()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = new_fig(size)
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return fig, ax


def pr_plot(y_true: Union[Sequence[float], np.ndarray],
            y_score: Union[Sequence[float], np.ndarray],
            *,
            size: Tuple[float, float] = (6, 4)) -> Tuple[Figure, Axes]:
    '''Plot a Precision–Recall curve with AP in legend.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels (0/1 or boolean).
    y_score : array-like of shape (n_samples,)
        Continuous prediction scores (higher = more positive).
    size : tuple(float, float), optional
        Figure size in inches (width, height). Default: (6, 4).

    Returns
    -------
    tuple(Figure, Axes)
        Matplotlib figure and axes objects.
    '''

    apply_basic_style()

    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = new_fig(size)
    ax.plot(r, p, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    
    return fig, ax


def enrichment_plot(y_true: Union[Sequence[float], np.ndarray],
                    y_score: Union[Sequence[float], np.ndarray],
                    fractions: Iterable[float] = (0.01, 0.02, 0.05, 0.1),
                    *,
                    size: Tuple[float, float] = (6, 4)) -> Tuple[Figure, Axes]:
    '''Plot an enrichment curve: normalized cumulative hits vs. ranked-list fraction.

    Draws vertical markers at requested fractions to ease reading the curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels (0/1 or boolean).
    y_score : array-like of shape (n_samples,)
        Continuous prediction scores (higher = more positive).
    fractions : Iterable[float], optional
        Fractions of the ranked list to annotate with vertical lines.
        Default: (0.01, 0.02, 0.05, 0.1).
    size : tuple(float, float), optional
        Figure size in inches (width, height). Default: (6, 4).

    Returns
    -------
    tuple(Figure, Axes)
        Matplotlib figure and axes objects.
    '''

    apply_basic_style()

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = int(len(y_true))
    order = np.argsort(-y_score)
    cum_hits = np.cumsum((y_true[order] == 1).astype(int))
    xs = np.linspace(1 / n, 1.0, n)
    fig, ax = new_fig(size)
    ax.plot(xs, cum_hits / max(1, int(cum_hits[-1])), label="Cumulative hits (norm)")

    for f in fractions:
        k = max(1, int(round(float(f) * n)))
        ax.axvline(x=k / n, linestyle=':', linewidth=1)

    ax.set_xlabel("Top fraction of ranked list")
    ax.set_ylabel("Normalized hits")
    ax.set_title("Enrichment Curve")
    ax.legend()
    return fig, ax
