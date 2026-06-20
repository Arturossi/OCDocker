#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Analysis.Plotting.MetricsPlots.
'''

# Imports
###############################################################################
import matplotlib.pyplot as plt
import numpy as np

import pytest

import OCDocker.OCScore.Analysis.Plotting.MetricsPlots as ocmetricsplots

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(275)
def test_enrichment_plot_returns_figure_and_axes():
    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=int)
    y_score = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3], dtype=float)

    fig, ax = ocmetricsplots.enrichment_plot(y_true, y_score, fractions=(0.1, 0.5), size=(4, 3))
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Enrichment Curve"
    assert len(ax.lines) >= 1
    plt.close(fig)


@pytest.mark.order(276)
def test_pr_and_roc_plots_return_expected_titles():
    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=int)
    y_score = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3], dtype=float)

    fig_pr, ax_pr = ocmetricsplots.pr_plot(y_true, y_score, size=(4, 3))
    fig_roc, ax_roc = ocmetricsplots.roc_plot(y_true, y_score, size=(4, 3))

    assert ax_pr.get_title() == "Precision-Recall Curve"
    assert ax_roc.get_title() == "ROC Curve"
    assert len(ax_pr.lines) >= 1
    assert len(ax_roc.lines) >= 1

    plt.close(fig_pr)
    plt.close(fig_roc)
