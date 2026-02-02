#!/usr/bin/env python3

# Description
###############################################################################
'''
Plotting package exports commonly used Analysis plotting utilities.

Usage:

import OCDocker.OCScore.Analysis.Plotting as ocstatplot

Modules
-------
- Colouring: Color palette helpers.
- Core: Matplotlib styling helpers.
- ImpactPlots: Feature impact plotting utilities.
- MetricsPlots: ROC/PR/enrichment plotting utilities.
- Stats: Statistical summary plots.
'''

from .Stats import (
    plot_combined_metric_scatter,
    plot_boxplots,
    plot_barplots,
    plot_scatterplot,
    plot_bar_with_significance,
    plot_heatmap,
    plot_normality_and_variance_diagnostics,
    plot_pca_importance_barplot,
    plot_pca_importance_histogram,
    save_pca_importance_groups,
    save_pca_importance_bins,
)

from .Colouring import set_color_mapping

__all__ = [
    'plot_combined_metric_scatter',
    'plot_boxplots',
    'plot_barplots',
    'plot_scatterplot',
    'plot_bar_with_significance',
    'plot_heatmap',
    'plot_normality_and_variance_diagnostics',
    'plot_pca_importance_barplot',
    'plot_pca_importance_histogram',
    'save_pca_importance_groups',
    'save_pca_importance_bins',
    'set_color_mapping',
]
