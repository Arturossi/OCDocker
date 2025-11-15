"""
Plotting package exports commonly used plotting utilities across Analysis.

Suggested import:
    import OCDocker.OCScore.Analysis.Plotting as ocstatplot
"""

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
