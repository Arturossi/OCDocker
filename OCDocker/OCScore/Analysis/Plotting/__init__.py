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

# Imports
###############################################################################
from .Colouring import set_color_mapping
from .Stats import plot_bar_with_significance
from .Stats import plot_barplots
from .Stats import plot_boxplots
from .Stats import plot_combined_metric_scatter
from .Stats import plot_heatmap
from .Stats import plot_normality_and_variance_diagnostics
from .Stats import plot_pca_importance_barplot
from .Stats import plot_pca_importance_histogram
from .Stats import plot_scatterplot
from .Stats import save_pca_importance_bins
from .Stats import save_pca_importance_groups

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

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

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
