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
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

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
