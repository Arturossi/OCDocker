#!/usr/bin/env python3

# Description
###############################################################################
'''
Unified exports for the OCScore Analysis package.

Usage:

import OCDocker.OCScore.Analysis as ocanalysis

Modules
-------
- Correlation: Correlation analysis helpers.
- FeatureImportance: SHAP-style feature importance utilities.
- Impact: Feature impact summaries and plots.
- Metrics: Metric computation helpers.
- NNUtils: Neural network helper utilities.
- PerformanceEvaluation: Performance evaluation workflows.
- Plotting: Plotting helpers for analyses.
- RankingMetrics: Ranking metrics and tables.
- SHAP: SHAP analysis workflows.
- StatTests: Statistical test helpers.
- StudyProcessing: Study parsing and aggregation utilities.
'''

# Imports
###############################################################################
from .Metrics import Ranking as RankingMetrics
from .Plotting import MetricsPlots as PlottingMetrics
from typing import Any

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

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

# Unified exports for Analysis package
run_shap_analysis: Any
OutputPaths: Any
StudyHandles: Any
BestSelections: Any
select_best_from_studies: Any
DataHandles: Any
load_and_prepare_data: Any
build_neural_net: Any
compute_shap_values: Any
shap_plots: Any

build_impact_overview: Any
plot_impact_arrows_inline_labels: Any
get_neutral_features: Any

try:
    from .SHAP import (
        run_shap_analysis as _run_shap_analysis, OutputPaths as _OutputPaths,
        StudyHandles as _StudyHandles, BestSelections as _BestSelections, select_best_from_studies as _select_best_from_studies,
        DataHandles as _DataHandles, load_and_prepare_data as _load_and_prepare_data,
        build_neural_net as _build_neural_net, compute_shap_values as _compute_shap_values, plots as _shap_plots,
    )
    run_shap_analysis = _run_shap_analysis
    OutputPaths = _OutputPaths
    StudyHandles = _StudyHandles
    BestSelections = _BestSelections
    select_best_from_studies = _select_best_from_studies
    DataHandles = _DataHandles
    load_and_prepare_data = _load_and_prepare_data
    build_neural_net = _build_neural_net
    compute_shap_values = _compute_shap_values
    shap_plots = _shap_plots
except Exception:
    # Keep optional dependency failures from breaking the package
    run_shap_analysis = None
    OutputPaths = None
    StudyHandles = None
    BestSelections = None
    select_best_from_studies = None
    DataHandles = None
    load_and_prepare_data = None
    build_neural_net = None
    compute_shap_values = None
    shap_plots = None

try:
    from .Impact import (
        build_impact_overview as _build_impact_overview,
        plot_impact_arrows_inline_labels as _plot_impact_arrows_inline_labels,
        get_neutral_features as _get_neutral_features,
    )
    build_impact_overview = _build_impact_overview
    plot_impact_arrows_inline_labels = _plot_impact_arrows_inline_labels
    get_neutral_features = _get_neutral_features
except Exception:
    build_impact_overview = None
    plot_impact_arrows_inline_labels = None
    get_neutral_features = None

# Public API
__all__ = [
    # SHAP exports (may be None if dependencies missing)
    'run_shap_analysis',
    'OutputPaths',
    'StudyHandles',
    'BestSelections',
    'select_best_from_studies',
    'DataHandles',
    'load_and_prepare_data',
    'build_neural_net',
    'compute_shap_values',
    'shap_plots',
    # Metrics exports
    'RankingMetrics',
    # Plotting exports
    'PlottingMetrics',
    # Impact exports (may be None if dependencies missing)
    'build_impact_overview',
    'plot_impact_arrows_inline_labels',
    'get_neutral_features',
]
