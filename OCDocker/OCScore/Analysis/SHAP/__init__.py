#!/usr/bin/env python3

# Description
###############################################################################
'''
Pipeline-native SHAP for exported ``best_model/`` bundles.
'''

# Imports
###############################################################################
from .ExportRunner import run_export_shap_analysis
from .Paths import OutputPaths
from .Plots import (
    DEFAULT_FEATURE_FAMILIES,
    assign_feature_families,
    save_shap_plot_suite,
    save_shap_plot_suite_from_paths,
)

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

__all__ = [
    "DEFAULT_FEATURE_FAMILIES",
    "OutputPaths",
    "assign_feature_families",
    "run_export_shap_analysis",
    "save_shap_plot_suite",
    "save_shap_plot_suite_from_paths",
]
