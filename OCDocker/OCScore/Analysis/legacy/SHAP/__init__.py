#!/usr/bin/env python3

# Description
###############################################################################
'''
Legacy four-study Optuna SHAP workflow.

Usage:

import OCDocker.OCScore.Analysis.legacy.SHAP as ocshap_legacy
'''

# Imports
###############################################################################
from . import Plots as plots
from .Data import DataHandles, load_and_prepare_data
from .Explain import compute_shap_values
from .Model import build_neural_net
from .Runner import OutputPaths, run_shap_analysis
from .Studies import BestSelections, StudyHandles, select_best_from_studies

__all__ = [
    "run_shap_analysis",
    "OutputPaths",
    "StudyHandles",
    "BestSelections",
    "select_best_from_studies",
    "DataHandles",
    "load_and_prepare_data",
    "build_neural_net",
    "compute_shap_values",
    "plots",
]
