#!/usr/bin/env python3

# Description
###############################################################################
'''
Re-export SHAP public API for convenience.

Usage:

import OCDocker.OCScore.Analysis.SHAP as ocshap

Modules
-------
- Cli: Command-line entry point for SHAP runs.
- Data: Data loading and preparation helpers.
- Explain: SHAP computation helpers.
- Model: Neural network builder for SHAP runs.
- Plots: SHAP visualization utilities.
- Runner: End-to-end SHAP workflow runner.
- Studies: Optuna study selection helpers.
'''

# Imports
###############################################################################
from . import Plots as plots
from .Data import DataHandles, load_and_prepare_data
from .Explain import compute_shap_values
from .Model import build_neural_net
from .Runner import OutputPaths, run_shap_analysis
from .Studies import BestSelections, StudyHandles, select_best_from_studies

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
