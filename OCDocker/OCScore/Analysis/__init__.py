#!/usr/bin/env python3

# Description
###############################################################################
'''
Unified exports for the OCScore Analysis package.

Usage:

import OCDocker.OCScore.Analysis as ocanalysis

Modules (current / staged pipeline)
-----------------------------------
- Metrics: Screening and regression metric helpers.
- Plotting: Cross-validation, baseline, and metrics plots.
- SHAP: Pipeline-native SHAP for exported model bundles.


'''

# Imports
###############################################################################
from .Metrics.Ranking import evaluate_screening_metrics

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

__all__ = [
    "evaluate_screening_metrics",
]
