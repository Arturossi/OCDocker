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

__all__ = [
    "evaluate_screening_metrics",
]
