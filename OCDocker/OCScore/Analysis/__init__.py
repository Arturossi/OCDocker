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
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

__all__ = [
    "evaluate_screening_metrics",
]
