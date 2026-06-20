#!/usr/bin/env python3

# Description
###############################################################################
'''
Output path container for export SHAP artifacts.
'''

# Imports
###############################################################################
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


@dataclass
class OutputPaths:
    """Container for SHAP analysis output file paths."""

    out_dir: str
    feature_importance_png: str
    beeswarm_png: str
    shap_values_npy: str
    shap_values_csv: Optional[str] = None


__all__ = ["OutputPaths"]
