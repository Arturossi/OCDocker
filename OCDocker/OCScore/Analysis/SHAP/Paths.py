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


@dataclass
class OutputPaths:
    """Container for SHAP analysis output file paths."""

    out_dir: str
    feature_importance_png: str
    beeswarm_png: str
    shap_values_npy: str
    shap_values_csv: Optional[str] = None


__all__ = ["OutputPaths"]
