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
