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

__all__ = ["run_export_shap_analysis", "OutputPaths"]
