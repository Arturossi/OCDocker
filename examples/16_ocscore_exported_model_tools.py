#!/usr/bin/env python3

# Description
###############################################################################
'''
Example: OCScore exported model tools.

CLI equivalents: ``ocdocker ocscore validate|load|retrain|cross-validate|plot|architecture-plot|shap|score``

See ``ocdocker ocscore --help`` or ``examples/README.md`` for usage.
'''

# Imports
###############################################################################
from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from OCDocker.OCScore.CLI import export_tools


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

# Functions
###############################################################################
## Public ##

if __name__ == "__main__":
    raise SystemExit(export_tools.main())
