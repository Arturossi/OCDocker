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
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Public ##

if __name__ == "__main__":
    raise SystemExit(export_tools.main())
