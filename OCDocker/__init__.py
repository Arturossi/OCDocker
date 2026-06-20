#!/usr/bin/env python3

# Description
###############################################################################
'''
The main OCDocker package.

Usage:

import OCDocker as ocdocker

Packages
--------
- CLI: Command-line interface helpers.
- DB: Database management utilities.
- Docking: Docking routines.
- OCScore: Scoring and ML utilities.
- Processing: Pre/post-processing workflows.
- Rescoring: Rescoring routines.
- Toolbox: Shared toolbox utilities.

Modules
-------
- Config: Configuration helpers.
- Error: Error reporting utilities.
- Initialise: Runtime initialization helpers.
- Ligand: Ligand model and descriptors.
- Receptor: Receptor model and descriptors.
'''

# Imports
###############################################################################

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
## Private ##

## Public ##

# Single source of truth for version (pyproject.toml reads this dynamically)
from ._version import __version__

# Public API: main package doesn't export modules directly
# Users should import from subpackages: e.g., `import OCDocker.Ligand as ocl`
__all__ = ['__version__']
