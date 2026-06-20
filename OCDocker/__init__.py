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
## Private ##

## Public ##

# Single source of truth for version (pyproject.toml reads this dynamically)
from ._version import __version__

# Public API: main package doesn't export modules directly
# Users should import from subpackages: e.g., `import OCDocker.Ligand as ocl`
__all__ = ['__version__']
