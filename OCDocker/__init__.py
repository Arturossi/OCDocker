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


# Functions
###############################################################################
## Private ##

## Public ##

# Single source of truth for version (pyproject.toml reads this dynamically)
from ._version import __version__

# Public API: main package doesn't export modules directly
# Users should import from subpackages: e.g., `import OCDocker.Ligand as ocl`
__all__ = ['__version__']
