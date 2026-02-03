#!/usr/bin/env python3

# Description
###############################################################################
'''
Database package.

Usage:

import OCDocker.DB as ocdb

Modules
-------
- baseDB: Base database helpers and shared workflows.
- DB: Database creation and ORM utilities.
- DBMinimal: Minimal database helpers.
- DUDEz: DUDE-Z dataset helpers.
- PDBbind: PDBbind dataset helpers.
- Models: ORM model definitions.
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
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

# Expose submodules so that Sphinx autodoc can import as `from OCDocker.DB import DB`.
try:  # optional during docs build
    from . import DB as DB  # type: ignore
except Exception:
    pass

__all__ = [
    'DB',
]
