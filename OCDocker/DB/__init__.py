#!/usr/bin/env python3

# Description
###############################################################################
"""
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
"""

# Imports
###############################################################################
from types import ModuleType
from typing import Optional

# License
###############################################################################
"""OCDocker
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
"""

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

# Expose submodules so that Sphinx autodoc can import as `from OCDocker.DB import DB`.
DB_IMPORT_ERROR: Optional[ImportError] = None
DB: Optional[ModuleType]

try:  # optional during docs build
    from . import DB as DB
except ImportError as exc:
    DB = None
    DB_IMPORT_ERROR = exc

__all__ = [
    "DB",
    "DB_IMPORT_ERROR",
]
