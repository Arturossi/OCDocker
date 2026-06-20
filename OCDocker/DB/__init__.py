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
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
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
