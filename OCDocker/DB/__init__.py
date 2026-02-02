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

# Expose submodules so that Sphinx autodoc can import as `from OCDocker.DB import DB`.
try:  # optional during docs build
    from . import DB as DB  # type: ignore
except Exception:
    pass

__all__ = [
    'DB',
]
