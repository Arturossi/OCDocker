"""
Database package.

Subpackages
-----------
- Models: Models of the database.

Modules
-------
- baseDB: Base classes for database models and operations.
- DB: Responsible for the database creation and session management.
- DBMinimal: Minimalist database interface for lightweight use cases.
- DUDEz: Database Utility for Data Extraction and Analysis in DUDEz database.
- PDBbind: Database Utility for Data Extraction and Analysis in PDBbind database.

"""

# Expose submodules so that Sphinx autodoc can import as `from OCDocker.DB import DB`.
try:  # optional during docs build
    from . import DB as DB  # type: ignore
except Exception:
    pass

__all__ = [
    'DB',
]
