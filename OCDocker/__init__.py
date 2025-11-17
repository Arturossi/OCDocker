"""
The main OCDocker package

Packages
--------
- DB: Database management for OCDocker.
- Docking: Docking routines.
- Processing: From pre to post processing data in OCDocker.
- Rescoring: Rescoring routines.
- Toolbox: Toolbox for OCDocker.
"""

# Keep in sync with pyproject.toml
__version__ = "0.11.1"

# Public API: main package doesn't export modules directly
# Users should import from subpackages: e.g., `import OCDocker.Ligand as ocl`
__all__ = ['__version__']
