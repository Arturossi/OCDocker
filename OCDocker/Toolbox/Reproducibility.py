#!/usr/bin/env python3

# Description
###############################################################################
'''
Reproducibility manifest helpers.

Usage:

import OCDocker.Toolbox.Reproducibility as ocrepro

manifest = ocrepro.generate_reproducibility_manifest()
_ = ocrepro.write_reproducibility_manifest("manifest.json")
'''

# Imports
###############################################################################
from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Dict

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

def generate_reproducibility_manifest(include_python_packages: bool = True) -> Dict[str, Any]:
    '''Generate a reproducibility manifest.

    Parameters
    ----------
    include_python_packages : bool, optional
        Whether to include installed Python package versions.

    Returns
    -------
    Dict[str, Any]
        Reproducibility manifest.
    '''

    # Keep CLI and Python API on the same implementation path.
    from OCDocker.CLI.__init__ import generate_reproducibility_manifest as _generate_manifest
    return _generate_manifest(include_python_packages=include_python_packages)


def write_reproducibility_manifest(output_path: str, include_python_packages: bool = True) -> Dict[str, Any]:
    '''Generate and write a reproducibility manifest to disk.

    Parameters
    ----------
    output_path : str
        Output JSON file path.
    include_python_packages : bool, optional
        Whether to include installed Python package versions.

    Returns
    -------
    Dict[str, Any]
        The manifest payload written to disk.
    '''

    manifest = generate_reproducibility_manifest(include_python_packages=include_python_packages)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def generate_manifest(include_python_packages: bool = True) -> Dict[str, Any]:
    '''Alias for ``generate_reproducibility_manifest``.

    Parameters
    ----------
    include_python_packages : bool, optional
        Whether to include installed Python package versions.

    Returns
    -------
    Dict[str, Any]
        Reproducibility manifest.
    '''

    return generate_reproducibility_manifest(include_python_packages=include_python_packages)
