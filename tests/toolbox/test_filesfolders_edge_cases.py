#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for extra FilesFolders helpers.

Usage:

pytest tests/test_filesfolders_extra.py
'''

# Imports
###############################################################################
import json

import pytest

import OCDocker.Toolbox.FilesFolders as ocff

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

@pytest.mark.order(65)
def test_empty_docking_digest_json(tmp_path):
    out = tmp_path / "digest.json"
    dct = ocff.empty_docking_digest(str(out), overwrite=True, digestFormat="json")
    assert isinstance(dct, dict)
    assert out.exists()
    loaded = json.loads(out.read_text())
    # Minimal keys present
    assert "vina_affinity" in loaded and "PLANTS_TOTAL_SCORE" in loaded


@pytest.mark.order(64)
def test_safe_create_and_remove(tmp_path):
    d = tmp_path / "d"
    # Create new
    rc1 = ocff.safe_create_dir(str(d))
    assert isinstance(rc1, int)
    # Idempotent
    rc2 = ocff.safe_create_dir(d)
    assert isinstance(rc2, int)
    # Remove dir
    rc3 = ocff.safe_remove_dir(str(d))
    assert isinstance(rc3, int)
