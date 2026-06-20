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
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
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
