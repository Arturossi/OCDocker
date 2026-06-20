#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Smina preparation helpers.
'''

# Imports
###############################################################################
from __future__ import annotations

import importlib

import pytest

from pathlib import Path

import OCDocker.Docking.Smina as smina

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

@pytest.mark.order(94)
def test_run_prepare_ligand_copy_fallback(tmp_path, monkeypatch):
    # Force copy fallback (pythonsh not available) by mocking Config
    from OCDocker.Config import get_config
    
    def mock_get_config():
        class MockToolsConfig:
            pythonsh = '/nonexistent/pythonsh'
            prepare_ligand = '/nonexistent/prepare_ligand4.py'
        class MockConfig:
            tools = MockToolsConfig()
        return MockConfig()
    
    monkeypatch.setattr(smina, 'get_config', mock_get_config)
    # Minimal input ligand (mol2)
    in_mol = tmp_path / 'ligand.mol2'
    in_mol.write_text('mol2')
    out_pdbqt = tmp_path / 'out' / 'lig.pdbqt'

    rc = smina.run_prepare_ligand(str(in_mol), str(out_pdbqt))
    assert rc == 0
    assert out_pdbqt.exists()
    assert out_pdbqt.read_text() == 'mol2'
