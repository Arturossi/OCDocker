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
