#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Vina preparation helpers.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from pathlib import Path

import OCDocker.Docking.Vina as vina

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

@pytest.mark.order(96)
def test_vina_run_prepare_fails_without_mgltools(tmp_path, monkeypatch):
    '''Check that ligand and receptor preparation fail without writing outputs when MGLTools is missing.'''

    import OCDocker.Error as ocerror
    import OCDocker.Toolbox.Preparation as ocprep

    def mock_get_config():
        class MockToolsConfig:
            pythonsh = '/nonexistent/pythonsh'
            prepare_ligand = '/nonexistent/prepare_ligand4.py'
            prepare_receptor = '/nonexistent/prepare_receptor4.py'
        class MockConfig:
            tools = MockToolsConfig()
        return MockConfig()

    # The MGLTools strategy reads the executables from the Preparation module config
    monkeypatch.setattr(ocprep, 'get_config', mock_get_config)

    lig_in = tmp_path / 'ligand.mol2'
    lig_in.write_text('L')
    lig_out = tmp_path / 'prep' / 'ligand.pdbqt'

    rec_in = tmp_path / 'rec.pdb'
    rec_in.write_text('R')
    rec_out = tmp_path / 'prep' / 'receptor.pdbqt'

    rc_l = vina.run_prepare_ligand(str(lig_in), str(lig_out))
    rc_r = vina.run_prepare_receptor(str(rec_in), str(rec_out))
    assert rc_l == ocerror.ErrorCode.SUBPROCESS and not lig_out.exists()
    assert rc_r == ocerror.ErrorCode.SUBPROCESS and not rec_out.exists()
