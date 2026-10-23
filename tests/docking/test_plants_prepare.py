#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for PLANTS preparation fallbacks.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from pathlib import Path

import OCDocker.Docking.PLANTS as plants

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

@pytest.mark.order(71)
def test_plants_prepare_fails_without_spores(tmp_path, monkeypatch):
    '''Check that ligand and receptor preparation fail without writing outputs when SPORES is missing.'''

    import OCDocker.Error as ocerror
    import OCDocker.Toolbox.Preparation as ocprep

    def mock_get_config():
        class MockToolsConfig:
            spores = '/nonexistent/spores'
        class MockConfig:
            tools = MockToolsConfig()
        return MockConfig()

    # The SPORES strategy reads the executable from the Preparation module config
    monkeypatch.setattr(ocprep, 'get_config', mock_get_config)

    lig_in = tmp_path / 'ligand.mol2'
    lig_in.write_text('L')
    lig_out = tmp_path / 'prep' / 'ligand.mol2'

    rec_in = tmp_path / 'rec.pdb'
    rec_in.write_text('R')
    rec_out = tmp_path / 'prep' / 'receptor.mol2'

    rc_l = plants.run_prepare_ligand(str(lig_in), str(lig_out))
    rc_r = plants.run_prepare_receptor(str(rec_in), str(rec_out))
    assert rc_l == ocerror.ErrorCode.SUBPROCESS and not lig_out.exists()
    assert rc_r == ocerror.ErrorCode.SUBPROCESS and not rec_out.exists()
