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

@pytest.mark.order(71)
def test_plants_prepare_copy_fallbacks(tmp_path, monkeypatch):
    # Force SPORES fallback (spores not available) by mocking Config
    from OCDocker.Config import get_config

    def mock_get_config():
        class MockToolsConfig:
            spores = '/nonexistent/spores'
        class MockConfig:
            tools = MockToolsConfig()
        return MockConfig()

    monkeypatch.setattr(plants, 'get_config', mock_get_config)

    lig_in = tmp_path / 'ligand.mol2'
    lig_in.write_text('L')
    lig_out = tmp_path / 'prep' / 'ligand.mol2'

    rec_in = tmp_path / 'rec.pdb'
    rec_in.write_text('R')
    rec_out = tmp_path / 'prep' / 'receptor.mol2'

    rc_l = plants.run_prepare_ligand(str(lig_in), str(lig_out))
    rc_r = plants.run_prepare_receptor(str(rec_in), str(rec_out))
    assert rc_l == 0 and lig_out.exists() and lig_out.read_text() == 'L'
    assert rc_r == 0 and rec_out.exists() and rec_out.read_text() == 'R'
