#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path

import pytest

import OCDocker.Docking.Vina as vina


@pytest.mark.order(96)
def test_vina_run_prepare_copy_fallbacks(tmp_path, monkeypatch):
    # Force copy fallback (pythonsh not available) by mocking Config
    from OCDocker.Config import get_config
    
    def mock_get_config():
        class MockToolsConfig:
            pythonsh = '/nonexistent/pythonsh'
            prepare_ligand = '/nonexistent/prepare_ligand4.py'
            prepare_receptor = '/nonexistent/prepare_receptor4.py'
        class MockConfig:
            tools = MockToolsConfig()
        return MockConfig()
    
    monkeypatch.setattr(vina, 'get_config', mock_get_config)

    lig_in = tmp_path / 'ligand.mol2'
    lig_in.write_text('L')
    lig_out = tmp_path / 'prep' / 'ligand.pdbqt'

    rec_in = tmp_path / 'rec.pdb'
    rec_in.write_text('R')
    rec_out = tmp_path / 'prep' / 'receptor.pdbqt'

    rc_l = vina.run_prepare_ligand(str(lig_in), str(lig_out))
    rc_r = vina.run_prepare_receptor(str(rec_in), str(rec_out))
    assert rc_l == 0 and lig_out.exists() and lig_out.read_text() == 'L'
    assert rc_r == 0 and rec_out.exists() and rec_out.read_text() == 'R'
