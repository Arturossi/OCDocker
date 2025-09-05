#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import importlib

import OCDocker.Docking.Smina as smina


def test_run_prepare_ligand_copy_fallback(tmp_path, monkeypatch):
    # Force copy fallback (pythonsh not available)
    monkeypatch.setattr(smina, 'pythonsh', '/nonexistent/pythonsh')
    # Minimal input ligand (mol2)
    in_mol = tmp_path / 'ligand.mol2'
    in_mol.write_text('mol2')
    out_pdbqt = tmp_path / 'out' / 'lig.pdbqt'

    rc = smina.run_prepare_ligand(str(in_mol), str(out_pdbqt))
    assert rc == 0
    assert out_pdbqt.exists()
    assert out_pdbqt.read_text() == 'mol2'

