#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import OCDocker.Docking.PLANTS as plants


def test_plants_prepare_copy_fallbacks(tmp_path, monkeypatch):
    # Force SPORES fallback (spores not available)
    monkeypatch.setattr(plants, 'spores', '/nonexistent/spores')

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

