#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for toolbox conversion utilities.

Usage:

pytest tests/test_conversion.py
'''

# Imports
###############################################################################
import math
import pytest

from rdkit import Chem
from rdkit.Chem import AllChem

import OCDocker.Toolbox.Constants as occ
import OCDocker.Toolbox.Conversion as occonversion

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

def _write_two_butanes_pdb(path):
    '''Write two n-butane molecules 20 angstroms apart as a PDB file; each has one rotatable bond.'''

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC.CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed = 1)
    conformer = mol.GetConformer()
    for idx in Chem.GetMolFrags(mol)[1]:
        pos = conformer.GetAtomPosition(idx)
        conformer.SetAtomPosition(idx, (pos.x + 20.0, pos.y, pos.z))
    Chem.MolToPDBFile(mol, str(path))


def _count_records(path, record):
    '''Count the PDBQT lines starting with the given record name.'''

    return sum(1 for line in path.read_text().splitlines() if line.startswith(record))

## Public ##

@pytest.mark.order(3)
def test_convert_mols_rigid_pdbqt_has_no_torsion_tree(tmp_path):
    pdb_path = tmp_path / "two_butanes.pdb"
    _write_two_butanes_pdb(pdb_path)
    flexible = tmp_path / "flexible.pdbqt"
    rigid = tmp_path / "rigid.pdbqt"

    assert occonversion.convert_mols(str(pdb_path), str(flexible)) == 0
    assert occonversion.convert_mols(str(pdb_path), str(rigid), rigid = True) == 0

    # The default keeps the ligand torsion tree; the rigid output has none
    assert _count_records(flexible, "ROOT") > 0
    for record in ("ROOT", "BRANCH", "TORSDOF"):
        assert _count_records(rigid, record) == 0
    assert _count_records(rigid, "ATOM") == _count_records(flexible, "ATOM")


@pytest.mark.order(4)
def test_convert_mols_rigid_pdbqt_combines_fragments(tmp_path):
    pdb_path = tmp_path / "two_butanes.pdb"
    _write_two_butanes_pdb(pdb_path)
    rigid = tmp_path / "rigid.pdbqt"

    assert occonversion.convert_mols(str(pdb_path), str(rigid), rigid = True) == 0

    # Both molecules end up in a single rigid block, as a receptor with ions or waters needs
    assert _count_records(rigid, "TER") == 1

@pytest.mark.order(2)
def test_convert_from_string_and_file(tmp_path):
    smiles = "CCO"
    out_from_str = tmp_path / "string_out.sdf"
    out_from_file = tmp_path / "file_out.sdf"

    # convert from SMILES string
    res_str = occonversion.convert_mols_from_string(smiles, str(out_from_str))
    assert res_str == 0 or res_str is True
    assert out_from_str.exists()

    # write smiles to file then convert
    smi_path = tmp_path / "mol.smi"
    smi_path.write_text(smiles)
    res_file = occonversion.convert_mols(str(smi_path), str(out_from_file))
    assert res_file == 0 or res_file is True
    assert out_from_file.exists()

@pytest.mark.order(1)
@pytest.mark.parametrize("kikd, order, factor", [
    (1.0, "nM", 1e9),
    (2.0, "uM", 1e6),
])
def test_kikd_to_deltag_various_orders(kikd, order, factor):
    expected = -occ.RJ * occ.ZERO_C_IN_K * math.log(kikd * factor)
    result = occonversion.kikd_to_deltag(kikd, T = occ.ZERO_C_IN_K, kikd_order = order)
    assert math.isclose(result, expected, rel_tol = 1e-5)
