#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for RMSD helpers in MoleculeProcessing.
'''

# Imports
###############################################################################
import pytest

from rdkit import Chem
from rdkit.Chem import AllChem

import OCDocker.Toolbox.MoleculeProcessing as ocmolproc

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

@pytest.fixture
def example_mols(tmp_path):
    '''Create three conformers of the same molecule and write to SDF files.'''
    # Three ethanol conformers with different embeddings
    mol1 = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol1, randomSeed=0xf00d) # type: ignore

    mol2 = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol2, randomSeed=0xcafe) # type: ignore

    mol3 = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol3, randomSeed=0xdead) # type: ignore

    files = []
    for idx, mol in enumerate((mol1, mol2, mol3), start=1):
        path = tmp_path / f"mol{idx}.sdf"
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()
        files.append(str(path))

    return files


@pytest.mark.order(1)
def test_get_rmsd(example_mols):
    mol_path = example_mols[0]
    rmsd = ocmolproc.get_rmsd(mol_path, mol_path)
    if isinstance(rmsd, list):
        rmsd = rmsd[0]
    assert pytest.approx(0.0, abs=1e-3) == rmsd


@pytest.mark.order(2)
def test_get_rmsd_matrix_symmetry(example_mols):
    matrix = ocmolproc.get_rmsd_matrix(example_mols)
    for i, m1 in enumerate(example_mols):
        for j, m2 in enumerate(example_mols):
            if i == j:
                assert matrix[m1][m2] == pytest.approx(0.0, abs=1e-3)
            else:
                assert matrix[m1][m2] == pytest.approx(matrix[m2][m1], abs=1e-6)


@pytest.fixture
def benzene_mols(tmp_path):
    '''Write two benzene conformers, whose heavy-atom graph has 12 automorphisms.'''
    files = []
    for idx, seed in enumerate((0xf00d, 0xcafe), start=1):
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=seed) # type: ignore
        path = tmp_path / f"benzene{idx}.sdf"
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()
        files.append(str(path))

    return files


@pytest.mark.order(2969)
def test_count_isomorphisms_stops_after_limit(benzene_mols):
    assert ocmolproc.count_isomorphisms(benzene_mols[0]) == 12
    assert ocmolproc.count_isomorphisms(benzene_mols[0], limit=5) == 6
    assert ocmolproc.count_isomorphisms(benzene_mols[0], limit=0) == 12


@pytest.mark.order(2970)
def test_get_rmsd_matrix_skips_highly_symmetric_ligand(benzene_mols, monkeypatch):
    monkeypatch.setattr(ocmolproc, "_MAX_RMSD_ISOMORPHISMS", 5)
    with pytest.raises(ocmolproc.TooManyIsomorphisms):
        ocmolproc.get_rmsd_matrix(benzene_mols)


@pytest.mark.order(2971)
def test_get_rmsd_matrix_zero_limit_disables_check(benzene_mols, monkeypatch):
    monkeypatch.setattr(ocmolproc, "_MAX_RMSD_ISOMORPHISMS", 0)
    monkeypatch.setattr(ocmolproc, "count_isomorphisms", lambda *_a, **_k: pytest.fail("count_isomorphisms called"))
    matrix = ocmolproc.get_rmsd_matrix(benzene_mols)
    assert matrix[benzene_mols[0]][benzene_mols[1]] == pytest.approx(matrix[benzene_mols[1]][benzene_mols[0]])
