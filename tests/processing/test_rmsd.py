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
