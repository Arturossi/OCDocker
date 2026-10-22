#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Pocket selection, descriptors and json caching.

Usage:

pytest tests/core/test_pocket.py
'''

# Imports
###############################################################################
import json
import pytest

import numpy as np

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import OCDocker.Pocket as ocpocket
import OCDocker.Receptor as ocr

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

def _pdb_atom(serial: int, name: str, resname: str, chain: str, resnum: int, x: float, y: float, z: float, element: str) -> str:
    '''Format one PDB ATOM record.'''

    return (
        f"ATOM  {serial:5d} {name:<4s} {resname:>3s} {chain}{resnum:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )

def _write_receptor(path: Path) -> None:
    '''Write a small receptor with residues at known distances from the origin.

    LYS A1 and ASP A2 sit next to the origin, SER B1 is 7 A away and
    GLY A3 is 20 A away, so an 8 A pocket around a ligand at the origin
    holds LYS, ASP and SER from two chains.
    '''

    residues = [
        ("LYS", "A", 1, [("N", 2.0, 0.0, 0.0, "N"), ("CA", 3.0, 0.0, 0.0, "C"), ("NZ", 1.5, 0.5, 0.0, "N")]),
        ("ASP", "A", 2, [("N", 0.0, 2.5, 0.0, "N"), ("CA", 0.0, 3.5, 0.0, "C"), ("OD1", 0.0, 2.0, 0.5, "O"), ("OD2", 0.0, 2.0, -0.5, "O")]),
        ("GLY", "A", 3, [("N", 20.0, 0.0, 0.0, "N"), ("CA", 21.0, 0.0, 0.0, "C")]),
        ("SER", "B", 1, [("N", 0.0, 0.0, 7.0, "N"), ("CA", 0.0, 0.0, 8.0, "C"), ("OG", 0.0, 0.0, 9.5, "O")]),
    ]
    lines = ["HEADER    TEST POCKET\n"]
    serial = 1
    for resname, chain, resnum, atoms in residues:
        for name, x, y, z, element in atoms:
            lines.append(_pdb_atom(serial, name, resname, chain, resnum, x, y, z, element))
            serial += 1
    path.write_text("".join(lines))

def _ligand_at(coordinates: list) -> Chem.rdchem.Mol:
    '''Build a carbon chain whose heavy atoms sit at the given coordinates.'''

    mol = Chem.RWMol()
    for _ in coordinates:
        mol.AddAtom(Chem.Atom(6))
    for i in range(len(coordinates) - 1):
        mol.AddBond(i, i + 1, Chem.BondType.SINGLE)
    conformer = Chem.Conformer(len(coordinates))
    for i, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(i, Point3D(x, y, z))
    mol.AddConformer(conformer, assignId=True)
    return mol.GetMol()

## Public ##

@pytest.fixture
def pocket_files(tmp_path):
    '''
    Fixture that writes a small receptor and a reference ligand at the origin.
    '''

    receptor_file = tmp_path / "receptor.pdb"
    _write_receptor(receptor_file)

    ligand = _ligand_at([(0.0, 0.0, 0.0)])
    ligand_file = tmp_path / "reference_ligand.mol"
    Chem.MolToMolFile(ligand, str(ligand_file))

    return {
        "receptor": receptor_file,
        "ligand": ligand_file,
        "ligand_mol": ligand,
        "dir": tmp_path,
    }


@pytest.mark.order(2950)
def test_select_pocket_residues_respects_cutoff(pocket_files):
    '''Only residues with a heavy atom within the cutoff are selected, in structure order.'''

    _, structure = ocr.load_mol(str(pocket_files["receptor"]), name="receptor", compute_sasa=False, clean=False, canonicalize_pdb=False)
    coordinates = np.array([[0.0, 0.0, 0.0]])

    selected = ocpocket.select_pocket_residues(structure, coordinates, cutoff=8.0)
    assert [(r.get_parent().id, r.id[1], r.get_resname()) for r in selected] == [("A", 1, "LYS"), ("A", 2, "ASP"), ("B", 1, "SER")]

    tight = ocpocket.select_pocket_residues(structure, coordinates, cutoff=4.0)
    assert [r.get_resname() for r in tight] == ["LYS", "ASP"]


@pytest.mark.order(2951)
def test_select_pocket_residues_empty_when_far(pocket_files):
    '''A ligand far from every residue gives an empty pocket.'''

    _, structure = ocr.load_mol(str(pocket_files["receptor"]), name="receptor", compute_sasa=False, clean=False, canonicalize_pdb=False)
    assert ocpocket.select_pocket_residues(structure, np.array([[100.0, 100.0, 100.0]]), cutoff=8.0) == []


@pytest.mark.order(2952)
def test_count_pocket_AAs(pocket_files):
    '''Pocket residues are counted per amino acid type.'''

    _, structure = ocr.load_mol(str(pocket_files["receptor"]), name="receptor", compute_sasa=False, clean=False, canonicalize_pdb=False)
    residues = ocpocket.select_pocket_residues(structure, np.array([[0.0, 0.0, 0.0]]), cutoff=8.0)

    counts = ocpocket.count_pocket_AAs(residues)
    assert counts["K"] == 1 and counts["D"] == 1 and counts["S"] == 1
    assert counts["G"] == 0 and counts["X"] == 0
    assert sum(counts.values()) == len(residues)


@pytest.mark.order(2953)
def test_compute_net_charge():
    '''Side-chain charges follow the Henderson-Hasselbalch equation without termini.'''

    assert ocpocket.compute_net_charge("K") == pytest.approx(1.0, abs=0.01)
    assert ocpocket.compute_net_charge("D") == pytest.approx(-1.0, abs=0.01)
    assert ocpocket.compute_net_charge("KD") == pytest.approx(0.0, abs=0.02)
    assert ocpocket.compute_net_charge("G") == 0.0
    assert 0.0 < ocpocket.compute_net_charge("H", ph=5.0) < 1.0


@pytest.mark.order(2954)
def test_count_hbond_atoms_only_lining_atoms(pocket_files):
    '''Only side-chain donor/acceptor atoms within the cutoff are counted.'''

    _, structure = ocr.load_mol(str(pocket_files["receptor"]), name="receptor", compute_sasa=False, clean=False, canonicalize_pdb=False)
    coordinates = np.array([[0.0, 0.0, 0.0]])
    residues = ocpocket.select_pocket_residues(structure, coordinates, cutoff=8.0)

    # LYS NZ (donor) and ASP OD1/OD2 (acceptors) are close; SER OG sits at 9.5 A
    assert ocpocket.count_hbond_atoms(residues, coordinates, cutoff=8.0) == (1, 2)
    # With a larger cutoff SER OG counts as both donor and acceptor
    assert ocpocket.count_hbond_atoms(residues, coordinates, cutoff=10.0) == (2, 3)


@pytest.mark.order(2955)
def test_get_ligand_heavy_atom_coordinates_excludes_hydrogens():
    '''Hydrogens are ignored and coordinates keep the conformer positions.'''

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=7)

    coordinates = ocpocket.get_ligand_heavy_atom_coordinates(mol)
    assert coordinates is not None
    assert coordinates.shape == (3, 3)


@pytest.mark.order(2956)
def test_get_ligand_heavy_atom_coordinates_from_file(pocket_files):
    '''A reference ligand file is read into its heavy-atom coordinates.'''

    coordinates = ocpocket.get_ligand_heavy_atom_coordinates(str(pocket_files["ligand"]))
    assert coordinates is not None
    assert np.allclose(coordinates, [[0.0, 0.0, 0.0]])


@pytest.mark.order(2957)
def test_get_ligand_heavy_atom_coordinates_without_conformer():
    '''A molecule without coordinates cannot define a pocket.'''

    assert ocpocket.get_ligand_heavy_atom_coordinates(Chem.MolFromSmiles("CCO")) is None


@pytest.mark.order(2958)
def test_get_ligand_heavy_atom_coordinates_wrong_type():
    '''Unsupported input types return None.'''

    assert ocpocket.get_ligand_heavy_atom_coordinates(12345) is None


@pytest.mark.order(2959)
def test_pocket_descriptors(pocket_files):
    '''A pocket built from a reference ligand has the expected descriptors.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), str(pocket_files["ligand"]), name="pocket", canonicalize_pdb=False)

    assert pocket.is_valid()
    assert pocket.residue_ids == ["A:1::LYS", "A:2::ASP", "B:1::SER"]
    assert pocket.residues == "KDS"

    descriptors = pocket.get_descriptors()
    assert set(descriptors) == set(ocpocket.Pocket.allDescriptors)
    assert descriptors["TotalAALength"] == 3
    assert descriptors["countChain"] == 2
    assert descriptors["countK"] == 1 and descriptors["countD"] == 1 and descriptors["countS"] == 1
    assert descriptors["countG"] == 0
    assert descriptors["NetCharge"] == pytest.approx(0.0, abs=0.02)
    assert descriptors["countHBondDonors"] == 1
    assert descriptors["countHBondAcceptors"] == 2
    assert descriptors["SASA"] > 0.0


@pytest.mark.order(2960)
def test_pocket_accepts_rdkit_mol(pocket_files):
    '''An RDKit Mol gives the same pocket as its file.'''

    from_file = ocpocket.Pocket(str(pocket_files["receptor"]), str(pocket_files["ligand"]), name="pocket", canonicalize_pdb=False)
    from_mol = ocpocket.Pocket(str(pocket_files["receptor"]), pocket_files["ligand_mol"], name="pocket", canonicalize_pdb=False)

    assert from_mol.is_valid()
    assert from_mol.residue_ids == from_file.residue_ids
    assert from_mol.reference_ligand_path == ""


@pytest.mark.order(2961)
def test_pocket_invalid_without_residues(pocket_files):
    '''A reference ligand far from the receptor gives an invalid pocket.'''

    far = pocket_files["dir"] / "far_ligand.mol"
    Chem.MolToMolFile(_ligand_at([(100.0, 100.0, 100.0)]), str(far))

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), str(far), name="pocket", canonicalize_pdb=False)
    assert not pocket.is_valid()


@pytest.mark.order(2962)
def test_pocket_invalid_with_empty_name(pocket_files):
    '''The pocket name is required.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), str(pocket_files["ligand"]), name="", canonicalize_pdb=False)
    assert not pocket.is_valid()


@pytest.mark.order(2963)
def test_pocket_invalid_with_missing_receptor(tmp_path, pocket_files):
    '''A missing receptor file gives an invalid pocket.'''

    pocket = ocpocket.Pocket(str(tmp_path / "missing.pdb"), str(pocket_files["ligand"]), name="pocket", canonicalize_pdb=False)
    assert not pocket.is_valid()


@pytest.mark.order(2964)
def test_pocket_json_roundtrip(pocket_files):
    '''Descriptors written with to_json are restored with from_json_descriptors.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), str(pocket_files["ligand"]), name="pocket", canonicalize_pdb=False)
    assert pocket.to_json() == 0

    json_file = pocket_files["dir"] / "pocket_descriptors.json"
    assert json_file.is_file()

    restored = ocpocket.Pocket(str(pocket_files["receptor"]), "", name="ignored", from_json_descriptors=str(json_file), canonicalize_pdb=False)
    assert restored.is_valid()
    assert restored.name == "pocket"
    assert restored.cutoff == pytest.approx(8.0)
    assert restored.residue_ids == pocket.residue_ids
    assert restored.residues == pocket.residues
    assert restored.get_descriptors() == pytest.approx(pocket.get_descriptors())


@pytest.mark.order(2965)
def test_pocket_to_json_respects_overwrite(pocket_files):
    '''to_json does not overwrite an existing file unless asked to.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), str(pocket_files["ligand"]), name="pocket", canonicalize_pdb=False)
    assert pocket.to_json() == 0
    assert pocket.to_json() != 0
    assert pocket.to_json(overwrite=True) == 0


@pytest.mark.order(2966)
def test_read_descriptors_from_json_missing_keys(tmp_path):
    '''A json file lacking descriptor keys is rejected.'''

    json_file = tmp_path / "broken_descriptors.json"
    json_file.write_text(json.dumps({"Name": "pocket", "Cutoff": 8.0}))

    assert ocpocket.read_descriptors_from_json(str(json_file)) is None


@pytest.mark.order(2967)
def test_read_descriptors_from_json_tuple_order(pocket_files):
    '''The tuple form follows Name, Cutoff, Residues and then allDescriptors.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), str(pocket_files["ligand"]), name="pocket", canonicalize_pdb=False)
    assert pocket.to_json() == 0

    data = ocpocket.read_descriptors_from_json(str(pocket_files["dir"] / "pocket_descriptors.json"))
    assert data is not None
    assert data[0] == "pocket"
    assert data[1] == pytest.approx(8.0)
    assert data[2] == "A:1::LYS;A:2::ASP;B:1::SER"
    assert len(data) == 3 + len(ocpocket.Pocket.allDescriptors)


@pytest.mark.order(2968)
def test_pocket_on_test_receptor():
    '''A pocket around a residue of the bundled test receptor is selected and valid.'''

    # Start from the current file location (assuming this code is in a test or module file)
    current_file = Path(__file__).resolve()

    # Traverse up to find the 'OCDocker' project root
    project_root = current_file
    while project_root.name != "OCDocker" and project_root != project_root.parent:
        project_root = project_root.parent

    if project_root.name != "OCDocker":
        raise RuntimeError("OCDocker directory not found in path hierarchy.")

    receptor_file = project_root / "test_files/test_ptn1/receptor.pdb"
    _, structure = ocr.load_mol(str(receptor_file), name="receptor", compute_sasa=False, clean=False, canonicalize_pdb=False)

    # Place the ligand on the CA of the first standard residue
    first = next(r for r in structure[0].get_residues() if r.id[0] == ' ')
    ligand = _ligand_at([tuple(float(v) for v in first["CA"].get_coord())])

    pocket = ocpocket.Pocket(structure, ligand, name="pocket")
    assert pocket.is_valid()
    assert pocket.totalAALength >= 1
    assert f"{first.get_parent().id}:{first.id[1]}:{first.id[2].strip()}:{first.get_resname()}" in pocket.residue_ids
