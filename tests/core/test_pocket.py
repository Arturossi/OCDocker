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
    GLY A3 is 20 A away, so an 8 A pocket around the origin holds LYS,
    ASP and SER from two chains.
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

def _write_box(path: Path, center: tuple) -> None:
    '''Write a docking box file in the OCDocker format with the given centre.'''

    x, y, z = center
    path.write_text(
        "HEADER    CORNERS OF BOX\n"
        f"REMARK    CENTER (X Y Z)      {x:8.3f}{y:8.3f}{z:8.3f}\n"
        "REMARK    DIMENSIONS (X Y Z)    10.000  10.000  10.000\n"
    )

## Public ##

@pytest.fixture
def pocket_files(tmp_path):
    '''
    Fixture that writes a small receptor and a docking box centred at the origin.
    '''

    receptor_file = tmp_path / "receptor.pdb"
    _write_receptor(receptor_file)

    box_file = tmp_path / "box0.pdb"
    _write_box(box_file, (0.0, 0.0, 0.0))

    return {
        "receptor": receptor_file,
        "box": box_file,
        "center": (0.0, 0.0, 0.0),
        "dir": tmp_path,
    }


@pytest.mark.order(2948)
def test_select_pocket_residues_respects_radius(pocket_files):
    '''Only residues with a heavy atom within the radius of the centre are selected, in structure order.'''

    _, structure = ocr.load_mol(str(pocket_files["receptor"]), name="receptor", compute_sasa=False, clean=False, canonicalize_pdb=False)
    center = np.array([0.0, 0.0, 0.0])

    selected = ocpocket.select_pocket_residues(structure, center, radius=8.0)
    assert [(r.get_parent().id, r.id[1], r.get_resname()) for r in selected] == [("A", 1, "LYS"), ("A", 2, "ASP"), ("B", 1, "SER")]

    tight = ocpocket.select_pocket_residues(structure, center, radius=4.0)
    assert [r.get_resname() for r in tight] == ["LYS", "ASP"]


@pytest.mark.order(2949)
def test_select_pocket_residues_empty_when_far(pocket_files):
    '''A centre far from every residue gives an empty pocket.'''

    _, structure = ocr.load_mol(str(pocket_files["receptor"]), name="receptor", compute_sasa=False, clean=False, canonicalize_pdb=False)
    assert ocpocket.select_pocket_residues(structure, np.array([100.0, 100.0, 100.0]), radius=8.0) == []


@pytest.mark.order(2950)
def test_count_pocket_AAs(pocket_files):
    '''Pocket residues are counted per amino acid type.'''

    _, structure = ocr.load_mol(str(pocket_files["receptor"]), name="receptor", compute_sasa=False, clean=False, canonicalize_pdb=False)
    residues = ocpocket.select_pocket_residues(structure, np.array([0.0, 0.0, 0.0]), radius=8.0)

    counts = ocpocket.count_pocket_AAs(residues)
    assert counts["K"] == 1 and counts["D"] == 1 and counts["S"] == 1
    assert counts["G"] == 0 and counts["X"] == 0
    assert sum(counts.values()) == len(residues)


@pytest.mark.order(2951)
def test_compute_net_charge():
    '''Side-chain charges follow the Henderson-Hasselbalch equation without termini.'''

    assert ocpocket.compute_net_charge("K") == pytest.approx(1.0, abs=0.01)
    assert ocpocket.compute_net_charge("D") == pytest.approx(-1.0, abs=0.01)
    assert ocpocket.compute_net_charge("KD") == pytest.approx(0.0, abs=0.02)
    assert ocpocket.compute_net_charge("G") == 0.0
    assert 0.0 < ocpocket.compute_net_charge("H", ph=5.0) < 1.0


@pytest.mark.order(2952)
def test_count_hbond_atoms_only_lining_atoms(pocket_files):
    '''Only side-chain donor/acceptor atoms within the radius of the centre are counted.'''

    _, structure = ocr.load_mol(str(pocket_files["receptor"]), name="receptor", compute_sasa=False, clean=False, canonicalize_pdb=False)
    center = np.array([0.0, 0.0, 0.0])
    residues = ocpocket.select_pocket_residues(structure, center, radius=8.0)

    # LYS NZ (donor) and ASP OD1/OD2 (acceptors) are close; SER OG sits at 9.5 A
    assert ocpocket.count_hbond_atoms(residues, center, radius=8.0) == (1, 2)
    # With a larger radius SER OG counts as both donor and acceptor
    assert ocpocket.count_hbond_atoms(residues, center, radius=10.0) == (2, 3)


@pytest.mark.order(2953)
def test_read_box_center(pocket_files, tmp_path):
    '''The centre is read from an OCDocker docking box file.'''

    shifted = tmp_path / "shifted_box.pdb"
    _write_box(shifted, (32.698, 38.446, -58.497))

    assert np.allclose(ocpocket.read_box_center(str(pocket_files["box"])), [0.0, 0.0, 0.0])
    assert np.allclose(ocpocket.read_box_center(str(shifted)), [32.698, 38.446, -58.497])


@pytest.mark.order(2954)
def test_read_box_center_with_touching_columns(tmp_path):
    '''Fixed-width coordinates that run together are still split correctly.'''

    box_file = tmp_path / "touching_box.pdb"
    box_file.write_text("REMARK    CENTER (X Y Z)        10.468-136.627  16.563\n")

    assert np.allclose(ocpocket.read_box_center(str(box_file)), [10.468, -136.627, 16.563])


@pytest.mark.order(2947)
def test_read_box_center_without_center_line(tmp_path):
    '''A box file without a CENTER line cannot define a pocket.'''

    box_file = tmp_path / "no_center.pdb"
    box_file.write_text("HEADER    CORNERS OF BOX\n")

    assert ocpocket.read_box_center(str(box_file)) is None


@pytest.mark.order(2955)
def test_read_box_center_missing_file(tmp_path):
    '''A missing box file returns None.'''

    assert ocpocket.read_box_center(str(tmp_path / "missing_box.pdb")) is None


@pytest.mark.order(2956)
def test_pocket_invalid_with_malformed_center(pocket_files):
    '''A centre that is not three finite coordinates gives an invalid pocket.'''

    for center in [(0.0, 0.0), (0.0, float("nan"), 0.0), 12345]:
        pocket = ocpocket.Pocket(str(pocket_files["receptor"]), center, name="pocket", canonicalize_pdb=False)
        assert not pocket.is_valid()


@pytest.mark.order(2957)
def test_pocket_descriptors(pocket_files):
    '''A pocket built from its centre has the expected descriptors.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), pocket_files["center"], name="pocket", radius=8.0, canonicalize_pdb=False)

    assert pocket.is_valid()
    assert pocket.center == (0.0, 0.0, 0.0)
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


@pytest.mark.order(2958)
def test_pocket_default_radius(pocket_files):
    '''The default radius is 12 A, which reaches SER OG but not GLY A3.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), pocket_files["center"], name="pocket", canonicalize_pdb=False)

    assert pocket.radius == pytest.approx(12.0)
    assert pocket.residues == "KDS"
    assert (pocket.countHBondDonors, pocket.countHBondAcceptors) == (2, 3)


@pytest.mark.order(2959)
def test_pocket_accepts_box_file(pocket_files):
    '''A docking box file gives the same pocket as its centre.'''

    from_center = ocpocket.Pocket(str(pocket_files["receptor"]), pocket_files["center"], name="pocket", canonicalize_pdb=False)
    from_box = ocpocket.Pocket(str(pocket_files["receptor"]), str(pocket_files["box"]), name="pocket", canonicalize_pdb=False)

    assert from_box.is_valid()
    assert from_box.residue_ids == from_center.residue_ids
    assert from_box.center == from_center.center


@pytest.mark.order(2960)
def test_pocket_invalid_without_residues(pocket_files):
    '''A centre far from the receptor gives an invalid pocket.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), (100.0, 100.0, 100.0), name="pocket", canonicalize_pdb=False)
    assert not pocket.is_valid()


@pytest.mark.order(2961)
def test_pocket_invalid_with_empty_name(pocket_files):
    '''The pocket name is required.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), pocket_files["center"], name="", canonicalize_pdb=False)
    assert not pocket.is_valid()


@pytest.mark.order(2962)
def test_pocket_invalid_with_missing_receptor(tmp_path, pocket_files):
    '''A missing receptor file gives an invalid pocket.'''

    pocket = ocpocket.Pocket(str(tmp_path / "missing.pdb"), pocket_files["center"], name="pocket", canonicalize_pdb=False)
    assert not pocket.is_valid()


@pytest.mark.order(2963)
def test_pocket_json_roundtrip(pocket_files):
    '''Descriptors written with to_json are restored with from_json_descriptors.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), pocket_files["center"], name="pocket", radius=8.0, canonicalize_pdb=False)
    assert pocket.to_json() == 0

    json_file = pocket_files["dir"] / "pocket_descriptors.json"
    assert json_file.is_file()

    restored = ocpocket.Pocket(str(pocket_files["receptor"]), "", name="ignored", from_json_descriptors=str(json_file), canonicalize_pdb=False)
    assert restored.is_valid()
    assert restored.name == "pocket"
    assert restored.radius == pytest.approx(8.0)
    assert restored.center == pytest.approx((0.0, 0.0, 0.0))
    assert restored.residue_ids == pocket.residue_ids
    assert restored.residues == pocket.residues
    assert restored.get_descriptors() == pytest.approx(pocket.get_descriptors())


@pytest.mark.order(2964)
def test_pocket_to_json_respects_overwrite(pocket_files):
    '''to_json does not overwrite an existing file unless asked to.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), pocket_files["center"], name="pocket", canonicalize_pdb=False)
    assert pocket.to_json() == 0
    assert pocket.to_json() != 0
    assert pocket.to_json(overwrite=True) == 0


@pytest.mark.order(2965)
def test_read_descriptors_from_json_missing_keys(tmp_path):
    '''A json file lacking descriptor keys is rejected.'''

    json_file = tmp_path / "broken_descriptors.json"
    json_file.write_text(json.dumps({"Name": "pocket", "Radius": 12.0}))

    assert ocpocket.read_descriptors_from_json(str(json_file)) is None


@pytest.mark.order(2966)
def test_read_descriptors_from_json_rejects_ligand_defined_pocket(tmp_path, pocket_files):
    '''A json written for a ligand-defined pocket (Cutoff, ReferenceLigand) is not accepted.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), pocket_files["center"], name="pocket", canonicalize_pdb=False)
    data = {"Name": "pocket", "ReferenceLigand": "ligand.sdf", "Cutoff": 8.0, "Residues": ";".join(pocket.residue_ids), **pocket.get_descriptors()}
    json_file = tmp_path / "old_descriptors.json"
    json_file.write_text(json.dumps(data))

    assert ocpocket.read_descriptors_from_json(str(json_file)) is None


@pytest.mark.order(2967)
def test_read_descriptors_from_json_tuple_order(pocket_files):
    '''The tuple form follows Name, Center, Radius, Residues and then allDescriptors.'''

    pocket = ocpocket.Pocket(str(pocket_files["receptor"]), pocket_files["center"], name="pocket", radius=8.0, canonicalize_pdb=False)
    assert pocket.to_json() == 0

    data = ocpocket.read_descriptors_from_json(str(pocket_files["dir"] / "pocket_descriptors.json"))
    assert data is not None
    assert data[0] == "pocket"
    assert data[1] == pytest.approx([0.0, 0.0, 0.0])
    assert data[2] == pytest.approx(8.0)
    assert data[3] == "A:1::LYS;A:2::ASP;B:1::SER"
    assert len(data) == 4 + len(ocpocket.Pocket.allDescriptors)


@pytest.mark.order(2968)
def test_pocket_on_test_receptor():
    '''A pocket centred on a residue of the bundled test receptor is selected and valid.'''

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

    # Centre the pocket on the CA of the first standard residue
    first = next(r for r in structure[0].get_residues() if r.id[0] == ' ')
    center = tuple(float(v) for v in first["CA"].get_coord())

    pocket = ocpocket.Pocket(structure, center, name="pocket")
    assert pocket.is_valid()
    assert pocket.totalAALength >= 1
    assert f"{first.get_parent().id}:{first.id[1]}:{first.id[2].strip()}:{first.get_resname()}" in pocket.residue_ids
