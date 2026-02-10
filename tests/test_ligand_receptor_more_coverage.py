#!/usr/bin/env python3

# Description
###############################################################################
'''
Additional targeted tests for Ligand and Receptor branch coverage.
'''

# Imports
###############################################################################
import os
from pathlib import Path

import pytest

from Bio.PDB.PDBExceptions import PDBException
from rdkit import Chem
from rdkit.Chem import AllChem

import OCDocker.Error as ocerror
import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr


# Functions
###############################################################################
## Private ##

def _project_root() -> Path:
    '''Return project root path.''' 

    current = Path(__file__).resolve()
    while current.name != "OCDocker" and current != current.parent:
        current = current.parent
    if current.name != "OCDocker":
        raise RuntimeError("OCDocker project root not found")
    return current


def _build_min_ligand(path: str, name: str = "lig_cov") -> ocl.Ligand:
    '''Build a minimal Ligand-like object for branch tests.'''

    lig = object.__new__(ocl.Ligand)
    lig.path = path
    lig.name = name
    lig.molecule = Chem.MolFromSmiles("CCO")
    lig.sanitize = True
    lig.from_json_descriptors = ""
    lig.box_path = os.path.join(os.path.dirname(path), "boxes/box0.pdb")

    for desc in ocl.Ligand.allDescriptors:
        setattr(lig, desc, 1.0)

    return lig


## Public ##

@pytest.fixture
def receptor_pdb_path() -> Path:
    '''Path to reference receptor test file.''' 

    return _project_root() / "test_files" / "test_ptn1" / "receptor.pdb"


@pytest.fixture
def receptor_structure(receptor_pdb_path: Path):
    '''Load reference receptor structure for helper tests.''' 

    structure = ocr.PDBParser().get_structure("rec_cov", str(receptor_pdb_path))
    return structure


def test_ligand_init_raises_when_load_fails(monkeypatch):
    '''Ligand constructor should raise if load_mol cannot create a molecule.'''

    monkeypatch.setattr(ocl, "load_mol", lambda *_a, **_k: ("", None))

    with pytest.raises(ValueError, match="could not be loaded"):
        ocl.Ligand("dummy.smi", name="bad")


def test_ligand_init_from_json_invalid_payload_raises(monkeypatch, tmp_path):
    '''Ligand constructor should raise when JSON descriptor payload is invalid.'''

    monkeypatch.setattr(ocl, "load_mol", lambda *_a, **_k: (str(tmp_path / "lig.mol2"), Chem.MolFromSmiles("CC")))
    monkeypatch.setattr(ocl, "read_descriptors_from_json", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="Problems while parsing json file"):
        ocl.Ligand("ignored.smi", name="bad_json", from_json_descriptors="broken.json")


def test_ligand_create_box_default_path_and_missing_radius(tmp_path):
    '''create_box should create default box dir and fail cleanly when radius is missing.'''

    lig_path = tmp_path / "input_ligand.smi"
    lig = _build_min_ligand(str(lig_path))
    lig.RadiusOfGyration = None

    rc = lig.create_box(centroid=(0.0, 1.0, 2.0), save_path="")

    assert isinstance(rc, int)
    assert rc != ocerror.Error.ok()
    assert (tmp_path / "boxes").exists()


def test_ligand_create_box_nonexistent_save_path(tmp_path):
    '''create_box should auto-create user-provided missing save_path.'''

    lig = _build_min_ligand(str(tmp_path / "lig.smi"))
    out_dir = tmp_path / "new_boxes_dir"
    rc = lig.create_box(centroid=(0.0, 0.0, 0.0), save_path=str(out_dir), overwrite=True)

    assert rc is None
    assert out_dir.exists()
    assert (out_dir / "box0.pdb").exists()


def test_ligand_create_box_centroid_error_branches(monkeypatch, tmp_path):
    '''create_box should handle centroid computation failures and malformed tuples.'''

    lig = _build_min_ligand(str(tmp_path / "lig.smi"))

    monkeypatch.setattr(lig, "get_centroid", lambda: (_ for _ in ()).throw(RuntimeError("centroid fail")))
    rc_fail = lig.create_box(save_path=str(tmp_path / "b1"))
    assert isinstance(rc_fail, int)

    rc_bad_len = lig.create_box(centroid=(1.0, 2.0), save_path=str(tmp_path / "b2"))
    assert isinstance(rc_bad_len, int)


def test_ligand_comparison_error_paths(monkeypatch, tmp_path):
    '''Comparison helpers should return expected fallback outputs on invalid inputs.'''

    lig = _build_min_ligand(str(tmp_path / "lig.smi"))

    rc = lig.is_same_molecule({"invalid": True})
    assert isinstance(rc, int)

    assert lig.is_same_molecule_SMILES(12345) is False

    monkeypatch.setattr(ocl, "load_mol", lambda *_a, **_k: ("", None))
    assert lig.is_same_molecule_SMILES(Chem.MolFromSmiles("CC")) is False


def test_ligand_to_json_branches(monkeypatch, tmp_path):
    '''Cover to_json path normalization, overwrite, write failure and outer exception branches.'''

    lig = _build_min_ligand(str(tmp_path / "lig.smi"), name="json_lig")

    # Empty base-dir branch (path without parent)
    lig.path = "json_lig.smi"
    monkeypatch.chdir(tmp_path)

    ok_rc = lig.to_json(overwrite=True)
    assert ok_rc == ocerror.Error.ok()

    exists_rc = lig.to_json(overwrite=False)
    assert exists_rc == ocerror.Error.file_exists()

    # Inner write exception branch
    import builtins
    real_open = builtins.open

    def broken_open(file, mode="r", *args, **kwargs):
        if str(file).endswith("json_lig_descriptors.json") and "w" in mode:
            raise OSError("write blocked")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", broken_open)
    write_rc = lig.to_json(overwrite=True)
    assert write_rc == ocerror.Error.write_file()

    # Outer exception branch
    lig.path = None  # type: ignore[assignment]
    unknown_rc = lig.to_json(overwrite=True)
    assert unknown_rc == ocerror.Error.unknown()


def test_ensure_3d_conformer_branches(monkeypatch):
    '''_ensure_3d_conformer should exercise existing-3D and fallback-failure paths.'''

    mol3d = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    embed = AllChem.EmbedMolecule(mol3d)
    assert embed == 0

    same = ocl._ensure_3d_conformer(mol3d, sanitize=True)
    assert same is not None
    assert same.GetNumConformers() > 0

    mol2d = Chem.MolFromSmiles("CC")
    monkeypatch.setattr(ocl, "_try_embed_rdkit", lambda *_a, **_k: False)
    monkeypatch.setattr(ocl, "_openbabel_3d_from_smiles", lambda *_a, **_k: None)

    none_result = ocl._ensure_3d_conformer(mol2d, sanitize=True, smiles_source="")
    assert none_result is None


def test_receptor_clean_path_and_cif_conversion_helper(receptor_structure, tmp_path):
    '''Cover helper path conversion and cif->pdb success/reuse branches.'''

    assert ocr._clean_pdb_path("/tmp/my_rec_clean.pdb") == "/tmp/my_rec_clean.pdb"
    assert ocr._clean_pdb_path("/tmp/my_rec.pdb").endswith("_clean.pdb")

    non_cif = ocr._convert_cif_to_pdb(str(tmp_path / "a.pdb"), receptor_structure)
    assert non_cif is None

    cif_path = tmp_path / "rec.cif"
    cif_path.write_text("data_test\n", encoding="utf-8")

    created = ocr._convert_cif_to_pdb(str(cif_path), receptor_structure, overwrite=True)
    assert created is not None
    assert Path(created).exists()

    reused = ocr._convert_cif_to_pdb(str(cif_path), receptor_structure, overwrite=False)
    assert reused == created


def test_receptor_convert_cif_failure_branch(monkeypatch, receptor_structure, tmp_path):
    '''_convert_cif_to_pdb should return None when PDBIO save fails.'''

    class FailingPDBIO:
        def set_structure(self, _structure):
            return None

        def save(self, _path):
            raise RuntimeError("forced save failure")

    monkeypatch.setattr(ocr, "PDBIO", lambda: FailingPDBIO())

    cif_path = tmp_path / "broken.cif"
    cif_path.write_text("data_test\n", encoding="utf-8")

    assert ocr._convert_cif_to_pdb(str(cif_path), receptor_structure, overwrite=True) is None


def test_receptor_count_aa_and_surface_missing_binary(monkeypatch, receptor_structure):
    '''Cover count_AAs_and_chains zero-chain error and missing DSSP executable branch.'''

    empty_structure = ocr.Bio.PDB.Structure.Structure("empty")
    assert ocr.count_AAs_and_chains(empty_structure) is None

    class DummyTools:
        dssp = ""

    class DummyConfig:
        tools = DummyTools()

    monkeypatch.setattr(ocr, "get_config", lambda: DummyConfig())
    monkeypatch.setattr(ocr.shutil, "which", lambda _name: None)
    monkeypatch.setattr(ocr.ocmolproc, "clean_for_dssp", lambda *_a, **_k: ocerror.Error.ok())

    result = ocr.count_surface_AA(receptor_structure, "missing_binary.pdb", cutoff=0.7)
    assert result is None


def test_receptor_count_surface_unknown_extension_and_fallback(monkeypatch, receptor_structure, tmp_path):
    '''Cover unknown extension handling, fallback command path and PDBException cleanup path.'''

    class DummyTools:
        dssp = "mkdssp"

    class DummyConfig:
        tools = DummyTools()

    class EmptyDSSP:
        def __init__(self, *_a, **_k):
            self.property_dict = {}

    monkeypatch.setattr(ocr, "get_config", lambda: DummyConfig())
    monkeypatch.setattr(ocr.shutil, "which", lambda _name: "/usr/bin/mkdssp")
    monkeypatch.setattr(ocr.ocmolproc, "clean_for_dssp", lambda *_a, **_k: ocerror.Error.ok())
    monkeypatch.setattr(ocr, "DSSP", EmptyDSSP)

    # First fallback: command failure
    monkeypatch.setattr(ocr.ocrun, "run", lambda *_a, **_k: 1)
    monkeypatch.setattr(ocr.os.path, "isfile", lambda _p: False)
    assert ocr.count_surface_AA(receptor_structure, "surface_weird.xyz", cutoff=0.1) is None

    # Second fallback: command ok + dssp file present but parser fails (PDBException branch)
    dssp_path = tmp_path / "surface_fail.dssp"
    dssp_path.write_text("DUMMY\n", encoding="utf-8")

    monkeypatch.setattr(ocr.ocrun, "run", lambda *_a, **_k: (0, ""))
    monkeypatch.setattr(ocr.os.path, "isfile", lambda _p: True)

    calls = {"n": 0}

    class TwoStageDSSP:
        def __init__(self, *_a, **_k):
            calls["n"] += 1
            if calls["n"] == 1:
                self.property_dict = {}
            else:
                raise PDBException("cannot parse dssp")

    monkeypatch.setattr(ocr, "DSSP", TwoStageDSSP)
    monkeypatch.setattr(ocr.os, "remove", lambda *_a, **_k: None)

    assert ocr.count_surface_AA(receptor_structure, str(tmp_path / "surface_fail.xyz"), cutoff=0.1) is None


def test_receptor_count_surface_unknown_aa_bucket(monkeypatch, receptor_structure):
    '''count_surface_AA should bucket unsupported amino-acid symbols into X.'''

    class DummyTools:
        dssp = "mkdssp"

    class DummyConfig:
        tools = DummyTools()

    class DSSPWithUnknownAA:
        def __init__(self, *_a, **_k):
            self.property_dict = {
                (0, 0): (0, "Z", "", 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            }

    monkeypatch.setattr(ocr, "get_config", lambda: DummyConfig())
    monkeypatch.setattr(ocr.shutil, "which", lambda _name: "/usr/bin/mkdssp")
    monkeypatch.setattr(ocr.ocmolproc, "clean_for_dssp", lambda *_a, **_k: ocerror.Error.ok())
    monkeypatch.setattr(ocr, "DSSP", DSSPWithUnknownAA)

    aas = ocr.count_surface_AA(receptor_structure, "protein.unknown", cutoff=0.0)
    assert aas is not None
    assert aas["X"] == 1


def test_receptor_constructor_branches(monkeypatch, receptor_structure, receptor_pdb_path):
    '''Cover Receptor constructor early-return and casting helpers in JSON mode.'''

    monkeypatch.setattr(ocr, "load_mol", lambda *_a, **_k: (str(receptor_pdb_path), receptor_structure))

    # Invalid descriptor payload branch
    monkeypatch.setattr(ocr, "read_descriptors_from_json", lambda *_a, **_k: None)
    rec_invalid = ocr.Receptor(str(receptor_pdb_path), name="r_invalid", from_json_descriptors="bad.json")
    assert rec_invalid.is_valid() is False

    # Valid tuple with mixed types to exercise _to_float/_to_int conversions
    tuple_data = (
        "r_json", "12.5", 1.0, "6.7", "5.4", "-0.2", "0.1", {"A": 1},
        True, 2.8, "3", "bad", 5, 6, 7, 8, 9, 10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, "10.9", "4.25", False,
    )
    monkeypatch.setattr(ocr, "read_descriptors_from_json", lambda *_a, **_k: tuple_data)

    rec_json = ocr.Receptor(str(receptor_pdb_path), name="r_json", from_json_descriptors="ok.json")
    assert rec_json.countA == 1
    assert rec_json.countR == 2
    assert rec_json.countN == 3
    assert rec_json.countD == 0
    assert rec_json.totalAALength == 10
    assert rec_json.avgAALength == 4.25
    assert rec_json.countChain == 0

    # load_mol failure branch
    monkeypatch.setattr(ocr, "load_mol", lambda *_a, **_k: ("", None))
    rec_none = ocr.Receptor("missing.pdb", name="r_none")
    assert rec_none.structure is None


def test_receptor_allow_missing_surface_and_is_valid_false(monkeypatch, receptor_structure, receptor_pdb_path):
    '''Cover allow_missing_surface fallback and explicit is_valid False branch.'''

    monkeypatch.setattr(ocr, "load_mol", lambda *_a, **_k: (str(receptor_pdb_path), receptor_structure))
    monkeypatch.setattr(ocr, "get_res", lambda *_a, **_k: "AAA")
    monkeypatch.setattr(ocr, "count_AAs_and_chains", lambda *_a, **_k: (3, 3.0, 1))
    monkeypatch.setattr(ocr, "compute_dipole_moment", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(ocr, "compute_isoelectric_point", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(ocr, "compute_instability_index", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(ocr, "compute_gravy", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(ocr, "compute_aromaticity", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(ocr, "count_surface_AA", lambda *_a, **_k: None)

    rec = ocr.Receptor(str(receptor_pdb_path), name="allow_surface", allow_missing_surface=True)
    assert rec.countA == 0
    assert rec.countV == 0

    rec.sasa = None
    assert rec.is_valid() is False
