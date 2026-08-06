#!/usr/bin/env python3

# Description
###############################################################################
'''
Second targeted coverage pass for Ligand and Receptor modules.
'''

# Imports
###############################################################################
import json
from pathlib import Path
from typing import Any

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
    current = Path(__file__).resolve()
    while current.name != "OCDocker" and current != current.parent:
        current = current.parent
    if current.name != "OCDocker":
        raise RuntimeError("OCDocker root not found")
    return current


def _simple_ligand(path: str, name: str = "lig_cov2") -> ocl.Ligand:
    lig = object.__new__(ocl.Ligand)
    lig.path = path
    lig.name = name
    lig.sanitize = True
    lig.from_json_descriptors = ""
    lig.box_path = str(Path(path).parent / "boxes" / "box0.pdb")
    lig.molecule = Chem.MolFromSmiles("CCO")

    for desc in ocl.Ligand.allDescriptors:
        setattr(lig, desc, 1.0)

    return lig


## Public ##

@pytest.fixture
def receptor_pdb_path() -> Path:
    return _project_root() / "test_files" / "test_ptn1" / "receptor.pdb"


@pytest.fixture
def receptor_structure(receptor_pdb_path: Path):
    return ocr.PDBParser().get_structure("rec_cov2", str(receptor_pdb_path))


def test_ligand_descriptor_factories_and_wrappers(monkeypatch, tmp_path):
    mol = Chem.MolFromSmiles("CCO")

    original = ocl.Descriptors.FpDensityMorgan1
    monkeypatch.setattr(ocl.Descriptors, "FpDensityMorgan1", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    compute = ocl.__descriptor_function_factory("FpDensityMorgan1")

    assert compute(mol) is None
    assert compute({"bad": True}) is None
    assert compute(None) is None

    monkeypatch.setattr(ocl.Descriptors, "FpDensityMorgan1", original)

    lig = _simple_ligand(str(tmp_path / "x.smi"))
    compute_cls = ocl.__descriptor_function_factory_class("FpDensityMorgan1")
    lig.molecule = {"bad": True}
    assert compute_cls(lig) is None
    lig.molecule = None
    assert compute_cls(lig) is None

    lig.molecule = Chem.MolFromSmiles("CCO")
    assert isinstance(ocl.findFpDensityMorgan1(lig.molecule), float)
    assert isinstance(ocl.findFpDensityMorgan2(lig.molecule), float)
    assert isinstance(ocl.findFpDensityMorgan3(lig.molecule), float)


def test_ligand_smiles_validity_and_same_smiles_exception(monkeypatch, tmp_path):
    lig = _simple_ligand(str(tmp_path / "lig.smi"))
    setattr(lig, ocl.Ligand.allDescriptors[0], None)
    assert lig.is_valid() is False

    assert isinstance(ocl.get_smiles("bad"), int)
    assert isinstance(ocl.get_smiles(None), int)

    monkeypatch.setattr(lig, "to_smiles", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert lig.is_same_molecule_SMILES(Chem.MolFromSmiles("CC")) is False


def test_ligand_get_centroid_failures(monkeypatch):
    monkeypatch.setattr(ocl, "load_mol", lambda *_a, **_k: ("", None))
    with pytest.raises(ValueError, match="Could not load molecule"):
        ocl.get_centroid("missing.smi")

    with pytest.raises(ValueError, match="not an RDKit Mol"):
        ocl.get_centroid(123)  # type: ignore[arg-type]

    mol = Chem.MolFromSmiles("CC")
    monkeypatch.setattr(ocl, "_ensure_3d_conformer", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="Could not generate a 3D conformer"):
        ocl.get_centroid(mol)


def test_ligand_optimize_and_embedding_helpers(monkeypatch):
    mol_no_conf = Chem.MolFromSmiles("CC")
    assert ocl._optimize_mol(mol_no_conf) is False

    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    assert AllChem.EmbedMolecule(mol) == 0

    monkeypatch.setattr(ocl.AllChem, "MMFFGetMoleculeProperties", lambda _m: object())
    monkeypatch.setattr(ocl.AllChem, "MMFFOptimizeMolecule", lambda *_a, **_k: 0)
    assert ocl._optimize_mol(mol) is True

    monkeypatch.setattr(ocl.AllChem, "MMFFGetMoleculeProperties", lambda _m: (_ for _ in ()).throw(RuntimeError("mmff")))
    monkeypatch.setattr(ocl.AllChem, "UFFOptimizeMolecule", lambda *_a, **_k: 0)
    assert ocl._optimize_mol(mol) is True

    monkeypatch.setattr(ocl.AllChem, "UFFOptimizeMolecule", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("uff")))
    assert ocl._optimize_mol(mol) is False

    # Each attempt now runs in a forked subprocess (see _embed_attempt_with_timeout),
    # so a plain-dict call counter mutated inside AllChem.EmbedMolecule would only
    # ever mutate the child's copy. Mock at the _embed_attempt_with_timeout boundary
    # instead, which is _try_embed_rdkit's actual retry-loop contract.
    state = {"calls": 0}

    def fake_embed_attempt(mol_obj, _etkdg_max_attempts, _seed):
        state["calls"] += 1
        if state["calls"] == 2:
            conf = Chem.Conformer(mol_obj.GetNumAtoms())
            conf.Set3D(True)
            mol_obj.AddConformer(conf)
            return True
        return False

    monkeypatch.setattr(ocl, "_embed_attempt_with_timeout", fake_embed_attempt)
    m2 = Chem.MolFromSmiles("CC")
    assert ocl._try_embed_rdkit(m2, max_attempts=3) is True
    assert state["calls"] == 2


def test_ligand_openbabel_builder_and_ensure_branches(monkeypatch):
    class OBConversionFailIn:
        def SetInFormat(self, _f):
            return False

        def SetOutFormat(self, _f):
            return True

    monkeypatch.setattr(ocl.openbabel, "OBConversion", OBConversionFailIn)
    assert ocl._openbabel_3d_from_smiles("CC", sanitize=True) is None

    class OBConversionFailOut:
        def SetInFormat(self, _f):
            return True

        def SetOutFormat(self, _f):
            return False

    monkeypatch.setattr(ocl.openbabel, "OBConversion", OBConversionFailOut)
    assert ocl._openbabel_3d_from_smiles("CC", sanitize=True) is None

    class OBConversionReadFail:
        def SetInFormat(self, _f):
            return True

        def SetOutFormat(self, _f):
            return True

        def ReadString(self, *_a, **_k):
            return False

    monkeypatch.setattr(ocl.openbabel, "OBConversion", OBConversionReadFail)
    monkeypatch.setattr(ocl.openbabel, "OBMol", lambda: object())
    assert ocl._openbabel_3d_from_smiles("CC", sanitize=True) is None

    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(False)
    mol.AddConformer(conf)

    monkeypatch.setattr(ocl, "_try_embed_rdkit", lambda *_a, **_k: False)
    monkeypatch.setattr(ocl.Chem, "MolToSmiles", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("smiles fail")))

    assert ocl._ensure_3d_conformer(mol, sanitize=True) is None


def test_ligand_load_mol_branch_matrix(monkeypatch, tmp_path):
    monkeypatch.setattr(ocl.occonversion, "convert_mols_from_string", lambda *_a, **_k: ocerror.Error.ok())

    rd = Chem.MolFromSmiles("CC")
    monkeypatch.setattr(ocl, "_ensure_3d_conformer", lambda *_a, **_k: None)
    p, m = ocl.load_mol(rd)
    assert p == "" and m is None

    monkeypatch.setattr(ocl, "_ensure_3d_conformer", lambda mol, **_k: Chem.AddHs(Chem.Mol(mol)))
    monkeypatch.setattr(ocl, "_optimize_mol", lambda *_a, **_k: False)
    p2, m2 = ocl.load_mol(Chem.MolFromSmiles("CC"))
    assert p2 == "" and m2 is not None

    comments = tmp_path / "comments.smi"
    comments.write_text("# only comments\n\n", encoding="utf-8")
    p3, m3 = ocl.load_mol(str(comments))
    assert p3 == "" and m3 is None

    broken_smiles = tmp_path / "broken.smi"
    broken_smiles.write_text("not_a_smiles\n", encoding="utf-8")
    p4, m4 = ocl.load_mol(str(broken_smiles))
    assert p4 == "" and m4 is None

    many = tmp_path / "many.smi"
    many.write_text("CC\nCCC\n", encoding="utf-8")
    monkeypatch.setattr(ocl, "_ensure_3d_conformer", lambda mol, **_k: Chem.AddHs(Chem.MolFromSmiles("CC")))
    p5, m5 = ocl.load_mol(str(many))
    assert p5.endswith("many.mol2")
    assert m5 is not None

    one = tmp_path / "one.smi"
    one.write_text("CC\n", encoding="utf-8")
    monkeypatch.setattr(ocl, "_ensure_3d_conformer", lambda *_a, **_k: None)
    p6, m6 = ocl.load_mol(str(one))
    assert p6 == "" and m6 is None

    sdf_path = tmp_path / "fake.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(Chem.MolFromSmiles("CC"))
    writer.write(Chem.MolFromSmiles("CCC"))
    writer.close()

    monkeypatch.setattr(ocl, "_ensure_3d_conformer", lambda mol, **_k: Chem.AddHs(Chem.MolFromSmiles("CC")))
    p7, m7 = ocl.load_mol(str(sdf_path))
    assert p7.endswith("fake.mol2") and m7 is not None

    mol2_path = tmp_path / "dummy.mol2"
    mol2_path.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    monkeypatch.setattr(ocl.Chem.rdmolfiles, "MolFromMol2File", lambda *_a, **_k: Chem.MolFromSmiles("CC"))
    monkeypatch.setattr(ocl, "_ensure_3d_conformer", lambda mol, **_k: Chem.AddHs(Chem.MolFromSmiles("CC")))
    p8, m8 = ocl.load_mol(str(mol2_path), sanitize=False)
    assert p8 == str(mol2_path)
    assert m8 is not None

    p9, m9 = ocl.load_mol(123)  # type: ignore[arg-type]
    assert p9 == "" and m9 is None


def test_ligand_multiple_sdf_and_json_and_split(monkeypatch, tmp_path):
    sdf_file = tmp_path / "multi.sdf"
    sdf_file.write_text("$$$$\n", encoding="utf-8")

    created = []

    class FakeLigand:
        def __init__(self, molecule, name, **_k):
            self.molecule = molecule
            self.name = name
            created.append(name)

    real_ligand_cls = ocl.Ligand
    real_split = ocl.split_molecules
    monkeypatch.setattr(ocl, "Ligand", FakeLigand)
    monkeypatch.setattr(ocl, "split_molecules", lambda *_a, **_k: ["a.mol2", "b.mol2"])

    ligs = ocl.multiple_molecules_sdf(str(sdf_file))
    assert len(ligs) == 2
    assert created == ["multi_1", "multi_2"]

    not_sdf = tmp_path / "not_sdf.txt"
    not_sdf.write_text("x", encoding="utf-8")
    assert ocl.multiple_molecules_sdf(str(not_sdf)) == []
    assert ocl.multiple_molecules_sdf(str(tmp_path / "missing.sdf")) == []

    mol = Chem.MolFromSmiles("CC")
    mol.SetProp("_Name", "named")
    out = ocl.multiple_molecules_sdf(mol)
    assert len(out) == 1

    assert ocl.multiple_molecules_sdf(123) == []  # type: ignore[arg-type]

    # Restore class for descriptor-json tests in this same function
    monkeypatch.setattr(ocl, "Ligand", real_ligand_cls)
    monkeypatch.setattr(ocl, "split_molecules", real_split)

    # read_descriptors_from_json missing keys path
    broken_json = tmp_path / "broken.json"
    broken_json.write_text("{}", encoding="utf-8")
    assert ocl.read_descriptors_from_json(str(broken_json)) is None

    # Force KeyError path with custom mapping behavior
    class SneakyDict(dict):
        def __contains__(self, _k):
            return True

        def __getitem__(self, key):
            if key == "RadiusOfGyration":
                raise KeyError("RadiusOfGyration")
            return super().__getitem__(key)

    valid = {"Name": "L"}
    for desc in ocl.Ligand.allDescriptors:
        valid[desc] = 1.0
    sneaky = SneakyDict(valid)

    monkeypatch.setattr(ocl.json, "load", lambda *_a, **_k: sneaky)
    key_path = tmp_path / "key.json"
    key_path.write_text("{}", encoding="utf-8")
    assert ocl.read_descriptors_from_json(str(key_path), return_data=False) is None

    # split_molecules invalid extension and write failure path
    monkeypatch.setattr(ocl.ocvalidation, "validate_obabel_extension", lambda *_a, **_k: 123)
    assert ocl.split_molecules(str(tmp_path / "x.bad")) == []

    class OBMol:
        pass

    class OBConv:
        def __init__(self):
            self._iter = 0

        def SetInAndOutFormats(self, *_a):
            return True

        def ReadFile(self, *_a):
            self._iter = 1
            return True

        def WriteFile(self, *_a):
            return False

        def Read(self, *_a):
            if self._iter == 1:
                self._iter = 2
                return False
            return False

    monkeypatch.setattr(ocl.ocvalidation, "validate_obabel_extension", lambda *_a, **_k: "smi")
    monkeypatch.setattr(ocl.openbabel, "OBConversion", OBConv)
    monkeypatch.setattr(ocl.openbabel, "OBMol", OBMol)

    source = tmp_path / "x.smi"
    source.write_text("CC\n", encoding="utf-8")
    assert ocl.split_molecules(str(source), output_dir=str(tmp_path / "out")) == []


def test_receptor_constructor_and_to_json_error_branches(monkeypatch, receptor_pdb_path, receptor_structure):
    clean_path = str(receptor_pdb_path).replace(".pdb", "_clean.pdb")
    monkeypatch.setattr(ocr, "load_mol", lambda *_a, **_k: (clean_path, receptor_structure))
    monkeypatch.setattr(ocr, "get_res", lambda *_a, **_k: "AAA")

    # JSON branch with invalid values (exercise _to_float/_to_int ValueError branches)
    tuple_data = (
        "rec", "bad_float", "bad_float", "bad_float", "bad_float", "bad_float", "bad_float", {"A": 1},
        "bad_int", "bad_int", "bad_int", "bad_int", "bad_int", "bad_int", "bad_int", "bad_int", "bad_int", "bad_int",
        "bad_int", "bad_int", "bad_int", "bad_int", "bad_int", "bad_int", "bad_int", "bad_int", "bad_int", "bad_int",
        "bad_int", "bad_float", "bad_int",
    )
    monkeypatch.setattr(ocr, "read_descriptors_from_json", lambda *_a, **_k: tuple_data)
    rec = ocr.Receptor(str(receptor_pdb_path), name="rec", from_json_descriptors="x.json")
    assert rec.clean_source_path.endswith(".pdb")

    # count_AAs_and_chains failure path
    monkeypatch.setattr(ocr, "read_descriptors_from_json", lambda *_a, **_k: None)
    monkeypatch.setattr(ocr, "count_AAs_and_chains", lambda *_a, **_k: None)
    rec2 = ocr.Receptor(str(receptor_pdb_path), name="rec2")
    assert rec2.is_valid() is False

    # allow_missing_surface=False branch
    monkeypatch.setattr(ocr, "count_AAs_and_chains", lambda *_a, **_k: (3, 3.0, 1))
    monkeypatch.setattr(ocr, "compute_sasa", lambda model, **_k: setattr(model, "sasa", 1.0))
    monkeypatch.setattr(ocr, "compute_dipole_moment", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(ocr, "compute_isoelectric_point", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(ocr, "compute_instability_index", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(ocr, "compute_gravy", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(ocr, "compute_aromaticity", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(ocr, "count_surface_AA", lambda *_a, **_k: None)
    rec3 = ocr.Receptor(str(receptor_pdb_path), name="rec3", allow_missing_surface=False)
    assert rec3.is_valid() is False

    # to_json inner/outer exception paths
    good = ocr.Receptor(str(receptor_pdb_path), name="rec_good", allow_missing_surface=True)
    monkeypatch.setattr(ocr.json, "dump", lambda *_a, **_k: (_ for _ in ()).throw(OSError("write fail")))
    assert good.to_json(overwrite=True) == ocerror.Error.write_file()
    good.path = None  # type: ignore[assignment]
    assert good.to_json(overwrite=True) == ocerror.Error.unknown()


def test_receptor_filter_dipole_sasa_and_surface_branches(monkeypatch, receptor_structure, tmp_path):
    ocr._warned_sequences.clear()
    assert ocr._filterSequence("AXX") == "A"

    monkeypatch.setattr(ocr.ocvalidation, "validate_obabel_extension", lambda *_a, **_k: 123)
    assert ocr.compute_dipole_moment(str(tmp_path / "a.bad")) is None

    class DummySR:
        def compute(self, *_a, **_k):
            return None

    monkeypatch.setattr(ocr.SASA, "ShrakeRupley", lambda **_k: DummySR())
    class DummyModel:
        id = "d"
    model = DummyModel()
    ocr.compute_sasa(model)
    assert hasattr(model, "sasa")

    # count_AAs_and_chains no-chain path
    empty_structure = ocr.Bio.PDB.Structure.Structure("empty")
    assert ocr.count_AAs_and_chains(empty_structure) is None

    # count_surface_AA absolute DSSP candidate and mmCIF branch
    class DummyTools:
        dssp = "/abs/path/mkdssp"

    class DummyConfig:
        tools = DummyTools()

    class DummyDSSP:
        def __init__(self, *_a, **_k):
            self.property_dict = {(0, 0): (0, "A", "", 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)}

    monkeypatch.setattr(ocr, "get_config", lambda: DummyConfig())
    monkeypatch.setattr(ocr.os.path, "isfile", lambda p: p == "/abs/path/mkdssp")
    monkeypatch.setattr(ocr.os, "access", lambda *_a, **_k: True)
    monkeypatch.setattr(ocr, "DSSP", DummyDSSP)

    out = ocr.count_surface_AA(receptor_structure, "test.cif", cutoff=0.2)
    assert out is not None and out["A"] == 1

    # fallback cleanup path where os.remove fails is swallowed
    class EmptyDSSP:
        def __init__(self, *_a, **_k):
            self.property_dict = {}

    calls = {"n": 0}

    class FailingSecondDSSP:
        def __init__(self, *_a, **_k):
            calls["n"] += 1
            if calls["n"] == 1:
                self.property_dict = {}
            else:
                raise PDBException("bad dssp")

    monkeypatch.setattr(ocr, "DSSP", FailingSecondDSSP)
    monkeypatch.setattr(ocr.ocrun, "run", lambda *_a, **_k: (0, ""))
    monkeypatch.setattr(ocr.os.path, "isfile", lambda _p: True)
    monkeypatch.setattr(ocr.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(ocr.os, "remove", lambda *_a, **_k: (_ for _ in ()).throw(OSError("deny")))

    assert ocr.count_surface_AA(receptor_structure, str(tmp_path / "fallback.unknown"), cutoff=0.0) is None


def test_receptor_load_mol_path_matrix(monkeypatch, receptor_pdb_path, receptor_structure, tmp_path):
    # Structure object branch: compute_sasa=False and clean=True with renumber None
    monkeypatch.setattr(ocr, "renumber_pdb_residues", lambda *_a, **_k: None)
    p0, s0 = ocr.load_mol(receptor_structure, compute_sasa=False, clean=True)
    assert p0 == "" and s0 is receptor_structure

    # Path branch with invalid canonicalize value and canonicalization failure
    monkeypatch.setattr(ocr.os.path, "isfile", lambda _p: True)

    class PdbParserStub:
        def get_structure(self, *_a, **_k):
            return receptor_structure

    monkeypatch.setattr(ocr, "PDBParser", lambda: PdbParserStub())
    monkeypatch.setattr(ocr.ocmolproc, "needs_canonical_pdb_fix", lambda *_a, **_k: True)
    monkeypatch.setattr(ocr.ocmolproc, "convert_pdb_charmm_to_canonical", lambda *_a, **_k: ocerror.Error.unknown())
    monkeypatch.setattr(ocr.ocmolproc, "clean_pdb_file", lambda *_a, **_k: ocerror.Error.ok())
    monkeypatch.setattr(ocr, "_clean_pdb_path", lambda _p: str(tmp_path / "cleaned.pdb"))
    monkeypatch.setattr(ocr, "renumber_pdb_residues", lambda s, *_a, **_k: s)
    monkeypatch.setattr(ocr, "compute_sasa", lambda *_a, **_k: None)

    p1, s1 = ocr.load_mol(str(receptor_pdb_path), canonicalize_pdb="invalid", clean=True)
    assert p1.endswith("cleaned.pdb")
    assert s1 is receptor_structure

    # Invalid canonicalize type branch + mol2 conversion + compute_sasa branch
    converted = []
    monkeypatch.setattr(ocr.occonversion, "convert_mols", lambda src, dst: converted.append((src, dst)) or ocerror.Error.ok())
    p2, s2 = ocr.load_mol(str(receptor_pdb_path), canonicalize_pdb=123, mol2_path=str(tmp_path / "r.mol2"), overwrite=True, clean=False)
    assert p2.endswith("receptor.pdb")
    assert s2 is receptor_structure
    assert converted

    # Unsupported extension branch
    unsupported = tmp_path / "r.xyz"
    unsupported.write_text("X", encoding="utf-8")
    p3, s3 = ocr.load_mol(str(unsupported), clean=False)
    assert p3 == "" and s3 is None

    # mmCIF branch including conversion and reparse
    mmcif = tmp_path / "r.cif"
    mmcif.write_text("data_x\n", encoding="utf-8")

    class MMCIFStub:
        def get_structure(self, *_a, **_k):
            return receptor_structure

    monkeypatch.setattr(ocr, "MMCIFParser", lambda: MMCIFStub())
    monkeypatch.setattr(ocr, "_convert_cif_to_pdb", lambda *_a, **_k: str(tmp_path / "r_from_cif.pdb"))
    monkeypatch.setattr(ocr.ocmolproc, "clean_pdb_file", lambda *_a, **_k: ocerror.Error.ok())

    p4, s4 = ocr.load_mol(str(mmcif), clean=True)
    assert p4.endswith("cleaned.pdb")
    assert s4 is receptor_structure
