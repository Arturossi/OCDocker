#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for OCDocker.Toolbox.Validation helpers.
'''

# Imports
###############################################################################
import sys
import types

import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Validation as ocvalidation

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##


def _patch_rdkit(monkeypatch, *, mol2=None, sdf=None, mol=None, smi=None):
    '''Patch rdkit import used by Validation.is_molecule_valid for deterministic parser tests.'''

    fake_rdmolfiles = types.SimpleNamespace(
        MolFromMol2File=mol2 if mol2 is not None else (lambda *a, **k: object()),
        SDMolSupplier=sdf if sdf is not None else (lambda *a, **k: [object()]),
        MolFromMolFile=mol if mol is not None else (lambda *a, **k: object()),
        MolFromSmiles=smi if smi is not None else (lambda *a, **k: object()),
    )
    fake_chem = types.SimpleNamespace(rdmolfiles=fake_rdmolfiles)
    fake_rdkit = types.SimpleNamespace(Chem=fake_chem)
    monkeypatch.setitem(sys.modules, "rdkit", fake_rdkit)

## Public ##

@pytest.mark.order(1)
@pytest.mark.parametrize("path,expected", [
    ("/tmp/ap", True),
    ("/tmp/not_allowed", False),
])
def test_is_algorithm_allowed(path, expected):
    assert ocvalidation.is_algorithm_allowed(path) is expected


@pytest.mark.order(6)
def test_is_molecule_valid_bad_extension(tmp_path):
    bad = tmp_path / "dummy.xyz"
    bad.write_text("dummy")
    assert not ocvalidation.is_molecule_valid(str(bad))


@pytest.mark.order(5)
def test_is_molecule_valid_missing_file(tmp_path):
    missing = tmp_path / "missing.pdb"
    assert not ocvalidation.is_molecule_valid(str(missing))


@pytest.mark.order(4)
def test_is_molecule_valid_pdb():
    pytest.importorskip("Bio.PDB")
    from pathlib import Path
    # Get absolute path to receptor file
    test_dir = Path(__file__).resolve().parent.parent.parent
    path = test_dir / "test_files/test_ptn1/receptor.pdb"
    assert ocvalidation.is_molecule_valid(str(path))


@pytest.mark.order(2)
def test_validate_digest_extension():
    # valid extension
    assert ocvalidation.validate_digest_extension("results.json", "json")
    # invalid extension should return False after warning
    assert not ocvalidation.validate_digest_extension("results.hdf5", "hdf5")


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "file_path,expected",
    [
        ("molecule.smi", "smi"),
        ("molecule.bad", ocerror.ErrorCode.UNSUPPORTED_EXTENSION),
    ],
)
def test_validate_obabel_extension(file_path, expected):
    result = ocvalidation.validate_obabel_extension(file_path)
    if isinstance(expected, str):
        assert result == expected
    else:
        assert result == expected


@pytest.mark.order(7)
def test_is_molecule_valid_mol2_none_parse(monkeypatch, tmp_path):
    path = tmp_path / "bad.mol2"
    path.write_text("@<TRIPOS>MOLECULE\nBAD\n")
    _patch_rdkit(monkeypatch, mol2=lambda *a, **k: None)
    assert not ocvalidation.is_molecule_valid(str(path))


@pytest.mark.order(8)
def test_is_molecule_valid_mol_none_parse(monkeypatch, tmp_path):
    path = tmp_path / "bad.mol"
    path.write_text("bad mol content")
    _patch_rdkit(monkeypatch, mol=lambda *a, **k: None)
    assert not ocvalidation.is_molecule_valid(str(path))


@pytest.mark.order(9)
def test_is_molecule_valid_sdf_all_none_parse(monkeypatch, tmp_path):
    path = tmp_path / "bad.sdf"
    path.write_text("bad sdf content\n$$$$\n")
    _patch_rdkit(monkeypatch, sdf=lambda *a, **k: [None, None])
    assert not ocvalidation.is_molecule_valid(str(path))


@pytest.mark.order(10)
def test_is_molecule_valid_sdf_with_one_valid_molecule(monkeypatch, tmp_path):
    path = tmp_path / "mixed.sdf"
    path.write_text("mixed sdf content\n$$$$\n")
    _patch_rdkit(monkeypatch, sdf=lambda *a, **k: [None, object()])
    assert ocvalidation.is_molecule_valid(str(path))


@pytest.mark.order(11)
def test_is_molecule_valid_smiles_none_parse(monkeypatch, tmp_path):
    path = tmp_path / "bad.smi"
    path.write_text("C1=CC=CC=C1\n")
    _patch_rdkit(monkeypatch, smi=lambda *a, **k: None)
    assert not ocvalidation.is_molecule_valid(str(path))


@pytest.mark.order(12)
def test_is_molecule_valid_sdf_supplier_none(monkeypatch, tmp_path):
    path = tmp_path / "supplier_none.sdf"
    path.write_text("bad sdf\n$$$$\n", encoding="utf-8")
    _patch_rdkit(monkeypatch, sdf=lambda *a, **k: None)
    assert not ocvalidation.is_molecule_valid(str(path))


@pytest.mark.order(13)
def test_is_molecule_valid_smiles_empty_file(monkeypatch, tmp_path):
    path = tmp_path / "empty.smi"
    path.write_text("", encoding="utf-8")
    _patch_rdkit(monkeypatch, smi=lambda *a, **k: object())
    assert not ocvalidation.is_molecule_valid(str(path))


@pytest.mark.order(14)
def test_is_molecule_valid_pdbqt_openbabel_readfile_false(monkeypatch, tmp_path):
    path = tmp_path / "bad.pdbqt"
    path.write_text("REMARK PDBQT", encoding="utf-8")
    _patch_rdkit(monkeypatch)

    class _OBMol:
        def NumAtoms(self):
            return 10

    class _OBConversion:
        def SetInFormat(self, _fmt):
            return True

        def ReadFile(self, _mol, _path):
            return False

    fake_ob = types.ModuleType("openbabel")
    fake_ob.openbabel = types.SimpleNamespace(OBConversion=_OBConversion, OBMol=_OBMol)
    monkeypatch.setitem(sys.modules, "openbabel", fake_ob)

    assert not ocvalidation.is_molecule_valid(str(path))


@pytest.mark.order(15)
def test_is_molecule_valid_pdbqt_openbabel_zero_atoms(monkeypatch, tmp_path):
    path = tmp_path / "zero_atoms.pdbqt"
    path.write_text("REMARK PDBQT", encoding="utf-8")
    _patch_rdkit(monkeypatch)

    class _OBMol:
        def NumAtoms(self):
            return 0

    class _OBConversion:
        def SetInFormat(self, _fmt):
            return True

        def ReadFile(self, _mol, _path):
            return True

    fake_ob = types.ModuleType("openbabel")
    fake_ob.openbabel = types.SimpleNamespace(OBConversion=_OBConversion, OBMol=_OBMol)
    monkeypatch.setitem(sys.modules, "openbabel", fake_ob)

    assert not ocvalidation.is_molecule_valid(str(path))


@pytest.mark.order(16)
def test_is_molecule_valid_with_retry_succeeds_after_initial_empty_file(monkeypatch, tmp_path):
    path = tmp_path / "retry.pdbqt"
    path.write_text("CONTENT", encoding="utf-8")

    sizes = [0, 32]
    sleeps = []
    checks = {"valid_calls": 0}

    monkeypatch.setattr(ocvalidation.os.path, "getsize", lambda *_a, **_k: sizes.pop(0))
    monkeypatch.setattr(
        ocvalidation,
        "is_molecule_valid",
        lambda *_a, **_k: checks.__setitem__("valid_calls", checks["valid_calls"] + 1) or True,
    )
    monkeypatch.setattr(ocvalidation.time, "sleep", lambda delay: sleeps.append(delay))

    assert ocvalidation.is_molecule_valid_with_retry(str(path), retries=2, delay=0.2)
    assert checks["valid_calls"] == 1
    assert sleeps == [0.2]
