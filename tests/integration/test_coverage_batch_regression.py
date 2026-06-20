#!/usr/bin/env python3

# Description
###############################################################################
'''
Targeted coverage batch for seven additional source files.
'''

# Imports
###############################################################################
import numpy as np
import pandas as pd

import pytest

from sqlalchemy import column

import OCDocker.DB.Models.Base as base_mod
import OCDocker.Docking.BaseVinaLike as ocbasevina
import OCDocker.Error as ocerror
import OCDocker.OCScore.Scoring as ocscoring
import OCDocker.Processing.Preprocessing.RMSDClustering as ocrmsd
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.Preparation as ocprep
import OCDocker.Toolbox.Printing as ocprinting
import OCDocker.Toolbox.Validation as ocvalidation

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


class _DummyPreparation(ocprep.PreparationStrategy):
    def prepare_ligand(self, input_path, output_path, log_file="", overwrite=False):
        _ = (input_path, output_path, log_file, overwrite)
        return ocerror.Error.ok()

    def prepare_receptor(self, input_path, output_path, log_file="", overwrite=False):
        _ = (input_path, output_path, log_file, overwrite)
        return ocerror.Error.ok()


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(443)
def test_preparation_helper_and_openbabel_smiles_branch(monkeypatch, tmp_path):
    helper = _DummyPreparation()

    in_file = tmp_path / "in.txt"
    in_file.write_text("abc", encoding="utf-8")
    out_file = tmp_path / "out.txt"

    # Shared helper success path
    rc_copy = helper._fallback_copy(str(in_file), str(out_file), "dummy")
    assert rc_copy == ocerror.Error.ok()
    assert out_file.read_text(encoding="utf-8") == "abc"

    # Shared helper error path
    monkeypatch.setattr(ocprep.shutil, "copyfile", lambda *_a, **_k: (_ for _ in ()).throw(OSError("deny")))
    rc_copy_err = helper._fallback_copy(str(in_file), str(out_file), "dummy")
    assert isinstance(rc_copy_err, int)
    assert rc_copy_err != ocerror.Error.ok()

    # Existing output handling
    existing = tmp_path / "already.pdbqt"
    existing.write_text("x", encoding="utf-8")
    rc_exists = helper._handle_existing_output(str(existing), overwrite=False, entity_label="ligand")
    assert rc_exists == ocerror.Error.ok()

    monkeypatch.setattr(ocprep.os, "remove", lambda *_a, **_k: (_ for _ in ()).throw(OSError("deny")))
    rc_overwrite = helper._handle_existing_output(str(existing), overwrite=True, entity_label="ligand")
    assert rc_overwrite is None

    # OpenBabel strategy smiles companion-mol2 missing branch
    strategy = ocprep.OpenBabelPreparationStrategy()
    smiles = tmp_path / "ligand.smi"
    smiles.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "ligand.pdbqt"

    monkeypatch.setattr(ocvalidation, "validate_obabel_extension", lambda _p: "smi")
    monkeypatch.setattr(ocprinting, "print_warning", lambda *_a, **_k: None)
    monkeypatch.setattr(ocprinting, "print_error", lambda *_a, **_k: None)

    rc_missing_companion = strategy.prepare_ligand(str(smiles), str(output), overwrite=True)
    assert rc_missing_companion == ocerror.Error.file_not_exist()


@pytest.mark.order(444)
def test_conversion_branches(monkeypatch, tmp_path):
    in_file = tmp_path / "mol_in.mol2"
    out_file = tmp_path / "mol_out.mol2"
    in_file.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    out_file.write_text("exists\n", encoding="utf-8")

    monkeypatch.setattr(occonversion.ocvalidation, "validate_obabel_extension", lambda _p: "mol2")
    rc_exists = occonversion.convert_mols(str(in_file), str(out_file), overwrite=False)
    assert rc_exists == ocerror.Error.file_exists()

    monkeypatch.setattr(occonversion.ocvalidation, "validate_obabel_extension", lambda _p: ocerror.Error.unsupported_extension())
    rc_bad_out = occonversion.convert_mols_from_string("CC", str(tmp_path / "bad.out"))
    assert rc_bad_out == ocerror.Error.unsupported_extension()

    class _OBConversionFail:
        def SetInAndOutFormats(self, *_a, **_k):
            return True

        def ReadFile(self, *_a, **_k):
            raise RuntimeError("forced error")

        def WriteFile(self, *_a, **_k):
            return True

    class _OBMolStub:
        def SetTitle(self, _title):
            return None

        def DeleteData(self, _key):
            return None

    monkeypatch.setattr(occonversion.ocvalidation, "validate_obabel_extension", lambda _p: "mol2")
    monkeypatch.setattr(occonversion.openbabel, "OBConversion", _OBConversionFail)
    monkeypatch.setattr(occonversion.openbabel, "OBMol", _OBMolStub)
    rc_subprocess = occonversion.convert_mols(str(in_file), str(tmp_path / "new_out.mol2"), overwrite=True)
    assert isinstance(rc_subprocess, int)
    assert rc_subprocess != ocerror.Error.ok()


@pytest.mark.order(445)
def test_db_base_dynamic_columns_and_opmap():
    TempCovModel = type("TempCovModel", (base_mod.Base,), {"__tablename__": "temp_cov_model"})

    assert TempCovModel.determine_column_type("countA").__class__.__name__.lower().startswith("integer")
    assert TempCovModel.determine_column_type("AFloatLike").__class__.__name__.lower().startswith("float")

    TempCovModel.add_dynamic_columns(["countB", "RingCount", "SomeFloat"])
    assert hasattr(TempCovModel, "countB")
    assert hasattr(TempCovModel, "SomeFloat")
    assert str(TempCovModel.__table__.c.countB.server_default.arg) == "0"

    expr = base_mod.OPMAP[">="](column("x"), 5)
    assert ">=" in str(expr)


@pytest.mark.order(446)
def test_scoring_model_validation_branches(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError):
        ocscoring.get_score(model_path=str(tmp_path / "missing_model.pkl"), data=pd.DataFrame())

    model_file = tmp_path / "model.pkl"
    model_file.write_text("x", encoding="utf-8")
    monkeypatch.setattr(ocscoring.os.path, "isfile", lambda _p: True)

    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: {"state_dict": {}})
    with pytest.raises(ValueError, match="state_dict"):
        ocscoring.get_score(model_path=str(model_file), enforce_reference_column_order=False)

    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: {"a": 1, "b": 2})
    with pytest.raises(ValueError, match="no model object"):
        ocscoring.get_score(model_path=str(model_file), enforce_reference_column_order=False)


@pytest.mark.order(447)
def test_rmsd_clustering_early_branches(monkeypatch):
    data = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]])
    rc = ocrmsd.cluster_rmsd(data, max_distance_threshold=1.0, min_distance_threshold=2.0)
    assert isinstance(rc, int)
    assert rc != ocerror.Error.ok()

    warnings = []
    monkeypatch.setattr(ocrmsd.ocprint, "print_warning", lambda msg: warnings.append(msg))
    one = pd.DataFrame([[0.0]])
    out = ocrmsd.cluster_rmsd(one)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [0.0]
    assert warnings


@pytest.mark.order(448)
def test_basevinalike_read_helpers(tmp_path):
    log_file = tmp_path / "vina.log"
    log_file.write_text(
        "header\n-----+------------+----------+----------+\n1 -8.00 0 0\n2 -7.50 0 0\n",
        encoding="utf-8",
    )

    best = ocbasevina._read_log_generic(
        str(log_file),
        scoring_key="vina_score",
        engine="vina",
        error_log="err.log",
        onlyBest=True,
    )
    assert isinstance(best, dict)
    assert len(best) == 1
    key = next(iter(best.keys()))
    assert key in {1, 2}
    assert "vina_score" in next(iter(best.values()))

    missing_rescore = ocbasevina._read_rescoring_log_generic(
        str(tmp_path / "missing_rescore.log"),
        start_string="Affinity:",
        engine="smina",
        error_log="err.log",
    )
    assert np.isnan(missing_rescore)


@pytest.mark.order(449)
def test_validation_safe_print_fallbacks(monkeypatch, capsys):
    printed = []

    class _PrintOnlyError:
        @staticmethod
        def print_error(msg):
            printed.append(msg)

    monkeypatch.setattr(ocvalidation, "ocprint", _PrintOnlyError)
    ocvalidation._safe_print_warning("warn-message")
    assert printed and printed[0].startswith("WARNING:")

    class _NoPrinting:
        pass

    monkeypatch.setattr(ocvalidation, "ocprint", _NoPrinting())
    ocvalidation._safe_print_warning("warn-stdout")
    ocvalidation._safe_print_error("err-stdout")
    captured = capsys.readouterr()
    assert "WARNING: warn-stdout" in captured.out
    assert "ERROR: err-stdout" in captured.out
