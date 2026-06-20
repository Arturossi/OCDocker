#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage-focused tests for branch-heavy paths in ``OCDocker.Toolbox.Preparation``.
'''

# Imports
###############################################################################
from __future__ import annotations

import os
import types

from pathlib import Path

import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Preparation as ocprep

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


class _DummyPreparationStrategy(ocprep.PreparationStrategy):
    def prepare_ligand(self, input_path: str, output_path: str, log_file: str = "", overwrite: bool = False):
        _ = (input_path, output_path, log_file, overwrite)
        return ocerror.ErrorCode.OK

    def prepare_receptor(self, input_path: str, output_path: str, log_file: str = "", overwrite: bool = False):
        _ = (input_path, output_path, log_file, overwrite)
        return ocerror.ErrorCode.OK


# Functions
###############################################################################
## Private ##


def _config() -> types.SimpleNamespace:
    tools = types.SimpleNamespace(
        pythonsh="/bin/echo",
        prepare_ligand="prepare_ligand4.py",
        prepare_receptor="prepare_receptor4.py",
        spores="/bin/echo",
        obabel="/bin/echo",
    )
    return types.SimpleNamespace(tools=tools)


## Public ##


@pytest.mark.order(430)
def test_preparation_base_helper_branches(tmp_path, monkeypatch):
    strategy = _DummyPreparationStrategy()

    monkeypatch.setattr(ocprep, "is_tool_available", lambda exe: exe == "ok_tool")
    assert strategy._check_tool_available("ok_tool") is True
    assert strategy._check_tool_available("missing_tool") is False

    out_file = tmp_path / "nested" / "out.pdbqt"
    strategy._ensure_output_dir(str(out_file))
    assert out_file.parent.exists()

    input_file = tmp_path / "ligand.mol2"
    input_file.write_text("LIG", encoding="utf-8")
    rc_copy = strategy._fallback_copy(str(input_file), str(out_file), "dummy")
    assert rc_copy == ocerror.ErrorCode.OK
    assert out_file.read_text(encoding="utf-8") == "LIG"

    monkeypatch.setattr(
        ocprep.shutil,
        "copyfile",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("copy failure")),
    )
    rc_copy_fail = strategy._fallback_copy(str(input_file), str(out_file), "dummy")
    assert rc_copy_fail == ocerror.ErrorCode.SUBPROCESS

    assert strategy.get_ligand_command("in", "out") == []
    assert strategy.get_receptor_command("in", "out") == []


@pytest.mark.order(431)
def test_handle_existing_output_overwrite_skip_and_remove_error(tmp_path, monkeypatch):
    strategy = _DummyPreparationStrategy()
    output_path = tmp_path / "prepared.pdbqt"
    output_path.write_text("OLD", encoding="utf-8")

    warnings = []
    monkeypatch.setattr(ocprep, "print_warning", lambda msg: warnings.append(msg))

    rc_skip = strategy._handle_existing_output(str(output_path), overwrite=False, entity_label="ligand")
    assert rc_skip == ocerror.ErrorCode.OK
    assert warnings
    assert output_path.exists()

    rc_overwrite = strategy._handle_existing_output(str(output_path), overwrite=True, entity_label="ligand")
    assert rc_overwrite is None
    assert not output_path.exists()

    output_path.write_text("OLD_AGAIN", encoding="utf-8")
    monkeypatch.setattr(
        ocprep.os,
        "remove",
        lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("remove denied")),
    )
    rc_remove_error = strategy._handle_existing_output(str(output_path), overwrite=True, entity_label="ligand")
    assert rc_remove_error is None
    assert output_path.exists()


@pytest.mark.order(432)
def test_mgltools_get_commands_and_prepare_run_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(ocprep, "get_config", _config)
    strategy = ocprep.MGLToolsPreparationStrategy()

    cmd_l = strategy.get_ligand_command("lig.mol2", "lig.pdbqt")
    cmd_r = strategy.get_receptor_command("rec.pdb", "rec.pdbqt")
    assert cmd_l[:2] == ["/bin/echo", "prepare_ligand4.py"]
    assert cmd_r[:2] == ["/bin/echo", "prepare_receptor4.py"]

    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    lig_in = input_dir / "lig.mol2"
    rec_in = input_dir / "rec.pdb"
    lig_in.write_text("LIG", encoding="utf-8")
    rec_in.write_text("REC", encoding="utf-8")
    lig_out = tmp_path / "prep" / "lig.pdbqt"
    rec_out = tmp_path / "prep" / "rec.pdbqt"

    monkeypatch.setattr(strategy, "_check_tool_available", lambda _exe: True)
    monkeypatch.setattr("OCDocker.Toolbox.Printing.printv", lambda *_a, **_k: None)

    calls = []
    monkeypatch.setattr(
        ocprep.ocrun,
        "run",
        lambda cmd, logFile="", cwd="": calls.append((list(cmd), logFile, cwd)) or 123,
    )

    rc_l = strategy.prepare_ligand(str(lig_in), str(lig_out), log_file="lig.log", overwrite=False)
    rc_r = strategy.prepare_receptor(str(rec_in), str(rec_out), log_file="rec.log", overwrite=False)

    assert rc_l == 123
    assert rc_r == 123
    assert any("prepare_ligand4.py" in c[0] for c in calls)
    assert any("prepare_receptor4.py" in c[0] for c in calls)
    assert all(c[2] == str(input_dir) for c in calls)


@pytest.mark.order(433)
def test_mgltools_prepare_early_return_and_unavailable_tool_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(ocprep, "get_config", _config)
    strategy = ocprep.MGLToolsPreparationStrategy()

    monkeypatch.setattr(strategy, "_handle_existing_output", lambda *_a, **_k: ocerror.ErrorCode.OK)
    assert strategy.prepare_ligand("lig.mol2", "lig.pdbqt") == ocerror.ErrorCode.OK
    assert strategy.prepare_receptor("rec.pdb", "rec.pdbqt") == ocerror.ErrorCode.OK

    lig_in = tmp_path / "lig.mol2"
    rec_in = tmp_path / "rec.pdb"
    lig_in.write_text("LIG", encoding="utf-8")
    rec_in.write_text("REC", encoding="utf-8")

    strategy2 = ocprep.MGLToolsPreparationStrategy()
    monkeypatch.setattr(strategy2, "_check_tool_available", lambda _exe: False)
    monkeypatch.setattr(strategy2, "_fallback_copy", lambda *_a, **_k: 777)
    rc_l = strategy2.prepare_ligand(str(lig_in), str(tmp_path / "out_lig.pdbqt"))
    rc_r = strategy2.prepare_receptor(str(rec_in), str(tmp_path / "out_rec.pdbqt"))
    assert rc_l == 777
    assert rc_r == 777


@pytest.mark.order(434)
def test_spores_prepare_and_command_branches(tmp_path, monkeypatch):
    monkeypatch.setattr(ocprep, "get_config", _config)
    strategy = ocprep.SPORESPreparationStrategy()

    cmd_l = strategy.get_ligand_command("lig.mol2", "lig_prepared.mol2")
    cmd_r = strategy.get_receptor_command("rec.pdb", "rec_prepared.mol2")
    assert cmd_l[0] == "/bin/echo"
    assert cmd_r == ["/bin/echo", "--mode", "complete", "rec.pdb", "rec_prepared.mol2"]

    monkeypatch.setattr(strategy, "_handle_existing_output", lambda *_a, **_k: ocerror.ErrorCode.OK)
    assert strategy._prepare("in", "out", "", False, "ligand") == ocerror.ErrorCode.OK

    strategy2 = ocprep.SPORESPreparationStrategy()
    monkeypatch.setattr(strategy2, "_check_tool_available", lambda _exe: True)
    monkeypatch.setattr("OCDocker.Toolbox.Printing.printv", lambda *_a, **_k: None)
    calls = []
    monkeypatch.setattr(ocprep.ocrun, "run", lambda cmd, logFile="": calls.append((list(cmd), logFile)) or 321)

    lig_in = tmp_path / "ligand.mol2"
    rec_in = tmp_path / "receptor.pdb"
    lig_in.write_text("LIG", encoding="utf-8")
    rec_in.write_text("REC", encoding="utf-8")
    rc_l = strategy2.prepare_ligand(str(lig_in), str(tmp_path / "prep_lig.mol2"), log_file="lig.log")
    rc_r = strategy2.prepare_receptor(str(rec_in), str(tmp_path / "prep_rec.mol2"), log_file="rec.log")
    assert rc_l == 321
    assert rc_r == 321
    assert len(calls) == 2
    assert all(c[0][1:3] == ["--mode", "complete"] for c in calls)


@pytest.mark.order(435)
def test_openbabel_get_commands_and_prepare_ligand_branches(tmp_path, monkeypatch):
    monkeypatch.setattr(ocprep, "get_config", _config)
    strategy = ocprep.OpenBabelPreparationStrategy()

    cmd_l = strategy.get_ligand_command("lig.mol2", "lig.pdbqt")
    cmd_r = strategy.get_receptor_command("rec.pdb", "rec.pdbqt")
    assert cmd_l == ["/bin/echo", "lig.mol2", "-O", "lig.pdbqt"]
    assert cmd_r == ["/bin/echo", "rec.pdb", "-O", "rec.pdbqt"]

    monkeypatch.setattr(strategy, "_handle_existing_output", lambda *_a, **_k: ocerror.ErrorCode.OK)
    assert strategy.prepare_ligand("lig.mol2", "lig.pdbqt") == ocerror.ErrorCode.OK

    strategy2 = ocprep.OpenBabelPreparationStrategy()
    monkeypatch.setattr("OCDocker.Toolbox.Validation.validate_obabel_extension", lambda *_a, **_k: ocerror.ErrorCode.WRONG_TYPE)
    errors = []
    monkeypatch.setattr("OCDocker.Toolbox.Printing.print_error", lambda msg: errors.append(msg))
    rc_invalid_ext = strategy2.prepare_ligand("lig.bad", str(tmp_path / "lig.pdbqt"))
    assert rc_invalid_ext == ocerror.ErrorCode.WRONG_TYPE
    assert errors

    strategy3 = ocprep.OpenBabelPreparationStrategy()
    monkeypatch.setattr("OCDocker.Toolbox.Validation.validate_obabel_extension", lambda *_a, **_k: "mol2")
    warnings = []
    monkeypatch.setattr("OCDocker.Toolbox.Printing.print_warning", lambda msg: warnings.append(msg))
    calls = []
    monkeypatch.setattr(
        "OCDocker.Toolbox.Conversion.convert_mols",
        lambda in_path, out_path, return_molecule=False, overwrite=False: calls.append((in_path, out_path, return_molecule, overwrite)) or 555,
    )
    rc_convert = strategy3.prepare_ligand("lig.mol2", str(tmp_path / "lig.mol2"), overwrite=True)
    assert rc_convert == 555
    assert warnings
    assert calls and calls[0][2] is False and calls[0][3] is True


@pytest.mark.order(436)
def test_openbabel_smiles_and_prepare_receptor_branches(tmp_path, monkeypatch):
    monkeypatch.setattr(ocprep, "get_config", _config)
    strategy = ocprep.OpenBabelPreparationStrategy()

    monkeypatch.setattr("OCDocker.Toolbox.Validation.validate_obabel_extension", lambda *_a, **_k: "smi")
    warnings = []
    errors = []
    monkeypatch.setattr("OCDocker.Toolbox.Printing.print_warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr("OCDocker.Toolbox.Printing.print_error", lambda msg: errors.append(msg))

    smiles_input = tmp_path / "ligand.smi"
    smiles_input.write_text("CCO", encoding="utf-8")
    rc_missing_companion = strategy.prepare_ligand(str(smiles_input), str(tmp_path / "out.pdbqt"))
    assert rc_missing_companion == ocerror.ErrorCode.FILE_NOT_EXIST
    assert warnings and errors

    companion = tmp_path / "ligand.mol2"
    companion.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    monkeypatch.setattr(
        "OCDocker.Toolbox.Conversion.convert_mols",
        lambda in_path, out_path, return_molecule=False, overwrite=False: (in_path, out_path, return_molecule, overwrite),
    )
    rc_smiles_ok = strategy.prepare_ligand(str(smiles_input), str(tmp_path / "ok_out.pdbqt"))
    assert rc_smiles_ok[0] == str(companion)

    strategy2 = ocprep.OpenBabelPreparationStrategy()
    monkeypatch.setattr(strategy2, "_handle_existing_output", lambda *_a, **_k: ocerror.ErrorCode.OK)
    assert strategy2.prepare_receptor("rec.pdb", "rec.pdbqt") == ocerror.ErrorCode.OK
