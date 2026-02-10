#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage-focused tests for branch-heavy paths in ``OCDocker.Docking.Vina``.

Usage:

pytest tests/test_vina_gap_branches.py
'''

# Imports
###############################################################################
from __future__ import annotations

import builtins
import os
import pytest

from pathlib import Path

import OCDocker.Docking.Vina as ocvina
import OCDocker.Error as ocerror

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


def _dummy_config(
    vina_executable: str = "/nonexistent/vina",
    scoring: str = "vina",
    scoring_functions: list[str] | None = None,
):
    class _VinaCfg:
        pass

    vina_cfg = _VinaCfg()
    vina_cfg.executable = vina_executable
    vina_cfg.energy_range = "10"
    vina_cfg.exhaustiveness = "8"
    vina_cfg.num_modes = "9"
    vina_cfg.scoring = scoring
    vina_cfg.scoring_functions = scoring_functions if scoring_functions is not None else [scoring]

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.vina = vina_cfg
    return cfg


def _valid_box_file(path: Path) -> None:
    path.write_text(
        "REMARK    CENTER (X Y Z)        1.000  2.000  3.000\n"
        "REMARK    DIMENSIONS (X Y Z)    4.000  5.000  6.000\n"
    )


## Public ##


@pytest.mark.order(151)
def test_vina_init_type_guards_and_overwrite_config(tmp_path, monkeypatch):
    class DummyReceptor:
        def __init__(self, path: str):
            self.path = path

    class DummyLigand:
        def __init__(self, path: str, name: str = "lig"):
            self.path = path
            self.name = name

    monkeypatch.setattr(ocvina.ocr, "Receptor", DummyReceptor)
    monkeypatch.setattr(ocvina.ocl, "Ligand", DummyLigand)

    invalid_receptor = ocvina.Vina(
        config_path=str(tmp_path / "invalid_receptor.conf"),
        box_file=str(tmp_path / "box.pdb"),
        receptor=123,  # type: ignore[arg-type]
        prepared_receptor_path=str(tmp_path / "receptor.pdbqt"),
        ligand=DummyLigand(str(tmp_path / "ligand.mol2")),
        prepared_ligand_path=str(tmp_path / "ligand.pdbqt"),
        vina_log=str(tmp_path / "vina.log"),
        output_vina=str(tmp_path / "out.pdbqt"),
    )
    assert not hasattr(invalid_receptor, "input_receptor")

    invalid_ligand = ocvina.Vina(
        config_path=str(tmp_path / "invalid_ligand.conf"),
        box_file=str(tmp_path / "box.pdb"),
        receptor=DummyReceptor(str(tmp_path / "receptor.pdb")),
        prepared_receptor_path=str(tmp_path / "receptor.pdbqt"),
        ligand=123,  # type: ignore[arg-type]
        prepared_ligand_path=str(tmp_path / "ligand.pdbqt"),
        vina_log=str(tmp_path / "vina.log"),
        output_vina=str(tmp_path / "out.pdbqt"),
    )
    assert not hasattr(invalid_ligand, "input_ligand")

    called = {"box_to_vina": 0}

    def _fake_box_to_vina(box_file: str, conf_file: str, receptor: str) -> int:
        _ = (box_file, conf_file, receptor)
        called["box_to_vina"] += 1
        return 0

    monkeypatch.setattr(ocvina, "box_to_vina", _fake_box_to_vina)
    monkeypatch.setattr(ocvina, "get_config", lambda: _dummy_config())

    valid_receptor = DummyReceptor(str(tmp_path / "receptor_valid.pdb"))
    valid_ligand = DummyLigand(str(tmp_path / "ligand_valid.mol2"))
    vina = ocvina.Vina(
        config_path=str(tmp_path / "overwrite.conf"),
        box_file=str(tmp_path / "box_overwrite.pdb"),
        receptor=valid_receptor,
        prepared_receptor_path=str(tmp_path / "receptor_out.pdbqt"),
        ligand=valid_ligand,
        prepared_ligand_path=str(tmp_path / "ligand_out.pdbqt"),
        vina_log=str(tmp_path / "vina_overwrite.log"),
        output_vina=str(tmp_path / "out_overwrite.pdbqt"),
        overwrite_config=True,
    )

    assert isinstance(vina, ocvina.Vina)
    assert called["box_to_vina"] == 1


@pytest.mark.order(152)
def test_vina_private_parse_helpers_and_process_ligand(tmp_path, monkeypatch):
    class DummyReceptor:
        def __init__(self, path: str):
            self.path = path

    class DummyLigand:
        def __init__(self, path: str, name: str = "lig"):
            self.path = path
            self.name = name

    monkeypatch.setattr(ocvina.ocr, "Receptor", DummyReceptor)
    monkeypatch.setattr(ocvina.ocl, "Ligand", DummyLigand)

    instance = ocvina.Vina.__new__(ocvina.Vina)

    mol2_file = tmp_path / "ligand.mol2"
    mol2_file.write_text("@<TRIPOS>MOLECULE\n")
    sdf_file = tmp_path / "ligand.sdf"
    sdf_file.write_text("dummy\n")
    receptor_file = tmp_path / "receptor.pdb"
    receptor_file.write_text("ATOM\n")

    def _fake_convert(inp: str, out: str, *args, **kwargs):
        _ = (inp, args, kwargs)
        Path(out).write_text("converted\n")
        return 0

    monkeypatch.setattr(ocvina.occonversion, "convert_mols", _fake_convert)

    ligand_obj = DummyLigand(str(tmp_path / "lig_obj.mol2"))
    assert instance._Vina__parse_ligand_path(ligand_obj) == ligand_obj.path
    assert instance._Vina__parse_ligand_path(str(mol2_file)) == str(mol2_file)
    converted = instance._Vina__parse_ligand_path(str(sdf_file))
    assert converted.endswith(".mol2")
    assert Path(converted).exists()
    assert instance._Vina__parse_ligand_path(str(tmp_path / "missing_ligand.sdf")) == ""
    assert instance._Vina__parse_ligand_path(3.14) == ""  # type: ignore[arg-type]

    receptor_obj = DummyReceptor(str(tmp_path / "rec_obj.pdb"))
    assert instance._Vina__parse_receptor_path(receptor_obj) == receptor_obj.path
    assert instance._Vina__parse_receptor_path(str(receptor_file)) == str(receptor_file)
    assert instance._Vina__parse_receptor_path(str(tmp_path / "missing_receptor.pdb")) == ""
    assert instance._Vina__parse_receptor_path(101) == ""  # type: ignore[arg-type]


@pytest.mark.order(153)
def test_vina_prepare_methods_delegate_to_strategy(tmp_path):
    class DummyPreparationStrategy:
        def prepare_ligand(
            self,
            input_ligand_path: str,
            prepared_ligand: str,
            log_file: str,
            overwrite: bool = False,
        ):
            return (input_ligand_path, prepared_ligand, log_file, overwrite)

        def prepare_receptor(
            self,
            input_receptor_path: str,
            prepared_receptor: str,
            log_file: str,
            overwrite: bool = False,
        ):
            return (input_receptor_path, prepared_receptor, log_file, overwrite)

    instance = ocvina.Vina.__new__(ocvina.Vina)
    instance.input_ligand_path = str(tmp_path / "ligand.mol2")
    instance.prepared_ligand = str(tmp_path / "ligand.pdbqt")
    instance.input_receptor_path = str(tmp_path / "receptor.pdb")
    instance.prepared_receptor = str(tmp_path / "receptor.pdbqt")
    instance.preparation_strategy = DummyPreparationStrategy()

    ligand_result = instance.run_prepare_ligand(
        logFile=str(tmp_path / "ligand.log"),
        useOpenBabel=False,
        overwrite=True,
    )
    receptor_result = instance.run_prepare_receptor(
        logFile=str(tmp_path / "receptor.log"),
        useOpenBabel=False,
        overwrite=True,
    )

    assert ligand_result == (
        instance.input_ligand_path,
        instance.prepared_ligand,
        str(tmp_path / "ligand.log"),
        True,
    )
    assert receptor_result == (
        instance.input_receptor_path,
        instance.prepared_receptor,
        str(tmp_path / "receptor.log"),
        True,
    )


@pytest.mark.order(154)
def test_vina_instance_run_vina_overwrite_and_not_set(tmp_path, monkeypatch):
    output_vina = tmp_path / "vina_out.pdbqt"
    vina_log = tmp_path / "vina.log"
    output_vina.write_text("old_out\n")
    vina_log.write_text("old_log\n")

    instance = ocvina.Vina.__new__(ocvina.Vina)
    instance.config = str(tmp_path / "vina.conf")
    instance.output_vina = str(output_vina)
    instance.vina_log = str(vina_log)
    instance.vina_cmd = None

    monkeypatch.setattr(ocvina.ocerror.Error, "not_set", lambda *a, **k: 777)

    result_not_set = instance.run_vina(overwrite=True)
    assert result_not_set == 777
    assert not output_vina.exists()
    assert not vina_log.exists()

    instance.vina_cmd = ["vina", "--config", "conf.txt"]
    monkeypatch.setattr(ocvina.ocrun, "run", lambda cmd, logFile="": (len(cmd), logFile))

    result_run = instance.run_vina(overwrite=False)
    assert result_run == (3, instance.vina_log)


@pytest.mark.order(155)
def test_box_to_vina_read_error_branch(tmp_path, monkeypatch):
    box_file = tmp_path / "box_bad.pdb"
    box_file.write_text("REMARK    CENTER (X Y Z)        1.000  2.000  3.000\n")
    conf_file = tmp_path / "conf_bad.txt"

    original_open = builtins.open

    def _failing_open(path, mode="r", *args, **kwargs):
        if str(path) == str(box_file) and "r" in mode:
            raise OSError("forced read failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _failing_open)

    result = ocvina.box_to_vina(str(box_file), str(conf_file), "receptor.pdbqt")
    assert isinstance(result, int)
    assert result != ocerror.Error.ok()  # type: ignore[arg-type]


@pytest.mark.order(156)
def test_box_to_vina_ignores_makedirs_errors(tmp_path, monkeypatch):
    box_file = tmp_path / "box_ok.pdb"
    _valid_box_file(box_file)
    conf_file = tmp_path / "vina_conf.txt"

    def _raise_os_error(*args, **kwargs):
        _ = (args, kwargs)
        raise OSError("forced makedirs failure")

    monkeypatch.setattr(ocvina.os, "makedirs", _raise_os_error)
    monkeypatch.setattr(ocvina, "get_config", lambda: _dummy_config())

    result = ocvina.box_to_vina(str(box_file), str(conf_file), "receptor.pdbqt")
    assert result == ocerror.Error.ok()  # type: ignore[arg-type]
    assert conf_file.exists()


@pytest.mark.order(157)
def test_read_rescore_logs_branchy_key_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ocvina,
        "get_config",
        lambda: _dummy_config(scoring="vina", scoring_functions=["vina"]),
    )

    warning_messages: list[str] = []

    def _warning_spy(*args, **kwargs):
        _ = args
        warning_messages.append(str(kwargs.get("message", "")))
        return 0

    monkeypatch.setattr(ocvina.ocerror.Error, "value_error", _warning_spy)

    logs = {
        "lig_vina_rescoring.log": "Estimated Free Energy of Binding    -7.10 (kcal/mol)\n",
        "lig_split_1.log": "Estimated Free Energy of Binding    -6.50 (kcal/mol)\n",
        "lig_split_A_vina_rescoring.log": "Estimated Free Energy of Binding    -8.25 (kcal/mol)\n",
        "lig_unknown_rescoring.log": "Estimated Free Energy of Binding    -5.50 (kcal/mol)\n",
    }

    log_paths = []
    for filename, content in logs.items():
        path = tmp_path / filename
        path.write_text(content)
        log_paths.append(str(path))

    data = ocvina.read_rescore_logs(log_paths, onlyBest=False)

    assert "vina_vina_rescoring" in data
    assert "rescoring_1" in data
    assert any("could not be found" in msg for msg in warning_messages)


@pytest.mark.order(158)
def test_run_rescore_handles_log_read_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ocvina,
        "get_config",
        lambda: _dummy_config(scoring="vina", scoring_functions=["vina"]),
    )

    conf_file = tmp_path / "vina.conf"
    conf_file.write_text("receptor = receptor.pdbqt\n")
    out_path = tmp_path / "rescoring"
    out_path.mkdir()
    ligand = out_path / "lig_split_1.pdbqt"
    ligand.write_text("MODEL 1\nENDMDL\n")

    def _run_stub(cmd, logFile=""):
        _ = cmd
        Path(logFile).write_text("run completed without marker\n")
        return 0

    monkeypatch.setattr(ocvina.ocrun, "run", _run_stub)

    removed: list[str] = []
    monkeypatch.setattr(ocvina.ocff, "safe_remove_file", lambda path: removed.append(path) or 0)

    original_open = builtins.open

    def _open_fail_on_read(path, mode="r", *args, **kwargs):
        if str(path).endswith("_vina_rescoring.log") and "r" in mode:
            raise OSError("forced read failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open_fail_on_read)
    monkeypatch.setattr(ocvina.ocprint, "print_error", lambda *a, **k: None)

    ocvina.run_rescore(
        confFile=str(conf_file),
        ligands=[str(ligand)],
        outPath=str(out_path),
        scoring_function="vina",
        splitLigand=False,
        overwrite=True,
    )

    assert removed


@pytest.mark.order(159)
def test_run_rescore_success_marker_keeps_log(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ocvina,
        "get_config",
        lambda: _dummy_config(scoring="vina", scoring_functions=["vina"]),
    )

    conf_file = tmp_path / "vina.conf"
    conf_file.write_text("receptor = receptor.pdbqt\n")
    out_path = tmp_path / "rescoring_ok"
    out_path.mkdir()
    ligand = out_path / "lig_split_1.pdbqt"
    ligand.write_text("MODEL 1\nENDMDL\n")

    def _run_stub(cmd, logFile=""):
        _ = cmd
        Path(logFile).write_text("Estimated Free Energy of Binding    -7.77 (kcal/mol)\n")
        return 0

    monkeypatch.setattr(ocvina.ocrun, "run", _run_stub)
    removed: list[str] = []
    monkeypatch.setattr(ocvina.ocff, "safe_remove_file", lambda path: removed.append(path) or 0)

    ocvina.run_rescore(
        confFile=str(conf_file),
        ligands=[str(ligand)],
        outPath=str(out_path),
        scoring_function="vina",
        splitLigand=False,
        overwrite=True,
    )

    expected_log = out_path / "lig_split_1_vina_rescoring.log"
    assert expected_log.exists()
    assert removed == []


@pytest.mark.order(160)
def test_run_vina_stub_without_log_file_branch(tmp_path, monkeypatch):
    monkeypatch.setattr(ocvina, "get_config", lambda: _dummy_config(vina_executable="/nonexistent/vina"))

    out_file = tmp_path / "vina_out.pdbqt"
    result = ocvina.run_vina(
        confFile=str(tmp_path / "vina.conf"),
        ligand=str(tmp_path / "ligand.pdbqt"),
        outPath=str(out_file),
        logFile="",
    )

    assert result == ocerror.Error.ok()  # type: ignore[arg-type]
    assert out_file.exists()


@pytest.mark.order(161)
def test_run_vina_stub_makedirs_and_write_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(ocvina, "get_config", lambda: _dummy_config(vina_executable="/nonexistent/vina"))

    def _raise_permission(*args, **kwargs):
        _ = (args, kwargs)
        raise PermissionError("forced directory creation failure")

    monkeypatch.setattr(ocvina.os, "makedirs", _raise_permission)

    result = ocvina.run_vina(
        confFile=str(tmp_path / "vina.conf"),
        ligand=str(tmp_path / "ligand.pdbqt"),
        outPath=str(tmp_path / "nested" / "vina_out.pdbqt"),
        logFile=str(tmp_path / "nested" / "vina.log"),
    )

    assert result == ocerror.Error.ok()  # type: ignore[arg-type]


@pytest.mark.order(162)
def test_run_vina_available_binary_path_calls_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(ocvina, "get_config", lambda: _dummy_config(vina_executable="/bin/echo"))
    monkeypatch.setattr(ocvina.ocrun, "run", lambda cmd, logFile="": (cmd, logFile))

    out_file = str(tmp_path / "vina_out.pdbqt")
    log_file = str(tmp_path / "vina.log")
    result = ocvina.run_vina(
        confFile=str(tmp_path / "vina.conf"),
        ligand=str(tmp_path / "ligand.pdbqt"),
        outPath=out_file,
        logFile=log_file,
    )

    assert isinstance(result, tuple)
    assert result[1] == log_file


@pytest.mark.order(163)
def test_vina_instance_run_vina_overwrite_remove_errors_are_ignored(tmp_path, monkeypatch):
    output_vina = tmp_path / "vina_out_keep.pdbqt"
    vina_log = tmp_path / "vina_keep.log"
    output_vina.write_text("out\n")
    vina_log.write_text("log\n")

    instance = ocvina.Vina.__new__(ocvina.Vina)
    instance.config = str(tmp_path / "vina.conf")
    instance.output_vina = str(output_vina)
    instance.vina_log = str(vina_log)
    instance.vina_cmd = None

    def _remove_fail(path):
        _ = path
        raise OSError("forced remove error")

    monkeypatch.setattr(ocvina.os, "remove", _remove_fail)
    monkeypatch.setattr(ocvina.ocerror.Error, "not_set", lambda *a, **k: 123)

    result = instance.run_vina(overwrite=True)
    assert result == 123


@pytest.mark.order(164)
def test_vina_instance_run_vina_overwrite_skips_empty_output_path(tmp_path, monkeypatch):
    vina_log = tmp_path / "vina.log"
    vina_log.write_text("log\n")

    instance = ocvina.Vina.__new__(ocvina.Vina)
    instance.config = str(tmp_path / "vina.conf")
    instance.output_vina = ""
    instance.vina_log = str(vina_log)
    instance.vina_cmd = None

    monkeypatch.setattr(ocvina.ocerror.Error, "not_set", lambda *a, **k: 321)

    result = instance.run_vina(overwrite=True)
    assert result == 321
