#!/usr/bin/env python3

# Description
###############################################################################
'''
Targeted branch coverage tests for Smina gaps.
'''

# Imports
###############################################################################
import builtins
import os

from pathlib import Path
from types import SimpleNamespace

import pytest

import OCDocker.Docking.Smina as ocsmina
import OCDocker.Error as ocerror
import OCDocker.Ligand as ocl
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

def _dummy_ligand(path: str, name: str = "lig") -> ocl.Ligand:
    lig = object.__new__(ocl.Ligand)
    lig.path = path
    lig.name = name
    return lig


def _dummy_receptor(path: str, name: str = "rec") -> ocr.Receptor:
    rec = object.__new__(ocr.Receptor)
    rec.path = path
    rec.name = name
    return rec


def _cfg(tmp_path: Path):
    smina_cfg = SimpleNamespace(
        executable="/missing/smina",
        local_only="yes",
        minimize="yes",
        randomize_only="yes",
        accurate_line="yes",
        minimize_early_term="yes",
        custom_scoring="no",
        custom_atoms="no",
        minimize_iters="no",
        approximation="spline",
        factor="32",
        force_cap="10",
        user_grid="no",
        user_grid_lambda="no",
        energy_range="10",
        exhaustiveness="5",
        num_modes="3",
        scoring="vinardo",
        scoring_functions=["vinardo", "dkoes_scoring"],
    )
    tools_cfg = SimpleNamespace(obabel="obabel")
    return SimpleNamespace(smina=smina_cfg, tools=tools_cfg, logdir=str(tmp_path))


## Public ##

def test_smina_init_and_private_parser_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(ocsmina.ocff, "safe_create_dir", lambda *_a, **_k: ocerror.Error.ok())
    monkeypatch.setattr(ocsmina, "get_config", lambda: _cfg(tmp_path))
    monkeypatch.setattr(ocsmina, "gen_smina_conf", lambda *_a, **_k: ocerror.Error.ok())

    lig = _dummy_ligand(str(tmp_path / "lig.smi"))
    rec = _dummy_receptor(str(tmp_path / "rec.pdb"))

    with pytest.raises(TypeError, match="Expected 'ocr.Receptor'"):
        _ = ocsmina.Smina(
            config_path=str(tmp_path / "smina.conf"),
            box_file=str(tmp_path / "box.pdb"),
            receptor=123,
            prepared_receptor_path=str(tmp_path / "prep_rec.pdbqt"),
            ligand=lig,
            prepared_ligand_path=str(tmp_path / "prep_lig.pdbqt"),
            smina_log=str(tmp_path / "smina.log"),
            output_smina=str(tmp_path / "out.pdbqt"),
        )

    with pytest.raises(TypeError, match="Expected 'ocl.Ligand'"):
        _ = ocsmina.Smina(
            config_path=str(tmp_path / "smina2.conf"),
            box_file=str(tmp_path / "box.pdb"),
            receptor=rec,
            prepared_receptor_path=str(tmp_path / "prep_rec.pdbqt"),
            ligand=123,
            prepared_ligand_path=str(tmp_path / "prep_lig.pdbqt"),
            smina_log=str(tmp_path / "smina.log"),
            output_smina=str(tmp_path / "out.pdbqt"),
        )

    inst = object.__new__(ocsmina.Smina)
    monkeypatch.setattr(inst, "_Smina__process_ligand", lambda _p: "processed.mol2")

    lig_file = tmp_path / "ligand_input.smi"
    lig_file.write_text("CCO\n", encoding="utf-8")

    assert inst._Smina__parse_ligand_path(str(lig_file)) == "processed.mol2"
    assert inst._Smina__parse_ligand_path(str(tmp_path / "missing.smi")) == ""
    assert inst._Smina__parse_ligand_path(123) == ""

    monkeypatch.setattr(ocsmina.os.path, "isfile", lambda _p: True)
    assert inst._Smina__parse_receptor_path(str(tmp_path / "rec.pdb")) == str(tmp_path / "rec.pdb")

    monkeypatch.setattr(ocsmina.os.path, "isfile", lambda _p: False)
    assert inst._Smina__parse_receptor_path(str(tmp_path / "missing.pdb")) == ""
    assert inst._Smina__parse_receptor_path(123) == ""

    assert inst._Smina__parse_receptor_path(rec) == rec.path

    inst2 = object.__new__(ocsmina.Smina)
    called = {}
    monkeypatch.setattr(ocsmina.occonversion, "convert_mols", lambda src, dst: called.update({"src": src, "dst": dst}) or ocerror.Error.ok())
    assert inst2._Smina__process_ligand("x.mol2") == "x.mol2"
    converted = inst2._Smina__process_ligand(str(lig_file))
    assert converted.endswith(".mol2")
    assert called["src"].endswith("ligand_input.smi")


def test_smina_cmd_and_prepare_function_branches(monkeypatch, tmp_path):
    inst = object.__new__(ocsmina.Smina)
    inst.config = str(tmp_path / "smina.conf")
    inst.prepared_ligand = str(tmp_path / "prep_lig.pdbqt")
    inst.output_smina = str(tmp_path / "out.pdbqt")
    inst.smina_log = str(tmp_path / "dock.log")

    monkeypatch.setattr(ocsmina, "get_config", lambda: _cfg(tmp_path))
    cmd = inst._Smina__smina_cmd()
    assert "--score_only" in cmd
    assert "--minimize" in cmd
    assert "--randomize_only" in cmd
    assert "--accurate_line" in cmd
    assert "--minimize_early_term" in cmd

    monkeypatch.setattr(ocsmina.ocvalidation, "validate_obabel_extension", lambda *_a, **_k: ocerror.Error.unsupported_extension())
    rc_bad_ext = ocsmina.run_prepare_ligand("input.bad", "out.pdbqt")
    assert isinstance(rc_bad_ext, int)

    captured = {}

    class StrategyOK:
        def prepare_ligand(self, input_ligand_path, prepared_ligand, _log, overwrite=False):
            captured["input"] = input_ligand_path
            captured["output"] = prepared_ligand
            captured["overwrite"] = overwrite
            return ocerror.Error.ok()

    monkeypatch.setattr(ocsmina.ocvalidation, "validate_obabel_extension", lambda *_a, **_k: "smi")
    monkeypatch.setattr(ocsmina, "MGLToolsPreparationStrategy", lambda: StrategyOK())

    rc_smi = ocsmina.run_prepare_ligand(str(tmp_path / "in.smi"), str(tmp_path / "out.mol2"), overwrite=True)
    assert rc_smi == ocerror.Error.ok()
    assert captured["input"].endswith("ligand.mol2")

    class StrategyFail:
        def prepare_ligand(self, *_a, **_k):
            raise RuntimeError("boom")

    monkeypatch.setattr(ocsmina.ocvalidation, "validate_obabel_extension", lambda *_a, **_k: "mol2")
    monkeypatch.setattr(ocsmina, "MGLToolsPreparationStrategy", lambda: StrategyFail())

    rc_fail = ocsmina.run_prepare_ligand(str(tmp_path / "in.mol2"), str(tmp_path / "out.pdbqt"))
    assert rc_fail == ocerror.Error.subprocess()


def test_smina_read_rescore_and_run_rescore_branches(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(ocsmina, "get_config", lambda: cfg)
    monkeypatch.setattr(ocsmina, "read_rescoring_log", lambda _p: -7.5)

    logs = [
        str(tmp_path / "lig_split_2_vinardo_rescoring.log"),
        str(tmp_path / "lig_split_1_vinardo_rescoring.log"),
        str(tmp_path / "lig_weird_rescoring.log"),
    ]

    data_best = ocsmina.read_rescore_logs(logs, onlyBest=True)
    assert "rescoring_vinardo_1" in data_best
    assert "rescoring_vinardo_2" not in data_best

    data_single = ocsmina.read_rescore_logs(str(tmp_path / "lig_split_1_dkoes_scoring_rescoring.log"), onlyBest=False)
    assert "rescoring_dkoes_scoring_1" in data_single

    removed = {"count": 0}
    monkeypatch.setattr(ocsmina.ocff, "normalize_path", lambda p: p)
    monkeypatch.setattr(ocsmina.os, "makedirs", lambda *_a, **_k: None)
    monkeypatch.setattr(ocsmina.ocff, "safe_remove_file", lambda *_a, **_k: removed.__setitem__("count", removed["count"] + 1))
    monkeypatch.setattr(ocsmina.ocrun, "run", lambda *_a, **_k: 0)
    monkeypatch.setattr(ocsmina.ocprint, "print_error", lambda *_a, **_k: None)
    monkeypatch.setattr(ocsmina.ocprint, "printv", lambda *_a, **_k: None)

    lig = tmp_path / "ligand_input.pdbqt"
    lig.write_text("MODEL\nENDMDL\n", encoding="utf-8")

    real_open = builtins.open

    def read_error_open(file, mode="r", *args, **kwargs):
        if str(file).endswith("_vinardo_rescoring.log") and "r" in mode:
            raise OSError("deny")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_error_open)
    monkeypatch.setattr(ocsmina.os.path, "isfile", lambda p: str(p).endswith("_vinardo_rescoring.log"))

    ocsmina.run_rescore(
        confFile=str(tmp_path / "smina.conf"),
        ligands=[str(lig)],
        outPath=str(tmp_path / "rescoring"),
        scoring_function="vinardo",
        logFile="",
        splitLigand=False,
        overwrite=True,
    )
    assert removed["count"] >= 1

    monkeypatch.setattr(ocsmina.os.path, "isfile", lambda _p: True)
    ocsmina.run_rescore(
        confFile=str(tmp_path / "smina.conf"),
        ligands=[str(lig)],
        outPath=str(tmp_path / "rescoring"),
        scoring_function="vinardo",
        logFile="",
        splitLigand=False,
        overwrite=False,
    )


def test_smina_run_smina_instance_and_global_stub_branches(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(ocsmina, "get_config", lambda: cfg)

    out_file = tmp_path / "dock" / "out.pdbqt"
    log_file = tmp_path / "dock" / "dock.log"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("existing", encoding="utf-8")
    log_file.write_text("existing", encoding="utf-8")

    inst = object.__new__(ocsmina.Smina)
    inst.output_smina = str(out_file)
    inst.smina_cmd = ["smina", "--config", "cfg"]

    monkeypatch.setattr(ocsmina.os, "remove", lambda *_a, **_k: (_ for _ in ()).throw(OSError("deny")))
    monkeypatch.setattr(ocsmina.os, "makedirs", lambda *_a, **_k: (_ for _ in ()).throw(OSError("deny")))

    real_open = builtins.open

    def write_error_open(file, mode="r", *args, **kwargs):
        if "w" in mode and (str(file).endswith("out.pdbqt") or str(file).endswith("dock.log")):
            raise OSError("deny")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", write_error_open)

    rc_inst = inst.run_smina(logFile=str(log_file), overwrite=True)
    assert rc_inst == ocerror.Error.ok()

    # Global helper branch matrix with all optional flags enabled and unavailable binary
    rc_global = ocsmina.run_smina(
        config=str(tmp_path / "cfg.txt"),
        prepared_ligand=str(tmp_path / "lig.pdbqt"),
        output_smina=str(tmp_path / "global" / "out.pdbqt"),
        smina_log=str(tmp_path / "global" / "run.log"),
        log_path=str(tmp_path / "global" / "exec.log"),
    )
    assert rc_global == ocerror.Error.ok()
