#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage-focused tests for branch-heavy paths in ``OCDocker.Docking.Gnina``.

Usage:

pytest tests/docking/test_gnina_branch_coverage.py
'''

# Imports
###############################################################################
from __future__ import annotations

import builtins
import os
import types

from pathlib import Path

import pytest

import OCDocker.Docking.Gnina as ocgnina
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
    *,
    executable: str = "/nonexistent/gnina",
    scoring: str = "default",
    scoring_functions: list[str] | None = None,
    cnn: str = "default",
    cnn_models: list[str] | None = None,
    cnn_scoring: str = "rescore",
    no_gpu: str = "no",
    device: str = "0",
):
    class _GninaCfg:
        pass

    gn = _GninaCfg()
    gn.executable = executable

    for attr, _flag in ocgnina._GNINA_FLAG_OPTIONS:
        setattr(gn, attr, "no")
    for attr, _flag in ocgnina._GNINA_BOOL_VALUE_OPTIONS:
        setattr(gn, attr, "")
    for attr, _flag, _skip_no, _skip_auto in ocgnina._GNINA_VALUE_OPTIONS:
        setattr(gn, attr, "")

    gn.scoring = scoring
    gn.scoring_functions = scoring_functions if scoring_functions is not None else [scoring]
    gn.cnn = cnn
    gn.cnn_models = cnn_models if cnn_models is not None else [cnn]
    gn.cnn_scoring = cnn_scoring
    gn.no_gpu = no_gpu
    gn.device = device
    return types.SimpleNamespace(gnina=gn)


def _valid_box_file(path: Path) -> None:
    path.write_text(
        "REMARK    CENTER (X Y Z)        1.000  2.000  3.000\n"
        "REMARK    DIMENSIONS (X Y Z)    4.000  5.000  6.000\n",
        encoding="utf-8",
    )


## Public ##

@pytest.mark.order(411)
def test_gnina_helper_functions_and_command_builder(monkeypatch):
    cfg = _dummy_config(executable="/bin/echo", no_gpu="yes", device="0")
    cfg.gnina.score_only = "yes"
    cfg.gnina.addH = "yes"
    cfg.gnina.stripH = "no"
    cfg.gnina.autobox_ligand = "auto"
    cfg.gnina.autobox_add = "4"
    cfg.gnina.autobox_extend = "no"
    cfg.gnina.cpu = "auto"   # skipped by skip_auto
    cfg.gnina.seed = "13"
    cfg.gnina.num_modes = "9"
    monkeypatch.setattr(ocgnina, "get_config", lambda: cfg)

    assert ocgnina._as_text(None) == ""
    assert ocgnina._is_true("YES") is True
    assert ocgnina._is_false("off") is True
    assert ocgnina._resolve_autobox_ligand("prepared_ligand", "ligand.pdbqt") == "ligand.pdbqt"
    assert ocgnina._resolve_autobox_ligand("no", "ligand.pdbqt") == ""
    assert ocgnina._resolve_autobox_ligand("custom_autobox.pdbqt", "ligand.pdbqt") == "custom_autobox.pdbqt"
    assert ocgnina._normalize_string_list(("a", "", "b"), ["x"]) == ["a", "b"]
    assert ocgnina._normalize_string_list(None, ["x", ""]) == ["x"]

    cmd: list[str] = []
    ocgnina._append_option(cmd, "--foo", "bar")
    ocgnina._append_option(cmd, "--skip_no", "no", skip_no=True)
    assert cmd == ["--foo", "bar"]

    built = ocgnina._build_gnina_cmd("conf.txt", "ligand.pdbqt", "out.pdbqt", "run.log")
    assert "--score_only" in built
    assert "--addH" in built and "1" in built
    assert "--stripH" in built and "0" in built
    assert "--autobox_ligand" in built
    assert "--autobox_add" in built
    assert "--autobox_extend" not in built
    assert "--device" not in built
    assert "--cpu" not in built

    cfg.gnina.stripH = "keep"
    built_custom_bool = ocgnina._build_gnina_cmd("conf.txt", "ligand.pdbqt", "out.pdbqt", "run.log")
    assert "--stripH" in built_custom_bool and "keep" in built_custom_bool


@pytest.mark.order(412)
def test_gnina_init_type_guards_and_overwrite_config(tmp_path, monkeypatch):
    class DummyReceptor:
        def __init__(self, path: str):
            self.path = path

    class DummyLigand:
        def __init__(self, path: str, name: str = "lig"):
            self.path = path
            self.name = name

    monkeypatch.setattr(ocgnina.ocr, "Receptor", DummyReceptor)
    monkeypatch.setattr(ocgnina.ocl, "Ligand", DummyLigand)
    monkeypatch.setattr(ocgnina, "get_config", lambda: _dummy_config(executable="/bin/echo"))
    monkeypatch.setattr(ocgnina.ocff, "safe_create_dir", lambda p, *_a, **_k: Path(p).mkdir(parents=True, exist_ok=True) or 0)

    with pytest.raises(TypeError, match="Expected 'ocr.Receptor'"):
        _ = ocgnina.Gnina(
            config_path=str(tmp_path / "bad_receptor.conf"),
            box_file=str(tmp_path / "box.pdb"),
            receptor=123,  # type: ignore[arg-type]
            prepared_receptor_path=str(tmp_path / "prep_rec.pdbqt"),
            ligand=DummyLigand(str(tmp_path / "lig.mol2")),
            prepared_ligand_path=str(tmp_path / "prep_lig.pdbqt"),
            gnina_log=str(tmp_path / "gnina.log"),
            output_gnina=str(tmp_path / "out.pdbqt"),
        )

    with pytest.raises(TypeError, match="Expected 'ocl.Ligand'"):
        _ = ocgnina.Gnina(
            config_path=str(tmp_path / "bad_ligand.conf"),
            box_file=str(tmp_path / "box.pdb"),
            receptor=DummyReceptor(str(tmp_path / "rec.pdb")),
            prepared_receptor_path=str(tmp_path / "prep_rec.pdbqt"),
            ligand=123,  # type: ignore[arg-type]
            prepared_ligand_path=str(tmp_path / "prep_lig.pdbqt"),
            gnina_log=str(tmp_path / "gnina.log"),
            output_gnina=str(tmp_path / "out.pdbqt"),
        )

    calls = {"gen": 0}
    monkeypatch.setattr(ocgnina, "gen_gnina_conf", lambda *_a, **_k: calls.__setitem__("gen", calls["gen"] + 1) or 0)

    receptor = DummyReceptor(str(tmp_path / "rec_valid.pdb"))
    ligand = DummyLigand(str(tmp_path / "lig_valid.mol2"))
    _ = ocgnina.Gnina(
        config_path=str(tmp_path / "overwrite.conf"),
        box_file=str(tmp_path / "box_overwrite.pdb"),
        receptor=receptor,
        prepared_receptor_path=str(tmp_path / "prep_rec.pdbqt"),
        ligand=ligand,
        prepared_ligand_path=str(tmp_path / "prep_lig.pdbqt"),
        gnina_log=str(tmp_path / "gnina.log"),
        output_gnina=str(tmp_path / "out.pdbqt"),
        overwrite_config=True,
    )
    assert calls["gen"] == 1


@pytest.mark.order(413)
def test_gnina_parse_helpers_and_process_ligand(tmp_path, monkeypatch):
    class DummyReceptor:
        def __init__(self, path: str):
            self.path = path

    class DummyLigand:
        def __init__(self, path: str, name: str = "lig"):
            self.path = path
            self.name = name

    monkeypatch.setattr(ocgnina.ocr, "Receptor", DummyReceptor)
    monkeypatch.setattr(ocgnina.ocl, "Ligand", DummyLigand)

    instance = ocgnina.Gnina.__new__(ocgnina.Gnina)

    mol2_file = tmp_path / "ligand.mol2"
    mol2_file.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    sdf_file = tmp_path / "ligand.sdf"
    sdf_file.write_text("dummy\n", encoding="utf-8")
    receptor_file = tmp_path / "receptor.pdb"
    receptor_file.write_text("ATOM\n", encoding="utf-8")

    monkeypatch.setattr(ocgnina.occonversion, "convert_mols", lambda inp, out: Path(out).write_text(f"from {inp}\n", encoding="utf-8"))

    lig_obj = DummyLigand(str(tmp_path / "lig_obj.mol2"))
    assert instance._Gnina__parse_ligand_path(lig_obj) == lig_obj.path
    assert instance._Gnina__parse_ligand_path(str(mol2_file)) == str(mol2_file)
    converted = instance._Gnina__parse_ligand_path(str(sdf_file))
    assert converted.endswith(".mol2")
    assert Path(converted).exists()
    assert instance._Gnina__parse_ligand_path(str(tmp_path / "missing.sdf")) == ""
    assert instance._Gnina__parse_ligand_path(3.14) == ""  # type: ignore[arg-type]

    rec_obj = DummyReceptor(str(tmp_path / "rec_obj.pdb"))
    assert instance._Gnina__parse_receptor_path(rec_obj) == rec_obj.path
    assert instance._Gnina__parse_receptor_path(str(receptor_file)) == str(receptor_file)
    assert instance._Gnina__parse_receptor_path(str(tmp_path / "missing.pdb")) == ""
    assert instance._Gnina__parse_receptor_path(7) == ""  # type: ignore[arg-type]


@pytest.mark.order(414)
def test_gnina_instance_methods_delegate_and_helpers(monkeypatch, tmp_path, capsys):
    class DummyPreparationStrategy:
        def get_ligand_command(self, in_path: str, out_path: str):
            return ["prep_lig", in_path, out_path]

        def get_receptor_command(self, in_path: str, out_path: str):
            return ["prep_rec", in_path, out_path]

        def prepare_ligand(self, in_path: str, out_path: str, log_file: str, overwrite: bool = False):
            return (in_path, out_path, log_file, overwrite)

        def prepare_receptor(self, in_path: str, out_path: str, log_file: str, overwrite: bool = False):
            return (in_path, out_path, log_file, overwrite)

    instance = ocgnina.Gnina.__new__(ocgnina.Gnina)
    instance.name = "gnina-instance"
    instance.config = str(tmp_path / "conf.conf")
    instance.input_receptor = object()
    instance.input_receptor_path = str(tmp_path / "inputs" / "receptor.pdb")
    instance.prepared_receptor = str(tmp_path / "prepared" / "receptor.pdbqt")
    instance.input_ligand = types.SimpleNamespace(name="lig")
    instance.input_ligand_path = str(tmp_path / "inputs" / "ligand.mol2")
    instance.prepared_ligand = str(tmp_path / "prepared" / "ligand.pdbqt")
    instance.gnina_log = str(tmp_path / "run" / "gnina.log")
    instance.output_gnina = str(tmp_path / "run" / "gnina_out.pdbqt")
    instance.gnina_cmd = ["gnina", "--config", "conf.conf"]
    instance.preparation_strategy = DummyPreparationStrategy()

    monkeypatch.setattr(ocgnina, "get_docked_poses", lambda path: [str(Path(path) / "pose1.pdbqt")])
    monkeypatch.setattr(ocgnina, "read_log", lambda path, onlyBest=False: {1: {"affinity": -7.0 if onlyBest else -6.0}})
    monkeypatch.setattr(ocgnina, "get_rescore_log_paths", lambda out: [f"{out}/log1.log"])
    monkeypatch.setattr(ocgnina, "read_rescore_logs", lambda logs, onlyBest=False: {"k": -8.0 if onlyBest else -7.5})

    assert instance.get_docked_poses()[0].endswith("pose1.pdbqt")
    assert instance.get_input_ligand_path().endswith("inputs")
    assert instance.get_input_receptor_path().endswith("inputs")
    assert instance.read_log(onlyBest=True)[1]["affinity"] == -7.0
    assert instance.read_rescore_logs(str(tmp_path), onlyBest=True)["k"] == -8.0

    lig_result = instance.run_prepare_ligand(overwrite=True)
    lig_cmd_result = instance.run_prepare_ligand_from_cmd(logFile=str(tmp_path / "lig.log"))
    rec_result = instance.run_prepare_receptor(overwrite=True)
    rec_cmd_result = instance.run_prepare_receptor_from_cmd(logFile=str(tmp_path / "rec.log"), overwrite=True)
    assert lig_result[-1] is True
    assert lig_cmd_result[2].endswith("lig.log")
    assert rec_result[-1] is True
    assert rec_cmd_result[2].endswith("rec.log")

    instance.print_attributes()
    captured = capsys.readouterr()
    assert "Gnina command:" in captured.out


@pytest.mark.order(415)
def test_gnina_instance_run_gnina_stub_and_available_paths(tmp_path, monkeypatch):
    instance = ocgnina.Gnina.__new__(ocgnina.Gnina)
    instance.gnina_cmd = ["gnina", "--config", "conf.txt"]
    instance.output_gnina = str(tmp_path / "out" / "gnina_out.pdbqt")
    instance.gnina_log = str(tmp_path / "out" / "gnina.log")

    extra_log = tmp_path / "out" / "run.log"
    monkeypatch.setattr(ocgnina, "get_config", lambda: _dummy_config(executable="/nonexistent/gnina"))
    monkeypatch.setattr(ocgnina.shutil, "which", lambda _exe: None)

    rc_stub = instance.run_gnina(logFile=str(extra_log), overwrite=True)
    assert rc_stub == ocerror.Error.ok()
    assert Path(instance.output_gnina).exists()
    assert Path(instance.gnina_log).exists()
    assert extra_log.exists()

    monkeypatch.setattr(ocgnina, "get_config", lambda: _dummy_config(executable="/bin/echo"))
    monkeypatch.setattr(ocgnina.ocrun, "run", lambda cmd, logFile="": (len(cmd), logFile))
    rc_exec = instance.run_gnina(logFile=str(extra_log), overwrite=False)
    assert rc_exec[1] == str(extra_log)


@pytest.mark.order(416)
def test_gnina_instance_run_gnina_ignores_remove_makedirs_and_stub_write_errors(tmp_path, monkeypatch):
    instance = ocgnina.Gnina.__new__(ocgnina.Gnina)
    instance.gnina_cmd = ["gnina", "--config", "conf.txt"]
    instance.output_gnina = str(tmp_path / "nested" / "out" / "gnina_out.pdbqt")
    instance.gnina_log = str(tmp_path / "nested" / "out" / "gnina.log")

    monkeypatch.setattr(ocgnina, "get_config", lambda: _dummy_config(executable="/nonexistent/gnina"))
    monkeypatch.setattr(ocgnina.shutil, "which", lambda _exe: None)
    monkeypatch.setattr(ocgnina.os, "remove", lambda _path: (_ for _ in ()).throw(OSError("forced remove failure")))
    monkeypatch.setattr(ocgnina.os, "makedirs", lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("forced mkdir failure")))

    rc = instance.run_gnina(logFile=str(tmp_path / "nested" / "out" / "run.log"), overwrite=True)
    assert rc == ocerror.Error.ok()


@pytest.mark.order(416)
def test_gnina_instance_run_gnina_overwrite_remove_errors_are_ignored(tmp_path, monkeypatch):
    out_file = tmp_path / "out.pdbqt"
    gnina_log = tmp_path / "gnina.log"
    extra_log = tmp_path / "exec.log"
    out_file.write_text("old\n", encoding="utf-8")
    gnina_log.write_text("old\n", encoding="utf-8")
    extra_log.write_text("old\n", encoding="utf-8")

    instance = ocgnina.Gnina.__new__(ocgnina.Gnina)
    instance.gnina_cmd = ["gnina", "--config", "conf.txt"]
    instance.output_gnina = str(out_file)
    instance.gnina_log = str(gnina_log)

    monkeypatch.setattr(ocgnina, "get_config", lambda: _dummy_config(executable="/nonexistent/gnina"))
    monkeypatch.setattr(ocgnina.shutil, "which", lambda _exe: None)
    monkeypatch.setattr(ocgnina.os, "remove", lambda _p: (_ for _ in ()).throw(OSError("forced remove failure")))

    rc = instance.run_gnina(logFile=str(extra_log), overwrite=True)
    assert rc == ocerror.Error.ok()


@pytest.mark.order(417)
def test_gnina_split_poses_uses_default_outpath(monkeypatch, tmp_path):
    instance = ocgnina.Gnina.__new__(ocgnina.Gnina)
    instance.output_gnina = str(tmp_path / "run" / "gnina_out.pdbqt")
    instance.input_ligand = types.SimpleNamespace(name="ligand_name")

    captured = {}

    def _split_stub(src, name, out, logFile="", suffix=""):
        captured["args"] = (src, name, out, logFile, suffix)
        return 0

    monkeypatch.setattr(ocgnina.ocmolproc, "split_poses", _split_stub)
    rc = instance.split_poses(outPath="", logFile="X.log")
    assert rc == 0
    assert captured["args"][2] == str(tmp_path / "run")
    assert captured["args"][4] == "_split_"


@pytest.mark.order(418)
def test_gen_gnina_conf_success_and_error_paths(tmp_path, monkeypatch):
    conf_file = tmp_path / "conf" / "conf_gnina.conf"
    box_file = tmp_path / "box.pdb"
    _valid_box_file(box_file)

    monkeypatch.setattr(ocgnina.ocff, "safe_create_dir", lambda p, *_a, **_k: Path(p).mkdir(parents=True, exist_ok=True) or 0)
    monkeypatch.setattr(ocgnina.ocprint, "printv", lambda *_a, **_k: None)
    rc_ok = ocgnina.gen_gnina_conf(str(box_file), str(conf_file), "receptor.pdbqt")
    assert rc_ok == ocerror.Error.ok()
    assert conf_file.exists()

    rc_missing = ocgnina.gen_gnina_conf(str(tmp_path / "missing_box.pdb"), str(conf_file), "receptor.pdbqt")
    assert rc_missing == ocerror.Error.file_not_exist()

    original_open = builtins.open

    def _open_read_fail(path, mode="r", *args, **kwargs):
        if str(path) == str(box_file) and "r" in mode:
            raise OSError("forced read failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open_read_fail)
    rc_read = ocgnina.gen_gnina_conf(str(box_file), str(conf_file), "receptor.pdbqt")
    assert rc_read == ocerror.Error.read_file()

    monkeypatch.setattr(builtins, "open", original_open)

    def _open_write_fail(path, mode="r", *args, **kwargs):
        if str(path) == str(conf_file) and "w" in mode:
            raise OSError("forced write failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open_write_fail)
    rc_write = ocgnina.gen_gnina_conf(str(box_file), str(conf_file), "receptor.pdbqt")
    assert rc_write == ocerror.Error.write_file()


@pytest.mark.order(419)
def test_gnina_pose_index_and_read_rescore_logs_branches(monkeypatch, tmp_path):
    assert ocgnina.get_pose_index_from_file_path("/tmp/lig_split_42.pdbqt") == 42

    cfg = _dummy_config(
        scoring="default",
        scoring_functions=["default", "ad4"],
        cnn_models=["dense_1_3"],
    )
    monkeypatch.setattr(ocgnina, "get_config", lambda: cfg)
    monkeypatch.setattr(ocgnina, "read_rescoring_log", lambda p: float(len(os.path.basename(p))))

    warnings = []
    monkeypatch.setattr(ocgnina.ocerror.Error, "value_error", lambda **kwargs: warnings.append(kwargs.get("message", "")) or 0)

    files = []
    for name in [
        "lig_split_1_default_rescoring.log",
        "lig_default_rescoring.log",
        "lig_split_2_cnn_dense_1_3_rescoring.log",
        "lig_cnn_dense_1_3_rescoring.log",
        "lig_unknown_rescoring.log",
    ]:
        p = tmp_path / name
        p.write_text("Affinity: -7.0 (kcal/mol)\n", encoding="utf-8")
        files.append(str(p))

    out_all = ocgnina.read_rescore_logs(files, onlyBest=False)
    assert "rescoring_default_1" in out_all
    assert "gnina_default_rescoring" in out_all
    assert "rescoring_cnn_dense_1_3_2" in out_all
    assert "gnina_cnn_dense_1_3_rescoring" in out_all
    assert any("could not be parsed" in msg for msg in warnings)

    out_best = ocgnina.read_rescore_logs(files, onlyBest=True)
    assert "rescoring_cnn_dense_1_3_2" not in out_best

    one = ocgnina.read_rescore_logs(files[0], onlyBest=False)
    assert isinstance(one, dict)


@pytest.mark.order(420)
def test_gnina_standalone_prepare_wrappers_and_run_gnina(monkeypatch, tmp_path):
    class DummyPreparationStrategy:
        def prepare_ligand(self, in_path, out_path, log_file, overwrite=False):
            return ("lig", in_path, out_path, log_file, overwrite)

        def prepare_receptor(self, in_path, out_path, log_file, overwrite=False):
            return ("rec", in_path, out_path, log_file, overwrite)

    monkeypatch.setattr(ocgnina, "OpenBabelPreparationStrategy", DummyPreparationStrategy)

    lig = ocgnina.run_prepare_ligand("a.sdf", "a.pdbqt", overwrite=True)
    lig_cmd = ocgnina.run_prepare_ligand_from_cmd("a.sdf", "a.pdbqt", log_file="l.log")
    rec = ocgnina.run_prepare_receptor("r.pdb", "r.pdbqt", overwrite=True)
    rec_cmd = ocgnina.run_prepare_receptor_from_cmd("r.pdb", "r.pdbqt", log_file="r.log")
    assert lig[0] == "lig"
    assert lig_cmd[3] == "l.log"
    assert rec[0] == "rec"
    assert rec_cmd[3] == "r.log"

    monkeypatch.setattr(ocgnina, "get_config", lambda: _dummy_config(executable="/nonexistent/gnina"))
    monkeypatch.setattr(ocgnina.shutil, "which", lambda _exe: None)
    out_file = tmp_path / "out" / "gnina_out.pdbqt"
    log_file = tmp_path / "out" / "gnina.log"
    exec_log = tmp_path / "out" / "run.log"
    rc_stub = ocgnina.run_gnina("conf.txt", "lig.pdbqt", str(out_file), str(log_file), str(exec_log))
    assert rc_stub == ocerror.Error.ok()
    assert out_file.exists()

    monkeypatch.setattr(ocgnina, "get_config", lambda: _dummy_config(executable="/bin/echo"))
    monkeypatch.setattr(ocgnina.ocrun, "run", lambda cmd, logFile="": (cmd, logFile))
    rc_run = ocgnina.run_gnina("conf.txt", "lig.pdbqt", str(out_file), str(log_file), str(exec_log))
    assert isinstance(rc_run, tuple)
    assert rc_run[1] == str(exec_log)


@pytest.mark.order(421)
def test_gnina_standalone_run_gnina_ignores_makedirs_and_write_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(ocgnina, "get_config", lambda: _dummy_config(executable="/nonexistent/gnina"))
    monkeypatch.setattr(ocgnina.shutil, "which", lambda _exe: None)
    monkeypatch.setattr(ocgnina.os, "makedirs", lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("forced mkdir failure")))

    rc = ocgnina.run_gnina(
        "conf.txt",
        "lig.pdbqt",
        str(tmp_path / "nested" / "out" / "gnina_out.pdbqt"),
        str(tmp_path / "nested" / "out" / "gnina.log"),
        str(tmp_path / "nested" / "out" / "run.log"),
    )
    assert rc == ocerror.Error.ok()


@pytest.mark.order(422)
def test_gnina_run_rescore_split_cleanup_cnn_and_skip_existing(monkeypatch, tmp_path):
    cfg = _dummy_config(
        executable="/bin/echo",
        scoring="default",
        scoring_functions=["default"],
        cnn_models=["dense"],
        cnn_scoring="rescore",
        no_gpu="yes",
        device="0",
    )
    monkeypatch.setattr(ocgnina, "get_config", lambda: cfg)
    monkeypatch.setattr(ocgnina.ocff, "normalize_path", lambda p: str(Path(p)))
    monkeypatch.setattr(ocgnina.ocprint, "printv", lambda *_a, **_k: None)
    monkeypatch.setattr(ocgnina.ocprint, "print_error", lambda *_a, **_k: None)

    conf_file = tmp_path / "conf_gnina.conf"
    conf_file.write_text("receptor = rec.pdbqt\n", encoding="utf-8")
    out_path = tmp_path / "rescoring"
    out_path.mkdir(parents=True, exist_ok=True)
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("MODEL 1\nENDMDL\n", encoding="utf-8")

    def _split_stub(_ligand, name, outPath, logFile="", suffix="_split_"):
        _ = logFile, suffix
        Path(outPath, f"{name}_split_1.pdbqt").write_text("MODEL 1\nENDMDL\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(ocgnina.ocmolproc, "split_poses", _split_stub)

    seen_cmds = []

    def _run_stub(cmd, logFile=""):
        seen_cmds.append(list(cmd))
        Path(logFile).write_text("log without affinity marker\n", encoding="utf-8")
        return 0

    removed = []
    monkeypatch.setattr(ocgnina.ocrun, "run", _run_stub)
    monkeypatch.setattr(ocgnina.ocff, "safe_remove_file", lambda p: removed.append(str(p)) or 0)

    ocgnina.run_rescore(
        confFile=str(conf_file),
        ligands=[str(ligand)],
        outPath=str(out_path),
        scoring_function="default",
        splitLigand=True,
        overwrite=True,
        disable_cnn=True,
    )

    assert seen_cmds
    assert "--cnn_scoring" in seen_cmds[0] and "none" in seen_cmds[0]
    assert "--no_gpu" in seen_cmds[0]
    assert removed

    cfg.gnina.no_gpu = "no"
    cfg.gnina.device = "2"
    seen_cmds.clear()
    removed.clear()

    # Existing log should skip execution when overwrite=False.
    existing = out_path / "ligand_cnn_dense_rescoring.log"
    existing.write_text("Affinity: -7.1 (kcal/mol)\n", encoding="utf-8")
    ocgnina.run_rescore(
        confFile=str(conf_file),
        ligands=str(ligand),
        outPath=str(out_path),
        scoring_function="default",
        splitLigand=False,
        overwrite=False,
        cnn_model="dense",
        disable_cnn=False,
    )
    assert seen_cmds == []

    existing.unlink()
    def _run_ok(cmd, logFile=""):
        seen_cmds.append(list(cmd))
        Path(logFile).write_text("Affinity: -7.9 (kcal/mol)\n", encoding="utf-8")
        return 0
    monkeypatch.setattr(ocgnina.ocrun, "run", _run_ok)

    ocgnina.run_rescore(
        confFile=str(conf_file),
        ligands=str(ligand),
        outPath=str(out_path),
        scoring_function="default",
        splitLigand=False,
        overwrite=True,
        cnn_model="dense",
        disable_cnn=False,
    )
    assert seen_cmds
    assert "--cnn" in seen_cmds[0]
    assert "--cnn_scoring" in seen_cmds[0]
    assert "--device" in seen_cmds[0]
    assert removed == []


@pytest.mark.order(423)
def test_gnina_instance_run_rescore_skips_empty_entries(monkeypatch):
    cfg = _dummy_config(
        scoring="default",
        scoring_functions=["", "default"],
        cnn_models=["", "dense"],
    )
    monkeypatch.setattr(ocgnina, "get_config", lambda: cfg)

    calls = []
    monkeypatch.setattr(
        ocgnina,
        "run_rescore",
        lambda confFile, ligands, outPath, scoring_function, **kwargs: calls.append((confFile, ligands, outPath, scoring_function, kwargs)),
    )

    instance = ocgnina.Gnina.__new__(ocgnina.Gnina)
    instance.config = "conf.txt"
    instance.run_rescore("out", "ligand.pdbqt", splitLigand=False, skipDefaultScoring=True)

    assert len(calls) == 1
    assert calls[0][3] == "default"
    assert calls[0][4]["cnn_model"] == "dense"


@pytest.mark.order(424)
def test_gnina_instance_run_rescore_skips_empty_entries_from_helper_outputs(monkeypatch):
    cfg = _dummy_config(scoring="default")
    monkeypatch.setattr(ocgnina, "get_config", lambda: cfg)
    monkeypatch.setattr(ocgnina, "_get_rescore_scoring_functions", lambda _cfg: ["", "default"])
    monkeypatch.setattr(ocgnina, "_get_rescore_cnn_models", lambda _cfg: ["", "dense"])

    calls = []
    monkeypatch.setattr(
        ocgnina,
        "run_rescore",
        lambda confFile, ligands, outPath, scoring_function, **kwargs: calls.append((confFile, ligands, outPath, scoring_function, kwargs)),
    )

    instance = ocgnina.Gnina.__new__(ocgnina.Gnina)
    instance.config = "conf.txt"
    instance.run_rescore("out", "ligand.pdbqt", splitLigand=False, skipDefaultScoring=True)

    assert len(calls) == 1
    assert calls[0][4]["cnn_model"] == "dense"


@pytest.mark.order(425)
def test_gnina_run_rescore_handles_log_read_errors(monkeypatch, tmp_path):
    cfg = _dummy_config(executable="/bin/echo", no_gpu="no", device="1")
    monkeypatch.setattr(ocgnina, "get_config", lambda: cfg)
    monkeypatch.setattr(ocgnina.ocff, "normalize_path", lambda p: str(Path(p)))
    monkeypatch.setattr(ocgnina.ocprint, "printv", lambda *_a, **_k: None)
    monkeypatch.setattr(ocgnina.ocprint, "print_error", lambda *_a, **_k: None)

    conf_file = tmp_path / "conf_gnina.conf"
    conf_file.write_text("receptor = rec.pdbqt\n", encoding="utf-8")
    out_path = tmp_path / "rescoring_read_error"
    out_path.mkdir(parents=True, exist_ok=True)
    ligand = out_path / "lig_split_1.pdbqt"
    ligand.write_text("MODEL 1\nENDMDL\n", encoding="utf-8")

    monkeypatch.setattr(ocgnina.ocrun, "run", lambda _cmd, logFile="": Path(logFile).write_text("anything\n", encoding="utf-8") or 0)
    removed = []
    monkeypatch.setattr(ocgnina.ocff, "safe_remove_file", lambda p: removed.append(str(p)) or 0)

    original_open = builtins.open

    def _open_fail(path, mode="r", *args, **kwargs):
        if str(path).endswith("_default_rescoring.log") and "r" in mode:
            raise OSError("forced read failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open_fail)

    ocgnina.run_rescore(
        confFile=str(conf_file),
        ligands=[str(ligand)],
        outPath=str(out_path),
        scoring_function="default",
        splitLigand=False,
        overwrite=True,
    )
    assert removed
