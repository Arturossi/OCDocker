#!/usr/bin/env python3

# Description
###############################################################################
'''
Branch-focused coverage tests for core CLI helpers and lightweight commands.

Usage:

pytest tests/test_cli_core_branches.py
'''

# Imports
###############################################################################
import argparse
import builtins
import runpy
import sys
import types

from pathlib import Path
from types import SimpleNamespace

import importlib.metadata
import pytest

import OCDocker.CLI.__init__ as cli

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


class _Parser:
    def __init__(self):
        self.seen_argv = None

    def parse_args(self, argv):
        self.seen_argv = list(argv)
        return argparse.Namespace(func=lambda _args: 7)


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(450)
def test_bootstrap_sets_config_and_calls_bootstrap(monkeypatch):
    seen = {"ns": None}
    fake_mod = types.SimpleNamespace(bootstrap=lambda ns: seen.__setitem__("ns", ns))
    monkeypatch.setattr(cli.importlib, "import_module", lambda _name: fake_mod)

    ns = argparse.Namespace(config_file="/tmp/custom.cfg")
    cli._bootstrap_ocdocker_env(ns)

    assert seen["ns"] is ns
    assert cli.os.environ["OCDOCKER_CONFIG"] == "/tmp/custom.cfg"


@pytest.mark.order(451)
def test_bootstrap_raises_when_bootstrap_not_found(monkeypatch):
    monkeypatch.setattr(cli.importlib, "import_module", lambda _name: types.SimpleNamespace())

    with pytest.raises(RuntimeError, match="bootstrap not found"):
        cli._bootstrap_ocdocker_env(argparse.Namespace(config_file=None))


@pytest.mark.order(452)
def test_box_sort_key_numeric_and_non_numeric():
    assert cli._box_sort_key(Path("box12.pdb")) == (0, 12)
    assert cli._box_sort_key(Path("boxx.pdb")) == (1, "boxx")
    assert cli._box_sort_key(Path("custom.pdb")) == (1, "custom")


@pytest.mark.order(453)
def test_ensure_mol2_poses_handles_passthrough_and_conversion(monkeypatch, tmp_path):
    lig_mol2 = tmp_path / "pose.mol2"
    lig_mol2.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    lig_pdbqt = tmp_path / "pose.pdbqt"
    lig_pdbqt.write_text("REMARK\n", encoding="utf-8")
    lig_sdf = tmp_path / "pose2.sdf"
    lig_sdf.write_text("$$$$\n", encoding="utf-8")

    fake_conv = types.ModuleType("OCDocker.Toolbox.Conversion")

    def _convert(src, out, overwrite=False):
        _ = (src, overwrite)
        Path(out).write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
        return out

    fake_conv.convert_mols = _convert  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Conversion", fake_conv)

    out_dir = tmp_path / "converted"
    paths, mapping = cli._ensure_mol2_poses(
        [str(lig_mol2), str(lig_pdbqt), str(lig_sdf)],
        out_dir,
        pose_engine_map={str(lig_pdbqt): "vina"},
    )

    assert str(lig_mol2) in paths
    assert str(out_dir / "vina_pose.mol2") in paths
    assert str(out_dir / "unknown_pose2.mol2") in paths
    assert mapping[str(lig_mol2)] == str(lig_mol2)
    assert mapping[str(out_dir / "vina_pose.mol2")] == str(lig_pdbqt)
    assert mapping[str(out_dir / "unknown_pose2.mol2")] == str(lig_sdf)


@pytest.mark.order(454)
def test_list_boxes_all_modes_and_resolve_fallback(monkeypatch, tmp_path):
    ligand_dir = tmp_path / "lig"
    box_dir = tmp_path / "boxes"
    ligand_dir.mkdir()
    box_dir.mkdir()

    box_main = box_dir / "box2.pdb"
    box_main.write_text("ATOM\n", encoding="utf-8")
    (ligand_dir / "box1.pdb").write_text("ATOM\n", encoding="utf-8")
    (ligand_dir / "box_bad.pdb").write_text("ATOM\n", encoding="utf-8")
    (box_dir / "box10.pdb").write_text("ATOM\n", encoding="utf-8")

    one = cli._list_boxes(ligand_dir, box_main, all_boxes=False)
    assert one == [box_main]

    original_resolve = Path.resolve

    def _resolve(path_obj, *args, **kwargs):
        if path_obj.name == "box_bad.pdb":
            raise OSError("cannot resolve")
        return original_resolve(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _resolve)
    boxes = cli._list_boxes(ligand_dir, box_main, all_boxes=True)

    names = [p.name for p in boxes]
    assert names == ["box1.pdb", "box2.pdb", "box10.pdb", "box_bad.pdb"]


@pytest.mark.order(455)
def test_preparse_global_args_extra_branches():
    ns = cli._preparse_global_args(
        [
            "vs",
            "--version",
            "--no-multiprocess",
            "--output-level",
            "not-an-int",
            "--log-file",
            "/tmp/ocdocker.log",
        ]
    )

    assert ns.version is True
    assert ns.multiprocess is False
    assert ns.output_level == 1
    assert ns.log_file == "/tmp/ocdocker.log"


@pytest.mark.order(456)
def test_cmd_init_config_example_not_found_anywhere(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    import OCDocker

    fake_pkg_init = tmp_path / "pkgroot" / "OCDocker" / "__init__.py"
    fake_pkg_init.parent.mkdir(parents=True, exist_ok=True)
    fake_pkg_init.write_text("", encoding="utf-8")
    monkeypatch.setattr(OCDocker, "__file__", str(fake_pkg_init), raising=False)

    rc = cli.cmd_init_config(SimpleNamespace(config_file=str(tmp_path / "new.cfg")))
    assert rc == 1
    assert "OCDocker.cfg.example not found" in capsys.readouterr().out


@pytest.mark.order(457)
def test_cmd_init_config_target_already_exists(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "OCDocker.cfg.example").write_text("sample=true\n", encoding="utf-8")
    existing = tmp_path / "OCDocker.cfg"
    existing.write_text("keep-me\n", encoding="utf-8")

    rc = cli.cmd_init_config(SimpleNamespace(config_file=str(existing)))
    assert rc == 0
    assert existing.read_text(encoding="utf-8") == "keep-me\n"
    assert "Config already exists" in capsys.readouterr().out


@pytest.mark.order(458)
def test_cmd_version_falls_back_to_importlib_metadata(monkeypatch, capsys):
    fake_oc = types.ModuleType("OCDocker")
    fake_oc.__version__ = None  # force metadata fallback
    monkeypatch.setitem(sys.modules, "OCDocker", fake_oc)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "9.9.9")

    rc = cli.cmd_version(SimpleNamespace())
    assert rc == 0
    assert capsys.readouterr().out.strip() == "9.9.9"


@pytest.mark.order(459)
def test_cmd_version_prints_unknown_when_imports_fail(monkeypatch, capsys):
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("OCDocker", "importlib.metadata"):
            raise ImportError("forced import failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    rc = cli.cmd_version(SimpleNamespace())
    assert rc == 0
    assert capsys.readouterr().out.strip() == "unknown"


@pytest.mark.order(460)
def test_main_dispatches_using_sys_argv_when_argv_none(monkeypatch):
    parser = _Parser()
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    monkeypatch.setattr(cli.sys, "argv", ["ocdocker", "cmd", "--x"])

    rc = cli.main()
    assert rc == 7
    assert parser.seen_argv == ["cmd", "--x"]


@pytest.mark.order(461)
def test_cli_module_main_guard_executes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ocdocker", "version"])
    cached = sys.modules.pop("OCDocker.CLI.__init__", None)
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("OCDocker.CLI.__init__", run_name="__main__")
        assert int(exc.value.code) == 0
    finally:
        if cached is not None:
            sys.modules["OCDocker.CLI.__init__"] = cached


@pytest.mark.order(462)
def test_list_boxes_all_boxes_with_missing_primary_box(tmp_path):
    ligand_dir = tmp_path / "ligand"
    ligand_dir.mkdir()
    (ligand_dir / "box3.pdb").write_text("ATOM\n", encoding="utf-8")
    missing_box = tmp_path / "missing_box.pdb"

    boxes = cli._list_boxes(ligand_dir, missing_box, all_boxes=True)
    assert [p.name for p in boxes] == ["box3.pdb"]


@pytest.mark.order(463)
def test_cmd_init_config_uses_package_example_when_cwd_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    import OCDocker

    fake_pkg_init = tmp_path / "fake_root" / "OCDocker" / "__init__.py"
    fake_pkg_init.parent.mkdir(parents=True, exist_ok=True)
    fake_pkg_init.write_text("", encoding="utf-8")
    pkg_example = fake_pkg_init.parent.parent / "OCDocker.cfg.example"
    pkg_example.write_text("from_package=true\n", encoding="utf-8")
    monkeypatch.setattr(OCDocker, "__file__", str(fake_pkg_init), raising=False)

    target = tmp_path / "generated.cfg"
    rc = cli.cmd_init_config(SimpleNamespace(config_file=str(target)))
    assert rc == 0
    assert target.read_text(encoding="utf-8") == "from_package=true\n"


@pytest.mark.order(464)
def test_bootstrap_passes_init_db_flag_when_supported(monkeypatch):
    seen = {"ns": None, "init_db": None}

    def _bootstrap(ns, init_db=True):
        seen["ns"] = ns
        seen["init_db"] = init_db

    fake_mod = types.SimpleNamespace(bootstrap=_bootstrap)
    monkeypatch.setattr(cli.importlib, "import_module", lambda _name: fake_mod)

    ns = argparse.Namespace(config_file=None, _ocdocker_init_db=False)
    cli._bootstrap_ocdocker_env(ns)

    assert seen["ns"] is ns
    assert seen["init_db"] is False


@pytest.mark.order(465)
def test_suggest_extra_for_missing_module_uses_ml_for_optuna():
    assert cli._suggest_extra_for_missing_module("optuna") == "ml"
    assert cli._suggest_extra_for_missing_module("optuna.samplers") == "ml"
    assert cli._suggest_extra_for_missing_module("torch") == "ml"
    assert cli._suggest_extra_for_missing_module("torch.nn") == "ml"
    assert cli._suggest_extra_for_missing_module("torchaudio") == "ml"
    assert cli._suggest_extra_for_missing_module("torchvision.transforms") == "ml"
    assert cli._suggest_extra_for_missing_module("xgboost") == "ml"
    assert cli._suggest_extra_for_missing_module("torchsummary") == "ml"
    assert cli._suggest_extra_for_missing_module("torchviz") == "ml"
    assert cli._suggest_extra_for_missing_module("visualtorch") == "ml"
    assert cli._suggest_extra_for_missing_module("sqlalchemy") == "db"
    assert cli._suggest_extra_for_missing_module("rdkit") == "docking"


@pytest.mark.order(466)
def test_print_optional_dependency_hint_reports_extra(capsys):
    try:
        __import__("definitely_missing_module_for_cli_hint_test")
    except ModuleNotFoundError as exc:
        rc = cli._print_optional_dependency_hint(feature="ML workflow", extra="ml", exc=exc)
    else:  # pragma: no cover
        raise AssertionError("expected import to fail")

    out = capsys.readouterr().out
    assert rc == 2
    assert "Error: missing optional dependency" in out
    assert 'Install with: pip install "ocdocker[ml]"' in out


@pytest.mark.order(467)
def test_db_dependencies_available_handles_missing_modules(monkeypatch):
    def _import_module(name):
        if name == "sqlalchemy":
            raise ModuleNotFoundError("No module named 'sqlalchemy'")
        return object()

    monkeypatch.setattr(cli.importlib, "import_module", _import_module)

    ok, exc = cli._db_dependencies_available()
    assert ok is False
    assert isinstance(exc, ModuleNotFoundError)
    assert "sqlalchemy" in str(exc)
