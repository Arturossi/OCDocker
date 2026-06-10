#!/usr/bin/env python3

# Description
###############################################################################
'''
CLI helper coverage for lightweight helpers.

Usage:

pytest tests/test_cli_helpers.py
'''

# Imports
###############################################################################
import argparse
import pytest

from pathlib import Path

from OCDocker.CLI.common import _preparse_global_args, _require_file
from OCDocker.CLI.parser import build_parser
from OCDocker.CLI.pipeline import cmd_pipeline
from OCDocker.CLI.script import cmd_script

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

## Public ##

@pytest.mark.order(21)
def test_build_parser_subcommands_and_parse():
    parser = build_parser()
    # A couple of subcommands should parse cleanly
    ns = parser.parse_args(["version"])  # sets func
    assert callable(getattr(ns, "func", None))
    ns2 = parser.parse_args(["init-config"])  # also sets func
    assert callable(getattr(ns2, "func", None))


def test_build_parser_scheduler_runtime_options(tmp_path):
    parser = build_parser()
    tmp_dir = tmp_path / "snakemake_tmp"
    common = [
        "--receptor",
        "receptor.pdb",
        "--ligand",
        "ligand.smi",
        "--box",
        "box0.pdb",
        "--strict-engines",
        "--done-marker",
        "done.json",
    ]

    before = parser.parse_args([
        "--threads",
        "4",
        "--tmp-dir",
        str(tmp_dir),
        "pipeline",
        *common,
    ])
    after = parser.parse_args([
        "pipeline",
        "--threads",
        "4",
        "--tmp-dir",
        str(tmp_dir),
        *common,
    ])

    for ns in (before, after):
        assert ns.threads == 4
        assert ns.tmp_dir == str(tmp_dir)
        assert ns.strict_engines is True
        assert ns.done_marker == "done.json"


@pytest.mark.order(19)
def test_preparse_global_args_reads_scattered_flags(tmp_path):
    cfg = tmp_path / "OCDocker.cfg"
    tmp_dir = tmp_path / "job_tmp"
    argv = [
        "vs", "--engine", "vina", "--output-level", "4",
        "--conf", str(cfg), "--overwrite", "--no-stdout-log", "--no-splash",
        "--threads", "3", "--tmp-dir", str(tmp_dir),
        "--multiprocess", "-u",
    ]
    ns = _preparse_global_args(argv)
    assert ns.output_level == 4
    assert ns.config_file == str(cfg)
    assert ns.overwrite is True
    assert ns.no_stdout_log is True
    assert ns.no_splash is True
    assert ns.threads == 3
    assert ns.tmp_dir == str(tmp_dir)
    assert ns.multiprocess is True
    assert ns.update is True


@pytest.mark.order(20)
def test_require_file_valid_and_errors(tmp_path):
    # Valid path returns Path
    f = tmp_path / "ok.txt"
    f.write_text("x")
    p = _require_file(str(f), "--file")
    assert isinstance(p, Path)
    assert p.exists()

    # Missing path raises SystemExit with code 2
    with pytest.raises(SystemExit) as ei:
        _require_file(str(tmp_path / "missing.txt"), "--file")
    assert ei.value.code == 2

    # Ellipsis placeholder triggers exit 2
    with pytest.raises(SystemExit) as ei2:
        _require_file("…/placeholder", "--file")
    assert ei2.value.code == 2


@pytest.mark.order(67)
def test_cmd_script_requires_security_opt_in(tmp_path, monkeypatch):
    script = tmp_path / "script.py"
    script.write_text("print('hello')", encoding="utf-8")

    monkeypatch.delenv("OCDOCKER_ALLOW_SCRIPT_EXEC", raising=False)
    args = argparse.Namespace(
        script_file=str(script),
        script_args=[],
        allow_unsafe_exec=False,
    )

    rc = cmd_script(args)
    assert rc == 2


def _mk_script_args(script_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        script_file=str(script_path),
        script_args=[],
        allow_unsafe_exec=True,
        log_file=None,
        no_stdout_log=False,
    )


@pytest.mark.order(77)
def test_cmd_script_success_path(tmp_path, monkeypatch):
    script = tmp_path / "ok.py"
    script.write_text("value = 123\n", encoding="utf-8")

    monkeypatch.setattr("OCDocker.CLI.common._preparse_global_args", lambda argv: argparse.Namespace())
    monkeypatch.setattr("OCDocker.CLI.common._bootstrap_ocdocker_env", lambda ns: None)

    rc = cmd_script(_mk_script_args(script))
    assert rc == 0


@pytest.mark.order(78)
def test_cmd_script_syntax_error(tmp_path, monkeypatch):
    script = tmp_path / "bad.py"
    script.write_text("if True print('x')\n", encoding="utf-8")

    monkeypatch.setattr("OCDocker.CLI.common._preparse_global_args", lambda argv: argparse.Namespace())
    monkeypatch.setattr("OCDocker.CLI.common._bootstrap_ocdocker_env", lambda ns: None)

    rc = cmd_script(_mk_script_args(script))
    assert rc == 1


@pytest.mark.order(79)
def test_cmd_script_propagates_system_exit(tmp_path, monkeypatch):
    script = tmp_path / "exit.py"
    script.write_text("import sys\nsys.exit(7)\n", encoding="utf-8")

    monkeypatch.setattr("OCDocker.CLI.common._preparse_global_args", lambda argv: argparse.Namespace())
    monkeypatch.setattr("OCDocker.CLI.common._bootstrap_ocdocker_env", lambda ns: None)

    rc = cmd_script(_mk_script_args(script))
    assert rc == 7


@pytest.mark.order(80)
def test_cmd_script_handles_keyboard_interrupt(tmp_path, monkeypatch):
    script = tmp_path / "kb.py"
    script.write_text("raise KeyboardInterrupt()\n", encoding="utf-8")

    monkeypatch.setattr("OCDocker.CLI.common._preparse_global_args", lambda argv: argparse.Namespace())
    monkeypatch.setattr("OCDocker.CLI.common._bootstrap_ocdocker_env", lambda ns: None)

    rc = cmd_script(_mk_script_args(script))
    assert rc == 130


@pytest.mark.order(81)
def test_cmd_script_handles_runtime_exception(tmp_path, monkeypatch):
    script = tmp_path / "boom.py"
    script.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    monkeypatch.setattr("OCDocker.CLI.common._preparse_global_args", lambda argv: argparse.Namespace())
    monkeypatch.setattr("OCDocker.CLI.common._bootstrap_ocdocker_env", lambda ns: None)

    rc = cmd_script(_mk_script_args(script))
    assert rc == 1

def test_build_parser_pipeline_step_stages(tmp_path):
    parser = build_parser()
    outdir = tmp_path / "pipeline_steps"
    for stage in ["prepare", "dock", "collect", "cluster", "rescore", "export"]:
        argv = ["pipeline", stage, "--outdir", str(outdir)]
        if stage in {"prepare", "dock"}:
            argv.extend(["--receptor", "r.pdbqt", "--ligand", "l.pdbqt", "--box", "box.txt"])
        ns = parser.parse_args(argv)
        assert ns.stage == stage
        assert ns.outdir == str(outdir)


def test_pipeline_prepare_collect_export_artifact_stages(tmp_path):
    receptor = tmp_path / "receptor.pdbqt"
    ligand = tmp_path / "ligand.pdbqt"
    box = tmp_path / "box.txt"
    receptor.write_text("RECEPTOR\n", encoding="utf-8")
    ligand.write_text("LIGAND\n", encoding="utf-8")
    box.write_text("BOX\n", encoding="utf-8")
    outdir = tmp_path / "out"

    parser = build_parser()
    prepare_args = parser.parse_args([
        "pipeline",
        "prepare",
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--box",
        str(box),
        "--outdir",
        str(outdir),
        "--engines",
        "vina,smina",
    ])
    assert cmd_pipeline(prepare_args) == 0
    assert (outdir / "prepare_manifest.json").is_file()

    engine_dir = outdir / "vinaFiles"
    engine_dir.mkdir(parents=True)
    pose = engine_dir / "pose_1.pdbqt"
    pose.write_text("POSE\n", encoding="utf-8")
    dock_manifest = {
        "engine": "vina",
        "engine_dir": str(engine_dir),
        "status": "complete",
        "box": str(box),
        "config": str(engine_dir / "conf_vina.txt"),
        "prepared_receptor": str(outdir / "prepared_receptor.pdbqt"),
        "prepared_ligand": str(outdir / "prepared_ligand.pdbqt"),
        "poses": [str(pose)],
    }
    import json
    (engine_dir / "dock_manifest.json").write_text(json.dumps(dock_manifest), encoding="utf-8")

    collect_args = parser.parse_args(["pipeline", "collect", "--outdir", str(outdir)])
    assert cmd_pipeline(collect_args) == 0
    assert (outdir / "pose_inventory.csv").is_file()
    assert (outdir / "collect_manifest.json").is_file()

    export_args = parser.parse_args(["pipeline", "export", "--outdir", str(outdir)])
    assert cmd_pipeline(export_args) == 0
    assert (outdir / "summary.json").is_file()

