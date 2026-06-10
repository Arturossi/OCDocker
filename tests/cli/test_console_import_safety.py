#!/usr/bin/env python3
"""Import safety and minimal behavior tests for OCDocker.Console."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_removed_ocdockerconsole_module_not_present():
    assert not (PROJECT_ROOT / "OCDockerConsole.py").exists()


def test_import_ocdocker_console_package_is_silent(capsys, monkeypatch):
    importlib.import_module("OCDocker.Console")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_console_main_exits_on_exit_command(monkeypatch):
    from OCDocker.Console import app as console_app

    inputs = iter(["help\n", "exit\n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    monkeypatch.setattr(console_app, "print_welcome_banner", lambda: None)
    monkeypatch.setattr(
        console_app,
        "build_namespace",
        lambda: {"print_args": lambda *a, **k: None},
    )

    code = console_app.run_console()
    assert code == 0


def test_help_command_prints_keywords(capsys, monkeypatch):
    from OCDocker.Console.session import run_interactive

    inputs = iter(["help\n", "exit\n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    run_interactive({})
    captured = capsys.readouterr()
    assert "help" in captured.out.lower()
    assert "exit" in captured.out.lower()


def test_ocdocker_help_lists_console_subcommand(capsys):
    from OCDocker.CLI import main as cli_main

    with pytest.raises(SystemExit) as exc:
        cli_main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "console" in captured.out

def test_console_print_args_and_clean_test_files(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    from OCDocker.Console import session

    cfg = SimpleNamespace(
        multiprocess=False,
        overwrite=True,
        output_level="BASIC",
        paths=SimpleNamespace(ocdb_path="/ocdb", pca_path="/pca"),
        logdir="/logs",
        oddt_models_dir="/oddt",
        vina=SimpleNamespace(
            executable="vina",
            scoring="vina",
            scoring_functions="vina",
            num_modes=9,
            energy_range=3,
            exhaustiveness=8,
        ),
        smina=SimpleNamespace(
            executable="smina",
            scoring="vinardo",
            scoring_functions="vinardo",
            num_modes=9,
            energy_range=3,
            exhaustiveness=8,
            custom_scoring="-",
            custom_atoms="-",
            local_only=False,
            minimize=False,
            randomize_only=False,
            minimize_iters=0,
            accurate_line=False,
            minimize_early_term=False,
            approximation="linear",
            factor=1.0,
            force_cap=0.0,
            user_grid="-",
            user_grid_lambda=0.0,
        ),
        plants=SimpleNamespace(
            executable="plants",
            cluster_structures=10,
            cluster_rmsd=2.0,
            search_speed="speed1",
            scoring="chemplp",
            scoring_functions="chemplp",
        ),
        gnina=SimpleNamespace(
            executable="gnina",
            exhaustiveness=8,
            num_modes=9,
            scoring="default",
            custom_scoring="-",
            custom_atoms="-",
            local_only=False,
            minimize=False,
            randomize_only=False,
            num_mc_steps=10,
            max_mc_steps=100,
            num_mc_saved=1,
            minimize_iters=0,
            simple_ascent=False,
            accurate_line=False,
            minimize_early_term=False,
            approximation="linear",
            factor=1.0,
            force_cap=0.0,
            user_grid="-",
            user_grid_lambda=0.0,
            no_gpu=True,
        ),
        tools=SimpleNamespace(
            obabel="obabel",
            pythonsh="pythonsh",
            prepare_ligand="prepare_ligand4.py",
            prepare_receptor="prepare_receptor4.py",
        ),
        oddt=SimpleNamespace(executable="oddt", seed=1, chunk_size=10, scoring_functions="rfscore"),
    )
    monkeypatch.setattr("OCDocker.Config.get_config", lambda: cfg)
    monkeypatch.setattr("OCDocker.Initialise.config_file", "cfg.yml", raising=False)
    monkeypatch.setattr("OCDocker.Initialise.update", False, raising=False)
    monkeypatch.setattr("OCDocker.Initialise.db_url", "sqlite:///db.sqlite", raising=False)
    monkeypatch.setattr("OCDocker.Initialise.optdb_url", "sqlite:///optuna.sqlite", raising=False)

    session.print_args("all")
    captured = capsys.readouterr().out
    assert "OCDocker Runtime Arguments" in captured
    assert "Docking Binaries" in captured
    assert "ODDT Parameters" in captured

    base_prot = tmp_path / "protein"
    base_lig = tmp_path / "ligands"
    base_dec = tmp_path / "decoys"
    base_can = tmp_path / "candidates"
    for base in (base_prot, base_lig, base_dec, base_can):
        base.mkdir()
    (base_prot / "receptor.pdb").write_text("keep", encoding="utf-8")
    (base_prot / "delete.pdbqt").write_text("delete", encoding="utf-8")
    for base in (base_lig, base_dec, base_can):
        item = base / "item"
        item.mkdir()
        (item / "ligand.smi").write_text("keep", encoding="utf-8")
        (item / "pose.pdbqt").write_text("delete", encoding="utf-8")
        boxes = item / "boxes"
        boxes.mkdir()
        tmp_dir = item / "tmp"
        tmp_dir.mkdir()

    session.clean_test_files(str(base_prot), str(base_lig), str(base_dec), str(base_can))

    assert (base_prot / "receptor.pdb").exists()
    assert not (base_prot / "delete.pdbqt").exists()
    for base in (base_lig, base_dec, base_can):
        item = base / "item"
        assert (item / "ligand.smi").exists()
        assert (item / "boxes").exists()
        assert not (item / "pose.pdbqt").exists()
        assert not (item / "tmp").exists()


def test_console_run_interactive_executes_python_and_handles_interrupts(monkeypatch, capsys):
    from OCDocker.Console import session

    inputs = iter(["value = 2", "print(value + 3)", "", KeyboardInterrupt(), "exit"])

    def fake_input(_prompt=""):
        item = next(inputs)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(session, "_setup_readline", lambda _namespace: None)
    monkeypatch.setattr(session, "_save_readline_history", lambda: None)

    rc = session.run_interactive({})

    assert rc == 0
    captured = capsys.readouterr().out
    assert "Launching OCDocker Console" in captured
    assert "5" in captured


def test_console_run_interactive_ipython_path(monkeypatch):
    import sys
    import types

    from OCDocker.Console import session

    calls = []
    ipython = types.ModuleType("IPython")
    ipython.embed = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm")

    rc = session.run_interactive({"x": 1}, use_ipython=True)

    assert rc == 0
    assert calls
    assert calls[0]["user_ns"]["x"] == 1
    assert calls[0]["colors"] == "Linux"

