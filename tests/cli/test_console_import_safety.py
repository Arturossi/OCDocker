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
