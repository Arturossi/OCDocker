#!/usr/bin/env python3

# Description
###############################################################################
'''
Branch-oriented tests for Toolbox.Logging.
'''

# Imports
###############################################################################
import builtins
import logging
import os
import sys
import types

import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Logging as oclogging

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

## Public ##

@pytest.fixture(autouse=True)
def _reset_logging_state():
    logger = oclogging._STATE["logger"]
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    oclogging._STATE["configured"] = False
    oclogging._STATE["use_rich"] = False
    oclogging._STATE["to_stdout"] = True
    oclogging._STATE["stream_handler"] = None
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    oclogging._STATE["configured"] = False
    oclogging._STATE["stream_handler"] = None


@pytest.mark.order(116)
def test_default_logdir_prefers_config_and_falls_back(monkeypatch, tmp_path):
    cfg_mod = types.ModuleType("OCDocker.Config")

    class _Cfg:
        logdir = str(tmp_path / "configured_logs")

    cfg_mod.get_config = lambda: _Cfg()
    monkeypatch.setitem(sys.modules, "OCDocker.Config", cfg_mod)
    assert oclogging._default_logdir() == str(tmp_path / "configured_logs")

    broken_mod = types.ModuleType("OCDocker.Config")
    broken_mod.get_config = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "OCDocker.Config", broken_mod)

    fallback = oclogging._default_logdir()
    assert fallback.endswith(os.path.join("logs"))


@pytest.mark.order(117)
def test_ensure_configured_reuses_and_replaces_handlers():
    oclogging._ensure_configured(to_stdout=True, use_rich=False)
    logger = oclogging._STATE["logger"]
    first_handler = oclogging._STATE["stream_handler"]
    assert first_handler in logger.handlers

    oclogging._ensure_configured(to_stdout=True, use_rich=False)
    assert oclogging._STATE["stream_handler"] is first_handler

    oclogging._ensure_configured(to_stdout=False, use_rich=False)
    second_handler = oclogging._STATE["stream_handler"]
    assert second_handler is not first_handler
    assert first_handler not in logger.handlers


@pytest.mark.order(118)
def test_build_stream_handler_rich_import_failure_falls_back(monkeypatch):
    real_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):
        if name.startswith("rich"):
            raise ImportError("rich not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _patched_import)
    handler, rich_ok = oclogging._build_stream_handler(to_stdout=True, use_rich=True)
    assert isinstance(handler, logging.StreamHandler)
    assert rich_ok is False


@pytest.mark.order(119)
def test_build_stream_handler_rich_signature_error_branch(monkeypatch):
    rich_console = types.ModuleType("rich.console")
    rich_logging = types.ModuleType("rich.logging")
    rich_theme = types.ModuleType("rich.theme")

    class _Console:
        def __init__(self, file=None, theme=None):
            self.file = file
            self.theme = theme

    class _Theme(dict):
        pass

    class _RichHandler(logging.Handler):
        def __init__(self, console=None, **_kwargs):
            super().__init__()
            self.console = console

        def emit(self, record):
            _ = record

    rich_console.Console = _Console
    rich_theme.Theme = _Theme
    rich_logging.RichHandler = _RichHandler

    monkeypatch.setitem(sys.modules, "rich.console", rich_console)
    monkeypatch.setitem(sys.modules, "rich.logging", rich_logging)
    monkeypatch.setitem(sys.modules, "rich.theme", rich_theme)
    monkeypatch.setattr(
        oclogging.inspect,
        "signature",
        lambda _obj: (_ for _ in ()).throw(TypeError("no signature")),
    )

    handler, rich_ok = oclogging._build_stream_handler(to_stdout=True, use_rich=True)
    assert isinstance(handler, _RichHandler)
    assert rich_ok is True


@pytest.mark.order(120)
def test_backup_and_clear_logs_handle_os_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(oclogging, "_default_logdir", lambda: str(tmp_path))

    src = tmp_path / "job.log"
    src.write_text("x", encoding="utf-8")
    monkeypatch.setattr(oclogging.os, "rename", lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("deny")))
    oclogging.backup_log("job")
    assert src.exists()

    past_dir = tmp_path / "keep_past"
    other_dir = tmp_path / "other_folder"
    past_dir.mkdir()
    other_dir.mkdir()
    monkeypatch.setattr(oclogging.shutil, "rmtree", lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("deny")))
    oclogging.clear_past_logs()
    assert past_dir.exists()
    assert other_dir.exists()


@pytest.mark.order(121)
def test_configure_with_level_none_and_makedirs_failure(monkeypatch, tmp_path):
    log_file = tmp_path / "config.log"
    monkeypatch.setattr(oclogging.os, "makedirs", lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("deny")))
    monkeypatch.setattr(ocerror.Error, "get_output_level", lambda: ocerror.ReportLevel.INFO)

    oclogging.configure(level=None, log_file=str(log_file), to_stdout=True, use_rich=None)
    logger = oclogging.get_logger("cfg")
    logger.info("configured")

    assert log_file.exists()
    assert "configured" in log_file.read_text(encoding="utf-8")


@pytest.mark.order(122)
def test_set_level_from_report_ignores_handlers_without_setlevel():
    oclogging._ensure_configured(to_stdout=True, use_rich=False)
    logger = oclogging._STATE["logger"]

    class _NoLevel:
        pass

    dummy = _NoLevel()
    # First in the list, so the handlers after it must still get the level
    logger.handlers.insert(0, dummy)  # type: ignore[arg-type]
    logger.setLevel(logging.WARNING)
    try:
        oclogging.set_level_from_report(ocerror.ReportLevel.INFO)
        assert logger.level == logging.INFO
        real_handlers = [h for h in logger.handlers if h is not dummy]
        assert real_handlers and all(h.level == logging.INFO for h in real_handlers)
    finally:
        if dummy in logger.handlers:
            logger.handlers.remove(dummy)


@pytest.mark.order(123)
def test_configure_same_log_file_twice_writes_each_message_once(tmp_path):
    log_file = tmp_path / "run.log"
    logger = oclogging._STATE["logger"]
    try:
        for _ in range(3):
            oclogging.configure(level=ocerror.ReportLevel.INFO, log_file=str(log_file), to_stdout=True, use_rich=False)
        oclogging.get_logger("dup").warning("only once")

        assert log_file.read_text(encoding="utf-8").count("only once") == 1
    finally:
        for handler in [h for h in logger.handlers if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)]:
            logger.removeHandler(handler)
            handler.close()


@pytest.mark.order(124)
def test_backup_log_keeps_only_the_newest_archives(monkeypatch, tmp_path):
    monkeypatch.setattr(oclogging, "_default_logdir", lambda: str(tmp_path))
    monkeypatch.setattr(oclogging, "_log_limits", lambda: (0, 0, 3), raising=False)

    # Five runs within the same second, each archiving the previous report
    for run in range(5):
        report = tmp_path / "job.log"
        report.write_text(f"run {run}", encoding="utf-8")
        os.utime(report, (1_000_000 + run, 1_000_000 + run))
        oclogging.backup_log("job")

    archived = sorted((tmp_path / "read_log_past").iterdir())
    assert sorted(path.read_text(encoding="utf-8") for path in archived) == ["run 2", "run 3", "run 4"]


@pytest.mark.order(125)
def test_configure_log_file_rotates_by_size(monkeypatch, tmp_path):
    from logging.handlers import RotatingFileHandler

    monkeypatch.setattr(oclogging, "_log_limits", lambda: (1234, 2, 10), raising=False)
    log_file = tmp_path / "rotating.log"
    logger = oclogging._STATE["logger"]
    try:
        oclogging.configure(level=ocerror.ReportLevel.INFO, log_file=str(log_file), to_stdout=True, use_rich=False)
        handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)]
        assert len(handlers) == 1
        assert isinstance(handlers[0], RotatingFileHandler)
        assert (handlers[0].maxBytes, handlers[0].backupCount) == (1234, 2)
    finally:
        for handler in [h for h in logger.handlers if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)]:
            logger.removeHandler(handler)
            handler.close()
